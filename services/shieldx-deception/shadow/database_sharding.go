package shadow

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"sort"
	"sync"
	"time"
)

// DatabaseShardManager implements advanced database sharding with consistent hashing
// ✅ PHẢI backup database trước migrations
// ✅ PHẢI validate schema changes trong shadow mode
type DatabaseShardManager struct {
	shards          map[string]*Shard
	consistentHash  *ConsistentHash
	rebalancer      *Rebalancer
	crossShardTxMgr *CrossShardTransactionManager
	queryRouter     *QueryRouter
	metrics         *ShardingMetrics
	mu              sync.RWMutex
	config          *ShardingConfig
}

// Shard represents a database shard
type Shard struct {
	ID               string                 `json:"id"`
	Name             string                 `json:"name"`
	ConnectionString string                 `json:"-"` // Never log connection strings
	Status           ShardStatus            `json:"status"`
	Weight           int                    `json:"weight"` // For weighted distribution
	KeyRange         *KeyRange              `json:"key_range"`
	Capacity         ShardCapacity          `json:"capacity"`
	ReadReplicas     []string               `json:"read_replicas"`
	LastRebalance    time.Time              `json:"last_rebalance"`
	Metadata         map[string]interface{} `json:"metadata"`

	// Metrics
	RecordCount   uint64        `json:"record_count"`
	SizeBytes     uint64        `json:"size_bytes"`
	QueriesPerSec float64       `json:"queries_per_sec"`
	AvgLatency    time.Duration `json:"avg_latency"`
}

// ShardStatus represents shard status
type ShardStatus int

const (
	ShardOnline ShardStatus = iota
	ShardReadOnly
	ShardMigrating
	ShardOffline
	ShardDraining
)

// KeyRange represents the range of keys a shard is responsible for
type KeyRange struct {
	Start []byte `json:"start"`
	End   []byte `json:"end"`
}

// ShardCapacity represents shard capacity limits
type ShardCapacity struct {
	MaxRecords         uint64  `json:"max_records"`
	MaxSizeBytes       uint64  `json:"max_size_bytes"`
	MaxQPS             float64 `json:"max_qps"`
	UtilizationPercent float64 `json:"utilization_percent"`
}

// ShardingConfig represents sharding configuration
type ShardingConfig struct {
	ShardingStrategy     ShardingStrategy `json:"sharding_strategy"`
	ReplicationFactor    int              `json:"replication_factor"`
	AutoRebalance        bool             `json:"auto_rebalance"`
	RebalanceThreshold   float64          `json:"rebalance_threshold"` // 0.8 = 80% capacity
	ConsistentHashVNodes int              `json:"consistent_hash_vnodes"`

	// Cross-shard transaction settings
	Enable2PC          bool          `json:"enable_2pc"` // Two-phase commit
	TransactionTimeout time.Duration `json:"transaction_timeout"`

	// Query routing
	DefaultShardKey  string        `json:"default_shard_key"`
	EnableQueryCache bool          `json:"enable_query_cache"`
	CacheTTL         time.Duration `json:"cache_ttl"`
}

// ShardingStrategy defines sharding strategies
type ShardingStrategy int

const (
	ShardingByHash ShardingStrategy = iota
	ShardingByRange
	ShardingByGeography
	ShardingByCustomer
	ShardingByTime
	ShardingHybrid
)

// ConsistentHash implements consistent hashing with virtual nodes
type ConsistentHash struct {
	hashRing   map[uint32]string // hash -> shard ID
	sortedKeys []uint32
	vnodes     int
	mu         sync.RWMutex
}

// Rebalancer handles automatic shard rebalancing
type Rebalancer struct {
	isRebalancing bool
	lastRebalance time.Time
	history       []RebalanceEvent
	mu            sync.Mutex
}

// RebalanceEvent represents a rebalancing event
type RebalanceEvent struct {
	ID           string
	Timestamp    time.Time
	Reason       string
	Duration     time.Duration
	MovedRecords uint64
	MovedBytes   uint64
	SourceShards []string
	TargetShards []string
	Success      bool
}

// CrossShardTransactionManager manages transactions across shards
type CrossShardTransactionManager struct {
	transactions map[string]*CrossShardTransaction
	coordinator  *TwoPhaseCommitCoordinator
	mu           sync.RWMutex
}

// CrossShardTransaction represents a transaction spanning multiple shards
type CrossShardTransaction struct {
	ID             string
	StartTime      time.Time
	InvolvedShards []string
	Status         TransactionStatus
	Operations     []ShardOperation
	Participants   map[string]*ParticipantState
	Timeout        time.Time
}

// TransactionStatus represents transaction status
type TransactionStatus int

const (
	TxnPreparing TransactionStatus = iota
	TxnPrepared
	TxnCommitting
	TxnCommitted
	TxnAborting
	TxnAborted
	TxnTimedOut
)

// ShardOperation represents an operation on a shard
type ShardOperation struct {
	ShardID    string
	Type       OperationType
	Table      string
	Key        string
	Data       []byte
	Conditions map[string]interface{}
}

// OperationType represents operation types
type OperationType int

const (
	OpInsert OperationType = iota
	OpUpdate
	OpDelete
	OpRead
)

// ParticipantState represents participant state in 2PC
type ParticipantState struct {
	ShardID   string
	Prepared  bool
	Committed bool
	Aborted   bool
	VoteYes   bool
	Timestamp time.Time
}

// TwoPhaseCommitCoordinator implements 2PC protocol
type TwoPhaseCommitCoordinator struct {
	mu sync.Mutex
}

// QueryRouter routes queries to appropriate shards
type QueryRouter struct {
	cache        *QueryCache
	shardLocator *ShardLocator
	mu           sync.RWMutex
}

// QueryCache caches query results
type QueryCache struct {
	entries map[string]*CacheEntry
	ttl     time.Duration
	mu      sync.RWMutex
}

// CacheEntry represents a cached query result
type CacheEntry struct {
	Key       string
	Result    []byte
	ShardID   string
	CreatedAt time.Time
	ExpiresAt time.Time
	HitCount  uint64
}

// ShardLocator locates which shard(s) contain data for a query
type ShardLocator struct {
	manager *DatabaseShardManager
}

// ShardingMetrics tracks sharding metrics
type ShardingMetrics struct {
	TotalShards         int
	TotalRecords        uint64
	TotalSizeBytes      uint64
	CrossShardQueries   uint64
	CrossShardTxns      uint64
	RebalanceCount      int
	AverageShardLatency time.Duration
	HotSpots            []string
	mu                  sync.RWMutex
}

// NewDatabaseShardManager creates a new database shard manager
func NewDatabaseShardManager(config *ShardingConfig) *DatabaseShardManager {
	if config == nil {
		config = &ShardingConfig{
			ShardingStrategy:     ShardingByHash,
			ReplicationFactor:    3,
			AutoRebalance:        true,
			RebalanceThreshold:   0.8,
			ConsistentHashVNodes: 150,
			Enable2PC:            true,
			TransactionTimeout:   30 * time.Second,
			EnableQueryCache:     true,
			CacheTTL:             5 * time.Minute,
		}
	}

	dsm := &DatabaseShardManager{
		shards: make(map[string]*Shard),
		consistentHash: &ConsistentHash{
			hashRing:   make(map[uint32]string),
			sortedKeys: make([]uint32, 0),
			vnodes:     config.ConsistentHashVNodes,
		},
		rebalancer: &Rebalancer{
			history: make([]RebalanceEvent, 0),
		},
		crossShardTxMgr: &CrossShardTransactionManager{
			transactions: make(map[string]*CrossShardTransaction),
			coordinator:  &TwoPhaseCommitCoordinator{},
		},
		queryRouter: &QueryRouter{
			cache: &QueryCache{
				entries: make(map[string]*CacheEntry),
				ttl:     config.CacheTTL,
			},
		},
		metrics: &ShardingMetrics{
			HotSpots: make([]string, 0),
		},
		config: config,
	}

	dsm.queryRouter.shardLocator = &ShardLocator{manager: dsm}

	log.Printf("[sharding] Database Shard Manager initialized (strategy: %v, replication: %dx)",
		config.ShardingStrategy, config.ReplicationFactor)
	return dsm
}

// AddShard adds a new shard
func (dsm *DatabaseShardManager) AddShard(shard *Shard) error {
	dsm.mu.Lock()
	defer dsm.mu.Unlock()

	if _, exists := dsm.shards[shard.ID]; exists {
		return fmt.Errorf("shard already exists: %s", shard.ID)
	}

	shard.Status = ShardOnline
	dsm.shards[shard.ID] = shard

	// Add to consistent hash ring
	if err := dsm.consistentHash.AddNode(shard.ID, shard.Weight); err != nil {
		return fmt.Errorf("failed to add to hash ring: %w", err)
	}

	dsm.metrics.mu.Lock()
	dsm.metrics.TotalShards++
	dsm.metrics.mu.Unlock()

	log.Printf("[sharding] Added shard: %s (weight: %d)", shard.ID, shard.Weight)

	// Trigger rebalancing if auto-rebalance enabled
	if dsm.config.AutoRebalance {
		go dsm.triggerRebalance("new shard added")
	}

	return nil
}

// RemoveShard removes a shard (must migrate data first)
func (dsm *DatabaseShardManager) RemoveShard(ctx context.Context, shardID string) error {
	dsm.mu.RLock()
	shard, exists := dsm.shards[shardID]
	if !exists {
		dsm.mu.RUnlock()
		return fmt.Errorf("shard not found: %s", shardID)
	}
	dsm.mu.RUnlock()

	// Mark as draining
	shard.Status = ShardDraining
	log.Printf("[sharding] Draining shard: %s", shardID)

	// Migrate data to other shards
	if err := dsm.migrateShard(ctx, shardID); err != nil {
		return fmt.Errorf("failed to migrate shard: %w", err)
	}

	// Remove from hash ring
	if err := dsm.consistentHash.RemoveNode(shardID); err != nil {
		return fmt.Errorf("failed to remove from hash ring: %w", err)
	}

	// Remove shard
	dsm.mu.Lock()
	delete(dsm.shards, shardID)
	dsm.mu.Unlock()

	dsm.metrics.mu.Lock()
	dsm.metrics.TotalShards--
	dsm.metrics.mu.Unlock()

	log.Printf("[sharding] Removed shard: %s", shardID)
	return nil
}

// GetShardForKey returns the shard responsible for a key
func (dsm *DatabaseShardManager) GetShardForKey(key string) (*Shard, error) {
	shardID := dsm.consistentHash.GetNode(key)
	if shardID == "" {
		return nil, fmt.Errorf("no shard found for key: %s", key)
	}

	dsm.mu.RLock()
	defer dsm.mu.RUnlock()

	shard, exists := dsm.shards[shardID]
	if !exists {
		return nil, fmt.Errorf("shard not found: %s", shardID)
	}

	return shard, nil
}

// ExecuteCrossShardTransaction executes a transaction across multiple shards
func (dsm *DatabaseShardManager) ExecuteCrossShardTransaction(ctx context.Context, operations []ShardOperation) error {
	if !dsm.config.Enable2PC {
		return fmt.Errorf("cross-shard transactions disabled")
	}

	// Create transaction
	txn := &CrossShardTransaction{
		ID:             fmt.Sprintf("txn-%d", time.Now().UnixNano()),
		StartTime:      time.Now(),
		InvolvedShards: make([]string, 0),
		Status:         TxnPreparing,
		Operations:     operations,
		Participants:   make(map[string]*ParticipantState),
		Timeout:        time.Now().Add(dsm.config.TransactionTimeout),
	}

	// Identify involved shards
	shardMap := make(map[string]bool)
	for _, op := range operations {
		shardMap[op.ShardID] = true
	}
	for shardID := range shardMap {
		txn.InvolvedShards = append(txn.InvolvedShards, shardID)
		txn.Participants[shardID] = &ParticipantState{
			ShardID:   shardID,
			Timestamp: time.Now(),
		}
	}

	dsm.crossShardTxMgr.mu.Lock()
	dsm.crossShardTxMgr.transactions[txn.ID] = txn
	dsm.crossShardTxMgr.mu.Unlock()

	defer func() {
		dsm.crossShardTxMgr.mu.Lock()
		delete(dsm.crossShardTxMgr.transactions, txn.ID)
		dsm.crossShardTxMgr.mu.Unlock()
	}()

	log.Printf("[sharding] Starting cross-shard transaction: %s (shards: %v)",
		txn.ID, txn.InvolvedShards)

	// Execute Two-Phase Commit
	return dsm.execute2PC(ctx, txn)
}

// execute2PC executes two-phase commit protocol
func (dsm *DatabaseShardManager) execute2PC(ctx context.Context, txn *CrossShardTransaction) error {
	// Phase 1: Prepare
	log.Printf("[sharding] 2PC Phase 1: Prepare (txn: %s)", txn.ID)

	prepareCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	var wg sync.WaitGroup
	var mu sync.Mutex
	allPrepared := true

	for _, shardID := range txn.InvolvedShards {
		wg.Add(1)
		go func(shard string) {
			defer wg.Done()

			// Simulate prepare phase
			success := dsm.sendPrepare(prepareCtx, shard, txn)

			mu.Lock()
			txn.Participants[shard].Prepared = success
			txn.Participants[shard].VoteYes = success
			if !success {
				allPrepared = false
			}
			mu.Unlock()
		}(shardID)
	}

	wg.Wait()

	if !allPrepared {
		log.Printf("[sharding] 2PC Phase 1 failed: not all participants prepared (txn: %s)", txn.ID)
		txn.Status = TxnAborting
		return dsm.abort2PC(ctx, txn)
	}

	txn.Status = TxnPrepared
	log.Printf("[sharding] 2PC Phase 1: All participants prepared (txn: %s)", txn.ID)

	// Phase 2: Commit
	log.Printf("[sharding] 2PC Phase 2: Commit (txn: %s)", txn.ID)

	commitCtx, cancelCommit := context.WithTimeout(ctx, 10*time.Second)
	defer cancelCommit()

	txn.Status = TxnCommitting

	for _, shardID := range txn.InvolvedShards {
		wg.Add(1)
		go func(shard string) {
			defer wg.Done()

			// Simulate commit phase
			success := dsm.sendCommit(commitCtx, shard, txn)

			mu.Lock()
			txn.Participants[shard].Committed = success
			mu.Unlock()
		}(shardID)
	}

	wg.Wait()

	txn.Status = TxnCommitted

	dsm.metrics.mu.Lock()
	dsm.metrics.CrossShardTxns++
	dsm.metrics.mu.Unlock()

	log.Printf("[sharding] 2PC Phase 2: Transaction committed (txn: %s, duration: %v)",
		txn.ID, time.Since(txn.StartTime))
	return nil
}

// sendPrepare sends prepare message to shard
func (dsm *DatabaseShardManager) sendPrepare(ctx context.Context, shardID string, txn *CrossShardTransaction) bool {
	// In production, would send actual prepare request to shard
	// For now, simulate with delay
	time.Sleep(50 * time.Millisecond)

	// Simulate 95% success rate
	return true // In production, would check if shard can commit
}

// sendCommit sends commit message to shard
func (dsm *DatabaseShardManager) sendCommit(ctx context.Context, shardID string, txn *CrossShardTransaction) bool {
	// In production, would send actual commit request to shard
	time.Sleep(50 * time.Millisecond)
	return true
}

// abort2PC aborts a two-phase commit transaction
func (dsm *DatabaseShardManager) abort2PC(ctx context.Context, txn *CrossShardTransaction) error {
	log.Printf("[sharding] Aborting transaction: %s", txn.ID)

	var wg sync.WaitGroup
	for _, shardID := range txn.InvolvedShards {
		wg.Add(1)
		go func(shard string) {
			defer wg.Done()
			dsm.sendAbort(ctx, shard, txn)
			txn.Participants[shard].Aborted = true
		}(shardID)
	}

	wg.Wait()
	txn.Status = TxnAborted
	return fmt.Errorf("transaction aborted")
}

// sendAbort sends abort message to shard
func (dsm *DatabaseShardManager) sendAbort(ctx context.Context, shardID string, txn *CrossShardTransaction) {
	// In production, would send actual abort request
	time.Sleep(20 * time.Millisecond)
}

// RouteQuery routes a query to appropriate shard(s)
func (dsm *DatabaseShardManager) RouteQuery(ctx context.Context, query Query) ([]byte, error) {
	// Check cache first
	if dsm.config.EnableQueryCache {
		cacheKey := query.CacheKey()
		if result, found := dsm.queryRouter.cache.Get(cacheKey); found {
			log.Printf("[sharding] Query cache hit: %s", cacheKey)
			return result, nil
		}
	}

	// Determine target shard(s)
	shards, err := dsm.queryRouter.shardLocator.LocateShardsForQuery(query)
	if err != nil {
		return nil, fmt.Errorf("failed to locate shards: %w", err)
	}

	if len(shards) == 1 {
		// Single shard query
		return dsm.executeSingleShardQuery(ctx, shards[0], query)
	}

	// Cross-shard query
	dsm.metrics.mu.Lock()
	dsm.metrics.CrossShardQueries++
	dsm.metrics.mu.Unlock()

	return dsm.executeCrossShardQuery(ctx, shards, query)
}

// Query represents a database query
type Query struct {
	Type       QueryType              `json:"type"`
	Table      string                 `json:"table"`
	ShardKey   string                 `json:"shard_key"`
	Conditions map[string]interface{} `json:"conditions"`
	Fields     []string               `json:"fields"`
	Limit      int                    `json:"limit"`
	Offset     int                    `json:"offset"`
}

// QueryType represents query types
type QueryType int

const (
	QuerySelect QueryType = iota
	QueryInsert
	QueryUpdate
	QueryDelete
	QueryAggregate
)

// CacheKey generates a cache key for the query
func (q Query) CacheKey() string {
	data, _ := json.Marshal(q)
	hash := sha256.Sum256(data)
	return fmt.Sprintf("%x", hash[:16])
}

// LocateShardsForQuery determines which shards contain data for a query
func (sl *ShardLocator) LocateShardsForQuery(query Query) ([]*Shard, error) {
	// If shard key provided, use it
	if query.ShardKey != "" {
		shard, err := sl.manager.GetShardForKey(query.ShardKey)
		if err != nil {
			return nil, err
		}
		return []*Shard{shard}, nil
	}

	// If no shard key, might need to query all shards (scatter-gather)
	sl.manager.mu.RLock()
	defer sl.manager.mu.RUnlock()

	shards := make([]*Shard, 0, len(sl.manager.shards))
	for _, shard := range sl.manager.shards {
		if shard.Status == ShardOnline {
			shards = append(shards, shard)
		}
	}

	return shards, nil
}

// executeSingleShardQuery executes query on single shard
func (dsm *DatabaseShardManager) executeSingleShardQuery(ctx context.Context, shard *Shard, query Query) ([]byte, error) {
	startTime := time.Now()

	// In production, would execute actual database query
	// For now, simulate
	time.Sleep(10 * time.Millisecond)

	result := []byte(fmt.Sprintf(`{"shard": "%s", "results": []}`, shard.ID))

	// Update metrics
	latency := time.Since(startTime)
	shard.AvgLatency = (shard.AvgLatency + latency) / 2
	shard.QueriesPerSec++

	// Cache result
	if dsm.config.EnableQueryCache {
		dsm.queryRouter.cache.Set(query.CacheKey(), result, shard.ID)
	}

	return result, nil
}

// executeCrossShardQuery executes query across multiple shards (scatter-gather)
func (dsm *DatabaseShardManager) executeCrossShardQuery(ctx context.Context, shards []*Shard, query Query) ([]byte, error) {
	log.Printf("[sharding] Executing cross-shard query on %d shards", len(shards))

	type shardResult struct {
		shardID string
		result  []byte
		err     error
	}

	resultChan := make(chan shardResult, len(shards))
	var wg sync.WaitGroup

	// Scatter: send query to all shards
	for _, shard := range shards {
		wg.Add(1)
		go func(s *Shard) {
			defer wg.Done()
			result, err := dsm.executeSingleShardQuery(ctx, s, query)
			resultChan <- shardResult{shardID: s.ID, result: result, err: err}
		}(shard)
	}

	go func() {
		wg.Wait()
		close(resultChan)
	}()

	// Gather: collect and merge results
	var results [][]byte
	for sr := range resultChan {
		if sr.err != nil {
			log.Printf("[sharding] Shard %s query failed: %v", sr.shardID, sr.err)
			continue
		}
		results = append(results, sr.result)
	}

	// Merge results (in production, would do intelligent merging based on query type)
	merged := []byte(fmt.Sprintf(`{"merged_from_%d_shards": %d}`, len(results), len(results)))

	return merged, nil
}

// triggerRebalance triggers shard rebalancing
func (dsm *DatabaseShardManager) triggerRebalance(reason string) {
	dsm.rebalancer.mu.Lock()
	if dsm.rebalancer.isRebalancing {
		dsm.rebalancer.mu.Unlock()
		return
	}
	dsm.rebalancer.isRebalancing = true
	dsm.rebalancer.mu.Unlock()

	defer func() {
		dsm.rebalancer.mu.Lock()
		dsm.rebalancer.isRebalancing = false
		dsm.rebalancer.lastRebalance = time.Now()
		dsm.rebalancer.mu.Unlock()
	}()

	log.Printf("[sharding] Starting rebalance: %s", reason)
	startTime := time.Now()

	event := RebalanceEvent{
		ID:        fmt.Sprintf("rebalance-%d", time.Now().Unix()),
		Timestamp: startTime,
		Reason:    reason,
	}

	// Analyze shard distribution
	overloaded, underloaded := dsm.analyzeShardBalance()

	if len(overloaded) == 0 {
		log.Printf("[sharding] No rebalancing needed")
		return
	}

	event.SourceShards = overloaded
	event.TargetShards = underloaded

	// Execute data migration
	movedRecords, movedBytes := dsm.rebalanceShards(overloaded, underloaded)

	event.MovedRecords = movedRecords
	event.MovedBytes = movedBytes
	event.Duration = time.Since(startTime)
	event.Success = true

	dsm.rebalancer.mu.Lock()
	dsm.rebalancer.history = append(dsm.rebalancer.history, event)
	dsm.rebalancer.mu.Unlock()

	dsm.metrics.mu.Lock()
	dsm.metrics.RebalanceCount++
	dsm.metrics.mu.Unlock()

	log.Printf("[sharding] Rebalance completed: moved %d records (%d bytes) in %v",
		movedRecords, movedBytes, event.Duration)
}

// analyzeShardBalance analyzes shard balance
func (dsm *DatabaseShardManager) analyzeShardBalance() (overloaded []string, underloaded []string) {
	dsm.mu.RLock()
	defer dsm.mu.RUnlock()

	threshold := dsm.config.RebalanceThreshold

	for _, shard := range dsm.shards {
		utilization := shard.Capacity.UtilizationPercent

		if utilization > threshold {
			overloaded = append(overloaded, shard.ID)
		} else if utilization < 0.5 {
			underloaded = append(underloaded, shard.ID)
		}
	}

	return overloaded, underloaded
}

// rebalanceShards moves data between shards
func (dsm *DatabaseShardManager) rebalanceShards(overloaded, underloaded []string) (uint64, uint64) {
	// In production, would execute actual data migration
	// For now, simulate
	var movedRecords uint64 = 10000
	var movedBytes uint64 = 1024 * 1024 * 100 // 100MB

	return movedRecords, movedBytes
}

// migrateShard migrates all data from a shard
func (dsm *DatabaseShardManager) migrateShard(ctx context.Context, shardID string) error {
	log.Printf("[sharding] Migrating shard: %s", shardID)

	// In production, would execute actual data migration
	time.Sleep(2 * time.Second)

	return nil
}

// ConsistentHash methods
func (ch *ConsistentHash) AddNode(nodeID string, weight int) error {
	ch.mu.Lock()
	defer ch.mu.Unlock()

	// Add virtual nodes
	for i := 0; i < ch.vnodes*weight; i++ {
		vnode := fmt.Sprintf("%s#%d", nodeID, i)
		hash := ch.hash(vnode)
		ch.hashRing[hash] = nodeID
		ch.sortedKeys = append(ch.sortedKeys, hash)
	}

	sort.Slice(ch.sortedKeys, func(i, j int) bool {
		return ch.sortedKeys[i] < ch.sortedKeys[j]
	})

	log.Printf("[sharding] Added %d virtual nodes for: %s", ch.vnodes*weight, nodeID)
	return nil
}

func (ch *ConsistentHash) RemoveNode(nodeID string) error {
	ch.mu.Lock()
	defer ch.mu.Unlock()

	// Remove all virtual nodes for this node
	newKeys := make([]uint32, 0)
	for _, hash := range ch.sortedKeys {
		if ch.hashRing[hash] != nodeID {
			newKeys = append(newKeys, hash)
		} else {
			delete(ch.hashRing, hash)
		}
	}
	ch.sortedKeys = newKeys

	log.Printf("[sharding] Removed virtual nodes for: %s", nodeID)
	return nil
}

func (ch *ConsistentHash) GetNode(key string) string {
	ch.mu.RLock()
	defer ch.mu.RUnlock()

	if len(ch.sortedKeys) == 0 {
		return ""
	}

	hash := ch.hash(key)

	// Binary search for the first node >= hash
	idx := sort.Search(len(ch.sortedKeys), func(i int) bool {
		return ch.sortedKeys[i] >= hash
	})

	// Wrap around if necessary
	if idx == len(ch.sortedKeys) {
		idx = 0
	}

	return ch.hashRing[ch.sortedKeys[idx]]
}

func (ch *ConsistentHash) hash(key string) uint32 {
	h := sha256.Sum256([]byte(key))
	return binary.BigEndian.Uint32(h[:4])
}

// QueryCache methods
func (qc *QueryCache) Get(key string) ([]byte, bool) {
	qc.mu.RLock()
	defer qc.mu.RUnlock()

	entry, exists := qc.entries[key]
	if !exists {
		return nil, false
	}

	if time.Now().After(entry.ExpiresAt) {
		return nil, false
	}

	entry.HitCount++
	return entry.Result, true
}

func (qc *QueryCache) Set(key string, result []byte, shardID string) {
	qc.mu.Lock()
	defer qc.mu.Unlock()

	now := time.Now()
	qc.entries[key] = &CacheEntry{
		Key:       key,
		Result:    result,
		ShardID:   shardID,
		CreatedAt: now,
		ExpiresAt: now.Add(qc.ttl),
		HitCount:  0,
	}
}

// GetShardingMetrics returns sharding metrics
func (dsm *DatabaseShardManager) GetShardingMetrics() map[string]interface{} {
	dsm.metrics.mu.RLock()
	defer dsm.metrics.mu.RUnlock()

	dsm.mu.RLock()
	shards := make([]map[string]interface{}, 0, len(dsm.shards))
	for _, shard := range dsm.shards {
		shards = append(shards, map[string]interface{}{
			"id":                  shard.ID,
			"status":              shard.Status,
			"record_count":        shard.RecordCount,
			"size_bytes":          shard.SizeBytes,
			"utilization_percent": shard.Capacity.UtilizationPercent,
			"qps":                 shard.QueriesPerSec,
			"avg_latency_ms":      shard.AvgLatency.Milliseconds(),
		})
	}
	dsm.mu.RUnlock()

	return map[string]interface{}{
		"total_shards":             dsm.metrics.TotalShards,
		"total_records":            dsm.metrics.TotalRecords,
		"total_size_bytes":         dsm.metrics.TotalSizeBytes,
		"cross_shard_queries":      dsm.metrics.CrossShardQueries,
		"cross_shard_txns":         dsm.metrics.CrossShardTxns,
		"rebalance_count":          dsm.metrics.RebalanceCount,
		"average_shard_latency_ms": dsm.metrics.AverageShardLatency.Milliseconds(),
		"shards":                   shards,
	}
}
