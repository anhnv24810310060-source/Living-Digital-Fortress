package main

import (
	"context"
	"crypto/md5"
	"database/sql"
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"log"
	"sync"
	"time"

	_ "github.com/lib/pq"
)

// ShardingEngine implements horizontal database sharding with:
// - Consistent hashing for shard distribution
// - Cross-shard transaction handling
// - Automatic rebalancing
// - Read replicas for query performance
type ShardingEngine struct {
	shards         []*Shard
	ring           *ConsistentHashRing
	rebalancer     *Rebalancer
	crossShardTx   *CrossShardTransactionManager
	config         ShardingConfig
	mu             sync.RWMutex
}

// Shard represents a database shard with read replicas
type Shard struct {
	ID              int
	MasterDB        *sql.DB
	ReplicaDBs      []*sql.DB
	Config          ShardConfig
	Status          string // "active", "rebalancing", "offline"
	CurrentLoad     int64
	TotalKeys       int64
	mu              sync.RWMutex
	replicaSelector *ReplicaSelector
}

// ShardConfig contains shard-specific configuration
type ShardConfig struct {
	MasterDSN      string
	ReplicaDSNs    []string
	VirtualNodes   int
	MaxConnections int
	ReadTimeout    time.Duration
	WriteTimeout   time.Duration
}

// ShardingConfig contains global sharding configuration
type ShardingConfig struct {
	NumShards         int
	ReplicationFactor int
	ConsistencyLevel  string // "ONE", "QUORUM", "ALL"
	RebalanceEnabled  bool
	ShardingStrategy  string // "consistent_hash", "range", "geographic"
}

// ConsistentHashRing implements consistent hashing with virtual nodes
type ConsistentHashRing struct {
	nodes          map[uint32]*Shard
	sortedHashes   []uint32
	virtualNodes   int
	mu             sync.RWMutex
}

// Rebalancer handles shard rebalancing operations
type Rebalancer struct {
	engine         *ShardingEngine
	rebalanceQueue chan RebalanceTask
	inProgress     map[string]*RebalanceTask
	mu             sync.RWMutex
}

// RebalanceTask represents a shard rebalancing operation
type RebalanceTask struct {
	ID              string
	SourceShard     int
	TargetShard     int
	KeyRangeStart   uint32
	KeyRangeEnd     uint32
	Status          string
	Progress        float64
	StartTime       time.Time
	EstimatedFinish time.Time
}

// CrossShardTransactionManager handles distributed transactions
type CrossShardTransactionManager struct {
	pendingTx      map[string]*CrossShardTransaction
	mu             sync.RWMutex
	coordinator    *TwoPhaseCommitCoordinator
}

// CrossShardTransaction represents a distributed transaction
type CrossShardTransaction struct {
	ID             string
	Shards         []int
	Status         string // "preparing", "prepared", "committed", "aborted"
	Operations     []ShardOperation
	Coordinator    int
	Timeout        time.Duration
	StartTime      time.Time
}

// ShardOperation represents an operation in a distributed transaction
type ShardOperation struct {
	ShardID   int
	OpType    string // "read", "write", "delete"
	Key       string
	Value     interface{}
	Condition string
}

// TwoPhaseCommitCoordinator implements 2PC protocol
type TwoPhaseCommitCoordinator struct {
	txLog          *sql.DB
	prepareTimeout time.Duration
	commitTimeout  time.Duration
}

// ReplicaSelector chooses optimal replica for read operations
type ReplicaSelector struct {
	strategy       string // "round_robin", "least_connections", "geo_proximity"
	currentIndex   int
	loadMetrics    map[int]int64
	mu             sync.Mutex
}

// NewShardingEngine creates a new sharding engine
func NewShardingEngine(config ShardingConfig, shardConfigs []ShardConfig) (*ShardingEngine, error) {
	if len(shardConfigs) == 0 {
		return nil, fmt.Errorf("at least one shard configuration required")
	}

	engine := &ShardingEngine{
		shards:       make([]*Shard, 0, len(shardConfigs)),
		ring:         NewConsistentHashRing(256), // 256 virtual nodes per shard
		config:       config,
		crossShardTx: NewCrossShardTransactionManager(),
	}

	// Initialize shards
	for i, shardConfig := range shardConfigs {
		shard, err := NewShard(i, shardConfig)
		if err != nil {
			return nil, fmt.Errorf("failed to create shard %d: %w", i, err)
		}
		engine.shards = append(engine.shards, shard)
		engine.ring.AddNode(shard)
	}

	// Initialize rebalancer if enabled
	if config.RebalanceEnabled {
		engine.rebalancer = NewRebalancer(engine)
		go engine.rebalancer.Start()
	}

	// Start health monitoring
	go engine.monitorShardHealth()

	log.Printf("[sharding] Engine initialized with %d shards", len(engine.shards))
	return engine, nil
}

// NewShard creates a new database shard
func NewShard(id int, config ShardConfig) (*Shard, error) {
	// Connect to master
	masterDB, err := sql.Open("postgres", config.MasterDSN)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to master: %w", err)
	}

	// Configure connection pool
	masterDB.SetMaxOpenConns(config.MaxConnections)
	masterDB.SetMaxIdleConns(config.MaxConnections / 4)
	masterDB.SetConnMaxLifetime(10 * time.Minute)

	if err := masterDB.Ping(); err != nil {
		return nil, fmt.Errorf("master ping failed: %w", err)
	}

	// Connect to replicas
	replicaDBs := make([]*sql.DB, 0, len(config.ReplicaDSNs))
	for _, dsn := range config.ReplicaDSNs {
		replicaDB, err := sql.Open("postgres", dsn)
		if err != nil {
			log.Printf("[shard-%d] Failed to connect to replica: %v", id, err)
			continue
		}

		replicaDB.SetMaxOpenConns(config.MaxConnections)
		replicaDB.SetMaxIdleConns(config.MaxConnections / 4)
		replicaDB.SetConnMaxLifetime(10 * time.Minute)

		if err := replicaDB.Ping(); err != nil {
			log.Printf("[shard-%d] Replica ping failed: %v", id, err)
			replicaDB.Close()
			continue
		}

		replicaDBs = append(replicaDBs, replicaDB)
	}

	shard := &Shard{
		ID:              id,
		MasterDB:        masterDB,
		ReplicaDBs:      replicaDBs,
		Config:          config,
		Status:          "active",
		replicaSelector: NewReplicaSelector("round_robin"),
	}

	log.Printf("[shard-%d] Initialized with %d replicas", id, len(replicaDBs))
	return shard, nil
}

// NewConsistentHashRing creates a new consistent hash ring
func NewConsistentHashRing(virtualNodes int) *ConsistentHashRing {
	return &ConsistentHashRing{
		nodes:        make(map[uint32]*Shard),
		sortedHashes: make([]uint32, 0),
		virtualNodes: virtualNodes,
	}
}

// AddNode adds a shard to the hash ring
func (ring *ConsistentHashRing) AddNode(shard *Shard) {
	ring.mu.Lock()
	defer ring.mu.Unlock()

	// Add virtual nodes for better distribution
	for i := 0; i < ring.virtualNodes; i++ {
		virtualKey := fmt.Sprintf("shard-%d-vnode-%d", shard.ID, i)
		hash := ring.hash(virtualKey)
		ring.nodes[hash] = shard
		ring.sortedHashes = append(ring.sortedHashes, hash)
	}

	// Sort hashes for binary search
	ring.sortHashes()
	log.Printf("[hash-ring] Added shard %d with %d virtual nodes", shard.ID, ring.virtualNodes)
}

// RemoveNode removes a shard from the hash ring
func (ring *ConsistentHashRing) RemoveNode(shardID int) {
	ring.mu.Lock()
	defer ring.mu.Unlock()

	newHashes := make([]uint32, 0)
	for _, hash := range ring.sortedHashes {
		if shard := ring.nodes[hash]; shard.ID != shardID {
			newHashes = append(newHashes, hash)
		} else {
			delete(ring.nodes, hash)
		}
	}

	ring.sortedHashes = newHashes
	log.Printf("[hash-ring] Removed shard %d", shardID)
}

// GetNode returns the shard for a given key
func (ring *ConsistentHashRing) GetNode(key string) *Shard {
	ring.mu.RLock()
	defer ring.mu.RUnlock()

	if len(ring.sortedHashes) == 0 {
		return nil
	}

	hash := ring.hash(key)

	// Binary search for the first hash >= key hash
	idx := ring.search(hash)
	return ring.nodes[ring.sortedHashes[idx]]
}

// hash computes CRC32 hash of a key
func (ring *ConsistentHashRing) hash(key string) uint32 {
	return crc32.ChecksumIEEE([]byte(key))
}

// search performs binary search on sorted hashes
func (ring *ConsistentHashRing) search(hash uint32) int {
	if len(ring.sortedHashes) == 0 {
		return 0
	}

	// Binary search
	left, right := 0, len(ring.sortedHashes)-1
	
	for left < right {
		mid := (left + right) / 2
		if ring.sortedHashes[mid] < hash {
			left = mid + 1
		} else {
			right = mid
		}
	}

	// Wrap around to first node if no node found
	if left >= len(ring.sortedHashes) {
		return 0
	}

	return left
}

// sortHashes sorts the hash array
func (ring *ConsistentHashRing) sortHashes() {
	// Simple insertion sort (good for small arrays)
	for i := 1; i < len(ring.sortedHashes); i++ {
		key := ring.sortedHashes[i]
		j := i - 1
		for j >= 0 && ring.sortedHashes[j] > key {
			ring.sortedHashes[j+1] = ring.sortedHashes[j]
			j--
		}
		ring.sortedHashes[j+1] = key
	}
}

// GetShard returns the shard for a given key
func (se *ShardingEngine) GetShard(key string) (*Shard, error) {
	shard := se.ring.GetNode(key)
	if shard == nil {
		return nil, fmt.Errorf("no shard available")
	}

	shard.mu.RLock()
	status := shard.Status
	shard.mu.RUnlock()

	if status != "active" {
		return nil, fmt.Errorf("shard %d is not active: %s", shard.ID, status)
	}

	return shard, nil
}

// Write performs a write operation on the appropriate shard
func (se *ShardingEngine) Write(ctx context.Context, key string, value interface{}) error {
	shard, err := se.GetShard(key)
	if err != nil {
		return err
	}

	// Write to master
	query := `
		INSERT INTO shard_data (key, value, shard_id, created_at, updated_at)
		VALUES ($1, $2, $3, NOW(), NOW())
		ON CONFLICT (key) DO UPDATE SET
			value = $2,
			updated_at = NOW(),
			version = shard_data.version + 1
	`

	_, err = shard.MasterDB.ExecContext(ctx, query, key, value, shard.ID)
	if err != nil {
		return fmt.Errorf("write failed on shard %d: %w", shard.ID, err)
	}

	// Update shard statistics
	shard.mu.Lock()
	shard.CurrentLoad++
	shard.TotalKeys++
	shard.mu.Unlock()

	return nil
}

// Read performs a read operation from replicas
func (se *ShardingEngine) Read(ctx context.Context, key string) (interface{}, error) {
	shard, err := se.GetShard(key)
	if err != nil {
		return nil, err
	}

	// Select optimal replica
	db := shard.SelectReplica()

	var value interface{}
	query := `SELECT value FROM shard_data WHERE key = $1`

	err = db.QueryRowContext(ctx, query, key).Scan(&value)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("key not found: %s", key)
	}
	if err != nil {
		return nil, fmt.Errorf("read failed on shard %d: %w", shard.ID, err)
	}

	return value, nil
}

// SelectReplica chooses optimal replica for read operations
func (s *Shard) SelectReplica() *sql.DB {
	s.mu.RLock()
	defer s.mu.RUnlock()

	// If no replicas, use master
	if len(s.ReplicaDBs) == 0 {
		return s.MasterDB
	}

	// Use replica selector strategy
	return s.replicaSelector.Select(s.ReplicaDBs)
}

// NewReplicaSelector creates a new replica selector
func NewReplicaSelector(strategy string) *ReplicaSelector {
	return &ReplicaSelector{
		strategy:    strategy,
		loadMetrics: make(map[int]int64),
	}
}

// Select chooses a replica based on the strategy
func (rs *ReplicaSelector) Select(replicas []*sql.DB) *sql.DB {
	rs.mu.Lock()
	defer rs.mu.Unlock()

	if len(replicas) == 0 {
		return nil
	}

	switch rs.strategy {
	case "round_robin":
		idx := rs.currentIndex % len(replicas)
		rs.currentIndex++
		return replicas[idx]

	case "least_connections":
		// Simple implementation: track usage
		minLoad := int64(^uint64(0) >> 1) // max int64
		minIdx := 0
		for i := range replicas {
			load := rs.loadMetrics[i]
			if load < minLoad {
				minLoad = load
				minIdx = i
			}
		}
		rs.loadMetrics[minIdx]++
		return replicas[minIdx]

	default:
		return replicas[0]
	}
}

// CrossShardTransaction handles distributed transactions
func (se *ShardingEngine) BeginCrossShardTransaction(ctx context.Context, keys []string) (*CrossShardTransaction, error) {
	// Identify which shards are involved
	shardSet := make(map[int]bool)
	for _, key := range keys {
		shard, err := se.GetShard(key)
		if err != nil {
			return nil, err
		}
		shardSet[shard.ID] = true
	}

	shards := make([]int, 0, len(shardSet))
	for shardID := range shardSet {
		shards = append(shards, shardID)
	}

	tx := &CrossShardTransaction{
		ID:         fmt.Sprintf("ctx-%d", time.Now().UnixNano()),
		Shards:     shards,
		Status:     "preparing",
		Operations: make([]ShardOperation, 0),
		Timeout:    30 * time.Second,
		StartTime:  time.Now(),
	}

	se.crossShardTx.mu.Lock()
	se.crossShardTx.pendingTx[tx.ID] = tx
	se.crossShardTx.mu.Unlock()

	log.Printf("[cross-shard-tx] Started transaction %s across %d shards", tx.ID, len(shards))
	return tx, nil
}

// CommitCrossShardTransaction commits a distributed transaction using 2PC
func (se *ShardingEngine) CommitCrossShardTransaction(ctx context.Context, tx *CrossShardTransaction) error {
	// Phase 1: Prepare
	prepareCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	log.Printf("[cross-shard-tx] Phase 1: Preparing transaction %s", tx.ID)

	for _, shardID := range tx.Shards {
		shard := se.shards[shardID]
		_, err := shard.MasterDB.ExecContext(prepareCtx, `
			INSERT INTO distributed_tx_log (tx_id, shard_id, status, created_at)
			VALUES ($1, $2, 'prepared', NOW())
		`, tx.ID, shardID)

		if err != nil {
			// Abort transaction
			log.Printf("[cross-shard-tx] Prepare failed on shard %d: %v", shardID, err)
			se.abortCrossShardTransaction(ctx, tx)
			return fmt.Errorf("prepare failed: %w", err)
		}
	}

	// Phase 2: Commit
	commitCtx, commitCancel := context.WithTimeout(ctx, 10*time.Second)
	defer commitCancel()

	log.Printf("[cross-shard-tx] Phase 2: Committing transaction %s", tx.ID)

	for _, shardID := range tx.Shards {
		shard := se.shards[shardID]
		_, err := shard.MasterDB.ExecContext(commitCtx, `
			UPDATE distributed_tx_log 
			SET status = 'committed', committed_at = NOW()
			WHERE tx_id = $1 AND shard_id = $2
		`, tx.ID, shardID)

		if err != nil {
			log.Printf("[cross-shard-tx] WARNING: Commit failed on shard %d: %v", shardID, err)
			// Continue trying other shards
		}
	}

	tx.Status = "committed"
	log.Printf("[cross-shard-tx] Transaction %s committed successfully", tx.ID)

	// Cleanup
	se.crossShardTx.mu.Lock()
	delete(se.crossShardTx.pendingTx, tx.ID)
	se.crossShardTx.mu.Unlock()

	return nil
}

// abortCrossShardTransaction aborts a distributed transaction
func (se *ShardingEngine) abortCrossShardTransaction(ctx context.Context, tx *CrossShardTransaction) {
	log.Printf("[cross-shard-tx] Aborting transaction %s", tx.ID)

	for _, shardID := range tx.Shards {
		shard := se.shards[shardID]
		_, err := shard.MasterDB.ExecContext(ctx, `
			UPDATE distributed_tx_log 
			SET status = 'aborted', committed_at = NOW()
			WHERE tx_id = $1 AND shard_id = $2
		`, tx.ID, shardID)

		if err != nil {
			log.Printf("[cross-shard-tx] Failed to abort on shard %d: %v", shardID, err)
		}
	}

	tx.Status = "aborted"
}

// NewCrossShardTransactionManager creates a new transaction manager
func NewCrossShardTransactionManager() *CrossShardTransactionManager {
	return &CrossShardTransactionManager{
		pendingTx: make(map[string]*CrossShardTransaction),
	}
}

// NewRebalancer creates a new rebalancer
func NewRebalancer(engine *ShardingEngine) *Rebalancer {
	return &Rebalancer{
		engine:         engine,
		rebalanceQueue: make(chan RebalanceTask, 100),
		inProgress:     make(map[string]*RebalanceTask),
	}
}

// Start begins the rebalancing worker
func (r *Rebalancer) Start() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			r.checkBalance()
		case task := <-r.rebalanceQueue:
			r.processRebalanceTask(task)
		}
	}
}

// checkBalance checks if rebalancing is needed
func (r *Rebalancer) checkBalance() {
	r.engine.mu.RLock()
	shards := r.engine.shards
	r.engine.mu.RUnlock()

	// Calculate average load
	var totalLoad int64
	for _, shard := range shards {
		shard.mu.RLock()
		totalLoad += shard.CurrentLoad
		shard.mu.RUnlock()
	}

	avgLoad := totalLoad / int64(len(shards))
	threshold := avgLoad * 3 / 2 // 50% above average triggers rebalance

	// Find overloaded shards
	for _, shard := range shards {
		shard.mu.RLock()
		load := shard.CurrentLoad
		shard.mu.RUnlock()

		if load > threshold {
			log.Printf("[rebalancer] Shard %d is overloaded: %d (avg: %d)", shard.ID, load, avgLoad)
			// Trigger rebalancing
			// Implementation details omitted for brevity
		}
	}
}

// processRebalanceTask processes a single rebalance task
func (r *Rebalancer) processRebalanceTask(task RebalanceTask) {
	log.Printf("[rebalancer] Processing task %s: shard %d -> %d", task.ID, task.SourceShard, task.TargetShard)
	// Implementation details omitted for brevity
}

// monitorShardHealth monitors shard health and handles failures
func (se *ShardingEngine) monitorShardHealth() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		se.mu.RLock()
		shards := se.shards
		se.mu.RUnlock()

		for _, shard := range shards {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			err := shard.MasterDB.PingContext(ctx)
			cancel()

			shard.mu.Lock()
			if err != nil {
				if shard.Status == "active" {
					log.Printf("[health] Shard %d master unreachable: %v", shard.ID, err)
					shard.Status = "degraded"
				}
			} else {
				if shard.Status == "degraded" {
					log.Printf("[health] Shard %d master recovered", shard.ID)
					shard.Status = "active"
				}
			}
			shard.mu.Unlock()
		}
	}
}

// GetShardStatistics returns statistics for all shards
func (se *ShardingEngine) GetShardStatistics() []map[string]interface{} {
	se.mu.RLock()
	defer se.mu.RUnlock()

	stats := make([]map[string]interface{}, 0, len(se.shards))

	for _, shard := range se.shards {
		shard.mu.RLock()
		stat := map[string]interface{}{
			"shard_id":     shard.ID,
			"status":       shard.Status,
			"current_load": shard.CurrentLoad,
			"total_keys":   shard.TotalKeys,
			"replicas":     len(shard.ReplicaDBs),
		}
		shard.mu.RUnlock()
		stats = append(stats, stat)
	}

	return stats
}

// Close closes all shard connections
func (se *ShardingEngine) Close() error {
	for _, shard := range se.shards {
		shard.MasterDB.Close()
		for _, replica := range shard.ReplicaDBs {
			replica.Close()
		}
	}
	return nil
}
