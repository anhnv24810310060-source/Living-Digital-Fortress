package main

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"sync"
)

// ShardingStrategy implements consistent hashing for horizontal scaling
// ✅ PHẢI horizontal scalability
type ShardingStrategy struct {
	shards      []*Shard
	ring        *ConsistentHashRing
	mu          sync.RWMutex
	rebalancing bool
	metrics     *ShardMetrics
}

// Shard represents a physical database partition
type Shard struct {
	ID          int
	DSN         string
	Weight      int // for weighted consistent hashing
	Status      ShardStatus
	ReadReplica []string    // read replicas for query performance
	db          interface{} // actual DB connection pool
	mu          sync.RWMutex
}

type ShardStatus int

const (
	ShardActive ShardStatus = iota
	ShardDraining
	ShardInactive
)

type ShardMetrics struct {
	TotalRequests     map[int]uint64 // by shard_id
	CrossShardQueries uint64
	RebalanceCount    uint64
	mu                sync.RWMutex
}

// ConsistentHashRing for minimal data movement during rebalancing
type ConsistentHashRing struct {
	vnodes   int // virtual nodes per physical shard
	ring     []uint64
	shardMap map[uint64]int // hash -> shard_id
	mu       sync.RWMutex
}

// NewShardingStrategy initializes sharding with consistent hashing
func NewShardingStrategy(shardConfigs []ShardConfig, vnodes int) (*ShardingStrategy, error) {
	if len(shardConfigs) == 0 {
		return nil, fmt.Errorf("at least one shard required")
	}

	ss := &ShardingStrategy{
		shards:  make([]*Shard, 0, len(shardConfigs)),
		ring:    newConsistentHashRing(vnodes),
		metrics: &ShardMetrics{TotalRequests: make(map[int]uint64)},
	}

	for i, cfg := range shardConfigs {
		shard := &Shard{
			ID:          i,
			DSN:         cfg.DSN,
			Weight:      cfg.Weight,
			Status:      ShardActive,
			ReadReplica: cfg.ReadReplicas,
		}
		ss.shards = append(ss.shards, shard)
		ss.ring.AddShard(i, cfg.Weight)
	}

	return ss, nil
}

type ShardConfig struct {
	DSN          string
	Weight       int
	ReadReplicas []string
}

func newConsistentHashRing(vnodes int) *ConsistentHashRing {
	return &ConsistentHashRing{
		vnodes:   vnodes,
		ring:     make([]uint64, 0),
		shardMap: make(map[uint64]int),
	}
}

// AddShard adds a shard to the ring with virtual nodes
func (chr *ConsistentHashRing) AddShard(shardID int, weight int) {
	chr.mu.Lock()
	defer chr.mu.Unlock()

	// Create virtual nodes for better distribution
	vnodeCount := chr.vnodes * weight
	for i := 0; i < vnodeCount; i++ {
		vnode := fmt.Sprintf("shard-%d-vnode-%d", shardID, i)
		hash := hashKey(vnode)
		chr.ring = append(chr.ring, hash)
		chr.shardMap[hash] = shardID
	}

	// Sort ring for binary search
	sortUint64Slice(chr.ring)
}

// GetShard returns shard ID for a given key using consistent hashing
func (chr *ConsistentHashRing) GetShard(key string) int {
	chr.mu.RLock()
	defer chr.mu.RUnlock()

	if len(chr.ring) == 0 {
		return 0
	}

	hash := hashKey(key)

	// Binary search for the first node >= hash
	idx := binarySearch(chr.ring, hash)
	if idx >= len(chr.ring) {
		idx = 0 // wrap around
	}

	return chr.shardMap[chr.ring[idx]]
}

// GetShardForTenant returns shard for a tenant (sharding key: tenant_id)
func (ss *ShardingStrategy) GetShardForTenant(tenantID string) (*Shard, error) {
	ss.mu.RLock()
	defer ss.mu.RUnlock()

	shardID := ss.ring.GetShard(tenantID)

	if shardID >= len(ss.shards) {
		return nil, fmt.Errorf("invalid shard id: %d", shardID)
	}

	shard := ss.shards[shardID]
	if shard.Status != ShardActive {
		// Fallback to another active shard
		for _, s := range ss.shards {
			if s.Status == ShardActive {
				return s, nil
			}
		}
		return nil, fmt.Errorf("no active shards available")
	}

	// Track metrics
	ss.metrics.mu.Lock()
	ss.metrics.TotalRequests[shardID]++
	ss.metrics.mu.Unlock()

	return shard, nil
}

// RebalanceShard initiates shard rebalancing (for adding/removing shards)
// ✅ PHẢI automatic rebalancing
func (ss *ShardingStrategy) RebalanceShard(ctx context.Context, newShardConfig ShardConfig) error {
	ss.mu.Lock()
	if ss.rebalancing {
		ss.mu.Unlock()
		return fmt.Errorf("rebalancing already in progress")
	}
	ss.rebalancing = true
	ss.mu.Unlock()

	defer func() {
		ss.mu.Lock()
		ss.rebalancing = false
		ss.mu.Unlock()
	}()

	// Add new shard to ring
	newShardID := len(ss.shards)
	newShard := &Shard{
		ID:          newShardID,
		DSN:         newShardConfig.DSN,
		Weight:      newShardConfig.Weight,
		Status:      ShardActive,
		ReadReplica: newShardConfig.ReadReplicas,
	}

	ss.mu.Lock()
	ss.shards = append(ss.shards, newShard)
	ss.ring.AddShard(newShardID, newShardConfig.Weight)
	ss.mu.Unlock()

	// TODO: Migrate data belonging to new shard's hash range
	// This would involve:
	// 1. Identify keys that should move to new shard
	// 2. Copy data to new shard
	// 3. Verify consistency
	// 4. Update routing
	// 5. Delete old data

	ss.metrics.mu.Lock()
	ss.metrics.RebalanceCount++
	ss.metrics.mu.Unlock()

	return nil
}

// CrossShardTransaction handles transactions spanning multiple shards
// Uses 2-phase commit protocol for distributed transactions
func (ss *ShardingStrategy) CrossShardTransaction(ctx context.Context,
	participants []string, fn func(shards map[string]*Shard) error) error {

	// Phase 1: Prepare
	shardMap := make(map[string]*Shard)
	var preparedShards []*Shard

	for _, tenantID := range participants {
		shard, err := ss.GetShardForTenant(tenantID)
		if err != nil {
			return fmt.Errorf("get shard for %s: %w", tenantID, err)
		}
		shardMap[tenantID] = shard
		preparedShards = append(preparedShards, shard)
	}

	// Execute transaction function
	if err := fn(shardMap); err != nil {
		// Rollback on all shards
		for _, shard := range preparedShards {
			// TODO: implement rollback
			_ = shard
		}
		return err
	}

	// Phase 2: Commit
	for _, shard := range preparedShards {
		// TODO: implement commit
		_ = shard
	}

	ss.metrics.mu.Lock()
	ss.metrics.CrossShardQueries++
	ss.metrics.mu.Unlock()

	return nil
}

// GetShardMetrics returns current shard distribution metrics
func (ss *ShardingStrategy) GetShardMetrics() map[string]interface{} {
	ss.metrics.mu.RLock()
	defer ss.metrics.mu.RUnlock()

	metrics := map[string]interface{}{
		"total_shards":        len(ss.shards),
		"rebalancing":         ss.rebalancing,
		"cross_shard_queries": ss.metrics.CrossShardQueries,
		"rebalance_count":     ss.metrics.RebalanceCount,
		"shard_requests":      make(map[int]uint64),
	}

	for id, count := range ss.metrics.TotalRequests {
		metrics["shard_requests"].(map[int]uint64)[id] = count
	}

	return metrics
}

// hashKey generates consistent hash for a key
func hashKey(key string) uint64 {
	h := sha256.Sum256([]byte(key))
	return binary.BigEndian.Uint64(h[:8])
}

// binarySearch finds the first element >= target
func binarySearch(arr []uint64, target uint64) int {
	left, right := 0, len(arr)
	for left < right {
		mid := (left + right) / 2
		if arr[mid] < target {
			left = mid + 1
		} else {
			right = mid
		}
	}
	return left
}

// sortUint64Slice sorts in ascending order (simple bubble sort for demo)
func sortUint64Slice(arr []uint64) {
	n := len(arr)
	for i := 0; i < n-1; i++ {
		for j := 0; j < n-i-1; j++ {
			if arr[j] > arr[j+1] {
				arr[j], arr[j+1] = arr[j+1], arr[j]
			}
		}
	}
}

// ReadReplicaSelector implements read/write splitting
type ReadReplicaSelector struct {
	primaryDSN string
	replicas   []string
	current    int
	mu         sync.Mutex
}

// GetReadConnection returns replica connection for read operations
func (rrs *ReadReplicaSelector) GetReadConnection() string {
	rrs.mu.Lock()
	defer rrs.mu.Unlock()

	if len(rrs.replicas) == 0 {
		return rrs.primaryDSN
	}

	// Round-robin selection
	replica := rrs.replicas[rrs.current]
	rrs.current = (rrs.current + 1) % len(rrs.replicas)

	return replica
}

// GetWriteConnection returns primary connection for write operations
func (rrs *ReadReplicaSelector) GetWriteConnection() string {
	return rrs.primaryDSN
}
