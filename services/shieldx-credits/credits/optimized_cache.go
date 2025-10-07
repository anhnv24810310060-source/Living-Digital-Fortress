package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"
)

// OptimizedCacheLayer provides multi-tier caching with write-through strategy
// Tier 1: In-memory LRU cache (hot data, microsecond latency)
// Tier 2: Redis cache (warm data, millisecond latency)
// Tier 3: PostgreSQL (cold data, full accuracy)
type OptimizedCacheLayer struct {
	rdb       *redis.Client
	localLRU  *LRUCache
	mu        sync.RWMutex
	hitStats  CacheStats
	isEnabled bool
}

type CacheStats struct {
	L1Hits    int64 // In-memory hits
	L2Hits    int64 // Redis hits
	L3Hits    int64 // DB hits (cache miss)
	TotalReqs int64
	mu        sync.RWMutex
}

func (cs *CacheStats) RecordHit(tier int) {
	cs.mu.Lock()
	defer cs.mu.Unlock()
	cs.TotalReqs++
	switch tier {
	case 1:
		cs.L1Hits++
	case 2:
		cs.L2Hits++
	case 3:
		cs.L3Hits++
	}
}

func (cs *CacheStats) GetHitRate() map[string]interface{} {
	cs.mu.RLock()
	defer cs.mu.RUnlock()
	if cs.TotalReqs == 0 {
		return map[string]interface{}{
			"l1_hit_rate": 0.0,
			"l2_hit_rate": 0.0,
			"l3_hit_rate": 0.0,
			"total_reqs":  0,
		}
	}
	return map[string]interface{}{
		"l1_hit_rate": float64(cs.L1Hits) / float64(cs.TotalReqs),
		"l2_hit_rate": float64(cs.L2Hits) / float64(cs.TotalReqs),
		"l3_hit_rate": float64(cs.L3Hits) / float64(cs.TotalReqs),
		"total_reqs":  cs.TotalReqs,
	}
}

// LRUCache: Simple in-memory LRU with TTL
type LRUCache struct {
	capacity int
	items    map[string]*cacheItem
	order    *dll // doubly-linked list for LRU
	mu       sync.RWMutex
}

type cacheItem struct {
	key       string
	value     interface{}
	expiresAt time.Time
	node      *dllNode
}

type dll struct {
	head *dllNode
	tail *dllNode
}

type dllNode struct {
	key  string
	prev *dllNode
	next *dllNode
}

func NewLRUCache(capacity int) *LRUCache {
	return &LRUCache{
		capacity: capacity,
		items:    make(map[string]*cacheItem, capacity),
		order:    &dll{},
	}
}

func (lru *LRUCache) Get(key string) (interface{}, bool) {
	lru.mu.Lock()
	defer lru.mu.Unlock()

	item, ok := lru.items[key]
	if !ok {
		return nil, false
	}
	if time.Now().After(item.expiresAt) {
		lru.removeNode(item.node)
		delete(lru.items, key)
		return nil, false
	}
	// Move to front (most recently used)
	lru.moveToFront(item.node)
	return item.value, true
}

func (lru *LRUCache) Set(key string, value interface{}, ttl time.Duration) {
	lru.mu.Lock()
	defer lru.mu.Unlock()

	if item, ok := lru.items[key]; ok {
		item.value = value
		item.expiresAt = time.Now().Add(ttl)
		lru.moveToFront(item.node)
		return
	}

	if len(lru.items) >= lru.capacity {
		// Evict LRU item
		if lru.order.tail != nil {
			delete(lru.items, lru.order.tail.key)
			lru.removeNode(lru.order.tail)
		}
	}

	node := &dllNode{key: key}
	lru.items[key] = &cacheItem{
		key:       key,
		value:     value,
		expiresAt: time.Now().Add(ttl),
		node:      node,
	}
	lru.addToFront(node)
}

func (lru *LRUCache) Delete(key string) {
	lru.mu.Lock()
	defer lru.mu.Unlock()
	if item, ok := lru.items[key]; ok {
		lru.removeNode(item.node)
		delete(lru.items, key)
	}
}

func (lru *LRUCache) addToFront(node *dllNode) {
	node.next = lru.order.head
	node.prev = nil
	if lru.order.head != nil {
		lru.order.head.prev = node
	}
	lru.order.head = node
	if lru.order.tail == nil {
		lru.order.tail = node
	}
}

func (lru *LRUCache) removeNode(node *dllNode) {
	if node.prev != nil {
		node.prev.next = node.next
	} else {
		lru.order.head = node.next
	}
	if node.next != nil {
		node.next.prev = node.prev
	} else {
		lru.order.tail = node.prev
	}
}

func (lru *LRUCache) moveToFront(node *dllNode) {
	lru.removeNode(node)
	lru.addToFront(node)
}

// NewOptimizedCacheLayer initializes multi-tier caching
func NewOptimizedCacheLayer(rdb *redis.Client, lruSize int) *OptimizedCacheLayer {
	return &OptimizedCacheLayer{
		rdb:       rdb,
		localLRU:  NewLRUCache(lruSize),
		isEnabled: rdb != nil,
	}
}

// GetBalance tries L1 (in-mem) -> L2 (Redis) -> L3 (DB)
func (ocl *OptimizedCacheLayer) GetBalance(ctx context.Context, tenantID string, dbFallback func() (int64, error)) (int64, error) {
	if !ocl.isEnabled {
		ocl.hitStats.RecordHit(3)
		return dbFallback()
	}

	key := "bal:" + tenantID

	// L1: In-memory LRU
	if val, ok := ocl.localLRU.Get(key); ok {
		if balance, ok := val.(int64); ok {
			ocl.hitStats.RecordHit(1)
			return balance, nil
		}
	}

	// L2: Redis
	if ocl.rdb != nil {
		if val, err := ocl.rdb.Get(ctx, key).Result(); err == nil {
			if balance, err := strconv.ParseInt(val, 10, 64); err == nil {
				// Promote to L1
				ocl.localLRU.Set(key, balance, 30*time.Second)
				ocl.hitStats.RecordHit(2)
				return balance, nil
			}
		}
	}

	// L3: Database (fallback)
	balance, err := dbFallback()
	if err != nil {
		return 0, err
	}

	// Write-through: populate L2 and L1
	if ocl.rdb != nil {
		_ = ocl.rdb.SetEx(ctx, key, strconv.FormatInt(balance, 10), 60*time.Second).Err()
	}
	ocl.localLRU.Set(key, balance, 30*time.Second)
	ocl.hitStats.RecordHit(3)

	return balance, nil
}

// InvalidateBalance clears all cache tiers
func (ocl *OptimizedCacheLayer) InvalidateBalance(ctx context.Context, tenantID string) {
	key := "bal:" + tenantID
	ocl.localLRU.Delete(key)
	if ocl.rdb != nil {
		_ = ocl.rdb.Del(ctx, key).Err()
	}
}

// GetStats returns cache performance metrics
func (ocl *OptimizedCacheLayer) GetStats() map[string]interface{} {
	return ocl.hitStats.GetHitRate()
}

// BatchInvalidate efficiently clears multiple tenant balances
func (ocl *OptimizedCacheLayer) BatchInvalidate(ctx context.Context, tenantIDs []string) error {
	if !ocl.isEnabled || len(tenantIDs) == 0 {
		return nil
	}

	// L1 invalidation
	for _, tid := range tenantIDs {
		ocl.localLRU.Delete("bal:" + tid)
	}

	// L2 batch invalidation (Redis pipeline)
	if ocl.rdb != nil {
		pipe := ocl.rdb.Pipeline()
		for _, tid := range tenantIDs {
			pipe.Del(ctx, "bal:"+tid)
		}
		_, err := pipe.Exec(ctx)
		return err
	}

	return nil
}

// WarmUpCache pre-loads hot data into cache tiers
func (ocl *OptimizedCacheLayer) WarmUpCache(ctx context.Context, hotTenants []string, dbLoader func(string) (int64, error)) error {
	if !ocl.isEnabled {
		return nil
	}

	for _, tid := range hotTenants {
		balance, err := dbLoader(tid)
		if err != nil {
			continue // Skip failed loads
		}

		key := "bal:" + tid
		if ocl.rdb != nil {
			_ = ocl.rdb.SetEx(ctx, key, strconv.FormatInt(balance, 10), 60*time.Second).Err()
		}
		ocl.localLRU.Set(key, balance, 30*time.Second)
	}

	return nil
}

// SerializeStats returns JSON-encoded cache statistics
func (ocl *OptimizedCacheLayer) SerializeStats() ([]byte, error) {
	stats := ocl.GetStats()
	return json.Marshal(stats)
}

// HealthCheck verifies cache tier connectivity
func (ocl *OptimizedCacheLayer) HealthCheck(ctx context.Context) error {
	if !ocl.isEnabled {
		return nil
	}

	if ocl.rdb != nil {
		if err := ocl.rdb.Ping(ctx).Err(); err != nil {
			return fmt.Errorf("redis health check failed: %w", err)
		}
	}

	return nil
}
