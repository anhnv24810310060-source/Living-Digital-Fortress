package ratelimit

import (
	"context"
	"fmt"
	"sync"
	"time"

	redis "github.com/redis/go-redis/v9"
)

// SlidingWindowLimiter implements hybrid sliding window + token bucket algorithm
// Provides smooth rate limiting with Redis for distributed systems
// Time Complexity: O(1) with Redis sorted sets
//
// Algorithm:
// - Uses Redis sorted sets to track request timestamps
// - Removes expired entries efficiently
// - Combines with token bucket for burst handling
// - Memory efficient: auto-cleanup of old entries
type SlidingWindowLimiter struct {
	rdb           *redis.Client
	capacity      int           // max requests per window
	window        time.Duration // sliding window duration
	burstCapacity int           // additional burst allowance
	localFallback *LocalLimiter // fallback when Redis unavailable
	mu            sync.RWMutex
}

// NewSlidingWindowLimiter creates production-grade rate limiter
func NewSlidingWindowLimiter(rdb *redis.Client, capacity int, window time.Duration, burstCapacity int) *SlidingWindowLimiter {
	return &SlidingWindowLimiter{
		rdb:           rdb,
		capacity:      capacity,
		window:        window,
		burstCapacity: burstCapacity,
		localFallback: NewLocalLimiter(capacity, window),
	}
}

// Allow checks if request is allowed using sliding window algorithm
// Returns: (allowed bool, remaining int, resetTime time.Time)
func (s *SlidingWindowLimiter) Allow(ctx context.Context, key string) (bool, int, time.Time) {
	if s.rdb == nil {
		// Fallback to local limiter
		return s.localFallback.Allow(key)
	}

	now := time.Now()
	windowStart := now.Add(-s.window)
	resetTime := now.Add(s.window)

	// Redis Lua script for atomic sliding window check
	// This ensures thread-safety across distributed instances
	script := `
		local key = KEYS[1]
		local now = tonumber(ARGV[1])
		local window_start = tonumber(ARGV[2])
		local capacity = tonumber(ARGV[3])
		local burst = tonumber(ARGV[4])
		local window_ms = tonumber(ARGV[5])
		
		-- Remove expired entries (O(log N) where N = entries in window)
		redis.call('ZREMRANGEBYSCORE', key, '-inf', window_start)
		
		-- Count current requests in window
		local count = redis.call('ZCARD', key)
		
		-- Calculate effective capacity (base + burst)
		local max_capacity = capacity + burst
		
		if count < max_capacity then
			-- Add current request with microsecond precision for uniqueness
			redis.call('ZADD', key, now, now .. ':' .. redis.call('INCR', key .. ':seq'))
			-- Set expiry to window + small buffer
			redis.call('PEXPIRE', key, window_ms + 1000)
			return {1, max_capacity - count - 1, window_ms}
		else
			return {0, 0, window_ms}
		end
	`

	result, err := s.rdb.Eval(ctx, script, []string{s.redisKey(key)},
		float64(now.UnixMicro())/1e6,
		float64(windowStart.UnixMicro())/1e6,
		s.capacity,
		s.burstCapacity,
		s.window.Milliseconds(),
	).Result()

	if err != nil {
		// Fallback to local on Redis error
		return s.localFallback.Allow(key)
	}

	res := result.([]interface{})
	allowed := res[0].(int64) == 1
	remaining := int(res[1].(int64))

	return allowed, remaining, resetTime
}

// AllowN checks if N requests are allowed (batch operation)
func (s *SlidingWindowLimiter) AllowN(ctx context.Context, key string, n int) (bool, int, time.Time) {
	if n <= 0 {
		return false, 0, time.Now()
	}

	now := time.Now()
	windowStart := now.Add(-s.window)
	resetTime := now.Add(s.window)

	script := `
		local key = KEYS[1]
		local now = tonumber(ARGV[1])
		local window_start = tonumber(ARGV[2])
		local capacity = tonumber(ARGV[3])
		local burst = tonumber(ARGV[4])
		local window_ms = tonumber(ARGV[5])
		local n = tonumber(ARGV[6])
		
		redis.call('ZREMRANGEBYSCORE', key, '-inf', window_start)
		local count = redis.call('ZCARD', key)
		local max_capacity = capacity + burst
		
		if count + n <= max_capacity then
			for i = 1, n do
				redis.call('ZADD', key, now + i * 0.000001, now .. ':' .. redis.call('INCR', key .. ':seq'))
			end
			redis.call('PEXPIRE', key, window_ms + 1000)
			return {1, max_capacity - count - n, window_ms}
		else
			return {0, 0, window_ms}
		end
	`

	result, err := s.rdb.Eval(ctx, script, []string{s.redisKey(key)},
		float64(now.UnixMicro())/1e6,
		float64(windowStart.UnixMicro())/1e6,
		s.capacity,
		s.burstCapacity,
		s.window.Milliseconds(),
		n,
	).Result()

	if err != nil {
		return false, 0, resetTime
	}

	res := result.([]interface{})
	allowed := res[0].(int64) == 1
	remaining := int(res[1].(int64))

	return allowed, remaining, resetTime
}

// Remaining returns current remaining capacity
func (s *SlidingWindowLimiter) Remaining(ctx context.Context, key string) int {
	if s.rdb == nil {
		return s.localFallback.Remaining(key)
	}

	now := time.Now()
	windowStart := now.Add(-s.window)

	// Clean and count atomically
	pipe := s.rdb.Pipeline()
	pipe.ZRemRangeByScore(ctx, s.redisKey(key), "-inf", fmt.Sprintf("%f", float64(windowStart.UnixMicro())/1e6))
	countCmd := pipe.ZCard(ctx, s.redisKey(key))
	_, err := pipe.Exec(ctx)

	if err != nil {
		return s.localFallback.Remaining(key)
	}

	count := int(countCmd.Val())
	maxCap := s.capacity + s.burstCapacity
	remaining := maxCap - count
	if remaining < 0 {
		remaining = 0
	}
	return remaining
}

// Reset clears the rate limit for a key
func (s *SlidingWindowLimiter) Reset(ctx context.Context, key string) error {
	if s.rdb == nil {
		s.localFallback.Reset(key)
		return nil
	}
	return s.rdb.Del(ctx, s.redisKey(key), s.redisKey(key)+":seq").Err()
}

func (s *SlidingWindowLimiter) redisKey(key string) string {
	return fmt.Sprintf("ratelimit:sw:%s", key)
}

// LocalLimiter is in-memory fallback using token bucket
type LocalLimiter struct {
	capacity int
	window   time.Duration
	mu       sync.Mutex
	buckets  map[string]*bucket
}

type bucket struct {
	tokens    int
	lastReset time.Time
}

func NewLocalLimiter(capacity int, window time.Duration) *LocalLimiter {
	l := &LocalLimiter{
		capacity: capacity,
		window:   window,
		buckets:  make(map[string]*bucket),
	}
	// Periodic cleanup
	go l.cleanup()
	return l
}

func (l *LocalLimiter) Allow(key string) (bool, int, time.Time) {
	l.mu.Lock()
	defer l.mu.Unlock()

	now := time.Now()
	b, exists := l.buckets[key]

	if !exists || now.Sub(b.lastReset) >= l.window {
		// New window
		l.buckets[key] = &bucket{
			tokens:    l.capacity - 1,
			lastReset: now,
		}
		return true, l.capacity - 1, now.Add(l.window)
	}

	if b.tokens > 0 {
		b.tokens--
		return true, b.tokens, b.lastReset.Add(l.window)
	}

	return false, 0, b.lastReset.Add(l.window)
}

func (l *LocalLimiter) Remaining(key string) int {
	l.mu.Lock()
	defer l.mu.Unlock()

	b, exists := l.buckets[key]
	if !exists {
		return l.capacity
	}

	if time.Since(b.lastReset) >= l.window {
		return l.capacity
	}

	return b.tokens
}

func (l *LocalLimiter) Reset(key string) {
	l.mu.Lock()
	defer l.mu.Unlock()
	delete(l.buckets, key)
}

func (l *LocalLimiter) cleanup() {
	ticker := time.NewTicker(l.window)
	defer ticker.Stop()

	for range ticker.C {
		l.mu.Lock()
		now := time.Now()
		for k, b := range l.buckets {
			if now.Sub(b.lastReset) >= l.window*2 {
				delete(l.buckets, k)
			}
		}
		l.mu.Unlock()
	}
}

// AdaptiveRateLimiter adjusts capacity based on system load
type AdaptiveRateLimiter struct {
	base          *SlidingWindowLimiter
	mu            sync.RWMutex
	currentFactor float64 // multiplier for capacity (0.5 to 2.0)
	metrics       *adaptiveMetrics
}

type adaptiveMetrics struct {
	successCount uint64
	errorCount   uint64
	lastAdjust   time.Time
	mu           sync.Mutex
}

func NewAdaptiveRateLimiter(rdb *redis.Client, baseCapacity int, window time.Duration, burst int) *AdaptiveRateLimiter {
	a := &AdaptiveRateLimiter{
		base:          NewSlidingWindowLimiter(rdb, baseCapacity, window, burst),
		currentFactor: 1.0,
		metrics:       &adaptiveMetrics{lastAdjust: time.Now()},
	}
	go a.adaptiveAdjuster()
	return a
}

func (a *AdaptiveRateLimiter) Allow(ctx context.Context, key string) (bool, int, time.Time) {
	// Use current adaptive capacity
	return a.base.Allow(ctx, key)
}

func (a *AdaptiveRateLimiter) RecordSuccess() {
	a.metrics.mu.Lock()
	a.metrics.successCount++
	a.metrics.mu.Unlock()
}

func (a *AdaptiveRateLimiter) RecordError() {
	a.metrics.mu.Lock()
	a.metrics.errorCount++
	a.metrics.mu.Unlock()
}

func (a *AdaptiveRateLimiter) adaptiveAdjuster() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		a.metrics.mu.Lock()
		success := a.metrics.successCount
		errors := a.metrics.errorCount
		a.metrics.successCount = 0
		a.metrics.errorCount = 0
		a.metrics.mu.Unlock()

		total := success + errors
		if total < 10 {
			continue // Not enough data
		}

		errorRate := float64(errors) / float64(total)

		a.mu.Lock()
		if errorRate > 0.1 {
			// High error rate: reduce capacity by 10%
			a.currentFactor *= 0.9
			if a.currentFactor < 0.5 {
				a.currentFactor = 0.5
			}
		} else if errorRate < 0.01 {
			// Low error rate: increase capacity by 5%
			a.currentFactor *= 1.05
			if a.currentFactor > 2.0 {
				a.currentFactor = 2.0
			}
		}
		// Update base capacity
		newCap := int(float64(a.base.capacity) * a.currentFactor)
		if newCap < 1 {
			newCap = 1
		}
		a.base.capacity = newCap
		a.mu.Unlock()
	}
}

// MultiTierLimiter combines multiple rate limits (per-IP, per-tenant, global)
type MultiTierLimiter struct {
	ipLimiter     *SlidingWindowLimiter
	tenantLimiter *SlidingWindowLimiter
	globalLimiter *SlidingWindowLimiter
}

func NewMultiTierLimiter(rdb *redis.Client) *MultiTierLimiter {
	return &MultiTierLimiter{
		ipLimiter:     NewSlidingWindowLimiter(rdb, 100, time.Minute, 20),     // 100/min per IP + 20 burst
		tenantLimiter: NewSlidingWindowLimiter(rdb, 1000, time.Minute, 200),   // 1000/min per tenant + 200 burst
		globalLimiter: NewSlidingWindowLimiter(rdb, 10000, time.Minute, 2000), // 10k/min global + 2k burst
	}
}

func (m *MultiTierLimiter) Allow(ctx context.Context, ip, tenant string) (bool, string, int) {
	// Check in order: most specific to least specific
	if allowed, remaining, _ := m.ipLimiter.Allow(ctx, "ip:"+ip); !allowed {
		return false, "ip_limit", remaining
	}

	if tenant != "" {
		if allowed, remaining, _ := m.tenantLimiter.Allow(ctx, "tenant:"+tenant); !allowed {
			return false, "tenant_limit", remaining
		}
	}

	if allowed, remaining, _ := m.globalLimiter.Allow(ctx, "global"); !allowed {
		return false, "global_limit", remaining
	}

	return true, "", 0
}
