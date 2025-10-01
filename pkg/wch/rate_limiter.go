package wch

import (
	"context"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"
)

// RateLimiter interface for rate limiting
type RateLimiter interface {
	Allow(ctx context.Context, key string) (bool, error)
	Reset(ctx context.Context, key string) error
	GetInfo(ctx context.Context, key string) (*RateLimitInfo, error)
}

// RateLimitInfo contains rate limit information
type RateLimitInfo struct {
	Allowed   bool
	Remaining int
	Limit     int
	ResetAt   time.Time
}

// RedisRateLimiter implements distributed rate limiting with Redis
type RedisRateLimiter struct {
	client       *redis.Client
	keyPrefix    string
	limit        int
	window       time.Duration
	burstLimit   int
	algorithm    string // "fixed_window", "sliding_window", "token_bucket"
}

// RateLimiterConfig configuration for rate limiter
type RateLimiterConfig struct {
	RedisAddr     string
	RedisPassword string
	RedisDB       int
	KeyPrefix     string
	Limit         int           // Requests per window
	Window        time.Duration // Time window
	BurstLimit    int           // Burst capacity
	Algorithm     string        // Algorithm type
}

// NewRedisRateLimiter creates a new Redis-based rate limiter
func NewRedisRateLimiter(config RateLimiterConfig) (*RedisRateLimiter, error) {
	client := redis.NewClient(&redis.Options{
		Addr:     config.RedisAddr,
		Password: config.RedisPassword,
		DB:       config.RedisDB,
	})

	// Test connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	
	if err := client.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("failed to connect to Redis: %w", err)
	}

	if config.KeyPrefix == "" {
		config.KeyPrefix = "ratelimit:"
	}
	if config.Limit == 0 {
		config.Limit = 100
	}
	if config.Window == 0 {
		config.Window = 1 * time.Minute
	}
	if config.BurstLimit == 0 {
		config.BurstLimit = config.Limit * 2
	}
	if config.Algorithm == "" {
		config.Algorithm = "sliding_window"
	}

	return &RedisRateLimiter{
		client:     client,
		keyPrefix:  config.KeyPrefix,
		limit:      config.Limit,
		window:     config.Window,
		burstLimit: config.BurstLimit,
		algorithm:  config.Algorithm,
	}, nil
}

// Allow checks if a request is allowed
func (rl *RedisRateLimiter) Allow(ctx context.Context, key string) (bool, error) {
	switch rl.algorithm {
	case "fixed_window":
		return rl.allowFixedWindow(ctx, key)
	case "sliding_window":
		return rl.allowSlidingWindow(ctx, key)
	case "token_bucket":
		return rl.allowTokenBucket(ctx, key)
	default:
		return rl.allowSlidingWindow(ctx, key)
	}
}

// allowFixedWindow implements fixed window algorithm
func (rl *RedisRateLimiter) allowFixedWindow(ctx context.Context, key string) (bool, error) {
	redisKey := rl.keyPrefix + key
	
	// Increment counter
	count, err := rl.client.Incr(ctx, redisKey).Result()
	if err != nil {
		return false, fmt.Errorf("failed to increment counter: %w", err)
	}

	// Set expiration on first request
	if count == 1 {
		rl.client.Expire(ctx, redisKey, rl.window)
	}

	return count <= int64(rl.limit), nil
}

// allowSlidingWindow implements sliding window algorithm
func (rl *RedisRateLimiter) allowSlidingWindow(ctx context.Context, key string) (bool, error) {
	redisKey := rl.keyPrefix + key
	now := time.Now()
	windowStart := now.Add(-rl.window)

	pipe := rl.client.Pipeline()
	
	// Remove old entries
	pipe.ZRemRangeByScore(ctx, redisKey, "0", fmt.Sprintf("%d", windowStart.UnixNano()))
	
	// Count current window
	countCmd := pipe.ZCard(ctx, redisKey)
	
	// Add current request
	pipe.ZAdd(ctx, redisKey, redis.Z{
		Score:  float64(now.UnixNano()),
		Member: fmt.Sprintf("%d", now.UnixNano()),
	})
	
	// Set expiration
	pipe.Expire(ctx, redisKey, rl.window+1*time.Second)
	
	_, err := pipe.Exec(ctx)
	if err != nil {
		return false, fmt.Errorf("failed to execute pipeline: %w", err)
	}

	count := countCmd.Val()
	return count < int64(rl.limit), nil
}

// allowTokenBucket implements token bucket algorithm
func (rl *RedisRateLimiter) allowTokenBucket(ctx context.Context, key string) (bool, error) {
	redisKey := rl.keyPrefix + key
	now := time.Now()

	// Lua script for atomic token bucket operation
	script := redis.NewScript(`
		local key = KEYS[1]
		local capacity = tonumber(ARGV[1])
		local refill_rate = tonumber(ARGV[2])
		local now = tonumber(ARGV[3])
		local requested = tonumber(ARGV[4])
		
		local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
		local tokens = tonumber(bucket[1])
		local last_refill = tonumber(bucket[2])
		
		if tokens == nil then
			tokens = capacity
			last_refill = now
		end
		
		-- Refill tokens based on elapsed time
		local elapsed = now - last_refill
		local refilled = math.floor(elapsed * refill_rate)
		tokens = math.min(capacity, tokens + refilled)
		last_refill = now
		
		-- Check if request can be fulfilled
		local allowed = 0
		if tokens >= requested then
			tokens = tokens - requested
			allowed = 1
		end
		
		-- Update bucket
		redis.call('HMSET', key, 'tokens', tokens, 'last_refill', last_refill)
		redis.call('EXPIRE', key, 3600)
		
		return {allowed, tokens}
	`)

	refillRate := float64(rl.limit) / rl.window.Seconds()
	
	result, err := script.Run(ctx, rl.client, []string{redisKey}, 
		rl.burstLimit, refillRate, now.Unix(), 1).Result()
	if err != nil {
		return false, fmt.Errorf("failed to execute token bucket script: %w", err)
	}

	resultSlice := result.([]interface{})
	allowed := resultSlice[0].(int64)
	
	return allowed == 1, nil
}

// Reset resets the rate limit for a key
func (rl *RedisRateLimiter) Reset(ctx context.Context, key string) error {
	redisKey := rl.keyPrefix + key
	return rl.client.Del(ctx, redisKey).Err()
}

// GetInfo returns rate limit information for a key
func (rl *RedisRateLimiter) GetInfo(ctx context.Context, key string) (*RateLimitInfo, error) {
	redisKey := rl.keyPrefix + key
	
	switch rl.algorithm {
	case "sliding_window":
		now := time.Now()
		windowStart := now.Add(-rl.window)
		
		count, err := rl.client.ZCount(ctx, redisKey, 
			fmt.Sprintf("%d", windowStart.UnixNano()), 
			fmt.Sprintf("%d", now.UnixNano())).Result()
		if err != nil {
			return nil, err
		}

		ttl, err := rl.client.TTL(ctx, redisKey).Result()
		if err != nil {
			return nil, err
		}

		remaining := rl.limit - int(count)
		if remaining < 0 {
			remaining = 0
		}

		return &RateLimitInfo{
			Allowed:   count < int64(rl.limit),
			Remaining: remaining,
			Limit:     rl.limit,
			ResetAt:   time.Now().Add(ttl),
		}, nil

	default:
		count, err := rl.client.Get(ctx, redisKey).Int()
		if err == redis.Nil {
			count = 0
		} else if err != nil {
			return nil, err
		}

		ttl, _ := rl.client.TTL(ctx, redisKey).Result()
		remaining := rl.limit - count
		if remaining < 0 {
			remaining = 0
		}

		return &RateLimitInfo{
			Allowed:   count < rl.limit,
			Remaining: remaining,
			Limit:     rl.limit,
			ResetAt:   time.Now().Add(ttl),
		}, nil
	}
}

// Close closes the Redis connection
func (rl *RedisRateLimiter) Close() error {
	return rl.client.Close()
}

// InMemoryRateLimiter implements in-memory rate limiting (for development)
type InMemoryRateLimiter struct {
	mu      sync.RWMutex
	buckets map[string]*bucket
	limit   int
	window  time.Duration
}

type bucket struct {
	count     int
	resetAt   time.Time
	mu        sync.Mutex
}

// NewInMemoryRateLimiter creates a new in-memory rate limiter
func NewInMemoryRateLimiter(limit int, window time.Duration) *InMemoryRateLimiter {
	limiter := &InMemoryRateLimiter{
		buckets: make(map[string]*bucket),
		limit:   limit,
		window:  window,
	}

	// Start cleanup goroutine
	go limiter.cleanup()

	return limiter
}

// Allow checks if a request is allowed
func (rl *InMemoryRateLimiter) Allow(ctx context.Context, key string) (bool, error) {
	rl.mu.Lock()
	b, exists := rl.buckets[key]
	if !exists {
		b = &bucket{
			count:   0,
			resetAt: time.Now().Add(rl.window),
		}
		rl.buckets[key] = b
	}
	rl.mu.Unlock()

	b.mu.Lock()
	defer b.mu.Unlock()

	now := time.Now()
	if now.After(b.resetAt) {
		b.count = 0
		b.resetAt = now.Add(rl.window)
	}

	if b.count >= rl.limit {
		return false, nil
	}

	b.count++
	return true, nil
}

// Reset resets the rate limit for a key
func (rl *InMemoryRateLimiter) Reset(ctx context.Context, key string) error {
	rl.mu.Lock()
	defer rl.mu.Unlock()
	delete(rl.buckets, key)
	return nil
}

// GetInfo returns rate limit information
func (rl *InMemoryRateLimiter) GetInfo(ctx context.Context, key string) (*RateLimitInfo, error) {
	rl.mu.RLock()
	b, exists := rl.buckets[key]
	rl.mu.RUnlock()

	if !exists {
		return &RateLimitInfo{
			Allowed:   true,
			Remaining: rl.limit,
			Limit:     rl.limit,
			ResetAt:   time.Now().Add(rl.window),
		}, nil
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	remaining := rl.limit - b.count
	if remaining < 0 {
		remaining = 0
	}

	return &RateLimitInfo{
		Allowed:   b.count < rl.limit,
		Remaining: remaining,
		Limit:     rl.limit,
		ResetAt:   b.resetAt,
	}, nil
}

// cleanup removes expired buckets
func (rl *InMemoryRateLimiter) cleanup() {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		rl.mu.Lock()
		now := time.Now()
		for key, b := range rl.buckets {
			b.mu.Lock()
			if now.After(b.resetAt.Add(rl.window)) {
				delete(rl.buckets, key)
			}
			b.mu.Unlock()
		}
		rl.mu.Unlock()
	}
}

// RateLimitMiddleware creates HTTP middleware for rate limiting
func RateLimitMiddleware(limiter RateLimiter) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Use client IP as key
			key := r.RemoteAddr
			
			allowed, err := limiter.Allow(r.Context(), key)
			if err != nil {
				http.Error(w, "Rate limiter error", http.StatusInternalServerError)
				return
			}

			// Get rate limit info
			info, _ := limiter.GetInfo(r.Context(), key)
			if info != nil {
				w.Header().Set("X-RateLimit-Limit", fmt.Sprintf("%d", info.Limit))
				w.Header().Set("X-RateLimit-Remaining", fmt.Sprintf("%d", info.Remaining))
				w.Header().Set("X-RateLimit-Reset", fmt.Sprintf("%d", info.ResetAt.Unix()))
			}

			if !allowed {
				w.Header().Set("Retry-After", fmt.Sprintf("%d", int(time.Until(info.ResetAt).Seconds())))
				http.Error(w, "Rate limit exceeded", http.StatusTooManyRequests)
				return
			}

			next.ServeHTTP(w, r)
		})
	}
}
