package gateway

import (
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"sync"
	"time"
)

// RateLimitConfig holds rate limiting configuration
type RateLimitConfig struct {
	RequestsPerMinute int
	BurstSize         int
	BypassPaths       []string
}

// RateLimiter implements token bucket rate limiting
type RateLimiter struct {
	config  RateLimitConfig
	buckets sync.Map // map[string]*tokenBucket
}

type tokenBucket struct {
	tokens     float64
	lastRefill time.Time
	mu         sync.Mutex
}

// NewRateLimiter creates a new rate limiter
func NewRateLimiter(config RateLimitConfig) *RateLimiter {
	if config.RequestsPerMinute == 0 {
		config.RequestsPerMinute = 60 // Default: 60 req/min
	}
	if config.BurstSize == 0 {
		config.BurstSize = 10 // Default: burst of 10
	}

	rl := &RateLimiter{config: config}

	// Cleanup goroutine
	go rl.cleanupLoop()

	return rl
}

// Limit applies rate limiting to HTTP requests
func (rl *RateLimiter) Limit(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Check if path should bypass rate limiting
		for _, path := range rl.config.BypassPaths {
			if r.URL.Path == path {
				next.ServeHTTP(w, r)
				return
			}
		}

		// Get client identifier (IP + User from context if available)
		clientID := rl.getClientID(r)

		// Check rate limit
		if !rl.allow(clientID) {
			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("X-RateLimit-Limit", fmt.Sprintf("%d", rl.config.RequestsPerMinute))
			w.Header().Set("X-RateLimit-Remaining", "0")
			w.Header().Set("Retry-After", "60")
			w.WriteHeader(http.StatusTooManyRequests)
			json.NewEncoder(w).Encode(map[string]string{
				"error":   "rate_limit_exceeded",
				"message": "too many requests, please try again later",
			})
			return
		}

		next.ServeHTTP(w, r)
	})
}

// getClientID extracts client identifier from request
func (rl *RateLimiter) getClientID(r *http.Request) string {
	// Try to get authenticated user first
	if claims, ok := ClaimsFromContext(r.Context()); ok {
		return fmt.Sprintf("user:%s:%s", claims.TenantID, claims.UserID)
	}

	// Fall back to IP address
	ip, _, _ := net.SplitHostPort(r.RemoteAddr)
	if ip == "" {
		ip = r.RemoteAddr
	}

	// Check X-Forwarded-For for proxied requests
	if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
		ip = xff
	}

	return fmt.Sprintf("ip:%s", ip)
}

// allow checks if request is allowed under rate limit
func (rl *RateLimiter) allow(clientID string) bool {
	bucket := rl.getBucket(clientID)
	bucket.mu.Lock()
	defer bucket.mu.Unlock()

	now := time.Now()
	elapsed := now.Sub(bucket.lastRefill).Seconds()

	// Refill tokens based on time elapsed
	refillRate := float64(rl.config.RequestsPerMinute) / 60.0 // tokens per second
	bucket.tokens += elapsed * refillRate

	// Cap at burst size
	maxTokens := float64(rl.config.BurstSize)
	if bucket.tokens > maxTokens {
		bucket.tokens = maxTokens
	}

	bucket.lastRefill = now

	// Check if we have tokens available
	if bucket.tokens >= 1.0 {
		bucket.tokens -= 1.0
		return true
	}

	return false
}

// getBucket retrieves or creates a token bucket for a client
func (rl *RateLimiter) getBucket(clientID string) *tokenBucket {
	if b, ok := rl.buckets.Load(clientID); ok {
		return b.(*tokenBucket)
	}

	bucket := &tokenBucket{
		tokens:     float64(rl.config.BurstSize),
		lastRefill: time.Now(),
	}

	rl.buckets.Store(clientID, bucket)
	return bucket
}

// cleanupLoop removes stale buckets
func (rl *RateLimiter) cleanupLoop() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		cutoff := time.Now().Add(-10 * time.Minute)

		rl.buckets.Range(func(key, value interface{}) bool {
			bucket := value.(*tokenBucket)
			bucket.mu.Lock()
			lastUsed := bucket.lastRefill
			bucket.mu.Unlock()

			if lastUsed.Before(cutoff) {
				rl.buckets.Delete(key)
			}
			return true
		})
	}
}

// GetStats returns rate limiter statistics
func (rl *RateLimiter) GetStats() map[string]interface{} {
	count := 0
	rl.buckets.Range(func(_, _ interface{}) bool {
		count++
		return true
	})

	return map[string]interface{}{
		"active_clients":      count,
		"requests_per_minute": rl.config.RequestsPerMinute,
		"burst_size":          rl.config.BurstSize,
	}
}
