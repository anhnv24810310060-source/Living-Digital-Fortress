package main

// P0 Enhancement: Advanced rate limiting and request filtering for Ingress
// Production-ready implementation with Redis support and distributed tracking

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	redis "github.com/redis/go-redis/v9"
)

// P0: Enhanced rate limiter with Redis backend for distributed rate limiting
type EnhancedRateLimiter struct {
	rdb          *redis.Client
	localBuckets *sync.Map // Fallback to local if Redis unavailable
	burst        int
	window       time.Duration
	useRedis     bool
}

func NewEnhancedRateLimiter(rdb *redis.Client, burst int, window time.Duration) *EnhancedRateLimiter {
	return &EnhancedRateLimiter{
		rdb:          rdb,
		localBuckets: &sync.Map{},
		burst:        burst,
		window:       window,
		useRedis:     rdb != nil,
	}
}

// Allow checks if request is allowed (token bucket algorithm)
// P0: Distributed across instances via Redis
func (erl *EnhancedRateLimiter) Allow(ctx context.Context, key string) (bool, error) {
	if erl.useRedis {
		return erl.allowRedis(ctx, key)
	}
	return erl.allowLocal(key), nil
}

// allowRedis implements distributed rate limiting using Redis
func (erl *EnhancedRateLimiter) allowRedis(ctx context.Context, key string) (bool, error) {
	redisKey := fmt.Sprintf("ratelimit:%s", key)

	// Lua script for atomic token bucket check
	script := `
		local key = KEYS[1]
		local burst = tonumber(ARGV[1])
		local window = tonumber(ARGV[2])
		local now = tonumber(ARGV[3])
		
		local bucket = redis.call('HGETALL', key)
		local tokens = burst
		local last_refill = now
		
		if #bucket > 0 then
			tokens = tonumber(bucket[2])
			last_refill = tonumber(bucket[4])
		end
		
		-- Refill tokens based on time elapsed
		local elapsed = now - last_refill
		local refill = math.floor(elapsed * burst / window)
		tokens = math.min(burst, tokens + refill)
		
		-- Check if request allowed
		if tokens > 0 then
			tokens = tokens - 1
			redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
			redis.call('EXPIRE', key, window)
			return 1
		else
			return 0
		end
	`

	nowMs := time.Now().UnixNano() / int64(time.Millisecond)
	windowMs := erl.window.Milliseconds()

	result, err := erl.rdb.Eval(ctx, script, []string{redisKey}, erl.burst, windowMs, nowMs).Result()
	if err != nil {
		// Fallback to local on Redis error
		return erl.allowLocal(key), nil
	}

	return result.(int64) == 1, nil
}

// allowLocal implements in-memory rate limiting (fallback)
func (erl *EnhancedRateLimiter) allowLocal(key string) bool {
	now := time.Now()

	value, loaded := erl.localBuckets.LoadOrStore(key, &localBucket{
		tokens:     erl.burst - 1,
		lastRefill: now,
		mu:         sync.Mutex{},
	})

	bucket := value.(*localBucket)
	bucket.mu.Lock()
	defer bucket.mu.Unlock()

	// Refill tokens based on elapsed time
	elapsed := now.Sub(bucket.lastRefill)
	refill := int(elapsed.Seconds() * float64(erl.burst) / erl.window.Seconds())

	if refill > 0 {
		bucket.tokens = min(erl.burst, bucket.tokens+refill)
		bucket.lastRefill = now
	}

	if !loaded {
		return true // First request always allowed
	}

	if bucket.tokens > 0 {
		bucket.tokens--
		return true
	}

	return false
}

type localBucket struct {
	tokens     int
	lastRefill time.Time
	mu         sync.Mutex
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// P0: Request filtering - deny list for malicious patterns
type RequestFilter struct {
	denyPaths      map[string]bool // Exact path matches to deny
	denyPathPrefix []string        // Path prefixes to deny
	denyQuery      []string        // Query parameter keys to deny
	denyHeaders    map[string]bool // Header names to deny
	mu             sync.RWMutex
}

func NewRequestFilter() *RequestFilter {
	return &RequestFilter{
		denyPaths: make(map[string]bool),
		denyPathPrefix: []string{
			"/.git/",
			"/.env",
			"/.aws/",
			"/admin/",
			"/phpmyadmin/",
			"/wp-admin/",
			"/.well-known/security.txt",
		},
		denyQuery: []string{
			"__proto__",
			"constructor",
			"prototype",
		},
		denyHeaders: map[string]bool{
			"x-forwarded-host": true, // Prevent host header injection
		},
	}
}

// CheckRequest returns true if request should be denied
// P0: Fast fail on suspicious patterns
func (rf *RequestFilter) CheckRequest(r *http.Request) (bool, string) {
	rf.mu.RLock()
	defer rf.mu.RUnlock()

	// Check exact path match
	if rf.denyPaths[r.URL.Path] {
		return true, "denied_path"
	}

	// Check path prefix
	path := strings.ToLower(r.URL.Path)
	for _, prefix := range rf.denyPathPrefix {
		if strings.HasPrefix(path, prefix) {
			return true, "denied_path_prefix"
		}
	}

	// Check query parameters
	query := r.URL.Query()
	for _, denyKey := range rf.denyQuery {
		if query.Has(denyKey) {
			return true, "denied_query"
		}
	}

	// Check headers
	for headerName := range rf.denyHeaders {
		if r.Header.Get(headerName) != "" {
			return true, "denied_header"
		}
	}

	// Check for suspicious patterns in path
	if containsSuspiciousPattern(path) {
		return true, "suspicious_pattern"
	}

	return false, ""
}

// AddDenyPath adds a path to the deny list
func (rf *RequestFilter) AddDenyPath(path string) {
	rf.mu.Lock()
	defer rf.mu.Unlock()
	rf.denyPaths[path] = true
}

// AddDenyPathPrefix adds a path prefix to the deny list
func (rf *RequestFilter) AddDenyPathPrefix(prefix string) {
	rf.mu.Lock()
	defer rf.mu.Unlock()
	rf.denyPathPrefix = append(rf.denyPathPrefix, prefix)
}

// containsSuspiciousPattern checks for common attack patterns
func containsSuspiciousPattern(path string) bool {
	suspicious := []string{
		"../",
		"..\\",
		"%2e%2e",
		"%252e",
		"//",
		"\\\\",
		"etc/passwd",
		"proc/self",
		"win.ini",
		"boot.ini",
	}

	for _, pattern := range suspicious {
		if strings.Contains(path, pattern) {
			return true
		}
	}

	return false
}

// P0: IP reputation tracking (simple implementation)
type IPReputation struct {
	scores   *sync.Map // IP -> score (0-100, lower is worse)
	rdb      *redis.Client
	useRedis bool
}

func NewIPReputation(rdb *redis.Client) *IPReputation {
	return &IPReputation{
		scores:   &sync.Map{},
		rdb:      rdb,
		useRedis: rdb != nil,
	}
}

// GetScore returns reputation score for IP (0-100)
func (ipr *IPReputation) GetScore(ctx context.Context, ip string) int {
	if ipr.useRedis {
		key := fmt.Sprintf("ip_reputation:%s", ip)
		val, err := ipr.rdb.Get(ctx, key).Int()
		if err == nil {
			return val
		}
	}

	// Fallback to local
	if val, ok := ipr.scores.Load(ip); ok {
		return val.(int)
	}

	return 100 // Default: good reputation
}

// DecrementScore decreases reputation score (on suspicious activity)
func (ipr *IPReputation) DecrementScore(ctx context.Context, ip string, delta int) {
	score := ipr.GetScore(ctx, ip)
	newScore := max(0, score-delta)

	if ipr.useRedis {
		key := fmt.Sprintf("ip_reputation:%s", ip)
		_ = ipr.rdb.Set(ctx, key, newScore, 24*time.Hour).Err()
	}

	ipr.scores.Store(ip, newScore)
}

// IncrementScore increases reputation score (on good behavior)
func (ipr *IPReputation) IncrementScore(ctx context.Context, ip string, delta int) {
	score := ipr.GetScore(ctx, ip)
	newScore := min(100, score+delta)

	if ipr.useRedis {
		key := fmt.Sprintf("ip_reputation:%s", ip)
		_ = ipr.rdb.Set(ctx, key, newScore, 24*time.Hour).Err()
	}

	ipr.scores.Store(ip, newScore)
}

// IsBlocked checks if IP should be blocked (score < 20)
func (ipr *IPReputation) IsBlocked(ctx context.Context, ip string) bool {
	return ipr.GetScore(ctx, ip) < 20
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// P0: Request fingerprinting for anomaly detection
type RequestFingerprint struct {
	Hash      string
	IP        string
	Path      string
	Method    string
	UserAgent string
	Timestamp time.Time
}

func GenerateRequestFingerprint(r *http.Request, ip string) *RequestFingerprint {
	// Create fingerprint hash from key attributes
	data := fmt.Sprintf("%s|%s|%s|%s", ip, r.Method, r.URL.Path, r.UserAgent())
	hash := sha256.Sum256([]byte(data))

	return &RequestFingerprint{
		Hash:      hex.EncodeToString(hash[:]),
		IP:        ip,
		Path:      r.URL.Path,
		Method:    r.Method,
		UserAgent: r.UserAgent(),
		Timestamp: time.Now(),
	}
}

// P0: Adaptive rate limiting based on threat level
func (erl *EnhancedRateLimiter) AllowWithThreatLevel(ctx context.Context, key string, threatLevel int) (bool, error) {
	// Adjust burst based on threat level (0-100)
	// High threat = lower burst
	adjustedBurst := erl.burst

	if threatLevel > 80 {
		adjustedBurst = max(1, erl.burst/10) // Severely restricted
	} else if threatLevel > 60 {
		adjustedBurst = max(5, erl.burst/5) // Heavily restricted
	} else if threatLevel > 40 {
		adjustedBurst = erl.burst / 2 // Moderately restricted
	}

	// Temporarily override burst for this check
	originalBurst := erl.burst
	erl.burst = adjustedBurst
	defer func() { erl.burst = originalBurst }()

	return erl.Allow(ctx, key)
}

// P0: Connection limiting per IP
type ConnectionLimiter struct {
	maxConns    int
	activeConns *sync.Map // IP -> count
	rdb         *redis.Client
	useRedis    bool
}

func NewConnectionLimiter(rdb *redis.Client, maxConns int) *ConnectionLimiter {
	return &ConnectionLimiter{
		maxConns:    maxConns,
		activeConns: &sync.Map{},
		rdb:         rdb,
		useRedis:    rdb != nil,
	}
}

// Acquire increments connection count for IP
func (cl *ConnectionLimiter) Acquire(ctx context.Context, ip string) bool {
	if cl.useRedis {
		key := fmt.Sprintf("conn_limit:%s", ip)
		val, err := cl.rdb.Incr(ctx, key).Result()
		if err != nil {
			// Fallback to local
			return cl.acquireLocal(ip)
		}

		// Set expiry on first increment
		if val == 1 {
			cl.rdb.Expire(ctx, key, 5*time.Minute)
		}

		if val > int64(cl.maxConns) {
			cl.rdb.Decr(ctx, key) // Rollback
			return false
		}

		return true
	}

	return cl.acquireLocal(ip)
}

func (cl *ConnectionLimiter) acquireLocal(ip string) bool {
	val, _ := cl.activeConns.LoadOrStore(ip, int32(0))
	count := val.(int32)

	if count >= int32(cl.maxConns) {
		return false
	}

	cl.activeConns.Store(ip, count+1)
	return true
}

// Release decrements connection count for IP
func (cl *ConnectionLimiter) Release(ctx context.Context, ip string) {
	if cl.useRedis {
		key := fmt.Sprintf("conn_limit:%s", ip)
		cl.rdb.Decr(ctx, key)
		return
	}

	if val, ok := cl.activeConns.Load(ip); ok {
		count := val.(int32)
		if count > 0 {
			cl.activeConns.Store(ip, count-1)
		}
	}
}
