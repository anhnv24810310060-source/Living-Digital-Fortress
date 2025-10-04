// Package adaptive provides adaptive rate limiting with ML-based threshold adjustment
// and multi-dimensional rate limiting (IP, user, endpoint, payload size).
package adaptive

import (
	"context"
	_ "crypto/sha256" // Keep for future use
	_ "encoding/binary" // Keep for future use
	_ "fmt" // Keep for future use
	"math"
	"sync"
	"sync/atomic"
	"time"
)

// LimiterConfig configures the adaptive rate limiter
type LimiterConfig struct {
	// Base capacity (requests per window)
	BaseCapacity      int
	Window            time.Duration
	
	// Adaptive parameters
	EnableMLAdaptive  bool          // Enable ML-based threshold adjustment
	AdaptInterval     time.Duration // How often to adjust thresholds
	MinCapacity       int           // Minimum capacity (safety bound)
	MaxCapacity       int           // Maximum capacity (safety bound)
	
	// Multi-dimensional limits
	EnableIPLimit     bool
	EnableUserLimit   bool
	EnableEndpointLimit bool
	EnablePayloadLimit  bool
	
	// Geolocation-aware (requires external GeoIP data)
	EnableGeoPolicy   bool
	GeoMultipliers    map[string]float64 // country code -> multiplier
}

// Limiter provides adaptive multi-dimensional rate limiting
type Limiter struct {
	cfg         LimiterConfig
	
	// Buckets for different dimensions
	ipBuckets       sync.Map // ip -> *TokenBucket
	userBuckets     sync.Map // userID -> *TokenBucket
	endpointBuckets sync.Map // endpoint -> *TokenBucket
	
	// Adaptive state
	requestHistory  []RequestEvent
	historyMu       sync.RWMutex
	currentCapacity int64 // Atomically updated
	
	// Reputation scoring
	reputationScores sync.Map // key -> *ReputationScore
	
	// Metrics
	allowed         uint64
	rejected        uint64
	adaptations     uint64
	
	ctx             context.Context
	cancel          context.CancelFunc
}

// TokenBucket implements token bucket algorithm with variable refill
type TokenBucket struct {
	capacity      int64
	tokens        int64 // atomic
	refillRate    int64 // tokens per second (atomic for adaptive)
	lastRefill    int64 // unix nano
	mu            sync.Mutex
}

// RequestEvent records a rate limit decision for ML analysis
type RequestEvent struct {
	Timestamp   time.Time
	Key         string
	Dimension   string // "ip", "user", "endpoint"
	Allowed     bool
	PayloadSize int
	Endpoint    string
	GeoCountry  string
}

// ReputationScore tracks behavior patterns for a key
type ReputationScore struct {
	mu             sync.RWMutex
	score          float64 // 0.0 (bad) to 1.0 (good)
	requestCount   uint64
	denialCount    uint64
	lastUpdate     time.Time
	
	// Behavioral features
	avgInterarrival time.Duration
	burstiness      float64
	diversityScore  float64 // endpoint diversity
}

// NewLimiter creates a new adaptive rate limiter
func NewLimiter(cfg LimiterConfig) *Limiter {
	if cfg.BaseCapacity == 0 {
		cfg.BaseCapacity = 100
	}
	if cfg.Window == 0 {
		cfg.Window = time.Minute
	}
	if cfg.MinCapacity == 0 {
		cfg.MinCapacity = 10
	}
	if cfg.MaxCapacity == 0 {
		cfg.MaxCapacity = 10000
	}
	if cfg.AdaptInterval == 0 {
		cfg.AdaptInterval = 1 * time.Minute
	}
	
	ctx, cancel := context.WithCancel(context.Background())
	
	l := &Limiter{
		cfg:             cfg,
		requestHistory:  make([]RequestEvent, 0, 10000),
		ctx:             ctx,
		cancel:          cancel,
	}
	
	atomic.StoreInt64(&l.currentCapacity, int64(cfg.BaseCapacity))
	
	// Start adaptive adjustment if enabled
	if cfg.EnableMLAdaptive {
		go l.adaptiveAdjuster()
	}
	
	// Start reputation scorer
	go l.reputationScorer()
	
	return l
}

// Allow checks if a request should be allowed
func (l *Limiter) Allow(key string, dimension string, attrs RequestAttributes) bool {
	// Multi-dimensional check
	dimensions := []string{}
	
	if l.cfg.EnableIPLimit && dimension == "ip" {
		dimensions = append(dimensions, "ip:"+key)
	}
	if l.cfg.EnableUserLimit && attrs.UserID != "" {
		dimensions = append(dimensions, "user:"+attrs.UserID)
	}
	if l.cfg.EnableEndpointLimit && attrs.Endpoint != "" {
		dimensions = append(dimensions, "endpoint:"+attrs.Endpoint)
	}
	
	// Check each dimension
	for _, dim := range dimensions {
		bucket := l.getOrCreateBucket(dim)
		
		// Apply reputation-based multiplier
		multiplier := l.getReputationMultiplier(dim)
		capacity := int64(float64(atomic.LoadInt64(&l.currentCapacity)) * multiplier)
		
		// Apply geolocation multiplier if enabled
		if l.cfg.EnableGeoPolicy && attrs.GeoCountry != "" {
			if geoMult, ok := l.cfg.GeoMultipliers[attrs.GeoCountry]; ok {
				capacity = int64(float64(capacity) * geoMult)
			}
		}
		
		if !bucket.Take(capacity) {
			// Dimension limit exceeded
			atomic.AddUint64(&l.rejected, 1)
			l.recordEvent(RequestEvent{
				Timestamp:   time.Now(),
				Key:         key,
				Dimension:   dimension,
				Allowed:     false,
				PayloadSize: attrs.PayloadSize,
				Endpoint:    attrs.Endpoint,
				GeoCountry:  attrs.GeoCountry,
			})
			l.updateReputation(dim, false)
			return false
		}
	}
	
	// Payload size check
	if l.cfg.EnablePayloadLimit && attrs.PayloadSize > 0 {
		// Cost = tokens proportional to payload size
		cost := attrs.PayloadSize / 1024 // 1 token per KB
		if cost > 0 {
			bucket := l.getOrCreateBucket("payload:" + key)
			for i := 0; i < cost && i < 100; i++ {
				if !bucket.Take(atomic.LoadInt64(&l.currentCapacity)) {
					atomic.AddUint64(&l.rejected, 1)
					return false
				}
			}
		}
	}
	
	atomic.AddUint64(&l.allowed, 1)
	l.recordEvent(RequestEvent{
		Timestamp:   time.Now(),
		Key:         key,
		Dimension:   dimension,
		Allowed:     true,
		PayloadSize: attrs.PayloadSize,
		Endpoint:    attrs.Endpoint,
		GeoCountry:  attrs.GeoCountry,
	})
	l.updateReputation(key, true)
	
	return true
}

// RequestAttributes provides additional context for rate limiting
type RequestAttributes struct {
	UserID      string
	Endpoint    string
	PayloadSize int
	GeoCountry  string
}

// getOrCreateBucket retrieves or creates a token bucket for a key
func (l *Limiter) getOrCreateBucket(key string) *TokenBucket {
	if val, ok := l.ipBuckets.Load(key); ok {
		return val.(*TokenBucket)
	}
	
	capacity := atomic.LoadInt64(&l.currentCapacity)
	bucket := &TokenBucket{
		capacity:   capacity,
		tokens:     capacity,
		refillRate: capacity / int64(l.cfg.Window.Seconds()),
		lastRefill: time.Now().UnixNano(),
	}
	
	actual, _ := l.ipBuckets.LoadOrStore(key, bucket)
	return actual.(*TokenBucket)
}

// Take attempts to consume a token from the bucket
func (tb *TokenBucket) Take(capacity int64) bool {
	tb.mu.Lock()
	defer tb.mu.Unlock()
	
	now := time.Now().UnixNano()
	lastRefill := atomic.LoadInt64(&tb.lastRefill)
	elapsed := now - lastRefill
	
	if elapsed > 0 {
		// Refill tokens based on elapsed time
		refillRate := atomic.LoadInt64(&tb.refillRate)
		tokensToAdd := (elapsed * refillRate) / int64(time.Second)
		newTokens := atomic.LoadInt64(&tb.tokens) + tokensToAdd
		
		if newTokens > capacity {
			newTokens = capacity
		}
		
		atomic.StoreInt64(&tb.tokens, newTokens)
		atomic.StoreInt64(&tb.lastRefill, now)
	}
	
	tokens := atomic.LoadInt64(&tb.tokens)
	if tokens > 0 {
		atomic.AddInt64(&tb.tokens, -1)
		return true
	}
	
	return false
}

// recordEvent records a rate limit decision for analysis
func (l *Limiter) recordEvent(event RequestEvent) {
	l.historyMu.Lock()
	defer l.historyMu.Unlock()
	
	l.requestHistory = append(l.requestHistory, event)
	
	// Keep only recent history (last 10000 events)
	if len(l.requestHistory) > 10000 {
		l.requestHistory = l.requestHistory[len(l.requestHistory)-10000:]
	}
}

// adaptiveAdjuster uses ML-like heuristics to adjust rate limits
func (l *Limiter) adaptiveAdjuster() {
	ticker := time.NewTicker(l.cfg.AdaptInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-l.ctx.Done():
			return
		case <-ticker.C:
			l.adjustThresholds()
		}
	}
}

// adjustThresholds analyzes request patterns and adjusts capacity
func (l *Limiter) adjustThresholds() {
	l.historyMu.RLock()
	events := make([]RequestEvent, len(l.requestHistory))
	copy(events, l.requestHistory)
	l.historyMu.RUnlock()
	
	if len(events) < 10 {
		return // Not enough data
	}
	
	// Calculate metrics over recent window
	windowStart := time.Now().Add(-l.cfg.AdaptInterval)
	recentEvents := []RequestEvent{}
	for _, e := range events {
		if e.Timestamp.After(windowStart) {
			recentEvents = append(recentEvents, e)
		}
	}
	
	if len(recentEvents) == 0 {
		return
	}
	
	// Calculate denial rate
	denials := 0
	for _, e := range recentEvents {
		if !e.Allowed {
			denials++
		}
	}
	denialRate := float64(denials) / float64(len(recentEvents))
	
	// Adjust capacity based on denial rate
	currentCap := atomic.LoadInt64(&l.currentCapacity)
	var newCap int64
	
	if denialRate > 0.1 {
		// High denial rate: increase capacity
		newCap = int64(float64(currentCap) * 1.2)
	} else if denialRate < 0.01 {
		// Very low denial rate: can reduce capacity slightly
		newCap = int64(float64(currentCap) * 0.95)
	} else {
		// Acceptable rate: maintain
		newCap = currentCap
	}
	
	// Apply safety bounds
	if newCap < int64(l.cfg.MinCapacity) {
		newCap = int64(l.cfg.MinCapacity)
	}
	if newCap > int64(l.cfg.MaxCapacity) {
		newCap = int64(l.cfg.MaxCapacity)
	}
	
	if newCap != currentCap {
		atomic.StoreInt64(&l.currentCapacity, newCap)
		atomic.AddUint64(&l.adaptations, 1)
	}
}

// reputationScorer updates reputation scores based on behavior
func (l *Limiter) reputationScorer() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-l.ctx.Done():
			return
		case <-ticker.C:
			l.updateAllReputations()
		}
	}
}

// updateReputation updates reputation score for a key
func (l *Limiter) updateReputation(key string, allowed bool) {
	var rep *ReputationScore
	if val, ok := l.reputationScores.Load(key); ok {
		rep = val.(*ReputationScore)
	} else {
		rep = &ReputationScore{
			score:      0.5, // Neutral initial score
			lastUpdate: time.Now(),
		}
		l.reputationScores.Store(key, rep)
	}
	
	rep.mu.Lock()
	defer rep.mu.Unlock()
	
	rep.requestCount++
	if !allowed {
		rep.denialCount++
	}
	
	// Update score (exponential moving average)
	alpha := 0.1
	if allowed {
		rep.score = (1-alpha)*rep.score + alpha*1.0
	} else {
		rep.score = (1-alpha)*rep.score + alpha*0.0
	}
	
	rep.lastUpdate = time.Now()
}

// updateAllReputations decays old reputation scores
func (l *Limiter) updateAllReputations() {
	now := time.Now()
	l.reputationScores.Range(func(key, value interface{}) bool {
		rep := value.(*ReputationScore)
		rep.mu.Lock()
		
		// Decay score towards neutral if inactive
		if now.Sub(rep.lastUpdate) > 5*time.Minute {
			rep.score = 0.9*rep.score + 0.1*0.5
		}
		
		rep.mu.Unlock()
		return true
	})
}

// getReputationMultiplier returns a capacity multiplier based on reputation
func (l *Limiter) getReputationMultiplier(key string) float64 {
	if val, ok := l.reputationScores.Load(key); ok {
		rep := val.(*ReputationScore)
		rep.mu.RLock()
		score := rep.score
		rep.mu.RUnlock()
		
		// Map score [0,1] to multiplier [0.5, 2.0]
		return 0.5 + 1.5*score
	}
	return 1.0 // Default multiplier
}

// GetReputationScore returns the current reputation score for a key
func (l *Limiter) GetReputationScore(key string) float64 {
	if val, ok := l.reputationScores.Load(key); ok {
		rep := val.(*ReputationScore)
		rep.mu.RLock()
		defer rep.mu.RUnlock()
		return rep.score
	}
	return 0.5 // Neutral
}

// Metrics returns current limiter metrics
func (l *Limiter) Metrics() map[string]interface{} {
	return map[string]interface{}{
		"allowed":          atomic.LoadUint64(&l.allowed),
		"rejected":         atomic.LoadUint64(&l.rejected),
		"adaptations":      atomic.LoadUint64(&l.adaptations),
		"current_capacity": atomic.LoadInt64(&l.currentCapacity),
		"history_size":     len(l.requestHistory),
	}
}

// Close stops the limiter
func (l *Limiter) Close() {
	l.cancel()
}

// ---------- Sliding Window with Exponential Decay ----------

type SlidingWindow struct {
	mu        sync.Mutex
	events    []time.Time
	capacity  int
	window    time.Duration
	decayRate float64 // Exponential decay factor
}

func NewSlidingWindow(capacity int, window time.Duration, decayRate float64) *SlidingWindow {
	return &SlidingWindow{
		events:    make([]time.Time, 0, capacity*2),
		capacity:  capacity,
		window:    window,
		decayRate: decayRate,
	}
}

func (sw *SlidingWindow) Allow() bool {
	sw.mu.Lock()
	defer sw.mu.Unlock()
	
	now := time.Now()
	cutoff := now.Add(-sw.window)
	
	// Remove expired events
	i := 0
	for i < len(sw.events) && sw.events[i].Before(cutoff) {
		i++
	}
	sw.events = sw.events[i:]
	
	// Apply exponential decay to effective count
	effectiveCount := 0.0
	for _, t := range sw.events {
		age := now.Sub(t).Seconds()
		weight := math.Exp(-sw.decayRate * age)
		effectiveCount += weight
	}
	
	if int(effectiveCount) < sw.capacity {
		sw.events = append(sw.events, now)
		return true
	}
	
	return false
}

// ---------- Leaky Bucket for Burst Handling ----------

type LeakyBucket struct {
	mu         sync.Mutex
	level      float64   // Current water level
	capacity   float64   // Maximum capacity
	leakRate   float64   // Leak rate (per second)
	lastLeak   time.Time
}

func NewLeakyBucket(capacity, leakRate float64) *LeakyBucket {
	return &LeakyBucket{
		capacity: capacity,
		leakRate: leakRate,
		lastLeak: time.Now(),
	}
}

func (lb *LeakyBucket) Allow(cost float64) bool {
	lb.mu.Lock()
	defer lb.mu.Unlock()
	
	now := time.Now()
	elapsed := now.Sub(lb.lastLeak).Seconds()
	
	// Leak water
	leaked := elapsed * lb.leakRate
	lb.level -= leaked
	if lb.level < 0 {
		lb.level = 0
	}
	lb.lastLeak = now
	
	// Try to add water
	if lb.level+cost <= lb.capacity {
		lb.level += cost
		return true
	}
	
	return false
}

// GetLevel returns current bucket level
func (lb *LeakyBucket) GetLevel() float64 {
	lb.mu.Lock()
	defer lb.mu.Unlock()
	return lb.level
}
