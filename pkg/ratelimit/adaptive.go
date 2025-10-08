// Package ratelimit provides adaptive rate limiting with ML-based threshold adjustment
// Implements multi-dimensional rate limiting with reputation scoring
package ratelimit

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"time"
)

// DimensionType defines rate limit dimensions
type DimensionType string

const (
	DimensionIP       DimensionType = "ip"
	DimensionUser     DimensionType = "user"
	DimensionEndpoint DimensionType = "endpoint"
	DimensionPayload  DimensionType = "payload_size"
	DimensionTenant   DimensionType = "tenant"
)

// AdaptiveLimiter implements intelligent rate limiting with risk assessment
type AdaptiveLimiter struct {
	// Base configuration
	baseRate   int             // Base requests/window
	window     time.Duration   // Time window
	dimensions []DimensionType // Active dimensions

	// Storage
	buckets    sync.Map // key -> *Bucket
	reputation sync.Map // key -> *ReputationScore

	// Adaptive parameters
	adaptEnabled bool
	learningRate float64 // Learning rate for threshold adjustment
	minRate      int     // Minimum allowed rate
	maxRate      int     // Maximum allowed rate

	// Geolocation awareness
	geoPolicy   map[string]int // country_code -> rate multiplier
	geoPolicyMu sync.RWMutex

	// Metrics
	allowed     uint64
	rejected    uint64
	adaptations uint64

	ctx    context.Context
	cancel context.CancelFunc
}

// Bucket represents a token bucket for rate limiting
type Bucket struct {
	capacity   int
	tokens     int
	lastRefill time.Time
	refillRate int // Tokens per window
	window     time.Duration

	// Adaptive adjustments
	multiplier float64 // Current rate multiplier (0.1 - 5.0)

	mu sync.Mutex
}

// ReputationScore tracks entity behavior for risk assessment
type ReputationScore struct {
	score         float64 // 0.0 (bad) to 1.0 (good)
	requests      uint64  // Total requests
	violations    uint64  // Rate limit violations
	lastViolation time.Time
	created       time.Time

	// Behavioral features
	avgInterval time.Duration // Average time between requests
	burstCount  uint64        // Number of burst patterns detected

	mu sync.RWMutex
}

// Config configures the adaptive limiter
type Config struct {
	BaseRate     int
	Window       time.Duration
	Dimensions   []DimensionType
	AdaptEnabled bool
	LearningRate float64
	MinRate      int
	MaxRate      int
	GeoPolicy    map[string]int
}

// NewAdaptiveLimiter creates a new adaptive rate limiter
func NewAdaptiveLimiter(cfg Config) *AdaptiveLimiter {
	if cfg.BaseRate <= 0 {
		cfg.BaseRate = 100
	}
	if cfg.Window == 0 {
		cfg.Window = time.Minute
	}
	if cfg.LearningRate == 0 {
		cfg.LearningRate = 0.1
	}
	if cfg.MinRate == 0 {
		cfg.MinRate = 10
	}
	if cfg.MaxRate == 0 {
		cfg.MaxRate = 10000
	}
	if len(cfg.Dimensions) == 0 {
		cfg.Dimensions = []DimensionType{DimensionIP}
	}

	ctx, cancel := context.WithCancel(context.Background())

	limiter := &AdaptiveLimiter{
		baseRate:     cfg.BaseRate,
		window:       cfg.Window,
		dimensions:   cfg.Dimensions,
		adaptEnabled: cfg.AdaptEnabled,
		learningRate: cfg.LearningRate,
		minRate:      cfg.MinRate,
		maxRate:      cfg.MaxRate,
		geoPolicy:    cfg.GeoPolicy,
		ctx:          ctx,
		cancel:       cancel,
	}

	// Start background workers
	go limiter.cleanupExpired()
	go limiter.adaptThresholds()

	return limiter
}

// Allow checks if request is allowed under rate limit
func (al *AdaptiveLimiter) Allow(ctx RequestContext) (*Decision, error) {
	// Generate composite key from active dimensions
	key := al.generateKey(ctx)

	// Get or create bucket
	bucket := al.getBucket(key)

	// Get reputation score
	rep := al.getReputation(key)

	// Calculate risk-adjusted rate
	adjustedRate := al.calculateAdjustedRate(bucket, rep, ctx)

	// Try to acquire token
	allowed := bucket.tryAcquire(adjustedRate)

	if allowed {
		atomic.AddUint64(&al.allowed, 1)
		rep.recordRequest(true)
	} else {
		atomic.AddUint64(&al.rejected, 1)
		rep.recordRequest(false)
	}

	decision := &Decision{
		Allowed:        allowed,
		RemainingQuota: bucket.remainingTokens(),
		ResetAt:        bucket.nextRefill(),
		Reason:         al.getReason(allowed, rep),
		RiskScore:      1.0 - rep.Score(),
	}

	return decision, nil
}

// Decision represents a rate limit decision
type Decision struct {
	Allowed        bool
	RemainingQuota int
	ResetAt        time.Time
	Reason         string
	RiskScore      float64
}

// RequestContext provides request attributes for rate limiting
type RequestContext struct {
	IP          string
	UserID      string
	Endpoint    string
	PayloadSize int
	Tenant      string
	Country     string
	Timestamp   time.Time
}

// generateKey creates a composite key from active dimensions
func (al *AdaptiveLimiter) generateKey(ctx RequestContext) string {
	// Build key from enabled dimensions
	parts := make([]string, 0, len(al.dimensions))

	for _, dim := range al.dimensions {
		switch dim {
		case DimensionIP:
			parts = append(parts, "ip:"+ctx.IP)
		case DimensionUser:
			parts = append(parts, "user:"+ctx.UserID)
		case DimensionEndpoint:
			parts = append(parts, "ep:"+ctx.Endpoint)
		case DimensionPayload:
			// Bucket by payload size range
			bucket := ctx.PayloadSize / 10000 // 10KB buckets
			parts = append(parts, fmt.Sprintf("payload:%d", bucket))
		case DimensionTenant:
			parts = append(parts, "tenant:"+ctx.Tenant)
		}
	}

	// Hash to fixed-length key
	h := sha256.Sum256([]byte(fmt.Sprintf("%v", parts)))
	return hex.EncodeToString(h[:])[:16]
}

// getBucket retrieves or creates a rate limit bucket
func (al *AdaptiveLimiter) getBucket(key string) *Bucket {
	if val, ok := al.buckets.Load(key); ok {
		return val.(*Bucket)
	}

	bucket := &Bucket{
		capacity:   al.baseRate,
		tokens:     al.baseRate,
		lastRefill: time.Now(),
		refillRate: al.baseRate,
		window:     al.window,
		multiplier: 1.0,
	}

	actual, _ := al.buckets.LoadOrStore(key, bucket)
	return actual.(*Bucket)
}

// getReputation retrieves or creates reputation score
func (al *AdaptiveLimiter) getReputation(key string) *ReputationScore {
	if val, ok := al.reputation.Load(key); ok {
		return val.(*ReputationScore)
	}

	rep := &ReputationScore{
		score:   0.5, // Neutral starting score
		created: time.Now(),
	}

	actual, _ := al.reputation.LoadOrStore(key, rep)
	return actual.(*ReputationScore)
}

// calculateAdjustedRate computes risk-adjusted rate limit
func (al *AdaptiveLimiter) calculateAdjustedRate(b *Bucket, rep *ReputationScore, ctx RequestContext) int {
	baseRate := float64(al.baseRate)

	// Apply bucket multiplier (adaptive learning)
	baseRate *= b.multiplier

	// Apply reputation score (good actors get higher limits)
	reputationMultiplier := 0.5 + rep.Score() // 0.5x to 1.5x
	baseRate *= reputationMultiplier

	// Apply geolocation policy
	if ctx.Country != "" {
		al.geoPolicyMu.RLock()
		if geoMult, ok := al.geoPolicy[ctx.Country]; ok {
			baseRate *= float64(geoMult)
		}
		al.geoPolicyMu.RUnlock()
	}

	// Clamp to min/max
	rate := int(baseRate)
	if rate < al.minRate {
		rate = al.minRate
	}
	if rate > al.maxRate {
		rate = al.maxRate
	}

	return rate
}

// tryAcquire attempts to acquire a token from bucket
func (b *Bucket) tryAcquire(requestedRate int) bool {
	b.mu.Lock()
	defer b.mu.Unlock()

	now := time.Now()

	// Refill tokens based on elapsed time
	elapsed := now.Sub(b.lastRefill)
	if elapsed >= b.window {
		tokensToAdd := int(float64(b.refillRate) * (elapsed.Seconds() / b.window.Seconds()))
		b.tokens += tokensToAdd
		if b.tokens > b.capacity {
			b.tokens = b.capacity
		}
		b.lastRefill = now
	}

	// Try to consume token
	if b.tokens > 0 {
		b.tokens--
		return true
	}

	return false
}

// remainingTokens returns current token count
func (b *Bucket) remainingTokens() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.tokens
}

// nextRefill returns time of next token refill
func (b *Bucket) nextRefill() time.Time {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.lastRefill.Add(b.window)
}

// recordRequest updates reputation based on request outcome
func (rep *ReputationScore) recordRequest(allowed bool) {
	rep.mu.Lock()
	defer rep.mu.Unlock()

	rep.requests++

	if !allowed {
		rep.violations++
		rep.lastViolation = time.Now()

		// Decay score on violation (exponential backoff)
		rep.score *= 0.9
		if rep.score < 0.0 {
			rep.score = 0.0
		}
	} else {
		// Gradually improve score for good behavior
		rep.score += 0.001 * (1.0 - rep.score)
		if rep.score > 1.0 {
			rep.score = 1.0
		}
	}
}

// Score returns current reputation score (0.0 to 1.0)
func (rep *ReputationScore) Score() float64 {
	rep.mu.RLock()
	defer rep.mu.RUnlock()
	return rep.score
}

// getReason returns human-readable rate limit reason
func (al *AdaptiveLimiter) getReason(allowed bool, rep *ReputationScore) string {
	if allowed {
		return "ok"
	}

	score := rep.Score()
	if score < 0.3 {
		return "rate_limit_low_reputation"
	}

	return "rate_limit_exceeded"
}

// cleanupExpired removes old buckets and reputation entries
func (al *AdaptiveLimiter) cleanupExpired() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-al.ctx.Done():
			return
		case <-ticker.C:
			cutoff := time.Now().Add(-30 * time.Minute)

			al.reputation.Range(func(key, value interface{}) bool {
				rep := value.(*ReputationScore)
				rep.mu.RLock()
				created := rep.created
				rep.mu.RUnlock()

				if created.Before(cutoff) {
					al.reputation.Delete(key)
					al.buckets.Delete(key)
				}
				return true
			})
		}
	}
}

// adaptThresholds continuously adjusts rate limits based on system health
func (al *AdaptiveLimiter) adaptThresholds() {
	if !al.adaptEnabled {
		return
	}

	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-al.ctx.Done():
			return
		case <-ticker.C:
			// Adjust bucket multipliers based on rejection rate
			allowed := atomic.LoadUint64(&al.allowed)
			rejected := atomic.LoadUint64(&al.rejected)
			total := allowed + rejected

			if total < 100 {
				continue // Not enough data
			}

			rejectionRate := float64(rejected) / float64(total)

			// Target rejection rate: 5-10%
			var adjustment float64
			if rejectionRate > 0.10 {
				// Too many rejections, increase limits
				adjustment = 1.0 + al.learningRate
			} else if rejectionRate < 0.05 {
				// Too few rejections, tighten limits
				adjustment = 1.0 - al.learningRate*0.5
			} else {
				continue // Within target range
			}

			// Apply adjustment to all buckets
			al.buckets.Range(func(key, value interface{}) bool {
				bucket := value.(*Bucket)
				bucket.mu.Lock()
				bucket.multiplier *= adjustment
				// Clamp multiplier
				if bucket.multiplier < 0.1 {
					bucket.multiplier = 0.1
				}
				if bucket.multiplier > 5.0 {
					bucket.multiplier = 5.0
				}
				bucket.mu.Unlock()
				return true
			})

			atomic.AddUint64(&al.adaptations, 1)

			// Reset counters
			atomic.StoreUint64(&al.allowed, 0)
			atomic.StoreUint64(&al.rejected, 0)
		}
	}
}

// UpdateGeoPolicy updates geolocation-based rate multipliers
func (al *AdaptiveLimiter) UpdateGeoPolicy(country string, multiplier int) {
	al.geoPolicyMu.Lock()
	defer al.geoPolicyMu.Unlock()

	if al.geoPolicy == nil {
		al.geoPolicy = make(map[string]int)
	}
	al.geoPolicy[country] = multiplier
}

// Metrics returns limiter metrics
func (al *AdaptiveLimiter) Metrics() map[string]interface{} {
	var bucketsCount int
	al.buckets.Range(func(_, _ interface{}) bool {
		bucketsCount++
		return true
	})

	var repCount int
	al.reputation.Range(func(_, _ interface{}) bool {
		repCount++
		return true
	})

	return map[string]interface{}{
		"allowed_total":       atomic.LoadUint64(&al.allowed),
		"rejected_total":      atomic.LoadUint64(&al.rejected),
		"adaptations_total":   atomic.LoadUint64(&al.adaptations),
		"active_buckets":      bucketsCount,
		"tracked_reputations": repCount,
	}
}

// Close stops the limiter
func (al *AdaptiveLimiter) Close() {
	al.cancel()
}

// Sliding Window Counter with Exponential Decay
type SlidingWindowLimiter struct {
	capacity  int
	window    time.Duration
	slots     []timeSlot
	slotCount int
	decay     float64 // Exponential decay factor

	mu sync.Mutex
}

type timeSlot struct {
	timestamp time.Time
	count     float64 // Float to support fractional counts with decay
}

// NewSlidingWindowLimiter creates a sliding window limiter with exponential decay
func NewSlidingWindowLimiter(capacity int, window time.Duration, slotCount int) *SlidingWindowLimiter {
	if slotCount == 0 {
		slotCount = 10
	}

	return &SlidingWindowLimiter{
		capacity:  capacity,
		window:    window,
		slots:     make([]timeSlot, slotCount),
		slotCount: slotCount,
		decay:     0.9, // Decay factor for older slots
	}
}

// Allow checks if request is allowed (sliding window with decay)
func (sw *SlidingWindowLimiter) Allow() bool {
	sw.mu.Lock()
	defer sw.mu.Unlock()

	now := time.Now()
	slotDuration := sw.window / time.Duration(sw.slotCount)
	currentSlot := int(now.UnixNano()/int64(slotDuration)) % sw.slotCount

	// Apply exponential decay to old slots and sum
	cutoff := now.Add(-sw.window)
	var weightedSum float64

	for i := range sw.slots {
		if sw.slots[i].timestamp.Before(cutoff) {
			sw.slots[i].count = 0
			continue
		}

		// Apply decay based on age
		age := now.Sub(sw.slots[i].timestamp).Seconds()
		decayFactor := math.Exp(-age / sw.window.Seconds())
		weightedSum += sw.slots[i].count * decayFactor
	}

	if int(weightedSum) >= sw.capacity {
		return false
	}

	// Increment current slot
	sw.slots[currentSlot].timestamp = now
	sw.slots[currentSlot].count++

	return true
}
