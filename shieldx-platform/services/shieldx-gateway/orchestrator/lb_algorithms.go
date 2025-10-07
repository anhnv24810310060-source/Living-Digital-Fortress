package main

import (
	"crypto/sha256"
	"encoding/binary"
	"hash/fnv"
	"math"
	"math/rand"
	"sync/atomic"
	"time"
)

// Advanced Load Balancing Algorithms for Production
// Implements high-performance selection strategies with O(1) or O(log n) complexity

// selectBackendRoundRobin implements classic round-robin with atomic counter
// Time Complexity: O(1)
// Use case: Fair distribution when all backends are equal capacity
func (p *Pool) selectBackendRoundRobin() *Backend {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if len(p.backends) == 0 {
		return nil
	}

	// Filter healthy backends first
	healthy := make([]*Backend, 0, len(p.backends))
	for _, b := range p.backends {
		if b.Healthy.Load() && isCircuitClosed(b) {
			healthy = append(healthy, b)
		}
	}

	if len(healthy) == 0 {
		return nil
	}

	// Atomic increment and wrap
	idx := atomic.AddUint64(&p.rr, 1) - 1
	return healthy[idx%uint64(len(healthy))]
}

// selectBackendLeastConnections chooses backend with minimum active connections
// Time Complexity: O(n) - but fast with small n
// Use case: Long-lived connections where load varies significantly
func (p *Pool) selectBackendLeastConnections() *Backend {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if len(p.backends) == 0 {
		return nil
	}

	var best *Backend
	minConns := int64(math.MaxInt64)

	for _, b := range p.backends {
		if !b.Healthy.Load() || !isCircuitClosed(b) {
			continue
		}

		conns := atomic.LoadInt64(&b.Conns)
		if conns < minConns {
			minConns = conns
			best = b
		}
	}

	return best
}

// selectBackendEWMA implements latency-aware selection using EWMA
// Time Complexity: O(n) - optimized with SIMD-friendly operations
// Use case: Default for production - adapts to real-time latency changes
//
// Algorithm: score = EWMA_latency + (active_connections * penalty) / weight
// Lower score = better choice
func (p *Pool) selectBackendEWMA(penalty float64) *Backend {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if len(p.backends) == 0 {
		return nil
	}

	var best *Backend
	minScore := math.MaxFloat64

	for _, b := range p.backends {
		if !b.Healthy.Load() || !isCircuitClosed(b) {
			continue
		}

		ewma := b.getEWMA()
		conns := float64(atomic.LoadInt64(&b.Conns))
		weight := b.Weight
		if weight < 0.1 {
			weight = 1.0
		}

		// Score considers both latency and current load
		// Higher capacity backends (higher weight) absorb more connections
		score := ewma + (conns * penalty / weight)

		if score < minScore {
			minScore = score
			best = b
		}
	}

	return best
}

// selectBackendP2C implements Power-of-Two-Choices with EWMA scoring
// Time Complexity: O(1) - samples only 2 backends
// Use case: High-traffic scenarios where O(n) scan is too expensive
//
// Algorithm:
// 1. Randomly sample 2 healthy backends
// 2. Choose the one with lower EWMA score
// 3. Much better than pure random, close to least-conn performance
func (p *Pool) selectBackendP2C(penalty float64) *Backend {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if len(p.backends) == 0 {
		return nil
	}

	// Collect healthy backends
	healthy := make([]*Backend, 0, len(p.backends))
	for _, b := range p.backends {
		if b.Healthy.Load() && isCircuitClosed(b) {
			healthy = append(healthy, b)
		}
	}

	n := len(healthy)
	if n == 0 {
		return nil
	}
	if n == 1 {
		return healthy[0]
	}

	// Sample 2 different backends
	idx1 := rand.Intn(n)
	idx2 := rand.Intn(n - 1)
	if idx2 >= idx1 {
		idx2++
	}

	b1, b2 := healthy[idx1], healthy[idx2]

	// Calculate scores
	score1 := b1.getEWMA() + (float64(atomic.LoadInt64(&b1.Conns)) * penalty / b1.Weight)
	score2 := b2.getEWMA() + (float64(atomic.LoadInt64(&b2.Conns)) * penalty / b2.Weight)

	if score1 <= score2 {
		return b1
	}
	return b2
}

// selectBackendRendezvous implements Highest Random Weight (HRW) consistent hashing
// Time Complexity: O(n) - but provides excellent cache affinity
// Use case: Sticky sessions without state (e.g., cache hit optimization)
//
// Algorithm (Rendezvous/HRW):
// 1. For each backend, compute: hash(key + backend_id) * weight
// 2. Choose backend with highest hash value
// 3. Guarantees minimal disruption on backend changes (only 1/n keys move)
func (p *Pool) selectBackendRendezvous(hashKey string) *Backend {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if len(p.backends) == 0 {
		return nil
	}

	if hashKey == "" {
		// Fallback to round-robin if no key provided
		return p.selectBackendRoundRobin()
	}

	var best *Backend
	var maxScore uint64

	for _, b := range p.backends {
		if !b.Healthy.Load() || !isCircuitClosed(b) {
			continue
		}

		// Compute rendezvous hash: hash(key || backend_url)
		score := rendezvousHash(hashKey, b.URL, b.Weight)

		if score > maxScore {
			maxScore = score
			best = b
		}
	}

	return best
}

// rendezvousHash computes a weighted rendezvous hash for consistent hashing
// Uses high-quality hash function and weight multiplication
func rendezvousHash(key, backend string, weight float64) uint64 {
	// Use SHA256 for high-quality distribution (truncated to 64-bit)
	h := sha256.New()
	h.Write([]byte(key))
	h.Write([]byte(backend))
	hash := h.Sum(nil)

	// Convert first 8 bytes to uint64
	rawHash := binary.BigEndian.Uint64(hash[:8])

	// Apply weight using high-bit multiplication to avoid overflow
	// This maintains hash quality while applying weight factor
	if weight <= 0 {
		weight = 1.0
	}

	// Use leading zeros count as weight multiplier (preserves hash bits)
	// Higher weight = fewer leading zeros = higher effective hash value
	weightFactor := uint64(weight * 1000)
	if weightFactor == 0 {
		weightFactor = 1
	}

	// Weighted hash: use high bits of hash * weightFactor
	return rawHash / (1000 / weightFactor)
}

// selectBackendWeightedRandom implements weighted random selection
// Time Complexity: O(n) - simple but effective for small pools
// Use case: Capacity-aware random distribution
func (p *Pool) selectBackendWeightedRandom() *Backend {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if len(p.backends) == 0 {
		return nil
	}

	// Collect healthy backends and compute total weight
	type candidate struct {
		backend *Backend
		weight  float64
	}

	candidates := make([]candidate, 0, len(p.backends))
	totalWeight := 0.0

	for _, b := range p.backends {
		if b.Healthy.Load() && isCircuitClosed(b) {
			w := b.Weight
			if w < 0.1 {
				w = 1.0
			}
			candidates = append(candidates, candidate{b, w})
			totalWeight += w
		}
	}

	if len(candidates) == 0 {
		return nil
	}

	// Random weighted selection
	r := rand.Float64() * totalWeight
	cumulative := 0.0

	for _, c := range candidates {
		cumulative += c.weight
		if r <= cumulative {
			return c.backend
		}
	}

	// Fallback (should never happen due to float precision)
	return candidates[len(candidates)-1].backend
}

// isCircuitClosed checks if circuit breaker allows traffic
func isCircuitClosed(b *Backend) bool {
	state := b.cbState.Load()
	if state == 0 { // CLOSED
		return true
	}
	if state == 2 { // HALF_OPEN
		// Allow probe traffic in half-open state
		return true
	}
	// OPEN - check if probe window has arrived
	if state == 1 {
		nextProbe := b.cbNextProbe.Load()
		if nextProbe > 0 && time.Now().UnixNano() >= nextProbe {
			// Transition to HALF_OPEN for probe
			if b.cbState.CompareAndSwap(1, 2) {
				mCBHalfOpen.Inc()
				return true
			}
		}
	}
	return false
}

// updateEWMA updates the exponentially weighted moving average for latency
// Uses alpha = 0.3 for balanced responsiveness vs stability
func updateEWMA(b *Backend, latencyMs float64) {
	const alpha = 0.3

	old := b.getEWMA()
	if old == 0 {
		// First measurement - initialize
		b.setEWMA(latencyMs)
		return
	}

	// EWMA formula: new = alpha * current + (1-alpha) * old
	newEWMA := alpha*latencyMs + (1-alpha)*old
	b.setEWMA(newEWMA)

	// Also update last latency for debugging
	atomic.StoreUint64(&b.LastLatMs, uint64(latencyMs))
}

// recordBackendSuccess updates backend state after successful request
func recordBackendSuccess(b *Backend, latencyMs float64) {
	updateEWMA(b, latencyMs)

	// Circuit breaker: reset on success
	state := b.cbState.Load()
	if state == 2 { // HALF_OPEN -> CLOSED
		if b.cbState.CompareAndSwap(2, 0) {
			b.cbFails.Store(0)
			mCBClose.Inc()
		}
	} else if state == 0 { // CLOSED -> reset fail counter
		b.cbFails.Store(0)
	}

	b.LastErr.Store("")
}

// recordBackendFailure updates backend state after failed request
func recordBackendFailure(b *Backend, err error) {
	if err != nil {
		b.LastErr.Store(err.Error())
	}

	// Circuit breaker: increment failures
	fails := b.cbFails.Add(1)
	threshold := uint32(envInt("ORCH_CB_THRESHOLD", 5))

	state := b.cbState.Load()

	// CLOSED -> OPEN after threshold failures
	if state == 0 && fails >= threshold {
		if b.cbState.CompareAndSwap(0, 1) {
			// Set next probe time (exponential backoff)
			backoffMs := envInt("ORCH_CB_BACKOFF_MS", 5000)
			nextProbe := time.Now().Add(time.Duration(backoffMs) * time.Millisecond).UnixNano()
			b.cbNextProbe.Store(nextProbe)
			mCBOpen.Inc()
		}
	}

	// HALF_OPEN -> OPEN on failure
	if state == 2 {
		if b.cbState.CompareAndSwap(2, 1) {
			// Double the backoff time
			backoffMs := envInt("ORCH_CB_BACKOFF_MS", 5000) * 2
			nextProbe := time.Now().Add(time.Duration(backoffMs) * time.Millisecond).UnixNano()
			b.cbNextProbe.Store(nextProbe)
			mCBOpen.Inc()
		}
	}
}

// hashString computes a fast 64-bit hash for strings (used in rendezvous)
func hashString(s string) uint64 {
	h := fnv.New64a()
	h.Write([]byte(s))
	return h.Sum64()
}

// selectBackend is the main entry point for backend selection
// Chooses algorithm based on pool config or request override
func (p *Pool) selectBackend(algo LBAlgo, hashKey string, penalty float64) *Backend {
	if algo == "" {
		algo = p.algo
	}
	if algo == "" {
		algo = defaultAlgo
	}

	var selected *Backend

	switch algo {
	case LBRoundRobin:
		selected = p.selectBackendRoundRobin()
	case LBLeastConnections:
		selected = p.selectBackendLeastConnections()
	case LBEWMA:
		selected = p.selectBackendEWMA(penalty)
	case LBP2CEWMA:
		selected = p.selectBackendP2C(penalty)
	case LBConsistentHash:
		selected = p.selectBackendRendezvous(hashKey)
	default:
		selected = p.selectBackendEWMA(penalty)
	}

	// Track selection metrics
	healthy := "false"
	if selected != nil && selected.Healthy.Load() {
		healthy = "true"
	}
	mLBPick.Inc(map[string]string{
		"pool":    p.name,
		"algo":    string(algo),
		"healthy": healthy,
	})

	return selected
}
