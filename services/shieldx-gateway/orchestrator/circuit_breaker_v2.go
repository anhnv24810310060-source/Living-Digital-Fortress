// Package main - Production Circuit Breaker với Adaptive Thresholds
// Implements intelligent failure detection and automatic recovery
package main

import (
	"fmt"
	"log"
	"math"
	"net/http"
	"sync"
	"sync/atomic"
	"time"
)

// CircuitBreakerV2 implements adaptive circuit breaker pattern
type CircuitBreakerV2 struct {
	mu sync.RWMutex

	// Backend reference
	backend *Backend

	// States
	state        atomic.Uint32 // 0=closed, 1=open, 2=half-open
	stateChanged time.Time

	// Failure tracking
	consecutiveFails atomic.Uint32
	totalRequests    atomic.Uint64
	totalFailures    atomic.Uint64
	recentFailures   *RingBuffer // For rate calculation

	// Adaptive thresholds
	failureThreshold atomic.Uint32 // Dynamic threshold
	minThreshold     uint32
	maxThreshold     uint32

	// Recovery parameters
	recoveryTimeout   time.Duration
	halfOpenRequests  uint32 // Number of test requests in half-open
	halfOpenSuccesses atomic.Uint32
	halfOpenFailures  atomic.Uint32

	// Metrics for adaptation
	historicalErrorRate atomic.Uint64 // x10000 (basis points)
	adaptationInterval  time.Duration
	lastAdaptation      time.Time

	// Timing
	nextProbeTime atomic.Int64 // Unix nano when next probe is allowed
}

// RingBuffer for tracking recent events
type RingBuffer struct {
	mu     sync.Mutex
	buffer []event
	index  int
	size   int
	full   bool
}

type event struct {
	timestamp time.Time
	failed    bool
}

// NewCircuitBreakerV2 creates an adaptive circuit breaker
func NewCircuitBreakerV2(backend *Backend) *CircuitBreakerV2 {
	cb := &CircuitBreakerV2{
		backend:            backend,
		minThreshold:       3,
		maxThreshold:       20,
		recoveryTimeout:    15 * time.Second,
		halfOpenRequests:   5,
		adaptationInterval: 60 * time.Second,
		recentFailures:     NewRingBuffer(100),
		lastAdaptation:     time.Now(),
	}

	// Initialize with middle threshold
	cb.failureThreshold.Store((cb.minThreshold + cb.maxThreshold) / 2)
	cb.state.Store(0) // Closed
	cb.stateChanged = time.Now()

	return cb
}

// NewRingBuffer creates a fixed-size ring buffer
func NewRingBuffer(size int) *RingBuffer {
	return &RingBuffer{
		buffer: make([]event, size),
		size:   size,
	}
}

// Add adds an event to the ring buffer
func (rb *RingBuffer) Add(evt event) {
	rb.mu.Lock()
	defer rb.mu.Unlock()

	rb.buffer[rb.index] = evt
	rb.index = (rb.index + 1) % rb.size

	if rb.index == 0 {
		rb.full = true
	}
}

// GetRecentFailureRate calculates failure rate in the last N seconds
func (rb *RingBuffer) GetRecentFailureRate(windowSec int) float64 {
	rb.mu.Lock()
	defer rb.mu.Unlock()

	now := time.Now()
	cutoff := now.Add(-time.Duration(windowSec) * time.Second)

	var total, failures int

	maxIdx := rb.size
	if !rb.full {
		maxIdx = rb.index
	}

	for i := 0; i < maxIdx; i++ {
		evt := rb.buffer[i]
		if evt.timestamp.After(cutoff) {
			total++
			if evt.failed {
				failures++
			}
		}
	}

	if total == 0 {
		return 0.0
	}

	return float64(failures) / float64(total)
}

// Call wraps a request with circuit breaker logic
func (cb *CircuitBreakerV2) Call(fn func() error) error {
	// Check if circuit is open
	if cb.isOpen() {
		return fmt.Errorf("circuit breaker open for %s", cb.backend.URL)
	}

	// Execute request
	cb.totalRequests.Add(1)
	err := fn()

	// Record result
	if err != nil {
		cb.recordFailure()
		return err
	}

	cb.recordSuccess()
	return nil
}

// isOpen checks if circuit is open (blocking requests)
func (cb *CircuitBreakerV2) isOpen() bool {
	state := cb.state.Load()

	switch state {
	case 0: // Closed
		return false

	case 1: // Open
		// Check if enough time has passed for probing
		now := time.Now()
		nextProbe := time.Unix(0, cb.nextProbeTime.Load())

		if now.After(nextProbe) {
			// Transition to half-open for probing
			if cb.state.CompareAndSwap(1, 2) {
				cb.mu.Lock()
				cb.stateChanged = now
				cb.mu.Unlock()
				cb.halfOpenSuccesses.Store(0)
				cb.halfOpenFailures.Store(0)
				mCBHalfOpen.Inc()
				return false // Allow probe
			}
		}
		return true

	case 2: // Half-open
		// Allow limited requests for testing
		successes := cb.halfOpenSuccesses.Load()
		failures := cb.halfOpenFailures.Load()

		if successes+failures >= cb.halfOpenRequests {
			// Enough samples, decide state
			if failures == 0 {
				// All successes, close circuit
				cb.transitionToClosed()
			} else if float64(failures)/float64(successes+failures) > 0.5 {
				// Too many failures, reopen
				cb.transitionToOpen()
			}
			// Otherwise stay half-open for more samples
		}

		return false // Allow request

	default:
		return false
	}
}

// recordFailure records a failed request
func (cb *CircuitBreakerV2) recordFailure() {
	cb.totalFailures.Add(1)
	fails := cb.consecutiveFails.Add(1)

	// Add to ring buffer
	cb.recentFailures.Add(event{
		timestamp: time.Now(),
		failed:    true,
	})

	state := cb.state.Load()

	if state == 0 { // Closed
		threshold := cb.failureThreshold.Load()
		if fails >= threshold {
			cb.transitionToOpen()
		}
	} else if state == 2 { // Half-open
		cb.halfOpenFailures.Add(1)
	}

	// Adapt threshold periodically
	cb.maybeAdaptThreshold()
}

// recordSuccess records a successful request
func (cb *CircuitBreakerV2) recordSuccess() {
	cb.consecutiveFails.Store(0)

	// Add to ring buffer
	cb.recentFailures.Add(event{
		timestamp: time.Now(),
		failed:    false,
	})

	state := cb.state.Load()

	if state == 2 { // Half-open
		cb.halfOpenSuccesses.Add(1)
	}
}

// transitionToOpen opens the circuit (blocking requests)
func (cb *CircuitBreakerV2) transitionToOpen() {
	if !cb.state.CompareAndSwap(0, 1) && !cb.state.CompareAndSwap(2, 1) {
		return // Already open or someone else transitioned
	}

	cb.mu.Lock()
	cb.stateChanged = time.Now()
	cb.mu.Unlock()

	// Schedule next probe attempt
	nextProbe := time.Now().Add(cb.recoveryTimeout)
	cb.nextProbeTime.Store(nextProbe.UnixNano())

	// Update backend state
	cb.backend.Healthy.Store(false)
	cb.backend.cbState.Store(1)
	cb.backend.cbNextProbe.Store(nextProbe.UnixNano())

	mCBOpen.Inc()
}

// transitionToClosed closes the circuit (normal operation)
func (cb *CircuitBreakerV2) transitionToClosed() {
	if !cb.state.CompareAndSwap(2, 0) && !cb.state.CompareAndSwap(1, 0) {
		return
	}

	cb.mu.Lock()
	cb.stateChanged = time.Now()
	cb.mu.Unlock()

	cb.consecutiveFails.Store(0)

	// Update backend state
	cb.backend.Healthy.Store(true)
	cb.backend.cbState.Store(0)
	cb.backend.cbFails.Store(0)

	mCBClose.Inc()
}

// maybeAdaptThreshold adjusts failure threshold based on historical error rate
func (cb *CircuitBreakerV2) maybeAdaptThreshold() {
	now := time.Now()

	cb.mu.RLock()
	lastAdapt := cb.lastAdaptation
	cb.mu.RUnlock()

	if now.Sub(lastAdapt) < cb.adaptationInterval {
		return // Too soon
	}

	cb.mu.Lock()
	defer cb.mu.Unlock()

	// Double-check after acquiring lock
	if now.Sub(cb.lastAdaptation) < cb.adaptationInterval {
		return
	}

	// Calculate historical error rate
	totalReqs := cb.totalRequests.Load()
	totalFails := cb.totalFailures.Load()

	if totalReqs < 100 {
		return // Insufficient data
	}

	errorRate := float64(totalFails) / float64(totalReqs)
	cb.historicalErrorRate.Store(uint64(errorRate * 10000))

	// Also check recent error rate from ring buffer
	recentErrorRate := cb.recentFailures.GetRecentFailureRate(60)

	// Adapt threshold based on error rates
	currentThreshold := cb.failureThreshold.Load()
	newThreshold := currentThreshold

	if errorRate < 0.01 && recentErrorRate < 0.05 {
		// Very stable, increase threshold (more tolerant)
		newThreshold = uint32(math.Min(float64(currentThreshold+2), float64(cb.maxThreshold)))
	} else if errorRate > 0.05 || recentErrorRate > 0.2 {
		// Unstable, decrease threshold (more sensitive)
		newThreshold = uint32(math.Max(float64(currentThreshold-2), float64(cb.minThreshold)))
	}

	if newThreshold != currentThreshold {
		cb.failureThreshold.Store(newThreshold)
		log.Printf("[cb] Adapted failure threshold for %s: %d -> %d (error_rate: %.2f%%, recent: %.2f%%)",
			cb.backend.URL, currentThreshold, newThreshold, errorRate*100, recentErrorRate*100)
	}

	cb.lastAdaptation = now
}

// GetState returns current circuit breaker state
func (cb *CircuitBreakerV2) GetState() string {
	switch cb.state.Load() {
	case 0:
		return "closed"
	case 1:
		return "open"
	case 2:
		return "half-open"
	default:
		return "unknown"
	}
}

// Metrics returns circuit breaker metrics
func (cb *CircuitBreakerV2) Metrics() map[string]interface{} {
	state := cb.GetState()

	cb.mu.RLock()
	stateChangedAgo := time.Since(cb.stateChanged)
	cb.mu.RUnlock()

	totalReqs := cb.totalRequests.Load()
	totalFails := cb.totalFailures.Load()

	errorRate := 0.0
	if totalReqs > 0 {
		errorRate = float64(totalFails) / float64(totalReqs)
	}

	recentErrorRate := cb.recentFailures.GetRecentFailureRate(60)

	return map[string]interface{}{
		"state":                 state,
		"state_changed_ago_sec": stateChangedAgo.Seconds(),
		"failure_threshold":     cb.failureThreshold.Load(),
		"consecutive_failures":  cb.consecutiveFails.Load(),
		"total_requests":        totalReqs,
		"total_failures":        totalFails,
		"error_rate":            errorRate,
		"recent_error_rate_60s": recentErrorRate,
		"half_open_successes":   cb.halfOpenSuccesses.Load(),
		"half_open_failures":    cb.halfOpenFailures.Load(),
	}
}

// ---------- Circuit Breaker Manager ----------

// CircuitBreakerManager manages circuit breakers for all backends
type CircuitBreakerManager struct {
	mu             sync.RWMutex
	breakers       map[string]*CircuitBreakerV2
	defaultTimeout time.Duration
}

// NewCircuitBreakerManager creates a circuit breaker manager
func NewCircuitBreakerManager() *CircuitBreakerManager {
	return &CircuitBreakerManager{
		breakers:       make(map[string]*CircuitBreakerV2),
		defaultTimeout: 15 * time.Second,
	}
}

// GetOrCreate returns existing or creates new circuit breaker for backend
func (cbm *CircuitBreakerManager) GetOrCreate(backend *Backend) *CircuitBreakerV2 {
	cbm.mu.RLock()
	if cb, ok := cbm.breakers[backend.URL]; ok {
		cbm.mu.RUnlock()
		return cb
	}
	cbm.mu.RUnlock()

	cbm.mu.Lock()
	defer cbm.mu.Unlock()

	// Double-check after acquiring write lock
	if cb, ok := cbm.breakers[backend.URL]; ok {
		return cb
	}

	cb := NewCircuitBreakerV2(backend)
	cbm.breakers[backend.URL] = cb

	return cb
}

// Remove removes circuit breaker for backend
func (cbm *CircuitBreakerManager) Remove(backendURL string) {
	cbm.mu.Lock()
	defer cbm.mu.Unlock()
	delete(cbm.breakers, backendURL)
}

// GetAll returns all circuit breakers
func (cbm *CircuitBreakerManager) GetAll() map[string]*CircuitBreakerV2 {
	cbm.mu.RLock()
	defer cbm.mu.RUnlock()

	result := make(map[string]*CircuitBreakerV2, len(cbm.breakers))
	for k, v := range cbm.breakers {
		result[k] = v
	}
	return result
}

// Metrics returns aggregated circuit breaker metrics
func (cbm *CircuitBreakerManager) Metrics() map[string]interface{} {
	cbm.mu.RLock()
	defer cbm.mu.RUnlock()

	var closed, open, halfOpen int

	for _, cb := range cbm.breakers {
		switch cb.state.Load() {
		case 0:
			closed++
		case 1:
			open++
		case 2:
			halfOpen++
		}
	}

	return map[string]interface{}{
		"total":     len(cbm.breakers),
		"closed":    closed,
		"open":      open,
		"half_open": halfOpen,
	}
}

// Global circuit breaker manager instance
var globalCBManager *CircuitBreakerManager

func init() {
	globalCBManager = NewCircuitBreakerManager()
}

// handleCircuitBreakerMetrics returns circuit breaker status for all backends
func handleCircuitBreakerMetrics(w http.ResponseWriter, r *http.Request) {
	breakers := globalCBManager.GetAll()

	metrics := make(map[string]interface{})
	metrics["summary"] = globalCBManager.Metrics()

	backends := make(map[string]interface{})
	for url, cb := range breakers {
		backends[url] = cb.Metrics()
	}
	metrics["backends"] = backends

	writeJSON(w, 200, metrics)
}
