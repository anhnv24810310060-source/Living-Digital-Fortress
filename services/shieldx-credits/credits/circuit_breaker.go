package main

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"sync"
	"time"
)

// CircuitBreaker implements the circuit breaker pattern for fault tolerance
// Prevents cascading failures by failing fast when downstream services are unhealthy
type CircuitBreaker struct {
	name         string
	maxFailures  int
	timeout      time.Duration
	resetTimeout time.Duration

	// State tracking
	state           CircuitState
	failures        int
	lastFailTime    time.Time
	lastStateChange time.Time

	// Statistics
	totalCalls      int64
	successfulCalls int64
	failedCalls     int64
	rejectedCalls   int64

	mutex sync.RWMutex
}

// CircuitState represents the state of the circuit breaker
type CircuitState string

const (
	StateClosed   CircuitState = "closed"    // Normal operation
	StateOpen     CircuitState = "open"      // Failing fast
	StateHalfOpen CircuitState = "half_open" // Testing recovery
)

// CircuitBreakerConfig holds configuration for circuit breaker
type CircuitBreakerConfig struct {
	Name         string        // Name for monitoring
	MaxFailures  int           // Consecutive failures before opening
	Timeout      time.Duration // How long to wait before attempting call
	ResetTimeout time.Duration // How long to stay open before trying half-open
}

// NewCircuitBreaker creates a new circuit breaker
func NewCircuitBreaker(config CircuitBreakerConfig) *CircuitBreaker {
	if config.MaxFailures <= 0 {
		config.MaxFailures = 5
	}
	if config.Timeout == 0 {
		config.Timeout = 30 * time.Second
	}
	if config.ResetTimeout == 0 {
		config.ResetTimeout = 60 * time.Second
	}

	return &CircuitBreaker{
		name:            config.Name,
		maxFailures:     config.MaxFailures,
		timeout:         config.Timeout,
		resetTimeout:    config.ResetTimeout,
		state:           StateClosed,
		lastStateChange: time.Now(),
	}
}

// Execute executes the function with circuit breaker protection
func (cb *CircuitBreaker) Execute(ctx context.Context, fn func(context.Context) error) error {
	// Check if we should attempt the call
	if err := cb.beforeCall(); err != nil {
		return err
	}

	// Create a context with timeout
	callCtx, cancel := context.WithTimeout(ctx, cb.timeout)
	defer cancel()

	// Execute the function
	errChan := make(chan error, 1)
	go func() {
		errChan <- fn(callCtx)
	}()

	// Wait for result or timeout
	select {
	case err := <-errChan:
		cb.afterCall(err)
		return err
	case <-callCtx.Done():
		err := callCtx.Err()
		cb.afterCall(err)
		return fmt.Errorf("circuit breaker timeout: %w", err)
	}
}

// beforeCall checks if the call should be allowed
func (cb *CircuitBreaker) beforeCall() error {
	cb.mutex.Lock()
	defer cb.mutex.Unlock()

	cb.totalCalls++

	switch cb.state {
	case StateClosed:
		// Normal operation - allow call
		return nil

	case StateOpen:
		// Check if it's time to try half-open
		if time.Since(cb.lastStateChange) >= cb.resetTimeout {
			cb.setState(StateHalfOpen)
			return nil
		}
		// Still open - reject call
		cb.rejectedCalls++
		return fmt.Errorf("circuit breaker %s is open", cb.name)

	case StateHalfOpen:
		// Allow one call to test if service recovered
		return nil

	default:
		return fmt.Errorf("unknown circuit breaker state: %s", cb.state)
	}
}

// afterCall updates circuit breaker state based on call result
func (cb *CircuitBreaker) afterCall(err error) {
	cb.mutex.Lock()
	defer cb.mutex.Unlock()

	if err != nil {
		// Call failed
		cb.failedCalls++
		cb.failures++
		cb.lastFailTime = time.Now()

		switch cb.state {
		case StateClosed:
			if cb.failures >= cb.maxFailures {
				cb.setState(StateOpen)
			}

		case StateHalfOpen:
			// Failed during recovery test - go back to open
			cb.setState(StateOpen)
		}
	} else {
		// Call succeeded
		cb.successfulCalls++

		switch cb.state {
		case StateClosed:
			// Reset failure count on success
			cb.failures = 0

		case StateHalfOpen:
			// Success in half-open - fully recover
			cb.failures = 0
			cb.setState(StateClosed)
		}
	}
}

// setState changes the circuit breaker state
func (cb *CircuitBreaker) setState(newState CircuitState) {
	if cb.state != newState {
		oldState := cb.state
		cb.state = newState
		cb.lastStateChange = time.Now()

		// Log state change
		fmt.Printf("[CircuitBreaker:%s] State changed: %s -> %s (failures: %d)\n",
			cb.name, oldState, newState, cb.failures)
	}
}

// GetState returns the current state (thread-safe)
func (cb *CircuitBreaker) GetState() CircuitState {
	cb.mutex.RLock()
	defer cb.mutex.RUnlock()
	return cb.state
}

// GetStats returns current statistics
func (cb *CircuitBreaker) GetStats() CircuitBreakerStats {
	cb.mutex.RLock()
	defer cb.mutex.RUnlock()

	return CircuitBreakerStats{
		Name:            cb.name,
		State:           string(cb.state),
		TotalCalls:      cb.totalCalls,
		SuccessfulCalls: cb.successfulCalls,
		FailedCalls:     cb.failedCalls,
		RejectedCalls:   cb.rejectedCalls,
		CurrentFailures: cb.failures,
		LastFailTime:    cb.lastFailTime,
		LastStateChange: cb.lastStateChange,
		SuccessRate:     cb.calculateSuccessRate(),
	}
}

// CircuitBreakerStats holds statistics about circuit breaker
type CircuitBreakerStats struct {
	Name            string    `json:"name"`
	State           string    `json:"state"`
	TotalCalls      int64     `json:"total_calls"`
	SuccessfulCalls int64     `json:"successful_calls"`
	FailedCalls     int64     `json:"failed_calls"`
	RejectedCalls   int64     `json:"rejected_calls"`
	CurrentFailures int       `json:"current_failures"`
	LastFailTime    time.Time `json:"last_fail_time"`
	LastStateChange time.Time `json:"last_state_change"`
	SuccessRate     float64   `json:"success_rate"`
}

// calculateSuccessRate computes the success rate
func (cb *CircuitBreaker) calculateSuccessRate() float64 {
	total := cb.successfulCalls + cb.failedCalls
	if total == 0 {
		return 0.0
	}
	return float64(cb.successfulCalls) / float64(total)
}

// Reset resets the circuit breaker to closed state
func (cb *CircuitBreaker) Reset() {
	cb.mutex.Lock()
	defer cb.mutex.Unlock()

	cb.state = StateClosed
	cb.failures = 0
	cb.lastStateChange = time.Now()
}

// CircuitBreakerPool manages multiple circuit breakers
type CircuitBreakerPool struct {
	breakers map[string]*CircuitBreaker
	mutex    sync.RWMutex
}

// NewCircuitBreakerPool creates a new pool
func NewCircuitBreakerPool() *CircuitBreakerPool {
	return &CircuitBreakerPool{
		breakers: make(map[string]*CircuitBreaker),
	}
}

// GetOrCreate gets or creates a circuit breaker
func (pool *CircuitBreakerPool) GetOrCreate(name string, config CircuitBreakerConfig) *CircuitBreaker {
	pool.mutex.RLock()
	if cb, exists := pool.breakers[name]; exists {
		pool.mutex.RUnlock()
		return cb
	}
	pool.mutex.RUnlock()

	pool.mutex.Lock()
	defer pool.mutex.Unlock()

	// Double-check after acquiring write lock
	if cb, exists := pool.breakers[name]; exists {
		return cb
	}

	config.Name = name
	cb := NewCircuitBreaker(config)
	pool.breakers[name] = cb
	return cb
}

// Get gets a circuit breaker by name
func (pool *CircuitBreakerPool) Get(name string) (*CircuitBreaker, error) {
	pool.mutex.RLock()
	defer pool.mutex.RUnlock()

	cb, exists := pool.breakers[name]
	if !exists {
		return nil, fmt.Errorf("circuit breaker %s not found", name)
	}
	return cb, nil
}

// GetAllStats returns stats for all circuit breakers
func (pool *CircuitBreakerPool) GetAllStats() []CircuitBreakerStats {
	pool.mutex.RLock()
	defer pool.mutex.RUnlock()

	stats := make([]CircuitBreakerStats, 0, len(pool.breakers))
	for _, cb := range pool.breakers {
		stats = append(stats, cb.GetStats())
	}
	return stats
}

// ResetAll resets all circuit breakers
func (pool *CircuitBreakerPool) ResetAll() {
	pool.mutex.RLock()
	defer pool.mutex.RUnlock()

	for _, cb := range pool.breakers {
		cb.Reset()
	}
}

// RateLimiter implements token bucket algorithm for rate limiting
type RateLimiter struct {
	capacity   int64 // Maximum tokens
	tokens     int64 // Current tokens
	refillRate int64 // Tokens per second
	lastRefill time.Time
	mutex      sync.Mutex
}

// NewRateLimiter creates a new rate limiter
func NewRateLimiter(capacity, refillRate int64) *RateLimiter {
	return &RateLimiter{
		capacity:   capacity,
		tokens:     capacity,
		refillRate: refillRate,
		lastRefill: time.Now(),
	}
}

// Allow checks if request is allowed (consumes 1 token)
func (rl *RateLimiter) Allow() bool {
	return rl.AllowN(1)
}

// AllowN checks if N tokens are available
func (rl *RateLimiter) AllowN(n int64) bool {
	rl.mutex.Lock()
	defer rl.mutex.Unlock()

	rl.refill()

	if rl.tokens >= n {
		rl.tokens -= n
		return true
	}
	return false
}

// refill adds tokens based on elapsed time
func (rl *RateLimiter) refill() {
	now := time.Now()
	elapsed := now.Sub(rl.lastRefill)
	tokensToAdd := int64(elapsed.Seconds()) * rl.refillRate

	if tokensToAdd > 0 {
		rl.tokens = min(rl.capacity, rl.tokens+tokensToAdd)
		rl.lastRefill = now
	}
}

// GetTokens returns current available tokens
func (rl *RateLimiter) GetTokens() int64 {
	rl.mutex.Lock()
	defer rl.mutex.Unlock()

	rl.refill()
	return rl.tokens
}

// Reset resets tokens to capacity
func (rl *RateLimiter) Reset() {
	rl.mutex.Lock()
	defer rl.mutex.Unlock()

	rl.tokens = rl.capacity
	rl.lastRefill = time.Now()
}

func min(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}

// RetryPolicy implements exponential backoff with jitter
type RetryPolicy struct {
	MaxRetries     int
	InitialBackoff time.Duration
	MaxBackoff     time.Duration
	Multiplier     float64
}

// DefaultRetryPolicy returns a sensible default retry policy
func DefaultRetryPolicy() RetryPolicy {
	return RetryPolicy{
		MaxRetries:     3,
		InitialBackoff: 100 * time.Millisecond,
		MaxBackoff:     10 * time.Second,
		Multiplier:     2.0,
	}
}

// ExecuteWithRetry executes a function with retry logic
func (rp RetryPolicy) ExecuteWithRetry(ctx context.Context, fn func(context.Context) error) error {
	var lastErr error
	backoff := rp.InitialBackoff

	for attempt := 0; attempt <= rp.MaxRetries; attempt++ {
		if attempt > 0 {
			// Add jitter to prevent thundering herd
			jitter := time.Duration(float64(backoff) * 0.1 * (2.0*rand.Float64() - 1.0))
			sleepDuration := backoff + jitter

			if sleepDuration > rp.MaxBackoff {
				sleepDuration = rp.MaxBackoff
			}

			select {
			case <-time.After(sleepDuration):
			case <-ctx.Done():
				return ctx.Err()
			}

			backoff = time.Duration(float64(backoff) * rp.Multiplier)
		}

		err := fn(ctx)
		if err == nil {
			return nil
		}

		// Check if error is retryable
		if !isRetryable(err) {
			return err
		}

		lastErr = err
	}

	return fmt.Errorf("max retries exceeded: %w", lastErr)
}

// isRetryable checks if an error should trigger a retry
func isRetryable(err error) bool {
	if err == nil {
		return false
	}

	// Add custom logic to determine if error is retryable
	// For now, retry on timeout and temporary errors
	if errors.Is(err, context.DeadlineExceeded) {
		return true
	}

	return false
}
