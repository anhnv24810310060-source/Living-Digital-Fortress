package circuitbreakerpackage circuitbreaker


import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"time"
)

// State represents circuit breaker state
type State int32

const (
	StateClosed State = iota
	StateOpen
	StateHalfOpen
)

func (s State) String() string {
	switch s {
	case StateClosed:
		return "closed"
	case StateOpen:
		return "open"
	case StateHalfOpen:
		return "half-open"
	default:
		return "unknown"
	}
}

var (
	// ErrCircuitOpen is returned when circuit is open
	ErrCircuitOpen = errors.New("circuit breaker is open")
	// ErrTooManyRequests is returned when too many requests in half-open state
	ErrTooManyRequests = errors.New("too many requests in half-open state")
)

// Settings for circuit breaker behavior
type Settings struct {
	// MaxRequests: max concurrent requests allowed in half-open state
	MaxRequests uint32
	// Interval: statistical window for closed state (count failures in this window)
	Interval time.Duration
	// Timeout: duration to stay in open state before transitioning to half-open
	Timeout time.Duration
	// FailureThreshold: number of consecutive failures to open circuit
	FailureThreshold uint32
	// SuccessThreshold: number of consecutive successes to close circuit from half-open
	SuccessThreshold uint32
	// OnStateChange: callback when state changes
	OnStateChange func(name string, from State, to State)
}

// DefaultSettings returns production-ready circuit breaker settings
func DefaultSettings() Settings {
	return Settings{
		MaxRequests:      1,                // allow 1 request to probe in half-open
		Interval:         60 * time.Second, // 1-minute statistical window
		Timeout:          30 * time.Second, // 30s before retry
		FailureThreshold: 5,                // 5 consecutive failures trigger open
		SuccessThreshold: 2,                // 2 consecutive successes close circuit
	}
}

// CircuitBreaker implements adaptive circuit breaker pattern
// Algorithm: State machine with exponential backoff and success rate tracking
type CircuitBreaker struct {
	name     string
	settings Settings

	state     atomic.Value // State
	mu        sync.Mutex
	counts    *counters
	expiry    time.Time // when to transition from Open to HalfOpen
	halfOpen  uint32    // atomic counter for half-open requests
	lastError error
}

type counters struct {
	requests       uint32
	successes      uint32
	failures       uint32
	consecutiveFail uint32
	consecutiveSucc uint32
}

// NewCircuitBreaker creates a production-grade circuit breaker
func NewCircuitBreaker(name string, settings Settings) *CircuitBreaker {
	if settings.FailureThreshold == 0 {
		settings.FailureThreshold = 5
	}
	if settings.SuccessThreshold == 0 {
		settings.SuccessThreshold = 2
	}
	if settings.Timeout == 0 {
		settings.Timeout = 30 * time.Second
	}
	if settings.Interval == 0 {
		settings.Interval = 60 * time.Second
	}
	if settings.MaxRequests == 0 {
		settings.MaxRequests = 1
	}

	cb := &CircuitBreaker{
		name:     name,
		settings: settings,
		counts:   &counters{},
	}
	cb.state.Store(StateClosed)
	return cb
}

// Execute runs function with circuit breaker protection
func (cb *CircuitBreaker) Execute(ctx context.Context, fn func() error) error {
	// Check if request is allowed
	generation, err := cb.beforeRequest()
	if err != nil {
		return err
	}

	// Execute function
	defer func() {
		if r := recover(); r != nil {
			cb.afterRequest(generation, false, errors.New("panic recovered"))
			panic(r) // re-panic after recording
		}
	}()

	err = fn()
	cb.afterRequest(generation, err == nil, err)
	return err
}

// Call is alias for Execute for backward compatibility
func (cb *CircuitBreaker) Call(fn func() error) error {
	return cb.Execute(context.Background(), fn)
}

// State returns current circuit breaker state
func (cb *CircuitBreaker) State() State {
	return cb.state.Load().(State)
}

// Counts returns current statistics
func (cb *CircuitBreaker) Counts() (requests, successes, failures, consecutiveFail, consecutiveSucc uint32) {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	return cb.counts.requests, cb.counts.successes, cb.counts.failures,
		cb.counts.consecutiveFail, cb.counts.consecutiveSucc
}

// Reset manually resets circuit breaker to closed state
func (cb *CircuitBreaker) Reset() {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	cb.toNewGeneration(time.Now())
	cb.setState(StateClosed)
}

// beforeRequest checks if request is allowed and increments generation counter
func (cb *CircuitBreaker) beforeRequest() (uint64, error) {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	now := time.Now()
	state := cb.currentState(now)

	switch state {
	case StateOpen:
		return 0, ErrCircuitOpen
	case StateHalfOpen:
		// Limit concurrent requests in half-open
		if atomic.LoadUint32(&cb.halfOpen) >= cb.settings.MaxRequests {
			return 0, ErrTooManyRequests
		}
		atomic.AddUint32(&cb.halfOpen, 1)
	case StateClosed:
		// Check if we need new generation (statistical window expired)
		if cb.expiry.Before(now) {
			cb.toNewGeneration(now)
		}
	}

	cb.counts.requests++
	generation := uint64(now.Unix()) // simple generation marker
	return generation, nil
}

// afterRequest records request result and updates state
func (cb *CircuitBreaker) afterRequest(generation uint64, success bool, err error) {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	now := time.Now()
	state := cb.currentState(now)

	if state == StateHalfOpen {
		atomic.AddUint32(&cb.halfOpen, ^uint32(0)) // decrement
	}

	if success {
		cb.counts.successes++
		cb.counts.consecutiveFail = 0
		cb.counts.consecutiveSucc++

		// Transition from HalfOpen to Closed if success threshold met
		if state == StateHalfOpen && cb.counts.consecutiveSucc >= cb.settings.SuccessThreshold {
			cb.toNewGeneration(now)
			cb.setState(StateClosed)
		}
	} else {
		cb.counts.failures++
		cb.counts.consecutiveSucc = 0
		cb.counts.consecutiveFail++
		cb.lastError = err

		// Transition to Open if failure threshold met
		if cb.counts.consecutiveFail >= cb.settings.FailureThreshold {
			cb.expiry = now.Add(cb.settings.Timeout)
			cb.setState(StateOpen)
		} else if state == StateHalfOpen {
			// Any failure in half-open immediately returns to open
			cb.expiry = now.Add(cb.settings.Timeout)
			cb.setState(StateOpen)
		}
	}
}

// currentState returns current state, transitioning from Open to HalfOpen if timeout passed
func (cb *CircuitBreaker) currentState(now time.Time) State {
	state := cb.State()
	if state == StateOpen && cb.expiry.Before(now) {
		cb.setState(StateHalfOpen)
		return StateHalfOpen
	}
	return state
}

// setState changes state and triggers callback
func (cb *CircuitBreaker) setState(newState State) {
	oldState := cb.State()
	if oldState == newState {
		return
	}
	cb.state.Store(newState)
	if cb.settings.OnStateChange != nil {
		go cb.settings.OnStateChange(cb.name, oldState, newState) // async to avoid blocking
	}
}

// toNewGeneration resets counters for new statistical window
func (cb *CircuitBreaker) toNewGeneration(now time.Time) {
	cb.counts = &counters{}
	cb.expiry = now.Add(cb.settings.Interval)
	atomic.StoreUint32(&cb.halfOpen, 0)
}

// AdaptiveSettings returns settings that adapt based on error rate
type AdaptiveSettings struct {
	BaseSettings     Settings
	ErrorRateWindow  time.Duration // window to calculate error rate
	HighErrorRate    float64       // e.g., 0.5 = 50% error rate
	AdaptiveTimeout  bool          // if true, increase timeout on repeated opens
	MaxTimeout       time.Duration
	TimeoutMultiplier float64 // multiply timeout by this on each consecutive open
}

// AdaptiveCircuitBreaker extends CircuitBreaker with adaptive behavior
type AdaptiveCircuitBreaker struct {
	*CircuitBreaker
	adaptive       AdaptiveSettings
	consecutiveOpen uint32
}

// NewAdaptiveCircuitBreaker creates circuit breaker with adaptive timeout
func NewAdaptiveCircuitBreaker(name string, adaptive AdaptiveSettings) *AdaptiveCircuitBreaker {
	base := adaptive.BaseSettings
	base.OnStateChange = func(n string, from, to State) {
		// Default behavior can be overridden
		if adaptive.BaseSettings.OnStateChange != nil {
			adaptive.BaseSettings.OnStateChange(n, from, to)
		}
	}
	
	cb := NewCircuitBreaker(name, base)
	return &AdaptiveCircuitBreaker{
		CircuitBreaker: cb,
		adaptive:       adaptive,
	}
}

// Execute with adaptive timeout adjustment
func (acb *AdaptiveCircuitBreaker) Execute(ctx context.Context, fn func() error) error {
	// Adjust timeout based on consecutive opens
	if acb.adaptive.AdaptiveTimeout && acb.State() == StateOpen {
		opens := atomic.LoadUint32(&acb.consecutiveOpen)
		if opens > 0 {
			multiplier := 1.0
			for i := uint32(0); i < opens && multiplier < 10.0; i++ {
				multiplier *= acb.adaptive.TimeoutMultiplier
			}
			newTimeout := time.Duration(float64(acb.adaptive.BaseSettings.Timeout) * multiplier)
			if newTimeout > acb.adaptive.MaxTimeout {
				newTimeout = acb.adaptive.MaxTimeout
			}
			acb.settings.Timeout = newTimeout
		}
	}

	// Track consecutive opens
	if acb.State() == StateOpen {
		atomic.AddUint32(&acb.consecutiveOpen, 1)
	} else if acb.State() == StateClosed {
		atomic.StoreUint32(&acb.consecutiveOpen, 0)
	}

	return acb.CircuitBreaker.Execute(ctx, fn)
}
