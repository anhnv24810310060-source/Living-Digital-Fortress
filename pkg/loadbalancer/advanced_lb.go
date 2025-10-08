package loadbalancer

import (
	"hash/fnv"
	"math"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"
)

// Backend represents a backend server with health and performance metrics
type Backend struct {
	URL       string
	Healthy   atomic.Bool
	Weight    float64      // Relative capacity (default 1.0)
	EWMA      uint64       // Exponential weighted moving average latency (stored as float64 bits)
	Conns     int64        // Active connections
	LastErr   atomic.Value // Last error (string)
	LastLatMs uint64       // Last observed latency in milliseconds

	// Circuit Breaker state
	cbState     atomic.Uint32 // 0=closed, 1=open, 2=half-open
	cbFails     atomic.Uint32 // Consecutive failures
	cbNextProbe atomic.Int64  // Unix nano when next probe allowed in OPEN state
}

func (b *Backend) GetEWMA() float64 {
	return math.Float64frombits(atomic.LoadUint64(&b.EWMA))
}

func (b *Backend) SetEWMA(v float64) {
	atomic.StoreUint64(&b.EWMA, math.Float64bits(v))
}

func (b *Backend) IncrConns() int64 {
	return atomic.AddInt64(&b.Conns, 1)
}

func (b *Backend) DecrConns() int64 {
	return atomic.AddInt64(&b.Conns, -1)
}

// LoadBalancer interface for pluggable algorithms
type LoadBalancer interface {
	Select(key string) *Backend
	Name() string
}

// Pool manages a collection of backends with load balancing
type Pool struct {
	Name      string
	Backends  []*Backend
	Algorithm LoadBalancer
	mu        sync.RWMutex

	// Round-robin counter
	rrCounter uint64

	// Rendezvous hash precomputed weights
	rendezvousHasher *RendezvousHasher
}

// NewPool creates a new backend pool
func NewPool(name string, backends []*Backend, algo LoadBalancer) *Pool {
	p := &Pool{
		Name:      name,
		Backends:  backends,
		Algorithm: algo,
	}

	// Initialize EWMA with small baseline
	for _, b := range backends {
		b.SetEWMA(10.0) // 10ms baseline
		b.Healthy.Store(true)
	}

	if rh, ok := algo.(*RendezvousHasher); ok {
		p.rendezvousHasher = rh
		p.rendezvousHasher.Rebuild(backends)
	}

	return p
}

// Select chooses a backend using the configured algorithm
func (p *Pool) Select(key string) *Backend {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if len(p.Backends) == 0 {
		return nil
	}

	return p.Algorithm.Select(key)
}

// GetHealthyBackends returns slice of healthy backends
func (p *Pool) GetHealthyBackends() []*Backend {
	p.mu.RLock()
	defer p.mu.RUnlock()

	healthy := make([]*Backend, 0, len(p.Backends))
	for _, b := range p.Backends {
		if b.Healthy.Load() && p.isCircuitClosed(b) {
			healthy = append(healthy, b)
		}
	}
	return healthy
}

func (p *Pool) isCircuitClosed(b *Backend) bool {
	state := b.cbState.Load()
	if state == 0 {
		return true // Closed = healthy
	}
	if state == 1 {
		// Open: check if probe time reached
		now := time.Now().UnixNano()
		nextProbe := b.cbNextProbe.Load()
		if now >= nextProbe {
			// Transition to half-open
			b.cbState.Store(2)
			return true
		}
		return false
	}
	// Half-open: allow one probe request
	return true
}

// RecordSuccess updates backend metrics after successful request
func (p *Pool) RecordSuccess(b *Backend, latencyMs float64) {
	if b == nil {
		return
	}

	// Update EWMA: α=0.3 for responsiveness to latency changes
	// EWMA_new = α * latency + (1-α) * EWMA_old
	alpha := 0.3
	oldEWMA := b.GetEWMA()
	newEWMA := alpha*latencyMs + (1-alpha)*oldEWMA
	b.SetEWMA(newEWMA)

	atomic.StoreUint64(&b.LastLatMs, uint64(latencyMs))
	b.LastErr.Store("")

	// Circuit breaker: reset on success
	b.cbFails.Store(0)
	if b.cbState.Load() != 0 {
		b.cbState.Store(0) // Close circuit
	}
}

// RecordFailure updates backend metrics after failed request
func (p *Pool) RecordFailure(b *Backend, err error) {
	if b == nil {
		return
	}

	errStr := ""
	if err != nil {
		errStr = err.Error()
	}
	b.LastErr.Store(errStr)

	// Circuit breaker: increment failures
	fails := b.cbFails.Add(1)

	// Open circuit after 5 consecutive failures
	if fails >= 5 && b.cbState.Load() == 0 {
		b.cbState.Store(1) // Open
		// Next probe in 10 seconds
		b.cbNextProbe.Store(time.Now().Add(10 * time.Second).UnixNano())
	}
}

// =====================================================
// ALGORITHM 1: Round Robin
// =====================================================
// Time Complexity: O(1)
// Space Complexity: O(1)
// Best for: Equal capacity backends, simple distribution

type RoundRobin struct {
	pool *Pool
}

func NewRoundRobin(pool *Pool) *RoundRobin {
	return &RoundRobin{pool: pool}
}

func (rr *RoundRobin) Select(key string) *Backend {
	healthy := rr.pool.GetHealthyBackends()
	if len(healthy) == 0 {
		return nil
	}

	idx := atomic.AddUint64(&rr.pool.rrCounter, 1) - 1
	return healthy[idx%uint64(len(healthy))]
}

func (rr *RoundRobin) Name() string {
	return "round_robin"
}

// =====================================================
// ALGORITHM 2: Least Connections
// =====================================================
// Time Complexity: O(n)
// Space Complexity: O(1)
// Best for: Long-lived connections with variable duration

type LeastConnections struct {
	pool *Pool
}

func NewLeastConnections(pool *Pool) *LeastConnections {
	return &LeastConnections{pool: pool}
}

func (lc *LeastConnections) Select(key string) *Backend {
	healthy := lc.pool.GetHealthyBackends()
	if len(healthy) == 0 {
		return nil
	}

	var best *Backend
	minConns := int64(math.MaxInt64)

	for _, b := range healthy {
		conns := atomic.LoadInt64(&b.Conns)
		if conns < minConns {
			minConns = conns
			best = b
		}
	}

	return best
}

func (lc *LeastConnections) Name() string {
	return "least_connections"
}

// =====================================================
// ALGORITHM 3: EWMA (Exponential Weighted Moving Average)
// =====================================================
// Time Complexity: O(n)
// Space Complexity: O(1)
// Best for: Latency-sensitive workloads, production default
//
// Score = EWMA + (active_connections * penalty) / weight
// Lower score = better choice

type EWMA struct {
	pool        *Pool
	connPenalty float64 // Penalty per connection (in ms)
}

func NewEWMA(pool *Pool, connPenalty float64) *EWMA {
	if connPenalty <= 0 {
		connPenalty = 5.0 // Default 5ms penalty per connection
	}
	return &EWMA{
		pool:        pool,
		connPenalty: connPenalty,
	}
}

func (e *EWMA) Select(key string) *Backend {
	healthy := e.pool.GetHealthyBackends()
	if len(healthy) == 0 {
		return nil
	}

	var best *Backend
	minScore := math.MaxFloat64

	for _, b := range healthy {
		ewma := b.GetEWMA()
		conns := float64(atomic.LoadInt64(&b.Conns))
		weight := b.Weight
		if weight < 0.1 {
			weight = 1.0
		}

		// Calculate weighted score
		score := ewma + (conns * e.connPenalty / weight)

		if score < minScore {
			minScore = score
			best = b
		}
	}

	return best
}

func (e *EWMA) Name() string {
	return "ewma"
}

// =====================================================
// ALGORITHM 4: Power of Two Choices (P2C) with EWMA
// =====================================================
// Time Complexity: O(1) - only examines 2 backends
// Space Complexity: O(1)
// Best for: Large backend pools (100+), high throughput
//
// Algorithm: Randomly pick 2 backends, choose the one with lower score
// Achieves near-optimal load distribution with minimal overhead

type PowerOfTwoChoices struct {
	pool        *Pool
	connPenalty float64
	rng         *rand.Rand
	mu          sync.Mutex
}

func NewPowerOfTwoChoices(pool *Pool, connPenalty float64) *PowerOfTwoChoices {
	if connPenalty <= 0 {
		connPenalty = 5.0
	}
	return &PowerOfTwoChoices{
		pool:        pool,
		connPenalty: connPenalty,
		rng:         rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

func (p2c *PowerOfTwoChoices) Select(key string) *Backend {
	healthy := p2c.pool.GetHealthyBackends()
	n := len(healthy)

	if n == 0 {
		return nil
	}
	if n == 1 {
		return healthy[0]
	}

	// Pick 2 random backends
	p2c.mu.Lock()
	i := p2c.rng.Intn(n)
	j := p2c.rng.Intn(n - 1)
	p2c.mu.Unlock()

	if j >= i {
		j++
	}

	b1, b2 := healthy[i], healthy[j]

	// Calculate scores
	score1 := p2c.calculateScore(b1)
	score2 := p2c.calculateScore(b2)

	if score1 <= score2 {
		return b1
	}
	return b2
}

func (p2c *PowerOfTwoChoices) calculateScore(b *Backend) float64 {
	ewma := b.GetEWMA()
	conns := float64(atomic.LoadInt64(&b.Conns))
	weight := b.Weight
	if weight < 0.1 {
		weight = 1.0
	}
	return ewma + (conns * p2c.connPenalty / weight)
}

func (p2c *PowerOfTwoChoices) Name() string {
	return "power_of_two"
}

// =====================================================
// ALGORITHM 5: Rendezvous Hashing (Highest Random Weight)
// =====================================================
// Time Complexity: O(n)
// Space Complexity: O(n)
// Best for: Consistent hashing, session affinity, cache distribution
//
// Provides excellent key distribution and minimal disruption on backend changes

type RendezvousHasher struct {
	pool     *Pool
	backends []*Backend
	mu       sync.RWMutex
}

func NewRendezvousHasher(pool *Pool) *RendezvousHasher {
	rh := &RendezvousHasher{pool: pool}
	return rh
}

func (rh *RendezvousHasher) Rebuild(backends []*Backend) {
	rh.mu.Lock()
	rh.backends = make([]*Backend, len(backends))
	copy(rh.backends, backends)
	rh.mu.Unlock()
}

func (rh *RendezvousHasher) Select(key string) *Backend {
	healthy := rh.pool.GetHealthyBackends()
	if len(healthy) == 0 {
		return nil
	}

	var best *Backend
	var maxHash uint64

	for _, b := range healthy {
		// Combine key with backend URL to compute hash
		h := fnv.New64a()
		h.Write([]byte(key))
		h.Write([]byte(b.URL))
		hash := h.Sum64()

		// Weight affects selection probability
		weightedHash := uint64(float64(hash) * b.Weight)

		if weightedHash > maxHash {
			maxHash = weightedHash
			best = b
		}
	}

	return best
}

func (rh *RendezvousHasher) Name() string {
	return "rendezvous"
}

// =====================================================
// ALGORITHM 6: Weighted Round Robin
// =====================================================
// Time Complexity: O(n)
// Space Complexity: O(n)
// Best for: Backends with different capacities

type WeightedRoundRobin struct {
	pool    *Pool
	current int
	weights []int
	maxGCD  int
	maxW    int
	mu      sync.Mutex
}

func NewWeightedRoundRobin(pool *Pool) *WeightedRoundRobin {
	wrr := &WeightedRoundRobin{
		pool:    pool,
		current: -1,
	}
	wrr.rebuild()
	return wrr
}

func (wrr *WeightedRoundRobin) rebuild() {
	backends := wrr.pool.GetHealthyBackends()
	wrr.weights = make([]int, len(backends))

	for i, b := range backends {
		wrr.weights[i] = int(b.Weight * 10) // Scale weights to integers
	}

	wrr.maxW = maxSlice(wrr.weights)
	wrr.maxGCD = gcdSlice(wrr.weights)
}

func (wrr *WeightedRoundRobin) Select(key string) *Backend {
	wrr.mu.Lock()
	defer wrr.mu.Unlock()

	backends := wrr.pool.GetHealthyBackends()
	if len(backends) == 0 {
		return nil
	}

	for {
		wrr.current = (wrr.current + 1) % len(backends)
		if wrr.current == 0 {
			wrr.maxW = wrr.maxW - wrr.maxGCD
			if wrr.maxW <= 0 {
				wrr.maxW = maxSlice(wrr.weights)
			}
		}

		if wrr.weights[wrr.current] >= wrr.maxW {
			return backends[wrr.current]
		}
	}
}

func (wrr *WeightedRoundRobin) Name() string {
	return "weighted_round_robin"
}

// Helper functions
func gcd(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

func gcdSlice(weights []int) int {
	if len(weights) == 0 {
		return 1
	}
	result := weights[0]
	for i := 1; i < len(weights); i++ {
		result = gcd(result, weights[i])
	}
	return result
}

func maxSlice(weights []int) int {
	if len(weights) == 0 {
		return 0
	}
	max := weights[0]
	for _, w := range weights[1:] {
		if w > max {
			max = w
		}
	}
	return max
}
