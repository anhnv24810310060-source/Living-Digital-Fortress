// Package main - Advanced Load Balancing Algorithms
// Production-ready LB with predictive selection and adaptive weighting
package main

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"
)

// LRT (Least Response Time) - Selects backend with lowest predicted response time
// Combines historical latency with in-flight request count for accurate prediction

type LRTSelector struct {
	mu               sync.RWMutex
	latencyPredictor *LatencyPredictor
	decayFactor      float64
	inflightWeight   float64
}

// LatencyPredictor uses exponentially weighted moving average with variance tracking
type LatencyPredictor struct {
	mu          sync.RWMutex
	predictions map[string]*LatencyStats
}

type LatencyStats struct {
	mean        atomic.Uint64 // Mean latency in microseconds
	variance    atomic.Uint64 // Variance for confidence interval
	sampleCount atomic.Uint64
	lastUpdate  time.Time
	p50         atomic.Uint64 // Median
	p95         atomic.Uint64 // 95th percentile
	p99         atomic.Uint64 // 99th percentile
	ringBuffer  []uint64      // For percentile calculation
	ringIdx     int
	mu          sync.Mutex
}

func NewLRTSelector() *LRTSelector {
	return &LRTSelector{
		latencyPredictor: NewLatencyPredictor(),
		decayFactor:      0.1, // Weight for new observations
		inflightWeight:   5.0, // Penalty per in-flight request (ms)
	}
}

func NewLatencyPredictor() *LatencyPredictor {
	return &LatencyPredictor{
		predictions: make(map[string]*LatencyStats),
	}
}

// SelectLRT chooses backend with lowest predicted response time
func (lb *LRTSelector) SelectLRT(backends []*Backend) *Backend {
	if len(backends) == 0 {
		return nil
	}
	if len(backends) == 1 {
		return backends[0]
	}

	lb.mu.RLock()
	defer lb.mu.RUnlock()

	var best *Backend
	var bestScore float64 = math.MaxFloat64

	for _, b := range backends {
		if !b.Healthy.Load() {
			continue
		}

		// Get predicted latency (with confidence interval)
		predictedLatency := lb.latencyPredictor.PredictLatency(b.URL)

		// Factor in current in-flight requests
		inflight := float64(atomic.LoadInt64(&b.Conns))

		// Calculate total predicted response time
		score := predictedLatency + (inflight * lb.inflightWeight)

		// Apply weight (capacity multiplier)
		weight := b.Weight
		if weight < 0.1 {
			weight = 0.1
		}
		score = score / weight

		if best == nil || score < bestScore {
			best = b
			bestScore = score
		}
	}

	if best == nil && len(backends) > 0 {
		// Fallback: return first healthy or any backend
		for _, b := range backends {
			if b.Healthy.Load() {
				return b
			}
		}
		return backends[0]
	}

	return best
}

// PredictLatency returns predicted latency in milliseconds with confidence
func (lp *LatencyPredictor) PredictLatency(backendURL string) float64 {
	lp.mu.RLock()
	stats, ok := lp.predictions[backendURL]
	lp.mu.RUnlock()

	if !ok || stats.sampleCount.Load() < 5 {
		// Insufficient data: return conservative estimate
		return 50.0 // 50ms default
	}

	// Use P95 for conservative prediction
	p95 := float64(stats.p95.Load()) / 1000.0 // Convert μs to ms

	// If no percentile data, fall back to mean + 2*stddev
	if p95 == 0 {
		mean := float64(stats.mean.Load()) / 1000.0
		variance := float64(stats.variance.Load()) / 1e6
		stddev := math.Sqrt(variance)
		return mean + 2.0*stddev // 95% confidence interval
	}

	return p95
}

// RecordLatency records observed latency for a backend
func (lp *LatencyPredictor) RecordLatency(backendURL string, latencyMs float64) {
	latencyUs := uint64(latencyMs * 1000)

	lp.mu.Lock()
	stats, ok := lp.predictions[backendURL]
	if !ok {
		stats = &LatencyStats{
			ringBuffer: make([]uint64, 1000), // Keep last 1000 samples for percentiles
			lastUpdate: time.Now(),
		}
		stats.mean.Store(latencyUs)
		lp.predictions[backendURL] = stats
		lp.mu.Unlock()
		return
	}
	lp.mu.Unlock()

	// Update statistics
	count := stats.sampleCount.Add(1)
	oldMean := float64(stats.mean.Load())

	// Exponentially weighted moving average
	alpha := 0.1
	newMean := alpha*float64(latencyUs) + (1.0-alpha)*oldMean
	stats.mean.Store(uint64(newMean))

	// Update variance (for Welford's online algorithm)
	oldVariance := float64(stats.variance.Load())
	delta := float64(latencyUs) - oldMean
	delta2 := float64(latencyUs) - newMean
	newVariance := ((float64(count)-1.0)*oldVariance + delta*delta2) / float64(count)
	stats.variance.Store(uint64(newVariance))

	// Update ring buffer for percentile calculation
	stats.mu.Lock()
	stats.ringBuffer[stats.ringIdx] = latencyUs
	stats.ringIdx = (stats.ringIdx + 1) % len(stats.ringBuffer)
	stats.lastUpdate = time.Now()

	// Calculate percentiles every 100 samples (expensive operation)
	if count%100 == 0 {
		lp.calculatePercentiles(stats)
	}
	stats.mu.Unlock()
}

// calculatePercentiles computes P50, P95, P99 from ring buffer
func (lp *LatencyPredictor) calculatePercentiles(stats *LatencyStats) {
	// Copy buffer for sorting (don't mutate original)
	samples := make([]uint64, len(stats.ringBuffer))
	copy(samples, stats.ringBuffer)

	// Quick sort
	quickSort(samples, 0, len(samples)-1)

	// Calculate percentiles
	if len(samples) > 0 {
		p50Idx := len(samples) * 50 / 100
		p95Idx := len(samples) * 95 / 100
		p99Idx := len(samples) * 99 / 100

		stats.p50.Store(samples[p50Idx])
		stats.p95.Store(samples[p95Idx])
		stats.p99.Store(samples[p99Idx])
	}
}

// quickSort is an in-place quick sort for uint64
func quickSort(arr []uint64, low, high int) {
	if low < high {
		pi := partition(arr, low, high)
		quickSort(arr, low, pi-1)
		quickSort(arr, pi+1, high)
	}
}

func partition(arr []uint64, low, high int) int {
	pivot := arr[high]
	i := low - 1
	for j := low; j < high; j++ {
		if arr[j] < pivot {
			i++
			arr[i], arr[j] = arr[j], arr[i]
		}
	}
	arr[i+1], arr[high] = arr[high], arr[i+1]
	return i + 1
}

// ---------- Enhanced Power-of-Two Choices (P2C) ----------
// Improved P2C with subsetting to reduce coordination and improve cache locality

type P2CEnhancedSelector struct {
	mu             sync.RWMutex
	subsetSize     int // Number of backends to consider (default: 5)
	randomSeed     uint64
	inflightWeight float64
}

func NewP2CEnhancedSelector(subsetSize int) *P2CEnhancedSelector {
	if subsetSize < 2 {
		subsetSize = 5
	}
	return &P2CEnhancedSelector{
		subsetSize:     subsetSize,
		randomSeed:     uint64(time.Now().UnixNano()),
		inflightWeight: 5.0, // ms penalty per connection
	}
}

// SelectP2CEnhanced uses bounded-load P2C with subsetting
func (p2c *P2CEnhancedSelector) SelectP2CEnhanced(backends []*Backend) *Backend {
	if len(backends) == 0 {
		return nil
	}
	if len(backends) == 1 {
		return backends[0]
	}

	// Filter healthy backends
	healthy := make([]*Backend, 0, len(backends))
	for _, b := range backends {
		if b.Healthy.Load() {
			healthy = append(healthy, b)
		}
	}

	if len(healthy) == 0 {
		// Fallback to all backends
		healthy = backends
	}

	if len(healthy) == 1 {
		return healthy[0]
	}

	// Select subset (bounded by subsetSize or available backends)
	subsetSize := p2c.subsetSize
	if subsetSize > len(healthy) {
		subsetSize = len(healthy)
	}

	// Randomly select subset without replacement using Fisher-Yates shuffle
	subset := make([]*Backend, len(healthy))
	copy(subset, healthy)

	for i := 0; i < subsetSize; i++ {
		j := i + rand.Intn(len(subset)-i)
		subset[i], subset[j] = subset[j], subset[i]
	}
	subset = subset[:subsetSize]

	// Find two best candidates from subset
	var first, second *Backend
	var firstCost, secondCost float64 = math.MaxFloat64, math.MaxFloat64

	for _, b := range subset {
		cost := p2c.calculateCost(b)

		if cost < firstCost {
			second = first
			secondCost = firstCost
			first = b
			firstCost = cost
		} else if cost < secondCost {
			second = b
			secondCost = cost
		}
	}

	// Randomly choose between top 2 with bias towards better one
	if second == nil {
		return first
	}

	// 70% chance to pick the better one, 30% for exploration
	if rand.Float64() < 0.7 {
		return first
	}
	return second
}

// calculateCost computes selection cost for P2C
func (p2c *P2CEnhancedSelector) calculateCost(b *Backend) float64 {
	// Get EWMA latency
	ewma := b.getEWMA()
	if ewma == 0 {
		ewma = 50.0 // Default 50ms
	}

	// Add penalty for in-flight connections
	inflight := float64(atomic.LoadInt64(&b.Conns))
	penalty := inflight * p2c.inflightWeight

	// Apply weight as capacity factor
	weight := b.Weight
	if weight < 0.1 {
		weight = 0.1
	}

	return (ewma + penalty) / weight
}

// ---------- Peak EWMA (Peak Exponentially Weighted Moving Average) ----------
// Tracks peak latency to avoid backends experiencing transient spikes

type PeakEWMASelector struct {
	mu            sync.RWMutex
	peakDecayRate float64 // How fast peaks decay (0-1)
	peaks         map[string]*PeakTracker
}

type PeakTracker struct {
	currentPeak   atomic.Uint64 // Current peak latency (μs)
	lastDecayTime time.Time
	mu            sync.Mutex
}

func NewPeakEWMASelector() *PeakEWMASelector {
	return &PeakEWMASelector{
		peakDecayRate: 0.05, // 5% decay per second
		peaks:         make(map[string]*PeakTracker),
	}
}

// SelectPeakEWMA chooses backend avoiding transient spikes
func (pe *PeakEWMASelector) SelectPeakEWMA(backends []*Backend) *Backend {
	if len(backends) == 0 {
		return nil
	}
	if len(backends) == 1 {
		return backends[0]
	}

	pe.mu.RLock()
	defer pe.mu.RUnlock()

	var best *Backend
	var bestScore float64 = math.MaxFloat64

	now := time.Now()

	for _, b := range backends {
		if !b.Healthy.Load() {
			continue
		}

		// Get current EWMA
		ewma := b.getEWMA()
		if ewma == 0 {
			ewma = 50.0
		}

		// Get peak latency for this backend
		tracker := pe.getPeakTracker(b.URL)
		peak := pe.getDecayedPeak(tracker, now)

		// Score is max of EWMA and decayed peak
		score := math.Max(ewma, peak)

		// Factor in connections
		inflight := float64(atomic.LoadInt64(&b.Conns))
		score += inflight * 5.0

		// Apply weight
		weight := b.Weight
		if weight < 0.1 {
			weight = 0.1
		}
		score = score / weight

		if best == nil || score < bestScore {
			best = b
			bestScore = score
		}
	}

	if best == nil && len(backends) > 0 {
		for _, b := range backends {
			if b.Healthy.Load() {
				return b
			}
		}
		return backends[0]
	}

	return best
}

// RecordPeak updates peak latency for a backend
func (pe *PeakEWMASelector) RecordPeak(backendURL string, latencyMs float64) {
	tracker := pe.getPeakTracker(backendURL)
	latencyUs := uint64(latencyMs * 1000)

	// Update peak if this latency is higher
	for {
		current := tracker.currentPeak.Load()
		if latencyUs <= current {
			break
		}
		if tracker.currentPeak.CompareAndSwap(current, latencyUs) {
			tracker.mu.Lock()
			tracker.lastDecayTime = time.Now()
			tracker.mu.Unlock()
			break
		}
	}
}

// getPeakTracker retrieves or creates peak tracker for backend
func (pe *PeakEWMASelector) getPeakTracker(backendURL string) *PeakTracker {
	pe.mu.RLock()
	tracker, ok := pe.peaks[backendURL]
	pe.mu.RUnlock()

	if ok {
		return tracker
	}

	pe.mu.Lock()
	defer pe.mu.Unlock()

	// Double-check after acquiring write lock
	if tracker, ok = pe.peaks[backendURL]; ok {
		return tracker
	}

	tracker = &PeakTracker{
		lastDecayTime: time.Now(),
	}
	tracker.currentPeak.Store(50000) // 50ms default in μs
	pe.peaks[backendURL] = tracker

	return tracker
}

// getDecayedPeak returns peak value with exponential decay
func (pe *PeakEWMASelector) getDecayedPeak(tracker *PeakTracker, now time.Time) float64 {
	tracker.mu.Lock()
	defer tracker.mu.Unlock()

	peak := float64(tracker.currentPeak.Load()) / 1000.0 // Convert to ms

	// Apply exponential decay based on time elapsed
	elapsed := now.Sub(tracker.lastDecayTime).Seconds()
	if elapsed > 0 {
		decayFactor := math.Pow(1.0-pe.peakDecayRate, elapsed)
		peak = peak * decayFactor

		// Update stored peak
		tracker.currentPeak.Store(uint64(peak * 1000))
		tracker.lastDecayTime = now
	}

	return peak
}

// ---------- Integration with main load balancer ----------

// pickBackendAdvanced uses advanced LB algorithms
func pickBackendAdvanced(p *Pool, algo LBAlgo, hashKey string, lrtSel *LRTSelector, p2cSel *P2CEnhancedSelector, peakSel *PeakEWMASelector) (*Backend, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if len(p.backends) == 0 {
		return nil, ErrEmptyPool
	}

	switch algo {
	case "lrt", "least_response_time":
		if lrtSel != nil {
			return lrtSel.SelectLRT(p.backends), nil
		}
		return pickBackend(p, LBEWMA, hashKey)

	case "p2c_enhanced", "p2c_subset":
		if p2cSel != nil {
			return p2cSel.SelectP2CEnhanced(p.backends), nil
		}
		return pickBackend(p, LBP2CEWMA, hashKey)

	case "peak_ewma":
		if peakSel != nil {
			return peakSel.SelectPeakEWMA(p.backends), nil
		}
		return pickBackend(p, LBEWMA, hashKey)

	default:
		return pickBackend(p, algo, hashKey)
	}
}

var (
	ErrEmptyPool = fmt.Errorf("empty backend pool")
)
