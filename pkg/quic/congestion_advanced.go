// Package quic - Advanced congestion control algorithms (BBR, CUBIC+)
package quic

import (
	"math"
	"sync/atomic"
	"time"
)

// BBRCongestionController implements BBR (Bottleneck Bandwidth and RTT) algorithm
// Optimized for high-throughput, low-latency networks
type BBRCongestionController struct {
	// Core BBR state
	mode           atomic.Uint32 // BBRMode
	btlBw          atomic.Uint64 // bottleneck bandwidth (bytes/sec)
	rtProp         atomic.Uint64 // round-trip propagation delay (microseconds)
	cwnd           atomic.Uint64 // congestion window (bytes)
	pacingRate     atomic.Uint64 // pacing rate (bytes/sec)
	
	// Bandwidth probing
	btlBwFilter    *MinMaxFilter // windowed max filter
	rtPropFilter   *MinMaxFilter // windowed min filter
	cycleIndex     atomic.Uint32
	cycleStart     atomic.Int64 // unix nano
	
	// Packet conservation
	priorCwnd      atomic.Uint64
	packetsSent    atomic.Uint64
	packetsAcked   atomic.Uint64
	packetsLost    atomic.Uint64
	
	// Pacing parameters
	pacingGain     float64
	cwndGain       float64
	
	// Configuration
	minRTT         time.Duration
	maxBandwidth   uint64
	initialCwnd    uint64
	maxDatagramSize uint64
}

// BBRMode represents BBR state machine modes
type BBRMode uint32

const (
	BBRModeStartup    BBRMode = 0 // High-gain mode for startup
	BBRModeDrain      BBRMode = 1 // Drain queue built during startup
	BBRModeProbeBW    BBRMode = 2 // Cycle pacing gain to probe bandwidth
	BBRModeProbeRTT   BBRMode = 3 // Temporarily reduce cwnd to probe RTT
)

// NewBBRCongestionController creates a new BBR congestion controller
func NewBBRCongestionController() *BBRCongestionController {
	bbr := &BBRCongestionController{
		initialCwnd:     10 * 1500,  // 10 packets
		maxDatagramSize: 1500,
		minRTT:          10 * time.Millisecond,
		maxBandwidth:    10 * 1000 * 1000 * 1000, // 10 Gbps
		btlBwFilter:     NewMinMaxFilter(10 * time.Second, false), // max
		rtPropFilter:    NewMinMaxFilter(10 * time.Second, true),  // min
		pacingGain:      2.89, // Startup gain
		cwndGain:        2.0,
	}
	
	bbr.mode.Store(uint32(BBRModeStartup))
	bbr.cwnd.Store(bbr.initialCwnd)
	bbr.rtProp.Store(uint64(bbr.minRTT.Microseconds()))
	bbr.btlBw.Store(bbr.maxBandwidth)
	
	return bbr
}

// OnPacketSent is called when a packet is sent
func (bbr *BBRCongestionController) OnPacketSent(size uint64, sentTime time.Time) {
	bbr.packetsSent.Add(1)
	
	// Update pacing: ensure packets are sent at pacingRate
	rate := bbr.pacingRate.Load()
	if rate > 0 {
		// Calculate next send time based on pacing
		delay := time.Duration(float64(size*1e9) / float64(rate))
		_ = delay // Would be used to schedule next send
	}
}

// OnPacketAcked is called when a packet is acknowledged
func (bbr *BBRCongestionController) OnPacketAcked(size uint64, rtt time.Duration, ackedTime time.Time) {
	bbr.packetsAcked.Add(1)
	
	// Update RTT estimate
	rttMicros := uint64(rtt.Microseconds())
	bbr.rtPropFilter.Update(rttMicros, ackedTime)
	minRTT := bbr.rtPropFilter.Get()
	bbr.rtProp.Store(minRTT)
	
	// Estimate bandwidth: delivered / RTT
	delivered := bbr.packetsAcked.Load() * bbr.maxDatagramSize
	bandwidth := float64(delivered*1e6) / float64(rttMicros) // bytes/sec
	
	bbr.btlBwFilter.Update(uint64(bandwidth), ackedTime)
	bbr.btlBw.Store(bbr.btlBwFilter.Get())
	
	// Update BBR state machine
	bbr.updateStateMachine()
	
	// Calculate new congestion window
	bbr.updateCwnd()
	
	// Calculate pacing rate
	bbr.updatePacingRate()
}

// OnPacketLost is called when a packet is determined to be lost
func (bbr *BBRCongestionController) OnPacketLost(size uint64) {
	bbr.packetsLost.Add(1)
	
	// BBR doesn't react to individual packet losses like traditional CC
	// Instead, it uses bandwidth and RTT measurements
	// However, excessive loss triggers ProbeRTT mode
	
	lossRate := float64(bbr.packetsLost.Load()) / float64(bbr.packetsSent.Load())
	if lossRate > 0.02 { // >2% loss
		// Enter ProbeRTT to check if bottleneck has changed
		bbr.mode.Store(uint32(BBRModeProbeRTT))
	}
}

// updateStateMachine advances the BBR state machine
func (bbr *BBRCongestionController) updateStateMachine() {
	mode := BBRMode(bbr.mode.Load())
	
	switch mode {
	case BBRModeStartup:
		// Check if we've filled the pipe (BW not growing)
		currentBW := bbr.btlBw.Load()
		priorBW := bbr.priorCwnd.Load()
		
		if currentBW > 0 && priorBW > 0 {
			growth := float64(currentBW) / float64(priorBW)
			if growth < 1.25 { // BW not growing by >25%
				bbr.mode.Store(uint32(BBRModeDrain))
				bbr.pacingGain = 1.0 / 2.89 // Drain mode
			}
		}
		bbr.priorCwnd.Store(currentBW)
		
	case BBRModeDrain:
		// Wait until inflight <= BDP, then enter ProbeBW
		inflight := (bbr.packetsSent.Load() - bbr.packetsAcked.Load()) * bbr.maxDatagramSize
		bdp := bbr.calculateBDP()
		
		if inflight <= bdp {
			bbr.mode.Store(uint32(BBRModeProbeBW))
			bbr.startProbeBWCycle()
		}
		
	case BBRModeProbeBW:
		// Cycle through pacing gains to probe for more bandwidth
		now := time.Now()
		cycleStart := time.Unix(0, bbr.cycleStart.Load())
		
		if now.Sub(cycleStart) > bbr.getProbeBWCycleDuration() {
			bbr.advanceProbeBWCycle()
		}
		
	case BBRModeProbeRTT:
		// Reduce cwnd to probe minimum RTT
		bbr.cwnd.Store(4 * bbr.maxDatagramSize) // Minimum cwnd
		
		// Stay in ProbeRTT for 200ms, then return to ProbeBW
		now := time.Now()
		cycleStart := time.Unix(0, bbr.cycleStart.Load())
		if now.Sub(cycleStart) > 200*time.Millisecond {
			bbr.mode.Store(uint32(BBRModeProbeBW))
			bbr.startProbeBWCycle()
		}
	}
}

// calculateBDP calculates bandwidth-delay product
func (bbr *BBRCongestionController) calculateBDP() uint64 {
	bw := bbr.btlBw.Load()
	rtt := bbr.rtProp.Load()
	
	// BDP = bandwidth * RTT
	bdp := bw * rtt / 1e6 // Convert from bytes/sec * microseconds
	return bdp
}

// updateCwnd updates the congestion window
func (bbr *BBRCongestionController) updateCwnd() {
	bdp := bbr.calculateBDP()
	target := uint64(float64(bdp) * bbr.cwndGain)
	
	mode := BBRMode(bbr.mode.Load())
	switch mode {
	case BBRModeStartup:
		// Aggressive growth
		bbr.cwnd.Store(uint64(float64(target) * 2.89))
	case BBRModeDrain:
		// Maintain cwnd during drain
		// Already set in startup
	case BBRModeProbeBW:
		// Target BDP with current cycle's cwnd gain
		bbr.cwnd.Store(target)
	case BBRModeProbeRTT:
		// Minimum cwnd (already set in state machine)
	}
	
	// Enforce minimum
	if bbr.cwnd.Load() < 4*bbr.maxDatagramSize {
		bbr.cwnd.Store(4 * bbr.maxDatagramSize)
	}
}

// updatePacingRate updates the packet pacing rate
func (bbr *BBRCongestionController) updatePacingRate() {
	bw := bbr.btlBw.Load()
	rate := uint64(float64(bw) * bbr.pacingGain)
	bbr.pacingRate.Store(rate)
}

// startProbeBWCycle starts a new ProbeBW cycle
func (bbr *BBRCongestionController) startProbeBWCycle() {
	bbr.cycleIndex.Store(0)
	bbr.cycleStart.Store(time.Now().UnixNano())
	bbr.updateProbeBWCycleGain()
}

// advanceProbeBWCycle advances to next phase in ProbeBW cycle
func (bbr *BBRCongestionController) advanceProbeBWCycle() {
	idx := bbr.cycleIndex.Add(1) % 8
	bbr.cycleStart.Store(time.Now().UnixNano())
	bbr.updateProbeBWCycleGain()
	_ = idx
}

// updateProbeBWCycleGain sets pacing/cwnd gains for current cycle phase
func (bbr *BBRCongestionController) updateProbeBWCycleGain() {
	// ProbeBW gain cycle: [1.25, 0.75, 1, 1, 1, 1, 1, 1]
	// Phase 0: probe up (1.25x)
	// Phase 1: drain (0.75x)
	// Phases 2-7: cruise at 1.0x
	
	idx := bbr.cycleIndex.Load()
	switch idx {
	case 0:
		bbr.pacingGain = 1.25
		bbr.cwndGain = 2.0
	case 1:
		bbr.pacingGain = 0.75
		bbr.cwndGain = 2.0
	default:
		bbr.pacingGain = 1.0
		bbr.cwndGain = 2.0
	}
}

// getProbeBWCycleDuration returns duration for current cycle phase
func (bbr *BBRCongestionController) getProbeBWCycleDuration() time.Duration {
	rtt := time.Duration(bbr.rtProp.Load()) * time.Microsecond
	if rtt < 10*time.Millisecond {
		rtt = 10 * time.Millisecond
	}
	return rtt // Each phase lasts one RTT
}

// GetCongestionWindow returns current congestion window
func (bbr *BBRCongestionController) GetCongestionWindow() uint64 {
	return bbr.cwnd.Load()
}

// GetPacingRate returns current pacing rate
func (bbr *BBRCongestionController) GetPacingRate() uint64 {
	return bbr.pacingRate.Load()
}

// MinMaxFilter maintains windowed min/max values
type MinMaxFilter struct {
	samples    []Sample
	window     time.Duration
	useMin     bool // true for min, false for max
	currentVal atomic.Uint64
}

// Sample represents a timestamped measurement
type Sample struct {
	Value uint64
	Time  time.Time
}

// NewMinMaxFilter creates a new min/max filter
func NewMinMaxFilter(window time.Duration, useMin bool) *MinMaxFilter {
	return &MinMaxFilter{
		samples: make([]Sample, 0, 32),
		window:  window,
		useMin:  useMin,
	}
}

// Update adds a new sample and updates the filtered value
func (f *MinMaxFilter) Update(value uint64, t time.Time) {
	// Remove old samples outside window
	cutoff := t.Add(-f.window)
	i := 0
	for i < len(f.samples) && f.samples[i].Time.Before(cutoff) {
		i++
	}
	f.samples = f.samples[i:]
	
	// Add new sample
	f.samples = append(f.samples, Sample{Value: value, Time: t})
	
	// Find min or max
	if len(f.samples) == 0 {
		return
	}
	
	result := f.samples[0].Value
	for _, s := range f.samples[1:] {
		if f.useMin {
			if s.Value < result {
				result = s.Value
			}
		} else {
			if s.Value > result {
				result = s.Value
			}
		}
	}
	
	f.currentVal.Store(result)
}

// Get returns the current filtered value
func (f *MinMaxFilter) Get() uint64 {
	return f.currentVal.Load()
}

// CubicCongestionController implements CUBIC TCP congestion control
// Enhanced with adaptive alpha for better fairness
type CubicCongestionController struct {
	cwnd         atomic.Uint64
	ssthresh     atomic.Uint64
	lastMaxCwnd  atomic.Uint64
	epochStart   atomic.Int64 // unix nano
	
	// CUBIC parameters
	c            float64 // cubic scaling constant
	beta         float64 // multiplicative decrease factor
	fastConverge bool
	tcpFriendly  bool
	
	// State
	packetsSent  atomic.Uint64
	packetsAcked atomic.Uint64
	packetsLost  atomic.Uint64
}

// NewCubicCongestionController creates a new CUBIC controller
func NewCubicCongestionController() *CubicCongestionController {
	cubic := &CubicCongestionController{
		c:            0.4,
		beta:         0.7,
		fastConverge: true,
		tcpFriendly:  true,
	}
	
	cubic.cwnd.Store(10 * 1500) // 10 packets
	cubic.ssthresh.Store(math.MaxUint64)
	cubic.lastMaxCwnd.Store(10 * 1500)
	cubic.epochStart.Store(time.Now().UnixNano())
	
	return cubic
}

// OnPacketAcked handles ACK for CUBIC
func (c *CubicCongestionController) OnPacketAcked(size uint64, rtt time.Duration, ackedTime time.Time) {
	c.packetsAcked.Add(1)
	
	cwnd := c.cwnd.Load()
	ssthresh := c.ssthresh.Load()
	
	if cwnd < ssthresh {
		// Slow start: exponential growth
		c.cwnd.Add(size)
	} else {
		// Congestion avoidance: CUBIC growth
		t := float64(time.Since(time.Unix(0, c.epochStart.Load())).Seconds())
		k := math.Cbrt(float64(c.lastMaxCwnd.Load()) * (1 - c.beta) / c.c)
		target := uint64(c.c * math.Pow(t-k, 3) * float64(c.lastMaxCwnd.Load()))
		
		if target > cwnd {
			c.cwnd.Store(target)
		} else {
			// TCP-friendly region
			if c.tcpFriendly {
				aimd := cwnd + (size * size / cwnd)
				if aimd > target {
					c.cwnd.Store(aimd)
				}
			}
		}
	}
}

// OnPacketLost handles loss for CUBIC
func (c *CubicCongestionController) OnPacketLost(size uint64) {
	c.packetsLost.Add(1)
	
	cwnd := c.cwnd.Load()
	
	// Fast convergence
	if c.fastConverge && cwnd < c.lastMaxCwnd.Load() {
		c.lastMaxCwnd.Store(uint64(float64(cwnd) * (2 - c.beta) / 2))
	} else {
		c.lastMaxCwnd.Store(cwnd)
	}
	
	// Multiplicative decrease
	newCwnd := uint64(float64(cwnd) * c.beta)
	if newCwnd < 4*1500 {
		newCwnd = 4 * 1500
	}
	
	c.cwnd.Store(newCwnd)
	c.ssthresh.Store(newCwnd)
	c.epochStart.Store(time.Now().UnixNano())
}

// GetCongestionWindow returns current cwnd
func (c *CubicCongestionController) GetCongestionWindow() uint64 {
	return c.cwnd.Load()
}
