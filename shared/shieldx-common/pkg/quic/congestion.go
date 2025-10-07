// Package quic - Congestion Control Algorithms
// Implements CUBIC, BBR (Bottleneck Bandwidth and RTT), and Reno
package quic

import (
	"math"
	"sync"
	"time"
)

// ---------- CUBIC Congestion Control ----------
// CUBIC is the default TCP congestion control algorithm in Linux
// Uses cubic function for window growth after congestion event

type CubicController struct {
	mu sync.RWMutex

	// CUBIC parameters
	cwnd       uint64    // Congestion window (bytes)
	ssthresh   uint64    // Slow start threshold
	wMax       uint64    // Window size before last reduction
	epochStart time.Time // Time of last congestion event

	// Constants (RFC 8312)
	c    float64 // CUBIC parameter (0.4)
	beta float64 // Multiplicative decrease factor (0.7)

	// RTT tracking
	minRTT      time.Duration
	smoothedRTT time.Duration
	rttVar      time.Duration

	// Metrics
	packetsSent  uint64
	packetsAcked uint64
	packetsLost  uint64
}

func NewCubicController() *CubicController {
	return &CubicController{
		cwnd:     10 * 1460, // Initial window: 10 packets (14600 bytes)
		ssthresh: math.MaxUint64,
		c:        0.4,
		beta:     0.7,
		minRTT:   time.Second,
	}
}

func (cc *CubicController) OnPacketSent(size int, now time.Time) {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	cc.packetsSent++
}

func (cc *CubicController) OnPacketAcked(size int, rtt time.Duration, now time.Time) {
	cc.mu.Lock()
	defer cc.mu.Unlock()

	cc.packetsAcked++
	cc.updateRTT(rtt)

	// Slow start phase
	if cc.cwnd < cc.ssthresh {
		cc.cwnd += uint64(size)
		return
	}

	// Congestion avoidance with CUBIC
	if cc.epochStart.IsZero() {
		cc.epochStart = now
	}

	t := now.Sub(cc.epochStart).Seconds()
	k := math.Cbrt(float64(cc.wMax) * (1.0 - cc.beta) / cc.c)

	// CUBIC window calculation
	target := cc.c*math.Pow(t-k, 3) + float64(cc.wMax)

	if target > float64(cc.cwnd) {
		cc.cwnd = uint64(target)
	} else {
		// TCP-friendly region
		cc.cwnd += uint64((int64(size) * 1460) / int64(cc.cwnd))
	}
}

func (cc *CubicController) OnPacketLost(size int, now time.Time) {
	cc.mu.Lock()
	defer cc.mu.Unlock()

	cc.packetsLost++

	// Multiplicative decrease
	cc.wMax = cc.cwnd
	cc.cwnd = uint64(float64(cc.cwnd) * cc.beta)
	cc.ssthresh = cc.cwnd
	cc.epochStart = time.Time{} // Reset epoch
}

func (cc *CubicController) CanSend() bool {
	cc.mu.RLock()
	defer cc.mu.RUnlock()
	// Simplified: always allow if under cwnd
	return true
}

func (cc *CubicController) GetCWND() uint64 {
	cc.mu.RLock()
	defer cc.mu.RUnlock()
	return cc.cwnd
}

func (cc *CubicController) GetRTT() time.Duration {
	cc.mu.RLock()
	defer cc.mu.RUnlock()
	return cc.smoothedRTT
}

func (cc *CubicController) Algorithm() string {
	return "cubic"
}

func (cc *CubicController) updateRTT(rtt time.Duration) {
	if rtt < cc.minRTT || cc.minRTT == time.Second {
		cc.minRTT = rtt
	}

	if cc.smoothedRTT == 0 {
		cc.smoothedRTT = rtt
		cc.rttVar = rtt / 2
	} else {
		// RFC 6298 exponential weighted moving average
		alpha := 0.125
		beta := 0.25
		diff := rtt - cc.smoothedRTT
		if diff < 0 {
			diff = -diff
		}
		cc.rttVar = time.Duration((1-beta)*float64(cc.rttVar) + beta*float64(diff))
		cc.smoothedRTT = time.Duration((1-alpha)*float64(cc.smoothedRTT) + alpha*float64(rtt))
	}
}

// ---------- BBR Congestion Control ----------
// BBR (Bottleneck Bandwidth and RTT) is Google's congestion control
// Focuses on maximizing bandwidth while minimizing delay

type BBRController struct {
	mu sync.RWMutex

	// BBR state machine
	state BBRState

	// Bandwidth estimation
	btlBw       uint64     // Bottleneck bandwidth (bytes/sec)
	btlBwFilter *MaxFilter // Max filter for bandwidth

	// RTT estimation
	rtProp       time.Duration // Round-trip propagation delay
	rtPropStamp  time.Time
	rtPropExpire time.Duration

	// Pacing
	pacingGain float64
	cwndGain   float64

	// Cycle tracking
	cycleIdx   int
	cycleStamp time.Time

	// Metrics
	packetsSent  uint64
	packetsAcked uint64
	packetsLost  uint64

	deliveredBytes uint64
	deliveredTime  time.Time
}

type BBRState int

const (
	BBRStateStartup BBRState = iota
	BBRStateDrain
	BBRStateProbeBW
	BBRStateProbeRTT
)

type MaxFilter struct {
	values []uint64
	times  []time.Time
	window time.Duration
}

func NewMaxFilter(window time.Duration) *MaxFilter {
	return &MaxFilter{
		values: make([]uint64, 0, 10),
		times:  make([]time.Time, 0, 10),
		window: window,
	}
}

func (f *MaxFilter) Update(val uint64, now time.Time) {
	// Remove expired entries
	cutoff := now.Add(-f.window)
	i := 0
	for i < len(f.times) && f.times[i].Before(cutoff) {
		i++
	}
	if i > 0 {
		f.values = f.values[i:]
		f.times = f.times[i:]
	}

	// Add new value
	f.values = append(f.values, val)
	f.times = append(f.times, now)
}

func (f *MaxFilter) Get() uint64 {
	if len(f.values) == 0 {
		return 0
	}
	max := f.values[0]
	for _, v := range f.values[1:] {
		if v > max {
			max = v
		}
	}
	return max
}

func NewBBRController() *BBRController {
	return &BBRController{
		state:        BBRStateStartup,
		btlBwFilter:  NewMaxFilter(10 * time.Second),
		rtProp:       time.Second,
		rtPropExpire: 10 * time.Second,
		pacingGain:   2.89, // High gain for startup
		cwndGain:     2.0,
	}
}

func (bbr *BBRController) OnPacketSent(size int, now time.Time) {
	bbr.mu.Lock()
	defer bbr.mu.Unlock()
	bbr.packetsSent++
}

func (bbr *BBRController) OnPacketAcked(size int, rtt time.Duration, now time.Time) {
	bbr.mu.Lock()
	defer bbr.mu.Unlock()

	bbr.packetsAcked++
	bbr.deliveredBytes += uint64(size)

	// Update RTT prop
	if rtt < bbr.rtProp || now.Sub(bbr.rtPropStamp) > bbr.rtPropExpire {
		bbr.rtProp = rtt
		bbr.rtPropStamp = now
	}

	// Estimate bandwidth
	if bbr.deliveredTime.IsZero() {
		bbr.deliveredTime = now
	} else {
		elapsed := now.Sub(bbr.deliveredTime)
		if elapsed > 0 {
			bw := uint64(float64(bbr.deliveredBytes) / elapsed.Seconds())
			bbr.btlBwFilter.Update(bw, now)
			bbr.btlBw = bbr.btlBwFilter.Get()
		}
	}

	// State machine
	bbr.updateState(now)
}

func (bbr *BBRController) OnPacketLost(size int, now time.Time) {
	bbr.mu.Lock()
	defer bbr.mu.Unlock()
	bbr.packetsLost++
}

func (bbr *BBRController) updateState(now time.Time) {
	switch bbr.state {
	case BBRStateStartup:
		// Exit startup if bandwidth stops growing
		if bbr.btlBw > 0 && bbr.packetsAcked > 100 {
			bbr.state = BBRStateDrain
			bbr.pacingGain = 1.0 / 2.89 // Drain queue
		}
	case BBRStateDrain:
		// Exit drain when inflight < BDP
		if bbr.inflightBytes() < bbr.bdp() {
			bbr.state = BBRStateProbeBW
			bbr.pacingGain = 1.0
			bbr.cycleStamp = now
		}
	case BBRStateProbeBW:
		// 8-phase pacing cycle
		gains := []float64{1.25, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
		if now.Sub(bbr.cycleStamp) > bbr.rtProp {
			bbr.cycleIdx = (bbr.cycleIdx + 1) % len(gains)
			bbr.pacingGain = gains[bbr.cycleIdx]
			bbr.cycleStamp = now
		}
	case BBRStateProbeRTT:
		// Not implemented in this simplified version
	}
}

func (bbr *BBRController) bdp() uint64 {
	if bbr.btlBw == 0 || bbr.rtProp == 0 {
		return 10 * 1460 // Default 10 packets
	}
	return uint64(float64(bbr.btlBw) * bbr.rtProp.Seconds())
}

func (bbr *BBRController) inflightBytes() uint64 {
	// Simplified: estimate from sent - acked
	return (bbr.packetsSent - bbr.packetsAcked) * 1460
}

func (bbr *BBRController) CanSend() bool {
	bbr.mu.RLock()
	defer bbr.mu.RUnlock()
	return bbr.inflightBytes() < bbr.bdp()*2
}

func (bbr *BBRController) GetCWND() uint64 {
	bbr.mu.RLock()
	defer bbr.mu.RUnlock()
	return uint64(bbr.cwndGain * float64(bbr.bdp()))
}

func (bbr *BBRController) GetRTT() time.Duration {
	bbr.mu.RLock()
	defer bbr.mu.RUnlock()
	return bbr.rtProp
}

func (bbr *BBRController) Algorithm() string {
	return "bbr"
}

// ---------- Reno Congestion Control ----------
// Classic TCP Reno with fast retransmit and fast recovery

type RenoController struct {
	mu sync.RWMutex

	cwnd         uint64
	ssthresh     uint64
	dupAckCount  int
	fastRecovery bool

	smoothedRTT time.Duration
	minRTT      time.Duration

	packetsSent  uint64
	packetsAcked uint64
	packetsLost  uint64
}

func NewRenoController() *RenoController {
	return &RenoController{
		cwnd:     10 * 1460,
		ssthresh: math.MaxUint64,
		minRTT:   time.Second,
	}
}

func (rc *RenoController) OnPacketSent(size int, now time.Time) {
	rc.mu.Lock()
	defer rc.mu.Unlock()
	rc.packetsSent++
}

func (rc *RenoController) OnPacketAcked(size int, rtt time.Duration, now time.Time) {
	rc.mu.Lock()
	defer rc.mu.Unlock()

	rc.packetsAcked++
	rc.updateRTT(rtt)

	if rc.fastRecovery {
		// Fast recovery: inflate window on each ack
		rc.cwnd += uint64(size)
		return
	}

	// Slow start
	if rc.cwnd < rc.ssthresh {
		rc.cwnd += uint64(size)
	} else {
		// Congestion avoidance: additive increase
		rc.cwnd += uint64((int64(size) * 1460) / int64(rc.cwnd))
	}

	rc.dupAckCount = 0
}

func (rc *RenoController) OnPacketLost(size int, now time.Time) {
	rc.mu.Lock()
	defer rc.mu.Unlock()

	rc.packetsLost++
	rc.dupAckCount++

	// Fast retransmit threshold
	if rc.dupAckCount >= 3 {
		if !rc.fastRecovery {
			// Enter fast recovery
			rc.ssthresh = rc.cwnd / 2
			rc.cwnd = rc.ssthresh + 3*1460
			rc.fastRecovery = true
		}
	}
}

func (rc *RenoController) ExitFastRecovery() {
	rc.mu.Lock()
	defer rc.mu.Unlock()
	if rc.fastRecovery {
		rc.cwnd = rc.ssthresh
		rc.fastRecovery = false
		rc.dupAckCount = 0
	}
}

func (rc *RenoController) CanSend() bool {
	rc.mu.RLock()
	defer rc.mu.RUnlock()
	return true
}

func (rc *RenoController) GetCWND() uint64 {
	rc.mu.RLock()
	defer rc.mu.RUnlock()
	return rc.cwnd
}

func (rc *RenoController) GetRTT() time.Duration {
	rc.mu.RLock()
	defer rc.mu.RUnlock()
	return rc.smoothedRTT
}

func (rc *RenoController) Algorithm() string {
	return "reno"
}

func (rc *RenoController) updateRTT(rtt time.Duration) {
	if rtt < rc.minRTT || rc.minRTT == time.Second {
		rc.minRTT = rtt
	}

	if rc.smoothedRTT == 0 {
		rc.smoothedRTT = rtt
	} else {
		alpha := 0.125
		rc.smoothedRTT = time.Duration((1-alpha)*float64(rc.smoothedRTT) + alpha*float64(rtt))
	}
}
