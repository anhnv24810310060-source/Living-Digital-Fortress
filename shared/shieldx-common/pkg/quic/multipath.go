// Package quic - Multipath QUIC implementation
// Enables simultaneous data transmission over multiple network paths
// Provides redundancy, increased throughput, and seamless failover
package quic

import (
	"context"
	"fmt"
	"net"
	"sync"
	"sync/atomic"
	"time"
)

// PathState represents the state of a QUIC path
type PathState int

const (
	PathStateActive  PathState = 0 // Path is actively used
	PathStateStandby PathState = 1 // Path is validated but not primary
	PathStateProbing PathState = 2 // Path is being validated
	PathStateFailed  PathState = 3 // Path has failed validation
)

// NetworkPath represents a single network path in multipath QUIC
type NetworkPath struct {
	ID         uint64
	LocalAddr  net.Addr
	RemoteAddr net.Addr
	State      atomic.Uint32 // PathState

	// Path metrics
	RTT           atomic.Uint64 // microseconds
	LossRate      atomic.Uint64 // basis points (0.01%)
	BandwidthBps  atomic.Uint64 // bits per second
	PacketsSent   atomic.Uint64
	PacketsAcked  atomic.Uint64
	PacketsLost   atomic.Uint64
	BytesSent     atomic.Uint64
	BytesReceived atomic.Uint64

	// Path scheduling
	Priority       int // Higher = preferred
	LastUsed       time.Time
	ValidationTime time.Time

	// Congestion control (per-path)
	CC CongestionController
}

// MultipathManager manages multiple QUIC paths for a connection
type MultipathManager struct {
	mu        sync.RWMutex
	connID    []byte
	paths     map[uint64]*NetworkPath
	primaryID uint64

	// Scheduling policy
	scheduler PathScheduler

	// Configuration
	config MultipathConfig

	// Metrics
	totalPaths     atomic.Uint64
	activePathsare atomic.Uint64
	failovers      atomic.Uint64
	pathMigrations atomic.Uint64
}

// MultipathConfig configures multipath behavior
type MultipathConfig struct {
	MaxPaths          int           // Maximum simultaneous paths
	MinPaths          int           // Minimum paths to maintain
	EnableAggregation bool          // Use multiple paths simultaneously
	Scheduler         string        // "roundrobin", "minrtt", "weighted"
	ProbeInterval     time.Duration // Path validation frequency
	FailureThreshold  int           // Consecutive failures before path disabled
}

// PathScheduler decides which path to use for each packet
type PathScheduler interface {
	SelectPath(paths map[uint64]*NetworkPath) *NetworkPath
	UpdateMetrics(pathID uint64, rtt time.Duration, lost bool)
	Name() string
}

// NewMultipathManager creates a new multipath manager
func NewMultipathManager(connID []byte, cfg MultipathConfig) *MultipathManager {
	if cfg.MaxPaths == 0 {
		cfg.MaxPaths = 4
	}
	if cfg.MinPaths == 0 {
		cfg.MinPaths = 1
	}
	if cfg.ProbeInterval == 0 {
		cfg.ProbeInterval = 5 * time.Second
	}
	if cfg.FailureThreshold == 0 {
		cfg.FailureThreshold = 5
	}

	var scheduler PathScheduler
	switch cfg.Scheduler {
	case "minrtt":
		scheduler = &MinRTTScheduler{}
	case "weighted":
		scheduler = &WeightedScheduler{}
	default:
		scheduler = &RoundRobinScheduler{}
	}

	return &MultipathManager{
		connID:    connID,
		paths:     make(map[uint64]*NetworkPath),
		scheduler: scheduler,
		config:    cfg,
	}
}

// AddPath adds a new network path
func (m *MultipathManager) AddPath(local, remote net.Addr, priority int) (uint64, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if len(m.paths) >= m.config.MaxPaths {
		return 0, fmt.Errorf("max paths reached: %d", m.config.MaxPaths)
	}

	pathID := uint64(len(m.paths) + 1)
	path := &NetworkPath{
		ID:         pathID,
		LocalAddr:  local,
		RemoteAddr: remote,
		Priority:   priority,
		LastUsed:   time.Now(),
		CC:         NewBBRController(), // BBR for multipath
	}
	path.State.Store(uint32(PathStateProbing))

	m.paths[pathID] = path
	m.totalPaths.Add(1)

	// Start path validation
	go m.validatePath(pathID)

	// If first path, make it primary
	if m.primaryID == 0 {
		m.primaryID = pathID
	}

	return pathID, nil
}

// RemovePath removes a network path
func (m *MultipathManager) RemovePath(pathID uint64) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, ok := m.paths[pathID]; !ok {
		return fmt.Errorf("path not found: %d", pathID)
	}

	delete(m.paths, pathID)

	// If primary path removed, select new primary
	if m.primaryID == pathID {
		m.selectNewPrimary()
	}

	return nil
}

// SendPacket sends a packet on the best available path
func (m *MultipathManager) SendPacket(data []byte) error {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if len(m.paths) == 0 {
		return fmt.Errorf("no paths available")
	}

	// Select path using scheduler
	path := m.scheduler.SelectPath(m.paths)
	if path == nil {
		return fmt.Errorf("no suitable path")
	}

	// Send packet (simplified - real implementation uses QUIC frames)
	path.PacketsSent.Add(1)
	path.BytesSent.Add(uint64(len(data)))
	path.LastUsed = time.Now()

	// Update congestion control
	if path.CC != nil {
		path.CC.OnPacketSent(len(data), time.Now())
	}

	return nil
}

// OnPacketAck handles packet acknowledgment
func (m *MultipathManager) OnPacketAck(pathID uint64, rtt time.Duration, bytes int) {
	m.mu.RLock()
	path, ok := m.paths[pathID]
	m.mu.RUnlock()

	if !ok {
		return
	}

	path.PacketsAcked.Add(1)
	path.BytesReceived.Add(uint64(bytes))

	// Update RTT (EWMA)
	oldRTT := time.Duration(path.RTT.Load()) * time.Microsecond
	if oldRTT == 0 {
		path.RTT.Store(uint64(rtt.Microseconds()))
	} else {
		newRTT := time.Duration(0.7*float64(oldRTT) + 0.3*float64(rtt))
		path.RTT.Store(uint64(newRTT.Microseconds()))
	}

	// Update congestion control
	if path.CC != nil {
		path.CC.OnPacketAcked(bytes, rtt, time.Now())
	}

	// Update scheduler metrics
	m.scheduler.UpdateMetrics(pathID, rtt, false)
}

// OnPacketLost handles packet loss
func (m *MultipathManager) OnPacketLost(pathID uint64, bytes int) {
	m.mu.RLock()
	path, ok := m.paths[pathID]
	m.mu.RUnlock()

	if !ok {
		return
	}

	path.PacketsLost.Add(1)

	// Update loss rate (basis points)
	sent := path.PacketsSent.Load()
	lost := path.PacketsLost.Load()
	if sent > 0 {
		lossRateBps := (lost * 10000) / sent
		path.LossRate.Store(lossRateBps)
	}

	// Update congestion control
	if path.CC != nil {
		path.CC.OnPacketLost(bytes, time.Now())
	}

	// Update scheduler metrics
	m.scheduler.UpdateMetrics(pathID, 0, true)

	// Check if path should be failed
	if lost > uint64(m.config.FailureThreshold) && float64(lost)/float64(sent) > 0.1 {
		m.markPathFailed(pathID)
	}
}

// validatePath performs path validation using PATH_CHALLENGE/RESPONSE
func (m *MultipathManager) validatePath(pathID uint64) {
	m.mu.RLock()
	path, ok := m.paths[pathID]
	m.mu.RUnlock()

	if !ok {
		return
	}

	// Send PATH_CHALLENGE (simplified)
	// Real implementation: send QUIC PATH_CHALLENGE frame
	time.Sleep(100 * time.Millisecond) // Simulate RTT

	// Mark as active if validation succeeds
	path.State.Store(uint32(PathStateActive))
	path.ValidationTime = time.Now()
	m.activePathsare.Add(1)
}

// markPathFailed marks a path as failed and triggers failover
func (m *MultipathManager) markPathFailed(pathID uint64) {
	m.mu.Lock()
	defer m.mu.Unlock()

	path, ok := m.paths[pathID]
	if !ok {
		return
	}

	path.State.Store(uint32(PathStateFailed))
	m.activePathsare.Add(^uint64(0)) // Decrement

	// If primary path failed, trigger failover
	if m.primaryID == pathID {
		m.selectNewPrimary()
		m.failovers.Add(1)
	}
}

// selectNewPrimary selects a new primary path after failover
func (m *MultipathManager) selectNewPrimary() {
	var bestPath *NetworkPath
	var bestScore float64

	for _, path := range m.paths {
		if PathState(path.State.Load()) != PathStateActive {
			continue
		}

		// Score based on RTT, loss rate, priority
		rtt := float64(path.RTT.Load())
		lossRate := float64(path.LossRate.Load()) / 10000.0
		priority := float64(path.Priority)

		score := (1000000.0 / (rtt + 1.0)) * (1.0 - lossRate) * (priority + 1.0)

		if bestPath == nil || score > bestScore {
			bestPath = path
			bestScore = score
		}
	}

	if bestPath != nil {
		m.primaryID = bestPath.ID
		m.pathMigrations.Add(1)
	}
}

// GetPrimaryPath returns the current primary path
func (m *MultipathManager) GetPrimaryPath() *NetworkPath {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.paths[m.primaryID]
}

// GetActivePaths returns all active paths
func (m *MultipathManager) GetActivePaths() []*NetworkPath {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var active []*NetworkPath
	for _, path := range m.paths {
		if PathState(path.State.Load()) == PathStateActive {
			active = append(active, path)
		}
	}
	return active
}

// Metrics returns multipath metrics
func (m *MultipathManager) Metrics() map[string]interface{} {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return map[string]interface{}{
		"total_paths":     m.totalPaths.Load(),
		"active_paths":    m.activePathsare.Load(),
		"failovers":       m.failovers.Load(),
		"path_migrations": m.pathMigrations.Load(),
		"primary_path_id": m.primaryID,
		"scheduler":       m.scheduler.Name(),
	}
}

// ---------- Path Schedulers ----------

// RoundRobinScheduler distributes packets evenly across paths
type RoundRobinScheduler struct {
	mu      sync.Mutex
	counter uint64
}

func (r *RoundRobinScheduler) SelectPath(paths map[uint64]*NetworkPath) *NetworkPath {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Filter active paths
	active := make([]*NetworkPath, 0, len(paths))
	for _, p := range paths {
		if PathState(p.State.Load()) == PathStateActive {
			active = append(active, p)
		}
	}

	if len(active) == 0 {
		return nil
	}

	idx := r.counter % uint64(len(active))
	r.counter++
	return active[idx]
}

func (r *RoundRobinScheduler) UpdateMetrics(pathID uint64, rtt time.Duration, lost bool) {}
func (r *RoundRobinScheduler) Name() string                                              { return "roundrobin" }

// MinRTTScheduler selects the path with lowest RTT
type MinRTTScheduler struct{}

func (m *MinRTTScheduler) SelectPath(paths map[uint64]*NetworkPath) *NetworkPath {
	var best *NetworkPath
	var minRTT uint64 = ^uint64(0)

	for _, p := range paths {
		if PathState(p.State.Load()) != PathStateActive {
			continue
		}

		rtt := p.RTT.Load()
		if rtt == 0 {
			rtt = 50000 // Default 50ms if unknown
		}

		if rtt < minRTT {
			minRTT = rtt
			best = p
		}
	}

	return best
}

func (m *MinRTTScheduler) UpdateMetrics(pathID uint64, rtt time.Duration, lost bool) {}
func (m *MinRTTScheduler) Name() string                                              { return "minrtt" }

// WeightedScheduler uses weighted round-robin based on bandwidth and priority
type WeightedScheduler struct {
	mu      sync.Mutex
	weights map[uint64]float64
}

func (w *WeightedScheduler) SelectPath(paths map[uint64]*NetworkPath) *NetworkPath {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.weights == nil {
		w.weights = make(map[uint64]float64)
	}

	// Calculate weights based on bandwidth estimate and priority
	for id, p := range paths {
		if PathState(p.State.Load()) != PathStateActive {
			continue
		}

		bw := float64(p.BandwidthBps.Load())
		if bw == 0 {
			bw = 1e6 // Default 1 Mbps
		}
		priority := float64(p.Priority + 1)
		w.weights[id] = bw * priority
	}

	// Weighted random selection
	totalWeight := 0.0
	for _, wt := range w.weights {
		totalWeight += wt
	}

	if totalWeight == 0 {
		return nil
	}

	r := float64(time.Now().UnixNano()%int64(totalWeight*1000)) / 1000.0
	cumulative := 0.0

	for id, wt := range w.weights {
		cumulative += wt
		if r <= cumulative {
			return paths[id]
		}
	}

	// Fallback
	for _, p := range paths {
		if PathState(p.State.Load()) == PathStateActive {
			return p
		}
	}

	return nil
}

func (w *WeightedScheduler) UpdateMetrics(pathID uint64, rtt time.Duration, lost bool) {
	// Could adjust weights based on recent performance
}

func (w *WeightedScheduler) Name() string { return "weighted" }

// ---------- Connection Migration ----------

// MigrateConnection handles seamless connection migration to a new path
func (m *MultipathManager) MigrateConnection(ctx context.Context, newRemote net.Addr) error {
	// Add new path
	local := m.GetPrimaryPath().LocalAddr
	pathID, err := m.AddPath(local, newRemote, 10) // High priority
	if err != nil {
		return err
	}

	// Wait for validation
	timeout := time.NewTimer(5 * time.Second)
	defer timeout.Stop()

	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-timeout.C:
			return fmt.Errorf("migration timeout")
		case <-ticker.C:
			m.mu.RLock()
			path, ok := m.paths[pathID]
			m.mu.RUnlock()

			if ok && PathState(path.State.Load()) == PathStateActive {
				// Switch to new path
				m.mu.Lock()
				m.primaryID = pathID
				m.mu.Unlock()
				m.pathMigrations.Add(1)
				return nil
			}
		}
	}
}
