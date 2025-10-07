package shadow
package main

import (
	"context"
	"fmt"
	"log"
	"math/rand"
	"sync"
	"time"
)

// ChaosEngineer implements automated chaos engineering for resilience testing
// ✅ PHẢI test rules trong shadow trước deploy (Chaos Engineering validates this)
type ChaosEngineer struct {
	experiments map[string]*ChaosExperiment
	active      map[string]*ActiveExperiment
	metrics     *ChaosMetrics
	mu          sync.RWMutex
	enabled     bool
}

// ChaosExperiment defines a chaos experiment
type ChaosExperiment struct {
	ID          string        `json:"id"`
	Name        string        `json:"name"`
	Type        ChaosType     `json:"type"`
	Target      string        `json:"target"` // service name or component
	Duration    time.Duration `json:"duration"`
	Probability float64       `json:"probability"` // 0.0 to 1.0
	Severity    ChaosSeverity `json:"severity"`
	Parameters  map[string]interface{} `json:"parameters"`
	Hypothesis  string        `json:"hypothesis"`
	Enabled     bool          `json:"enabled"`
}

// ActiveExperiment represents a running chaos experiment
type ActiveExperiment struct {
	Experiment  *ChaosExperiment
	StartTime   time.Time
	EndTime     time.Time
	ObservedImpact map[string]interface{}
	Status      ExperimentStatus
	cancel      context.CancelFunc
}

// ChaosType represents different types of chaos experiments
type ChaosType string

const (
	// Service failures
	ChaosServiceKill       ChaosType = "SERVICE_KILL"
	ChaosServiceRestart    ChaosType = "SERVICE_RESTART"
	ChaosServiceSlow       ChaosType = "SERVICE_SLOW"
	ChaosServiceCrash      ChaosType = "SERVICE_CRASH"
	
	// Network chaos
	ChaosNetworkPartition  ChaosType = "NETWORK_PARTITION"
	ChaosNetworkLatency    ChaosType = "NETWORK_LATENCY"
	ChaosNetworkPacketLoss ChaosType = "NETWORK_PACKET_LOSS"
	ChaosNetworkBandwidth  ChaosType = "NETWORK_BANDWIDTH_LIMIT"
	
	// Resource exhaustion
	ChaosCPUStress         ChaosType = "CPU_STRESS"
	ChaosMemoryStress      ChaosType = "MEMORY_STRESS"
	ChaosDiskStress        ChaosType = "DISK_STRESS"
	ChaosIOStress          ChaosType = "IO_STRESS"
	
	// Dependency failures
	ChaosDatabaseFailure   ChaosType = "DATABASE_FAILURE"
	ChaosCacheFailure      ChaosType = "CACHE_FAILURE"
	ChaosAPIFailure        ChaosType = "API_FAILURE"
	
	// Data chaos
	ChaosDataCorruption    ChaosType = "DATA_CORRUPTION"
	ChaosDataLoss          ChaosType = "DATA_LOSS"
	
	// Time chaos
	ChaosClockSkew         ChaosType = "CLOCK_SKEW"
)

// ChaosSeverity represents severity levels
type ChaosSeverity int

const (
	SeverityLow ChaosSeverity = iota
	SeverityMedium
	SeverityHigh
	SeverityCritical
)

// ExperimentStatus represents experiment status
type ExperimentStatus int

const (
	StatusPending ExperimentStatus = iota
	StatusRunning
	StatusCompleted
	StatusFailed
	StatusAborted
)

// ChaosMetrics tracks chaos engineering metrics
type ChaosMetrics struct {
	TotalExperiments     int
	SuccessfulExperiments int
	FailedExperiments    int
	ServicesRecovered    int
	MeanRecoveryTime     time.Duration
	ImpactedServices     map[string]int
	mu                   sync.RWMutex
}

// NewChaosEngineer creates a new chaos engineer
func NewChaosEngineer() *ChaosEngineer {
	ce := &ChaosEngineer{
		experiments: make(map[string]*ChaosExperiment),
		active:      make(map[string]*ActiveExperiment),
		metrics:     &ChaosMetrics{ImpactedServices: make(map[string]int)},
		enabled:     false, // Disabled by default - must be explicitly enabled
	}

	// Register default experiments
	ce.registerDefaultExperiments()

	log.Printf("[chaos] Chaos Engineering initialized (disabled by default)")
	return ce
}

// registerDefaultExperiments registers built-in chaos experiments
func (ce *ChaosEngineer) registerDefaultExperiments() {
	// Service failure experiments
	ce.RegisterExperiment(&ChaosExperiment{
		ID:          "exp-001",
		Name:        "Credits Service Failure",
		Type:        ChaosServiceKill,
		Target:      "credits",
		Duration:    30 * time.Second,
		Probability: 0.1,
		Severity:    SeverityHigh,
		Hypothesis:  "System should gracefully handle credits service failure with proper fallbacks",
		Enabled:     false,
	})

	ce.RegisterExperiment(&ChaosExperiment{
		ID:          "exp-002",
		Name:        "Database Connection Failure",
		Type:        ChaosDatabaseFailure,
		Target:      "postgresql",
		Duration:    20 * time.Second,
		Probability: 0.05,
		Severity:    SeverityCritical,
		Parameters: map[string]interface{}{
			"failure_mode": "connection_timeout",
			"timeout_ms":   5000,
		},
		Hypothesis: "Services should retry with exponential backoff and use circuit breakers",
		Enabled:    false,
	})

	// Network chaos experiments
	ce.RegisterExperiment(&ChaosExperiment{
		ID:          "exp-003",
		Name:        "Network Partition Between Services",
		Type:        ChaosNetworkPartition,
		Target:      "orchestrator<->guardian",
		Duration:    45 * time.Second,
		Probability: 0.05,
		Severity:    SeverityHigh,
		Parameters: map[string]interface{}{
			"partition_services": []string{"orchestrator", "guardian"},
		},
		Hypothesis: "System should detect partition and reroute traffic appropriately",
		Enabled:    false,
	})

	ce.RegisterExperiment(&ChaosExperiment{
		ID:          "exp-004",
		Name:        "High Network Latency",
		Type:        ChaosNetworkLatency,
		Target:      "all",
		Duration:    60 * time.Second,
		Probability: 0.2,
		Severity:    SeverityMedium,
		Parameters: map[string]interface{}{
			"latency_ms": 500,
			"jitter_ms":  100,
		},
		Hypothesis: "Timeouts should be properly configured and requests should not cascade",
		Enabled:    false,
	})

	// Resource exhaustion experiments
	ce.RegisterExperiment(&ChaosExperiment{
		ID:          "exp-005",
		Name:        "Memory Pressure",
		Type:        ChaosMemoryStress,
		Target:      "shadow",
		Duration:    30 * time.Second,
		Probability: 0.1,
		Severity:    SeverityHigh,
		Parameters: map[string]interface{}{
			"memory_percentage": 90,
		},
		Hypothesis: "Service should handle memory pressure gracefully without OOM kills",
		Enabled:    false,
	})

	// Cache failure
	ce.RegisterExperiment(&ChaosExperiment{
		ID:          "exp-006",
		Name:        "Redis Cache Failure",
		Type:        ChaosCacheFailure,
		Target:      "redis",
		Duration:    40 * time.Second,
		Probability: 0.15,
		Severity:    SeverityMedium,
		Parameters: map[string]interface{}{
			"failure_mode": "unavailable",
		},
		Hypothesis: "Services should fall back to database queries without cascading failures",
		Enabled:    false,
	})

	log.Printf("[chaos] Registered %d default chaos experiments", len(ce.experiments))
}

// RegisterExperiment registers a chaos experiment
func (ce *ChaosEngineer) RegisterExperiment(exp *ChaosExperiment) error {
	ce.mu.Lock()
	defer ce.mu.Unlock()

	if _, exists := ce.experiments[exp.ID]; exists {
		return fmt.Errorf("experiment %s already registered", exp.ID)
	}

	ce.experiments[exp.ID] = exp
	log.Printf("[chaos] Registered experiment: %s (%s)", exp.Name, exp.ID)
	return nil
}

// Enable enables chaos engineering
func (ce *ChaosEngineer) Enable() {
	ce.mu.Lock()
	defer ce.mu.Unlock()
	
	ce.enabled = true
	log.Printf("[chaos] Chaos Engineering ENABLED")
}

// Disable disables chaos engineering
func (ce *ChaosEngineer) Disable() {
	ce.mu.Lock()
	defer ce.mu.Unlock()
	
	ce.enabled = false
	
	// Stop all active experiments
	for _, active := range ce.active {
		if active.cancel != nil {
			active.cancel()
		}
	}
	
	ce.active = make(map[string]*ActiveExperiment)
	log.Printf("[chaos] Chaos Engineering DISABLED")
}

// RunExperiment runs a specific chaos experiment
func (ce *ChaosEngineer) RunExperiment(ctx context.Context, experimentID string) error {
	ce.mu.RLock()
	if !ce.enabled {
		ce.mu.RUnlock()
		return fmt.Errorf("chaos engineering is disabled")
	}
	
	exp, exists := ce.experiments[experimentID]
	if !exists {
		ce.mu.RUnlock()
		return fmt.Errorf("experiment %s not found", experimentID)
	}
	
	if !exp.Enabled {
		ce.mu.RUnlock()
		return fmt.Errorf("experiment %s is disabled", experimentID)
	}
	ce.mu.RUnlock()

	// Check if experiment is already running
	ce.mu.Lock()
	if _, running := ce.active[experimentID]; running {
		ce.mu.Unlock()
		return fmt.Errorf("experiment %s is already running", experimentID)
	}

	// Create active experiment
	ctx, cancel := context.WithTimeout(ctx, exp.Duration)
	active := &ActiveExperiment{
		Experiment:     exp,
		StartTime:      time.Now(),
		EndTime:        time.Now().Add(exp.Duration),
		ObservedImpact: make(map[string]interface{}),
		Status:         StatusRunning,
		cancel:         cancel,
	}
	ce.active[experimentID] = active
	ce.mu.Unlock()

	// Update metrics
	ce.metrics.mu.Lock()
	ce.metrics.TotalExperiments++
	ce.metrics.ImpactedServices[exp.Target]++
	ce.metrics.mu.Unlock()

	log.Printf("[chaos] Starting experiment: %s on target %s for %v", exp.Name, exp.Target, exp.Duration)

	// Run the experiment
	go ce.executeExperiment(ctx, active)

	return nil
}

// executeExperiment executes the chaos experiment
func (ce *ChaosEngineer) executeExperiment(ctx context.Context, active *ActiveExperiment) {
	exp := active.Experiment
	startTime := time.Now()

	defer func() {
		duration := time.Since(startTime)
		
		ce.mu.Lock()
		delete(ce.active, exp.ID)
		ce.mu.Unlock()

		ce.metrics.mu.Lock()
		if active.Status == StatusCompleted {
			ce.metrics.SuccessfulExperiments++
			ce.metrics.ServicesRecovered++
			
			// Update mean recovery time
			n := ce.metrics.SuccessfulExperiments
			ce.metrics.MeanRecoveryTime = (ce.metrics.MeanRecoveryTime*time.Duration(n-1) + duration) / time.Duration(n)
		} else {
			ce.metrics.FailedExperiments++
		}
		ce.metrics.mu.Unlock()

		log.Printf("[chaos] Experiment %s completed with status: %v (duration: %v)", 
			exp.Name, active.Status, duration)
	}()

	// Apply chaos based on experiment type
	switch exp.Type {
	case ChaosServiceKill:
		ce.injectServiceFailure(ctx, active)
	case ChaosNetworkLatency:
		ce.injectNetworkLatency(ctx, active)
	case ChaosNetworkPartition:
		ce.injectNetworkPartition(ctx, active)
	case ChaosDatabaseFailure:
		ce.injectDatabaseFailure(ctx, active)
	case ChaosMemoryStress:
		ce.injectMemoryStress(ctx, active)
	case ChaosCacheFailure:
		ce.injectCacheFailure(ctx, active)
	default:
		log.Printf("[chaos] Unsupported chaos type: %s", exp.Type)
		active.Status = StatusFailed
		return
	}

	// Wait for experiment duration or cancellation
	select {
	case <-ctx.Done():
		active.Status = StatusCompleted
	case <-time.After(exp.Duration):
		active.Status = StatusCompleted
	}

	// Cleanup/restore
	ce.restoreService(active)
}

// injectServiceFailure simulates service failure
func (ce *ChaosEngineer) injectServiceFailure(ctx context.Context, active *ActiveExperiment) {
	exp := active.Experiment
	log.Printf("[chaos] Injecting service failure for: %s", exp.Target)

	// Record pre-failure metrics
	active.ObservedImpact["pre_failure_timestamp"] = time.Now()
	
	// Simulate service kill (in production, would actually stop the service)
	// For now, we simulate by setting a flag or calling service API
	
	// Monitor system behavior
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			// Collect metrics during failure
			metrics := ce.collectSystemMetrics(exp.Target)
			active.ObservedImpact["metrics"] = metrics
			
			// Check if system recovered gracefully
			if ce.isSystemHealthy(exp.Target) {
				active.ObservedImpact["graceful_degradation"] = true
			}
		}
	}
}

// injectNetworkLatency simulates network latency
func (ce *ChaosEngineer) injectNetworkLatency(ctx context.Context, active *ActiveExperiment) {
	exp := active.Experiment
	latencyMs := 100
	if l, ok := exp.Parameters["latency_ms"].(int); ok {
		latencyMs = l
	}

	log.Printf("[chaos] Injecting %dms network latency for: %s", latencyMs, exp.Target)

	// In production, would use tc (traffic control) or toxiproxy
	// Simulate by adding delays to requests
	active.ObservedImpact["injected_latency_ms"] = latencyMs
	active.ObservedImpact["start_time"] = time.Now()

	<-ctx.Done()
}

// injectNetworkPartition simulates network partition
func (ce *ChaosEngineer) injectNetworkPartition(ctx context.Context, active *ActiveExperiment) {
	exp := active.Experiment
	log.Printf("[chaos] Injecting network partition for: %s", exp.Target)

	// In production, would use iptables rules or network policies
	// services, _ := exp.Parameters["partition_services"].([]string)
	
	active.ObservedImpact["partition_start"] = time.Now()
	active.ObservedImpact["partition_detected"] = false

	// Monitor for partition detection
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			// Check if system detected the partition
			if ce.hasDetectedPartition(exp.Target) {
				active.ObservedImpact["partition_detected"] = true
				active.ObservedImpact["detection_time"] = time.Since(active.ObservedImpact["partition_start"].(time.Time))
			}
		}
	}
}

// injectDatabaseFailure simulates database failure
func (ce *ChaosEngineer) injectDatabaseFailure(ctx context.Context, active *ActiveExperiment) {
	exp := active.Experiment
	log.Printf("[chaos] Injecting database failure for: %s", exp.Target)

	active.ObservedImpact["failure_start"] = time.Now()
	active.ObservedImpact["circuit_breaker_opened"] = false

	// In production, would block database connections or kill postgres process
	
	// Monitor for circuit breaker activation
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			// Check if circuit breaker opened
			if ce.isCircuitBreakerOpen("database") {
				active.ObservedImpact["circuit_breaker_opened"] = true
				active.ObservedImpact["circuit_breaker_time"] = time.Since(active.ObservedImpact["failure_start"].(time.Time))
			}
		}
	}
}

// injectMemoryStress simulates memory pressure
func (ce *ChaosEngineer) injectMemoryStress(ctx context.Context, active *ActiveExperiment) {
	exp := active.Experiment
	log.Printf("[chaos] Injecting memory stress for: %s", exp.Target)

	// In production, would use stress-ng or allocate large memory blocks
	percentage := 80
	if p, ok := exp.Parameters["memory_percentage"].(int); ok {
		percentage = p
	}

	active.ObservedImpact["memory_stress_percentage"] = percentage
	active.ObservedImpact["oom_killed"] = false

	<-ctx.Done()
}

// injectCacheFailure simulates cache failure
func (ce *ChaosEngineer) injectCacheFailure(ctx context.Context, active *ActiveExperiment) {
	exp := active.Experiment
	log.Printf("[chaos] Injecting cache failure for: %s", exp.Target)

	active.ObservedImpact["cache_failure_start"] = time.Now()
	active.ObservedImpact["fallback_activated"] = false

	// Monitor for database fallback
	ticker := time.NewTicker(3 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if ce.isDatabaseFallbackActive() {
				active.ObservedImpact["fallback_activated"] = true
				active.ObservedImpact["fallback_time"] = time.Since(active.ObservedImpact["cache_failure_start"].(time.Time))
			}
		}
	}
}

// restoreService restores service after experiment
func (ce *ChaosEngineer) restoreService(active *ActiveExperiment) {
	log.Printf("[chaos] Restoring service: %s", active.Experiment.Target)
	
	// In production, would restart services, restore network rules, etc.
	active.ObservedImpact["restore_time"] = time.Now()
}

// Helper methods for monitoring
func (ce *ChaosEngineer) collectSystemMetrics(target string) map[string]interface{} {
	// In production, would query Prometheus or metrics API
	return map[string]interface{}{
		"timestamp":    time.Now(),
		"target":       target,
		"request_rate": rand.Float64() * 1000,
		"error_rate":   rand.Float64() * 0.1,
		"latency_p99":  rand.Float64() * 500,
	}
}

func (ce *ChaosEngineer) isSystemHealthy(target string) bool {
	// Simulate health check
	return rand.Float64() > 0.3
}

func (ce *ChaosEngineer) hasDetectedPartition(target string) bool {
	// Simulate partition detection
	return rand.Float64() > 0.5
}

func (ce *ChaosEngineer) isCircuitBreakerOpen(service string) bool {
	// Simulate circuit breaker check
	return rand.Float64() > 0.6
}

func (ce *ChaosEngineer) isDatabaseFallbackActive() bool {
	// Simulate fallback detection
	return rand.Float64() > 0.4
}

// GetExperimentResults returns results for a completed experiment
func (ce *ChaosEngineer) GetExperimentResults(experimentID string) (map[string]interface{}, error) {
	ce.mu.RLock()
	defer ce.mu.RUnlock()

	exp, exists := ce.experiments[experimentID]
	if !exists {
		return nil, fmt.Errorf("experiment %s not found", experimentID)
	}

	results := map[string]interface{}{
		"experiment_id":   exp.ID,
		"experiment_name": exp.Name,
		"hypothesis":      exp.Hypothesis,
		"status":          "not_run",
	}

	// Check if experiment was run
	if active, running := ce.active[experimentID]; running {
		results["status"] = "running"
		results["start_time"] = active.StartTime
		results["observed_impact"] = active.ObservedImpact
	}

	return results, nil
}

// GetChaosMetrics returns chaos engineering metrics
func (ce *ChaosEngineer) GetChaosMetrics() map[string]interface{} {
	ce.metrics.mu.RLock()
	defer ce.metrics.mu.RUnlock()

	return map[string]interface{}{
		"total_experiments":      ce.metrics.TotalExperiments,
		"successful_experiments": ce.metrics.SuccessfulExperiments,
		"failed_experiments":     ce.metrics.FailedExperiments,
		"services_recovered":     ce.metrics.ServicesRecovered,
		"mean_recovery_time_ms":  ce.metrics.MeanRecoveryTime.Milliseconds(),
		"impacted_services":      ce.metrics.ImpactedServices,
		"enabled":                ce.enabled,
		"active_experiments":     len(ce.active),
	}
}

// RunContinuousChaos runs chaos experiments continuously in the background
func (ce *ChaosEngineer) RunContinuousChaos(ctx context.Context) {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	log.Printf("[chaos] Starting continuous chaos testing")

	for {
		select {
		case <-ctx.Done():
			log.Printf("[chaos] Stopping continuous chaos testing")
			return
		case <-ticker.C:
			ce.runRandomExperiment(ctx)
		}
	}
}

// runRandomExperiment runs a random enabled experiment
func (ce *ChaosEngineer) runRandomExperiment(ctx context.Context) {
	ce.mu.RLock()
	if !ce.enabled {
		ce.mu.RUnlock()
		return
	}

	// Collect enabled experiments
	var enabled []*ChaosExperiment
	for _, exp := range ce.experiments {
		if exp.Enabled && rand.Float64() < exp.Probability {
			enabled = append(enabled, exp)
		}
	}
	ce.mu.RUnlock()

	if len(enabled) == 0 {
		return
	}

	// Select random experiment
	exp := enabled[rand.Intn(len(enabled))]
	
	// Run it
	if err := ce.RunExperiment(ctx, exp.ID); err != nil {
		log.Printf("[chaos] Failed to run experiment %s: %v", exp.ID, err)
	}
}
