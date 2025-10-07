package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"sync"
	"time"
)

// ChaosEngine implements Chaos Engineering automation for proactive resilience testing
// This provides:
// - Service failure injection (Chaos Monkey)
// - Network partitioning simulation
// - Resource exhaustion testing
// - Dependency failure injection
type ChaosEngine struct {
	db               *sql.DB
	config           ChaosConfig
	experiments      map[string]*ChaosExperiment
	mu               sync.RWMutex
	enabled          bool
	safetyChecks     []SafetyCheck
	metricsCollector *ChaosMetricsCollector
}

// ChaosConfig contains chaos engineering configuration
type ChaosConfig struct {
	Enabled                  bool          `json:"enabled"`
	TargetServices           []string      `json:"target_services"`
	ExperimentInterval       time.Duration `json:"experiment_interval"`
	MaxConcurrentExperiments int           `json:"max_concurrent_experiments"`
	SafeHours                []int         `json:"safe_hours"` // Hours when chaos is allowed (0-23)
	ProductionEnabled        bool          `json:"production_enabled"`
	AutoRollbackEnabled      bool          `json:"auto_rollback_enabled"`
	MaxImpactThreshold       float64       `json:"max_impact_threshold"` // Max % of requests affected
}

// ChaosExperiment represents a single chaos experiment
type ChaosExperiment struct {
	ID              string                 `json:"id"`
	Name            string                 `json:"name"`
	Type            ExperimentType         `json:"type"`
	TargetService   string                 `json:"target_service"`
	Parameters      map[string]interface{} `json:"parameters"`
	Status          string                 `json:"status"` // "scheduled", "running", "completed", "failed", "rolled_back"
	StartTime       time.Time              `json:"start_time"`
	EndTime         time.Time              `json:"end_time"`
	Duration        time.Duration          `json:"duration"`
	Results         ExperimentResults      `json:"results"`
	SafetyViolation bool                   `json:"safety_violation"`
	RollbackReason  string                 `json:"rollback_reason,omitempty"`
}

// ExperimentType defines types of chaos experiments
type ExperimentType string

const (
	// Service-level experiments
	ExperimentServiceFailure     ExperimentType = "service_failure"
	ExperimentHighLatency        ExperimentType = "high_latency"
	ExperimentResourceExhaustion ExperimentType = "resource_exhaustion"
	ExperimentCPUSpike           ExperimentType = "cpu_spike"
	ExperimentMemoryLeak         ExperimentType = "memory_leak"

	// Network-level experiments
	ExperimentNetworkPartition ExperimentType = "network_partition"
	ExperimentPacketLoss       ExperimentType = "packet_loss"
	ExperimentBandwidthLimit   ExperimentType = "bandwidth_limit"
	ExperimentDNSFailure       ExperimentType = "dns_failure"

	// Database-level experiments
	ExperimentDBSlowQuery       ExperimentType = "db_slow_query"
	ExperimentDBConnectionLimit ExperimentType = "db_connection_limit"
	ExperimentDBFailover        ExperimentType = "db_failover"

	// Dependency experiments
	ExperimentDependencyFailure ExperimentType = "dependency_failure"
	ExperimentCacheFailure      ExperimentType = "cache_failure"
	ExperimentAPITimeout        ExperimentType = "api_timeout"
)

// ExperimentResults contains experiment metrics
type ExperimentResults struct {
	TotalRequests       int64         `json:"total_requests"`
	FailedRequests      int64         `json:"failed_requests"`
	AverageLatency      time.Duration `json:"average_latency"`
	P95Latency          time.Duration `json:"p95_latency"`
	P99Latency          time.Duration `json:"p99_latency"`
	ErrorRate           float64       `json:"error_rate"`
	ImpactPercentage    float64       `json:"impact_percentage"`
	RecoveryTime        time.Duration `json:"recovery_time"`
	CircuitBreakerTrips int           `json:"circuit_breaker_trips"`
	Observations        []string      `json:"observations"`
}

// SafetyCheck validates experiment safety before execution
type SafetyCheck func(ctx context.Context, experiment *ChaosExperiment) (bool, string)

// ChaosMetricsCollector collects metrics during experiments
type ChaosMetricsCollector struct {
	requestCounts map[string]int64
	errorCounts   map[string]int64
	latencies     map[string][]time.Duration
	mu            sync.RWMutex
}

// NewChaosEngine creates a new chaos engineering engine
func NewChaosEngine(db *sql.DB, config ChaosConfig) (*ChaosEngine, error) {
	engine := &ChaosEngine{
		db:               db,
		config:           config,
		experiments:      make(map[string]*ChaosExperiment),
		enabled:          config.Enabled,
		metricsCollector: NewChaosMetricsCollector(),
	}

	// Initialize schema
	if err := engine.initializeSchema(); err != nil {
		return nil, fmt.Errorf("failed to initialize chaos schema: %w", err)
	}

	// Register default safety checks
	engine.registerDefaultSafetyChecks()

	// Start experiment scheduler if enabled
	if config.Enabled {
		go engine.scheduleExperiments()
	}

	log.Printf("[chaos] Engine initialized (enabled=%v, production=%v)", config.Enabled, config.ProductionEnabled)
	return engine, nil
}

// initializeSchema creates necessary tables
func (ce *ChaosEngine) initializeSchema() error {
	schema := `
	CREATE TABLE IF NOT EXISTS chaos_experiments (
		id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
		name VARCHAR(255) NOT NULL,
		experiment_type VARCHAR(100) NOT NULL,
		target_service VARCHAR(255) NOT NULL,
		parameters JSONB,
		status VARCHAR(50) NOT NULL DEFAULT 'scheduled',
		start_time TIMESTAMP WITH TIME ZONE,
		end_time TIMESTAMP WITH TIME ZONE,
		duration_seconds INT,
		results JSONB,
		safety_violation BOOLEAN DEFAULT false,
		rollback_reason TEXT,
		created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
		updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
	);

	CREATE INDEX IF NOT EXISTS idx_chaos_experiments_status 
		ON chaos_experiments(status, start_time DESC);
	CREATE INDEX IF NOT EXISTS idx_chaos_experiments_type 
		ON chaos_experiments(experiment_type);

	-- Chaos experiment metrics table
	CREATE TABLE IF NOT EXISTS chaos_metrics (
		id BIGSERIAL PRIMARY KEY,
		experiment_id UUID NOT NULL REFERENCES chaos_experiments(id),
		metric_name VARCHAR(100) NOT NULL,
		metric_value DOUBLE PRECISION NOT NULL,
		timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
		metadata JSONB
	);

	CREATE INDEX IF NOT EXISTS idx_chaos_metrics_experiment 
		ON chaos_metrics(experiment_id, timestamp);

	-- Safety violations log
	CREATE TABLE IF NOT EXISTS chaos_safety_violations (
		id BIGSERIAL PRIMARY KEY,
		experiment_id UUID REFERENCES chaos_experiments(id),
		check_name VARCHAR(255) NOT NULL,
		violation_message TEXT NOT NULL,
		severity VARCHAR(50) NOT NULL, -- "low", "medium", "high", "critical"
		detected_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
		auto_rollback BOOLEAN DEFAULT false
	);
	`

	_, err := ce.db.Exec(schema)
	return err
}

// registerDefaultSafetyChecks adds standard safety validations
func (ce *ChaosEngine) registerDefaultSafetyChecks() {
	// Check 1: Production hours restriction
	ce.safetyChecks = append(ce.safetyChecks, func(ctx context.Context, exp *ChaosExperiment) (bool, string) {
		if !ce.config.ProductionEnabled {
			return false, "chaos engineering disabled in production"
		}

		hour := time.Now().Hour()
		safeHour := false
		for _, h := range ce.config.SafeHours {
			if h == hour {
				safeHour = true
				break
			}
		}

		if !safeHour {
			return false, fmt.Sprintf("current hour %d not in safe hours %v", hour, ce.config.SafeHours)
		}

		return true, ""
	})

	// Check 2: Concurrent experiment limit
	ce.safetyChecks = append(ce.safetyChecks, func(ctx context.Context, exp *ChaosExperiment) (bool, string) {
		ce.mu.RLock()
		runningCount := 0
		for _, e := range ce.experiments {
			if e.Status == "running" {
				runningCount++
			}
		}
		ce.mu.RUnlock()

		if runningCount >= ce.config.MaxConcurrentExperiments {
			return false, fmt.Sprintf("max concurrent experiments reached: %d", runningCount)
		}

		return true, ""
	})

	// Check 3: System health check
	ce.safetyChecks = append(ce.safetyChecks, func(ctx context.Context, exp *ChaosExperiment) (bool, string) {
		// Query current error rate from metrics
		var errorRate float64
		err := ce.db.QueryRowContext(ctx, `
			SELECT COALESCE(
				(SELECT COUNT(*) FROM request_log WHERE timestamp > NOW() - INTERVAL '5 minutes' AND status >= 500)::float /
				NULLIF((SELECT COUNT(*) FROM request_log WHERE timestamp > NOW() - INTERVAL '5 minutes'), 0),
				0
			) AS error_rate
		`).Scan(&errorRate)

		if err != nil && err != sql.ErrNoRows {
			return false, fmt.Sprintf("failed to check system health: %v", err)
		}

		// Don't inject chaos if already experiencing high error rate
		if errorRate > 0.05 { // 5% error rate threshold
			return false, fmt.Sprintf("system already unhealthy: %.2f%% error rate", errorRate*100)
		}

		return true, ""
	})

	// Check 4: Impact threshold check
	ce.safetyChecks = append(ce.safetyChecks, func(ctx context.Context, exp *ChaosExperiment) (bool, string) {
		// For certain experiment types, validate impact won't exceed threshold
		if exp.Type == ExperimentServiceFailure {
			impactPercent, ok := exp.Parameters["impact_percentage"].(float64)
			if ok && impactPercent > ce.config.MaxImpactThreshold {
				return false, fmt.Sprintf("impact %.1f%% exceeds threshold %.1f%%", impactPercent, ce.config.MaxImpactThreshold)
			}
		}

		return true, ""
	})
}

// ScheduleExperiment schedules a new chaos experiment
func (ce *ChaosEngine) ScheduleExperiment(ctx context.Context, experiment *ChaosExperiment) error {
	if !ce.enabled {
		return fmt.Errorf("chaos engine is disabled")
	}

	// Run safety checks
	for _, check := range ce.safetyChecks {
		safe, reason := check(ctx, experiment)
		if !safe {
			log.Printf("[chaos] Safety check failed for experiment %s: %s", experiment.Name, reason)

			// Log safety violation
			ce.logSafetyViolation(ctx, experiment, "pre-execution", reason, "high")

			return fmt.Errorf("safety check failed: %s", reason)
		}
	}

	// Generate ID if not set
	if experiment.ID == "" {
		experiment.ID = fmt.Sprintf("chaos-%d", time.Now().UnixNano())
	}

	experiment.Status = "scheduled"

	// Persist to database
	parametersJSON, _ := json.Marshal(experiment.Parameters)

	_, err := ce.db.ExecContext(ctx, `
		INSERT INTO chaos_experiments (
			id, name, experiment_type, target_service, parameters, 
			status, duration_seconds
		) VALUES ($1, $2, $3, $4, $5, $6, $7)
	`, experiment.ID, experiment.Name, experiment.Type, experiment.TargetService,
		parametersJSON, experiment.Status, int(experiment.Duration.Seconds()))

	if err != nil {
		return fmt.Errorf("failed to schedule experiment: %w", err)
	}

	// Store in memory
	ce.mu.Lock()
	ce.experiments[experiment.ID] = experiment
	ce.mu.Unlock()

	log.Printf("[chaos] Scheduled experiment: %s (%s) on %s for %v",
		experiment.Name, experiment.Type, experiment.TargetService, experiment.Duration)

	return nil
}

// ExecuteExperiment runs a scheduled experiment
func (ce *ChaosEngine) ExecuteExperiment(ctx context.Context, experimentID string) error {
	ce.mu.RLock()
	experiment, ok := ce.experiments[experimentID]
	ce.mu.RUnlock()

	if !ok {
		return fmt.Errorf("experiment not found: %s", experimentID)
	}

	// Final safety check before execution
	for _, check := range ce.safetyChecks {
		safe, reason := check(ctx, experiment)
		if !safe {
			return ce.abortExperiment(ctx, experiment, reason)
		}
	}

	// Update status
	experiment.Status = "running"
	experiment.StartTime = time.Now()

	_, err := ce.db.ExecContext(ctx, `
		UPDATE chaos_experiments 
		SET status = 'running', start_time = $1, updated_at = NOW()
		WHERE id = $2
	`, experiment.StartTime, experiment.ID)

	if err != nil {
		return fmt.Errorf("failed to update experiment status: %w", err)
	}

	log.Printf("[chaos] Executing experiment: %s (%s)", experiment.Name, experiment.Type)

	// Execute experiment based on type
	go ce.runExperiment(ctx, experiment)

	return nil
}

// runExperiment executes the actual chaos injection
func (ce *ChaosEngine) runExperiment(ctx context.Context, experiment *ChaosExperiment) {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("[chaos] Experiment %s panicked: %v", experiment.ID, r)
			ce.abortExperiment(ctx, experiment, fmt.Sprintf("panic: %v", r))
		}
	}()

	// Start metrics collection
	ce.metricsCollector.StartCollection(experiment.ID)

	// Create timeout context
	timeoutCtx, cancel := context.WithTimeout(ctx, experiment.Duration)
	defer cancel()

	// Execute chaos based on type
	var err error
	switch experiment.Type {
	case ExperimentHighLatency:
		err = ce.injectLatency(timeoutCtx, experiment)
	case ExperimentServiceFailure:
		err = ce.injectServiceFailure(timeoutCtx, experiment)
	case ExperimentResourceExhaustion:
		err = ce.injectResourceExhaustion(timeoutCtx, experiment)
	case ExperimentDBSlowQuery:
		err = ce.injectDBSlowdown(timeoutCtx, experiment)
	case ExperimentCacheFailure:
		err = ce.injectCacheFailure(timeoutCtx, experiment)
	default:
		err = fmt.Errorf("unsupported experiment type: %s", experiment.Type)
	}

	// Wait for experiment duration
	<-timeoutCtx.Done()

	// Stop metrics collection and compute results
	experiment.EndTime = time.Now()
	experiment.Results = ce.metricsCollector.ComputeResults(experiment.ID)

	// Check if safety threshold was violated during execution
	if experiment.Results.ErrorRate > ce.config.MaxImpactThreshold/100 {
		experiment.SafetyViolation = true
		ce.logSafetyViolation(ctx, experiment, "during-execution",
			fmt.Sprintf("error rate %.2f%% exceeded threshold %.2f%%",
				experiment.Results.ErrorRate*100, ce.config.MaxImpactThreshold), "critical")

		if ce.config.AutoRollbackEnabled {
			ce.rollbackExperiment(ctx, experiment)
		}
	}

	// Mark as completed or failed
	if err != nil {
		experiment.Status = "failed"
		log.Printf("[chaos] Experiment %s failed: %v", experiment.ID, err)
	} else {
		experiment.Status = "completed"
		log.Printf("[chaos] Experiment %s completed successfully", experiment.ID)
	}

	// Persist results
	ce.persistExperimentResults(ctx, experiment)
}

// injectLatency adds artificial latency to requests
func (ce *ChaosEngine) injectLatency(ctx context.Context, experiment *ChaosExperiment) error {
	latencyMS, ok := experiment.Parameters["latency_ms"].(float64)
	if !ok {
		return fmt.Errorf("latency_ms parameter required")
	}

	impactPercent, _ := experiment.Parameters["impact_percentage"].(float64)
	if impactPercent == 0 {
		impactPercent = 100 // Default: affect all requests
	}

	log.Printf("[chaos] Injecting %vms latency to %.1f%% of requests on %s",
		latencyMS, impactPercent, experiment.TargetService)

	// In real implementation, this would:
	// 1. Add middleware to inject delays
	// 2. Use traffic routing to affect only certain percentage
	// 3. Monitor impact and auto-rollback if needed

	// Simulate latency injection
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			log.Printf("[chaos] Latency injection stopped for %s", experiment.ID)
			return nil
		case <-ticker.C:
			// Record metrics
			ce.metricsCollector.RecordMetric(experiment.ID, "latency_injected", latencyMS)
		}
	}
}

// injectServiceFailure simulates service failures
func (ce *ChaosEngine) injectServiceFailure(ctx context.Context, experiment *ChaosExperiment) error {
	impactPercent, ok := experiment.Parameters["impact_percentage"].(float64)
	if !ok {
		impactPercent = 10 // Default: 10% failure rate
	}

	log.Printf("[chaos] Injecting %.1f%% service failures on %s", impactPercent, experiment.TargetService)

	// Simulate service failure
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return nil
		case <-ticker.C:
			// Simulate failure injection
			if rand.Float64()*100 < impactPercent {
				ce.metricsCollector.RecordError(experiment.ID)
			}
			ce.metricsCollector.RecordRequest(experiment.ID)
		}
	}
}

// injectResourceExhaustion simulates resource exhaustion
func (ce *ChaosEngine) injectResourceExhaustion(ctx context.Context, experiment *ChaosExperiment) error {
	resourceType, ok := experiment.Parameters["resource_type"].(string)
	if !ok {
		resourceType = "memory"
	}

	log.Printf("[chaos] Injecting %s exhaustion on %s", resourceType, experiment.TargetService)

	// In real implementation, this would:
	// 1. Allocate large amounts of memory/CPU
	// 2. Monitor system impact
	// 3. Release resources after experiment

	<-ctx.Done()
	return nil
}

// injectDBSlowdown simulates slow database queries
func (ce *ChaosEngine) injectDBSlowdown(ctx context.Context, experiment *ChaosExperiment) error {
	slowdownFactor, ok := experiment.Parameters["slowdown_factor"].(float64)
	if !ok {
		slowdownFactor = 2.0 // Default: 2x slower
	}

	log.Printf("[chaos] Injecting DB slowdown (factor: %.1fx) on %s", slowdownFactor, experiment.TargetService)

	// Simulate DB slowdown
	<-ctx.Done()
	return nil
}

// injectCacheFailure simulates cache failures
func (ce *ChaosEngine) injectCacheFailure(ctx context.Context, experiment *ChaosExperiment) error {
	log.Printf("[chaos] Injecting cache failures on %s", experiment.TargetService)

	// Simulate cache miss rate increase
	<-ctx.Done()
	return nil
}

// abortExperiment aborts an experiment due to safety violation
func (ce *ChaosEngine) abortExperiment(ctx context.Context, experiment *ChaosExperiment, reason string) error {
	experiment.Status = "aborted"
	experiment.RollbackReason = reason
	experiment.SafetyViolation = true

	_, err := ce.db.ExecContext(ctx, `
		UPDATE chaos_experiments 
		SET status = 'aborted', rollback_reason = $1, safety_violation = true, updated_at = NOW()
		WHERE id = $2
	`, reason, experiment.ID)

	log.Printf("[chaos] Aborted experiment %s: %s", experiment.ID, reason)
	return err
}

// rollbackExperiment rolls back an experiment
func (ce *ChaosEngine) rollbackExperiment(ctx context.Context, experiment *ChaosExperiment) error {
	log.Printf("[chaos] Rolling back experiment %s", experiment.ID)

	// Stop chaos injection
	// Restore normal operation
	// Wait for recovery

	experiment.Status = "rolled_back"

	_, err := ce.db.ExecContext(ctx, `
		UPDATE chaos_experiments 
		SET status = 'rolled_back', updated_at = NOW()
		WHERE id = $1
	`, experiment.ID)

	return err
}

// logSafetyViolation logs safety violations
func (ce *ChaosEngine) logSafetyViolation(ctx context.Context, experiment *ChaosExperiment,
	checkName, message, severity string) {

	_, err := ce.db.ExecContext(ctx, `
		INSERT INTO chaos_safety_violations (
			experiment_id, check_name, violation_message, severity, auto_rollback
		) VALUES ($1, $2, $3, $4, $5)
	`, experiment.ID, checkName, message, severity, ce.config.AutoRollbackEnabled)

	if err != nil {
		log.Printf("[chaos] Failed to log safety violation: %v", err)
	}
}

// persistExperimentResults saves experiment results
func (ce *ChaosEngine) persistExperimentResults(ctx context.Context, experiment *ChaosExperiment) error {
	resultsJSON, _ := json.Marshal(experiment.Results)

	_, err := ce.db.ExecContext(ctx, `
		UPDATE chaos_experiments 
		SET status = $1, end_time = $2, results = $3, 
		    safety_violation = $4, updated_at = NOW()
		WHERE id = $5
	`, experiment.Status, experiment.EndTime, resultsJSON,
		experiment.SafetyViolation, experiment.ID)

	return err
}

// scheduleExperiments continuously schedules experiments
func (ce *ChaosEngine) scheduleExperiments() {
	ticker := time.NewTicker(ce.config.ExperimentInterval)
	defer ticker.Stop()

	for range ticker.C {
		if !ce.enabled {
			continue
		}

		// Create random experiment
		experiment := ce.generateRandomExperiment()

		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		if err := ce.ScheduleExperiment(ctx, experiment); err != nil {
			log.Printf("[chaos] Failed to schedule experiment: %v", err)
		} else {
			// Execute immediately
			if err := ce.ExecuteExperiment(ctx, experiment.ID); err != nil {
				log.Printf("[chaos] Failed to execute experiment: %v", err)
			}
		}
		cancel()
	}
}

// generateRandomExperiment creates a random chaos experiment
func (ce *ChaosEngine) generateRandomExperiment() *ChaosExperiment {
	experimentTypes := []ExperimentType{
		ExperimentHighLatency,
		ExperimentServiceFailure,
		ExperimentDBSlowQuery,
		ExperimentCacheFailure,
	}

	expType := experimentTypes[rand.Intn(len(experimentTypes))]

	var parameters map[string]interface{}
	switch expType {
	case ExperimentHighLatency:
		parameters = map[string]interface{}{
			"latency_ms":        float64(100 + rand.Intn(400)),
			"impact_percentage": float64(10 + rand.Intn(20)),
		}
	case ExperimentServiceFailure:
		parameters = map[string]interface{}{
			"impact_percentage": float64(5 + rand.Intn(15)),
		}
	case ExperimentDBSlowQuery:
		parameters = map[string]interface{}{
			"slowdown_factor": float64(2 + rand.Intn(3)),
		}
	default:
		parameters = map[string]interface{}{}
	}

	targetService := ce.config.TargetServices[rand.Intn(len(ce.config.TargetServices))]

	return &ChaosExperiment{
		Name:          fmt.Sprintf("auto_%s_%s", expType, time.Now().Format("20060102_150405")),
		Type:          expType,
		TargetService: targetService,
		Parameters:    parameters,
		Duration:      time.Duration(30+rand.Intn(90)) * time.Second,
	}
}

// NewChaosMetricsCollector creates a new metrics collector
func NewChaosMetricsCollector() *ChaosMetricsCollector {
	return &ChaosMetricsCollector{
		requestCounts: make(map[string]int64),
		errorCounts:   make(map[string]int64),
		latencies:     make(map[string][]time.Duration),
	}
}

// StartCollection begins metric collection for an experiment
func (cmc *ChaosMetricsCollector) StartCollection(experimentID string) {
	cmc.mu.Lock()
	defer cmc.mu.Unlock()

	cmc.requestCounts[experimentID] = 0
	cmc.errorCounts[experimentID] = 0
	cmc.latencies[experimentID] = make([]time.Duration, 0)
}

// RecordRequest records a request
func (cmc *ChaosMetricsCollector) RecordRequest(experimentID string) {
	cmc.mu.Lock()
	defer cmc.mu.Unlock()

	cmc.requestCounts[experimentID]++
}

// RecordError records an error
func (cmc *ChaosMetricsCollector) RecordError(experimentID string) {
	cmc.mu.Lock()
	defer cmc.mu.Unlock()

	cmc.errorCounts[experimentID]++
}

// RecordLatency records latency measurement
func (cmc *ChaosMetricsCollector) RecordLatency(experimentID string, latency time.Duration) {
	cmc.mu.Lock()
	defer cmc.mu.Unlock()

	cmc.latencies[experimentID] = append(cmc.latencies[experimentID], latency)
}

// RecordMetric records a custom metric
func (cmc *ChaosMetricsCollector) RecordMetric(experimentID, name string, value float64) {
	// Implementation for custom metrics
}

// ComputeResults computes final experiment results
func (cmc *ChaosMetricsCollector) ComputeResults(experimentID string) ExperimentResults {
	cmc.mu.RLock()
	defer cmc.mu.RUnlock()

	totalRequests := cmc.requestCounts[experimentID]
	failedRequests := cmc.errorCounts[experimentID]
	latencies := cmc.latencies[experimentID]

	var avgLatency, p95, p99 time.Duration
	if len(latencies) > 0 {
		// Calculate average
		var total time.Duration
		for _, l := range latencies {
			total += l
		}
		avgLatency = total / time.Duration(len(latencies))

		// Calculate percentiles (simple approximation)
		p95Idx := int(float64(len(latencies)) * 0.95)
		p99Idx := int(float64(len(latencies)) * 0.99)
		if p95Idx < len(latencies) {
			p95 = latencies[p95Idx]
		}
		if p99Idx < len(latencies) {
			p99 = latencies[p99Idx]
		}
	}

	errorRate := float64(0)
	if totalRequests > 0 {
		errorRate = float64(failedRequests) / float64(totalRequests)
	}

	return ExperimentResults{
		TotalRequests:    totalRequests,
		FailedRequests:   failedRequests,
		AverageLatency:   avgLatency,
		P95Latency:       p95,
		P99Latency:       p99,
		ErrorRate:        errorRate,
		ImpactPercentage: errorRate * 100,
		Observations:     []string{},
	}
}

// GetExperimentHistory returns historical experiments
func (ce *ChaosEngine) GetExperimentHistory(ctx context.Context, limit int) ([]*ChaosExperiment, error) {
	rows, err := ce.db.QueryContext(ctx, `
		SELECT id, name, experiment_type, target_service, parameters,
		       status, start_time, end_time, duration_seconds, results,
		       safety_violation, rollback_reason
		FROM chaos_experiments
		ORDER BY created_at DESC
		LIMIT $1
	`, limit)

	if err != nil {
		return nil, err
	}
	defer rows.Close()

	experiments := make([]*ChaosExperiment, 0)
	for rows.Next() {
		var exp ChaosExperiment
		var parametersJSON, resultsJSON []byte
		var durationSeconds int
		var startTime, endTime sql.NullTime
		var rollbackReason sql.NullString

		err := rows.Scan(
			&exp.ID, &exp.Name, &exp.Type, &exp.TargetService, &parametersJSON,
			&exp.Status, &startTime, &endTime, &durationSeconds, &resultsJSON,
			&exp.SafetyViolation, &rollbackReason,
		)

		if err != nil {
			continue
		}

		json.Unmarshal(parametersJSON, &exp.Parameters)
		json.Unmarshal(resultsJSON, &exp.Results)

		if startTime.Valid {
			exp.StartTime = startTime.Time
		}
		if endTime.Valid {
			exp.EndTime = endTime.Time
		}
		if rollbackReason.Valid {
			exp.RollbackReason = rollbackReason.String
		}

		exp.Duration = time.Duration(durationSeconds) * time.Second

		experiments = append(experiments, &exp)
	}

	return experiments, nil
}

// Enable enables chaos engineering
func (ce *ChaosEngine) Enable() {
	ce.mu.Lock()
	defer ce.mu.Unlock()
	ce.enabled = true
	log.Printf("[chaos] Chaos engineering enabled")
}

// Disable disables chaos engineering
func (ce *ChaosEngine) Disable() {
	ce.mu.Lock()
	defer ce.mu.Unlock()
	ce.enabled = false
	log.Printf("[chaos] Chaos engineering disabled")
}

// Close closes the chaos engine
func (ce *ChaosEngine) Close() error {
	ce.Disable()
	return nil
}
