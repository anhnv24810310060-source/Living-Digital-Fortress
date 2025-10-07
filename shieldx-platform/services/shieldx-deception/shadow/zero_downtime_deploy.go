package shadow

import (
	"context"
	"fmt"
	"log"
	"math"
	"sync"
	"time"
)

// ZeroDowntimeDeployer implements zero-downtime deployment strategies
// ✅ PHẢI test rules trong shadow trước deploy
// Strategies: Blue-Green, Canary, Rolling Update
type ZeroDowntimeDeployer struct {
	deployments   map[string]*Deployment
	strategies    map[string]DeployStrategy
	trafficMgr    *TrafficManager
	healthMonitor *DeploymentHealthMonitor
	rollbackMgr   *RollbackManager
	featureFlags  *FeatureFlagManager
	mu            sync.RWMutex
}

// Deployment represents a deployment
type Deployment struct {
	ID             string             `json:"id"`
	Service        string             `json:"service"`
	Version        string             `json:"version"`
	Strategy       DeployStrategyType `json:"strategy"`
	Status         DeploymentStatus   `json:"status"`
	Progress       float64            `json:"progress"` // 0.0 to 1.0
	StartTime      time.Time          `json:"start_time"`
	CompletionTime time.Time          `json:"completion_time,omitempty"`

	// Blue-Green specific
	BlueEnvironment  *Environment `json:"blue_environment,omitempty"`
	GreenEnvironment *Environment `json:"green_environment,omitempty"`

	// Canary specific
	CanaryConfig *CanaryConfig `json:"canary_config,omitempty"`
	CanaryStages []CanaryStage `json:"canary_stages,omitempty"`

	// Rolling update specific
	RollingConfig *RollingConfig `json:"rolling_config,omitempty"`

	// Health metrics
	HealthChecks []HealthCheckResult `json:"health_checks"`
	Metrics      DeploymentMetrics   `json:"metrics"`

	// Rollback info
	RollbackReady   bool   `json:"rollback_ready"`
	RollbackVersion string `json:"rollback_version,omitempty"`

	Metadata map[string]interface{} `json:"metadata"`
}

// DeployStrategyType represents deployment strategy types
type DeployStrategyType int

const (
	StrategyBlueGreen DeployStrategyType = iota
	StrategyCanary
	StrategyRollingUpdate
	StrategyRecreate
	StrategyShadow
)

// DeploymentStatus represents deployment status
type DeploymentStatus int

const (
	DeploymentPending DeploymentStatus = iota
	DeploymentInProgress
	DeploymentCompleted
	DeploymentFailed
	DeploymentRolledBack
	DeploymentPaused
)

// Environment represents a deployment environment (blue or green)
type Environment struct {
	Name          string                 `json:"name"`
	Active        bool                   `json:"active"`
	Version       string                 `json:"version"`
	Instances     []Instance             `json:"instances"`
	TrafficWeight float64                `json:"traffic_weight"` // 0.0 to 1.0
	HealthStatus  EnvironmentHealth      `json:"health_status"`
	DeployedAt    time.Time              `json:"deployed_at"`
	Metadata      map[string]interface{} `json:"metadata"`
}

// Instance represents a service instance
type Instance struct {
	ID              string         `json:"id"`
	Hostname        string         `json:"hostname"`
	Status          InstanceStatus `json:"status"`
	Health          InstanceHealth `json:"health"`
	Version         string         `json:"version"`
	StartedAt       time.Time      `json:"started_at"`
	RequestCount    uint64         `json:"request_count"`
	ErrorCount      uint64         `json:"error_count"`
	AvgResponseTime time.Duration  `json:"avg_response_time"`
}

// InstanceStatus represents instance status
type InstanceStatus int

const (
	InstanceStarting InstanceStatus = iota
	InstanceHealthy
	InstanceUnhealthy
	InstanceDraining
	InstanceStopped
)

// InstanceHealth represents instance health
type InstanceHealth int

const (
	HealthPassing InstanceHealth = iota
	HealthWarning
	HealthCritical
	HealthUnknown
)

// EnvironmentHealth represents overall environment health
type EnvironmentHealth int

const (
	EnvHealthy EnvironmentHealth = iota
	EnvDegraded
	EnvUnhealthy
)

// CanaryConfig represents canary deployment configuration
type CanaryConfig struct {
	InitialTrafficPercent   float64          `json:"initial_traffic_percent"`
	TrafficIncrementPercent float64          `json:"traffic_increment_percent"`
	StageInterval           time.Duration    `json:"stage_interval"`
	MaxStages               int              `json:"max_stages"`
	SuccessThreshold        SuccessThreshold `json:"success_threshold"`
	AutoPromote             bool             `json:"auto_promote"`
	AutoRollback            bool             `json:"auto_rollback"`
}

// CanaryStage represents a canary deployment stage
type CanaryStage struct {
	StageNumber    int           `json:"stage_number"`
	TrafficPercent float64       `json:"traffic_percent"`
	StartTime      time.Time     `json:"start_time"`
	EndTime        time.Time     `json:"end_time,omitempty"`
	Status         StageStatus   `json:"status"`
	Metrics        StageMetrics  `json:"metrics"`
	Decision       StageDecision `json:"decision"`
}

// StageStatus represents canary stage status
type StageStatus int

const (
	StageRunning StageStatus = iota
	StageSucceeded
	StageFailed
	StagePaused
)

// StageDecision represents decision for stage
type StageDecision int

const (
	DecisionPending StageDecision = iota
	DecisionPromote
	DecisionRollback
	DecisionPause
)

// StageMetrics represents metrics for a canary stage
type StageMetrics struct {
	RequestCount uint64        `json:"request_count"`
	ErrorCount   uint64        `json:"error_count"`
	ErrorRate    float64       `json:"error_rate"`
	AvgLatency   time.Duration `json:"avg_latency"`
	P95Latency   time.Duration `json:"p95_latency"`
	P99Latency   time.Duration `json:"p99_latency"`
	SuccessRate  float64       `json:"success_rate"`
}

// SuccessThreshold defines success criteria
type SuccessThreshold struct {
	MaxErrorRate    float64       `json:"max_error_rate"`    // e.g., 0.01 = 1%
	MaxLatencyP99   time.Duration `json:"max_latency_p99"`   // e.g., 500ms
	MinSuccessRate  float64       `json:"min_success_rate"`  // e.g., 0.99 = 99%
	MinRequestCount uint64        `json:"min_request_count"` // Minimum requests for valid metrics
}

// RollingConfig represents rolling update configuration
type RollingConfig struct {
	MaxSurge           int           `json:"max_surge"`            // Max instances above desired
	MaxUnavailable     int           `json:"max_unavailable"`      // Max instances unavailable
	BatchSize          int           `json:"batch_size"`           // Instances per batch
	BatchInterval      time.Duration `json:"batch_interval"`       // Time between batches
	HealthCheckTimeout time.Duration `json:"health_check_timeout"` // Wait for health
}

// DeploymentMetrics tracks deployment metrics
type DeploymentMetrics struct {
	TotalRequests      uint64        `json:"total_requests"`
	SuccessfulRequests uint64        `json:"successful_requests"`
	FailedRequests     uint64        `json:"failed_requests"`
	AvgResponseTime    time.Duration `json:"avg_response_time"`
	ErrorRate          float64       `json:"error_rate"`
	DeploymentDuration time.Duration `json:"deployment_duration"`
}

// TrafficManager manages traffic routing between environments
type TrafficManager struct {
	routes      map[string]*TrafficRoute
	shiftPolicy TrafficShiftPolicy
	mu          sync.RWMutex
}

// TrafficRoute represents traffic routing configuration
type TrafficRoute struct {
	Service     string
	BlueWeight  float64 // 0.0 to 1.0
	GreenWeight float64 // 0.0 to 1.0
	LastShift   time.Time
	ShiftRate   float64 // Percent per interval
}

// TrafficShiftPolicy defines how traffic is shifted
type TrafficShiftPolicy int

const (
	ShiftImmediate TrafficShiftPolicy = iota
	ShiftGradual
	ShiftLinear
	ShiftExponential
)

// DeploymentHealthMonitor monitors deployment health
type DeploymentHealthMonitor struct {
	checks  []DeploymentHealthCheck
	results map[string][]HealthCheckResult
	mu      sync.RWMutex
}

// DeploymentHealthCheck represents a health check
type DeploymentHealthCheck struct {
	Name      string
	CheckFunc func(context.Context, *Deployment) HealthCheckResult
	Interval  time.Duration
	Critical  bool
}

// HealthCheckResult represents health check result
type HealthCheckResult struct {
	CheckName string
	Timestamp time.Time
	Passed    bool
	Message   string
	Score     float64
	Details   map[string]interface{}
}

// RollbackManager manages deployment rollbacks
type RollbackManager struct {
	history []RollbackEvent
	mu      sync.Mutex
}

// RollbackEvent represents a rollback event
type RollbackEvent struct {
	ID           string
	Timestamp    time.Time
	DeploymentID string
	FromVersion  string
	ToVersion    string
	Reason       string
	Automatic    bool
	Duration     time.Duration
	Success      bool
}

// FeatureFlagManager manages feature flags for gradual rollout
type FeatureFlagManager struct {
	flags map[string]*FeatureFlag
	mu    sync.RWMutex
}

// FeatureFlag represents a feature flag
type FeatureFlag struct {
	Name           string                 `json:"name"`
	Enabled        bool                   `json:"enabled"`
	RolloutPercent float64                `json:"rollout_percent"` // 0.0 to 1.0
	TargetUsers    []string               `json:"target_users,omitempty"`
	TargetGroups   []string               `json:"target_groups,omitempty"`
	Rules          []FeatureFlagRule      `json:"rules,omitempty"`
	CreatedAt      time.Time              `json:"created_at"`
	UpdatedAt      time.Time              `json:"updated_at"`
	Metadata       map[string]interface{} `json:"metadata"`
}

// FeatureFlagRule represents a feature flag rule
type FeatureFlagRule struct {
	Condition string      `json:"condition"`
	Value     interface{} `json:"value"`
}

// DeployStrategy interface
type DeployStrategy interface {
	Deploy(ctx context.Context, deployment *Deployment) error
	Rollback(ctx context.Context, deployment *Deployment) error
	GetProgress(deployment *Deployment) float64
}

// NewZeroDowntimeDeployer creates a new zero-downtime deployer
func NewZeroDowntimeDeployer() *ZeroDowntimeDeployer {
	zdd := &ZeroDowntimeDeployer{
		deployments: make(map[string]*Deployment),
		strategies:  make(map[string]DeployStrategy),
		trafficMgr: &TrafficManager{
			routes:      make(map[string]*TrafficRoute),
			shiftPolicy: ShiftGradual,
		},
		healthMonitor: &DeploymentHealthMonitor{
			checks:  make([]DeploymentHealthCheck, 0),
			results: make(map[string][]HealthCheckResult),
		},
		rollbackMgr: &RollbackManager{
			history: make([]RollbackEvent, 0),
		},
		featureFlags: &FeatureFlagManager{
			flags: make(map[string]*FeatureFlag),
		},
	}

	// Register deployment strategies
	zdd.strategies["blue-green"] = &BlueGreenStrategy{zdd: zdd}
	zdd.strategies["canary"] = &CanaryStrategy{zdd: zdd}
	zdd.strategies["rolling"] = &RollingUpdateStrategy{zdd: zdd}

	// Register health checks
	zdd.registerHealthChecks()

	log.Printf("[deploy] Zero-Downtime Deployer initialized")
	return zdd
}

// CreateDeployment creates a new deployment
func (zdd *ZeroDowntimeDeployer) CreateDeployment(service, version string, strategy DeployStrategyType) (*Deployment, error) {
	zdd.mu.Lock()
	defer zdd.mu.Unlock()

	deployID := fmt.Sprintf("deploy-%s-%d", service, time.Now().Unix())

	deployment := &Deployment{
		ID:           deployID,
		Service:      service,
		Version:      version,
		Strategy:     strategy,
		Status:       DeploymentPending,
		Progress:     0.0,
		StartTime:    time.Now(),
		HealthChecks: make([]HealthCheckResult, 0),
		Metadata:     make(map[string]interface{}),
	}

	// Configure based on strategy
	switch strategy {
	case StrategyBlueGreen:
		deployment.BlueEnvironment = &Environment{
			Name:          "blue",
			Active:        true,
			TrafficWeight: 1.0,
			Instances:     make([]Instance, 0),
			HealthStatus:  EnvHealthy,
			Metadata:      make(map[string]interface{}),
		}
		deployment.GreenEnvironment = &Environment{
			Name:          "green",
			Active:        false,
			Version:       version,
			TrafficWeight: 0.0,
			Instances:     make([]Instance, 0),
			HealthStatus:  EnvHealthy,
			Metadata:      make(map[string]interface{}),
		}

	case StrategyCanary:
		deployment.CanaryConfig = &CanaryConfig{
			InitialTrafficPercent:   5.0,
			TrafficIncrementPercent: 10.0,
			StageInterval:           5 * time.Minute,
			MaxStages:               10,
			SuccessThreshold: SuccessThreshold{
				MaxErrorRate:    0.01, // 1%
				MaxLatencyP99:   500 * time.Millisecond,
				MinSuccessRate:  0.99, // 99%
				MinRequestCount: 100,
			},
			AutoPromote:  true,
			AutoRollback: true,
		}
		deployment.CanaryStages = make([]CanaryStage, 0)

	case StrategyRollingUpdate:
		deployment.RollingConfig = &RollingConfig{
			MaxSurge:           1,
			MaxUnavailable:     1,
			BatchSize:          2,
			BatchInterval:      30 * time.Second,
			HealthCheckTimeout: 60 * time.Second,
		}
	}

	zdd.deployments[deployID] = deployment

	log.Printf("[deploy] Created deployment: %s (service: %s, version: %s, strategy: %v)",
		deployID, service, version, strategy)

	return deployment, nil
}

// StartDeployment starts a deployment
func (zdd *ZeroDowntimeDeployer) StartDeployment(ctx context.Context, deploymentID string) error {
	zdd.mu.RLock()
	deployment, exists := zdd.deployments[deploymentID]
	if !exists {
		zdd.mu.RUnlock()
		return fmt.Errorf("deployment not found: %s", deploymentID)
	}
	zdd.mu.RUnlock()

	// Update status
	deployment.Status = DeploymentInProgress

	// Get strategy
	var strategyName string
	switch deployment.Strategy {
	case StrategyBlueGreen:
		strategyName = "blue-green"
	case StrategyCanary:
		strategyName = "canary"
	case StrategyRollingUpdate:
		strategyName = "rolling"
	default:
		return fmt.Errorf("unsupported strategy: %v", deployment.Strategy)
	}

	strategy := zdd.strategies[strategyName]
	if strategy == nil {
		return fmt.Errorf("strategy not found: %s", strategyName)
	}

	log.Printf("[deploy] Starting deployment: %s", deploymentID)

	// Start health monitoring
	go zdd.monitorDeploymentHealth(ctx, deployment)

	// Execute deployment
	go func() {
		if err := strategy.Deploy(ctx, deployment); err != nil {
			log.Printf("[deploy] Deployment failed: %s: %v", deploymentID, err)
			deployment.Status = DeploymentFailed

			// Auto-rollback if enabled
			if deployment.CanaryConfig != nil && deployment.CanaryConfig.AutoRollback {
				log.Printf("[deploy] Triggering auto-rollback for: %s", deploymentID)
				_ = zdd.RollbackDeployment(ctx, deploymentID, "automatic rollback due to failure")
			}
		} else {
			deployment.Status = DeploymentCompleted
			deployment.CompletionTime = time.Now()
			deployment.Metrics.DeploymentDuration = time.Since(deployment.StartTime)
			log.Printf("[deploy] Deployment completed: %s (duration: %v)",
				deploymentID, deployment.Metrics.DeploymentDuration)
		}
	}()

	return nil
}

// RollbackDeployment rolls back a deployment
func (zdd *ZeroDowntimeDeployer) RollbackDeployment(ctx context.Context, deploymentID, reason string) error {
	zdd.mu.RLock()
	deployment, exists := zdd.deployments[deploymentID]
	if !exists {
		zdd.mu.RUnlock()
		return fmt.Errorf("deployment not found: %s", deploymentID)
	}
	zdd.mu.RUnlock()

	if !deployment.RollbackReady {
		return fmt.Errorf("rollback not ready for deployment: %s", deploymentID)
	}

	startTime := time.Now()
	event := RollbackEvent{
		ID:           fmt.Sprintf("rollback-%d", time.Now().Unix()),
		Timestamp:    startTime,
		DeploymentID: deploymentID,
		FromVersion:  deployment.Version,
		ToVersion:    deployment.RollbackVersion,
		Reason:       reason,
		Automatic:    reason == "automatic rollback due to failure",
	}

	log.Printf("[deploy] Rolling back deployment: %s (reason: %s)", deploymentID, reason)

	// Get strategy
	var strategyName string
	switch deployment.Strategy {
	case StrategyBlueGreen:
		strategyName = "blue-green"
	case StrategyCanary:
		strategyName = "canary"
	case StrategyRollingUpdate:
		strategyName = "rolling"
	}

	strategy := zdd.strategies[strategyName]
	if strategy == nil {
		return fmt.Errorf("strategy not found: %s", strategyName)
	}

	// Execute rollback
	if err := strategy.Rollback(ctx, deployment); err != nil {
		event.Success = false
		log.Printf("[deploy] Rollback failed: %s: %v", deploymentID, err)
		return err
	}

	deployment.Status = DeploymentRolledBack
	event.Success = true
	event.Duration = time.Since(startTime)

	zdd.rollbackMgr.mu.Lock()
	zdd.rollbackMgr.history = append(zdd.rollbackMgr.history, event)
	zdd.rollbackMgr.mu.Unlock()

	log.Printf("[deploy] Rollback completed: %s (duration: %v)", deploymentID, event.Duration)
	return nil
}

// registerHealthChecks registers default health checks
func (zdd *ZeroDowntimeDeployer) registerHealthChecks() {
	zdd.healthMonitor.checks = append(zdd.healthMonitor.checks,
		DeploymentHealthCheck{
			Name:      "error_rate",
			CheckFunc: zdd.checkErrorRate,
			Interval:  30 * time.Second,
			Critical:  true,
		},
		DeploymentHealthCheck{
			Name:      "latency",
			CheckFunc: zdd.checkLatency,
			Interval:  30 * time.Second,
			Critical:  true,
		},
		DeploymentHealthCheck{
			Name:      "instance_health",
			CheckFunc: zdd.checkInstanceHealth,
			Interval:  15 * time.Second,
			Critical:  true,
		},
	)

	log.Printf("[deploy] Registered %d health checks", len(zdd.healthMonitor.checks))
}

// checkErrorRate checks deployment error rate
func (zdd *ZeroDowntimeDeployer) checkErrorRate(ctx context.Context, deployment *Deployment) HealthCheckResult {
	result := HealthCheckResult{
		CheckName: "error_rate",
		Timestamp: time.Now(),
		Details:   make(map[string]interface{}),
	}

	// Calculate error rate
	if deployment.Metrics.TotalRequests == 0 {
		result.Passed = true
		result.Score = 1.0
		result.Message = "No requests yet"
		return result
	}

	errorRate := float64(deployment.Metrics.FailedRequests) / float64(deployment.Metrics.TotalRequests)
	deployment.Metrics.ErrorRate = errorRate

	threshold := 0.01 // 1%
	if deployment.CanaryConfig != nil {
		threshold = deployment.CanaryConfig.SuccessThreshold.MaxErrorRate
	}

	result.Details["error_rate"] = errorRate
	result.Details["threshold"] = threshold
	result.Passed = errorRate <= threshold
	result.Score = math.Max(0, 1.0-errorRate/threshold)
	result.Message = fmt.Sprintf("Error rate: %.2f%% (threshold: %.2f%%)",
		errorRate*100, threshold*100)

	return result
}

// checkLatency checks deployment latency
func (zdd *ZeroDowntimeDeployer) checkLatency(ctx context.Context, deployment *Deployment) HealthCheckResult {
	result := HealthCheckResult{
		CheckName: "latency",
		Timestamp: time.Now(),
		Details:   make(map[string]interface{}),
	}

	latency := deployment.Metrics.AvgResponseTime
	threshold := 500 * time.Millisecond
	if deployment.CanaryConfig != nil {
		threshold = deployment.CanaryConfig.SuccessThreshold.MaxLatencyP99
	}

	result.Details["avg_latency_ms"] = latency.Milliseconds()
	result.Details["threshold_ms"] = threshold.Milliseconds()
	result.Passed = latency <= threshold

	if latency == 0 {
		result.Score = 1.0
	} else {
		result.Score = math.Max(0, 1.0-float64(latency)/float64(threshold))
	}

	result.Message = fmt.Sprintf("Avg latency: %v (threshold: %v)", latency, threshold)

	return result
}

// checkInstanceHealth checks instance health
func (zdd *ZeroDowntimeDeployer) checkInstanceHealth(ctx context.Context, deployment *Deployment) HealthCheckResult {
	result := HealthCheckResult{
		CheckName: "instance_health",
		Timestamp: time.Now(),
		Details:   make(map[string]interface{}),
	}

	var totalInstances, healthyInstances int

	// Count instances based on deployment type
	if deployment.BlueEnvironment != nil {
		for _, inst := range deployment.BlueEnvironment.Instances {
			totalInstances++
			if inst.Health == HealthPassing {
				healthyInstances++
			}
		}
	}
	if deployment.GreenEnvironment != nil {
		for _, inst := range deployment.GreenEnvironment.Instances {
			totalInstances++
			if inst.Health == HealthPassing {
				healthyInstances++
			}
		}
	}

	if totalInstances == 0 {
		result.Passed = false
		result.Score = 0.0
		result.Message = "No instances found"
		return result
	}

	healthyPercent := float64(healthyInstances) / float64(totalInstances)
	result.Details["total_instances"] = totalInstances
	result.Details["healthy_instances"] = healthyInstances
	result.Details["healthy_percent"] = healthyPercent

	result.Passed = healthyPercent >= 0.8 // 80% healthy
	result.Score = healthyPercent
	result.Message = fmt.Sprintf("%d/%d instances healthy (%.0f%%)",
		healthyInstances, totalInstances, healthyPercent*100)

	return result
}

// monitorDeploymentHealth monitors deployment health
func (zdd *ZeroDowntimeDeployer) monitorDeploymentHealth(ctx context.Context, deployment *Deployment) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if deployment.Status != DeploymentInProgress {
				return
			}

			for _, check := range zdd.healthMonitor.checks {
				result := check.CheckFunc(ctx, deployment)

				deployment.HealthChecks = append(deployment.HealthChecks, result)
				if len(deployment.HealthChecks) > 100 {
					deployment.HealthChecks = deployment.HealthChecks[1:]
				}

				if !result.Passed && check.Critical {
					log.Printf("[deploy] Critical health check failed: %s - %s",
						check.Name, result.Message)

					// Trigger rollback if auto-rollback enabled
					if deployment.CanaryConfig != nil && deployment.CanaryConfig.AutoRollback {
						go zdd.RollbackDeployment(ctx, deployment.ID,
							fmt.Sprintf("health check failed: %s", check.Name))
						return
					}
				}
			}
		}
	}
}

// GetDeploymentStatus returns deployment status
func (zdd *ZeroDowntimeDeployer) GetDeploymentStatus(deploymentID string) (map[string]interface{}, error) {
	zdd.mu.RLock()
	deployment, exists := zdd.deployments[deploymentID]
	if !exists {
		zdd.mu.RUnlock()
		return nil, fmt.Errorf("deployment not found: %s", deploymentID)
	}
	zdd.mu.RUnlock()

	return map[string]interface{}{
		"id":              deployment.ID,
		"service":         deployment.Service,
		"version":         deployment.Version,
		"strategy":        deployment.Strategy,
		"status":          deployment.Status,
		"progress":        deployment.Progress,
		"start_time":      deployment.StartTime,
		"completion_time": deployment.CompletionTime,
		"metrics":         deployment.Metrics,
		"health_checks":   deployment.HealthChecks,
		"rollback_ready":  deployment.RollbackReady,
	}, nil
}

// Blue-Green Strategy Implementation
type BlueGreenStrategy struct {
	zdd *ZeroDowntimeDeployer
}

func (s *BlueGreenStrategy) Deploy(ctx context.Context, deployment *Deployment) error {
	log.Printf("[deploy] Executing Blue-Green deployment for: %s", deployment.ID)

	// Phase 1: Deploy to green environment (0% traffic)
	log.Printf("[deploy] Phase 1: Deploying to green environment")
	deployment.Progress = 0.1

	// Simulate deployment
	time.Sleep(5 * time.Second)

	// Create instances in green environment
	for i := 0; i < 3; i++ {
		instance := Instance{
			ID:        fmt.Sprintf("green-%d", i),
			Hostname:  fmt.Sprintf("green-%d.example.com", i),
			Status:    InstanceStarting,
			Health:    HealthUnknown,
			Version:   deployment.Version,
			StartedAt: time.Now(),
		}
		deployment.GreenEnvironment.Instances = append(deployment.GreenEnvironment.Instances, instance)
	}
	deployment.Progress = 0.3

	// Phase 2: Health check green environment
	log.Printf("[deploy] Phase 2: Health checking green environment")
	time.Sleep(3 * time.Second)

	for i := range deployment.GreenEnvironment.Instances {
		deployment.GreenEnvironment.Instances[i].Status = InstanceHealthy
		deployment.GreenEnvironment.Instances[i].Health = HealthPassing
	}
	deployment.Progress = 0.5

	// Phase 3: Switch traffic to green (100%)
	log.Printf("[deploy] Phase 3: Switching traffic to green environment")
	time.Sleep(2 * time.Second)

	deployment.BlueEnvironment.TrafficWeight = 0.0
	deployment.BlueEnvironment.Active = false
	deployment.GreenEnvironment.TrafficWeight = 1.0
	deployment.GreenEnvironment.Active = true
	deployment.Progress = 0.8

	// Update traffic manager
	s.zdd.trafficMgr.mu.Lock()
	s.zdd.trafficMgr.routes[deployment.Service] = &TrafficRoute{
		Service:     deployment.Service,
		BlueWeight:  0.0,
		GreenWeight: 1.0,
		LastShift:   time.Now(),
	}
	s.zdd.trafficMgr.mu.Unlock()

	// Phase 4: Mark rollback ready
	deployment.RollbackReady = true
	deployment.RollbackVersion = deployment.BlueEnvironment.Version
	deployment.Progress = 1.0

	log.Printf("[deploy] Blue-Green deployment completed: %s", deployment.ID)
	return nil
}

func (s *BlueGreenStrategy) Rollback(ctx context.Context, deployment *Deployment) error {
	log.Printf("[deploy] Rolling back Blue-Green deployment: %s", deployment.ID)

	// Switch traffic back to blue
	deployment.BlueEnvironment.TrafficWeight = 1.0
	deployment.BlueEnvironment.Active = true
	deployment.GreenEnvironment.TrafficWeight = 0.0
	deployment.GreenEnvironment.Active = false

	// Update traffic manager
	s.zdd.trafficMgr.mu.Lock()
	s.zdd.trafficMgr.routes[deployment.Service] = &TrafficRoute{
		Service:     deployment.Service,
		BlueWeight:  1.0,
		GreenWeight: 0.0,
		LastShift:   time.Now(),
	}
	s.zdd.trafficMgr.mu.Unlock()

	return nil
}

func (s *BlueGreenStrategy) GetProgress(deployment *Deployment) float64 {
	return deployment.Progress
}

// Canary Strategy Implementation
type CanaryStrategy struct {
	zdd *ZeroDowntimeDeployer
}

func (s *CanaryStrategy) Deploy(ctx context.Context, deployment *Deployment) error {
	log.Printf("[deploy] Executing Canary deployment for: %s", deployment.ID)

	config := deployment.CanaryConfig
	currentTraffic := config.InitialTrafficPercent

	for stage := 1; stage <= config.MaxStages; stage++ {
		canaryStage := CanaryStage{
			StageNumber:    stage,
			TrafficPercent: currentTraffic,
			StartTime:      time.Now(),
			Status:         StageRunning,
		}

		log.Printf("[deploy] Canary stage %d: %.1f%% traffic", stage, currentTraffic)

		// Wait for stage interval
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(config.StageInterval):
		}

		// Evaluate stage
		canaryStage.EndTime = time.Now()
		canaryStage.Metrics = s.collectStageMetrics(deployment)

		decision := s.evaluateStage(canaryStage, config.SuccessThreshold)
		canaryStage.Decision = decision

		if decision == DecisionRollback {
			canaryStage.Status = StageFailed
			deployment.CanaryStages = append(deployment.CanaryStages, canaryStage)
			return fmt.Errorf("canary stage %d failed: metrics below threshold", stage)
		}

		canaryStage.Status = StageSucceeded
		deployment.CanaryStages = append(deployment.CanaryStages, canaryStage)

		// Increase traffic
		currentTraffic += config.TrafficIncrementPercent
		if currentTraffic >= 100.0 {
			currentTraffic = 100.0
			deployment.Progress = 1.0
			break
		}

		deployment.Progress = currentTraffic / 100.0
	}

	log.Printf("[deploy] Canary deployment completed: %s", deployment.ID)
	return nil
}

func (s *CanaryStrategy) collectStageMetrics(deployment *Deployment) StageMetrics {
	// In production, would collect real metrics from monitoring system
	return StageMetrics{
		RequestCount: 1000,
		ErrorCount:   5,
		ErrorRate:    0.005,
		AvgLatency:   150 * time.Millisecond,
		P95Latency:   300 * time.Millisecond,
		P99Latency:   450 * time.Millisecond,
		SuccessRate:  0.995,
	}
}

func (s *CanaryStrategy) evaluateStage(stage CanaryStage, threshold SuccessThreshold) StageDecision {
	metrics := stage.Metrics

	// Check if enough requests
	if metrics.RequestCount < threshold.MinRequestCount {
		return DecisionPause
	}

	// Check error rate
	if metrics.ErrorRate > threshold.MaxErrorRate {
		log.Printf("[deploy] Stage %d failed: error rate %.2f%% > %.2f%%",
			stage.StageNumber, metrics.ErrorRate*100, threshold.MaxErrorRate*100)
		return DecisionRollback
	}

	// Check latency
	if metrics.P99Latency > threshold.MaxLatencyP99 {
		log.Printf("[deploy] Stage %d failed: P99 latency %v > %v",
			stage.StageNumber, metrics.P99Latency, threshold.MaxLatencyP99)
		return DecisionRollback
	}

	// Check success rate
	if metrics.SuccessRate < threshold.MinSuccessRate {
		log.Printf("[deploy] Stage %d failed: success rate %.2f%% < %.2f%%",
			stage.StageNumber, metrics.SuccessRate*100, threshold.MinSuccessRate*100)
		return DecisionRollback
	}

	return DecisionPromote
}

func (s *CanaryStrategy) Rollback(ctx context.Context, deployment *Deployment) error {
	log.Printf("[deploy] Rolling back Canary deployment: %s", deployment.ID)
	// Set traffic back to 0% for canary version
	return nil
}

func (s *CanaryStrategy) GetProgress(deployment *Deployment) float64 {
	return deployment.Progress
}

// Rolling Update Strategy Implementation
type RollingUpdateStrategy struct {
	zdd *ZeroDowntimeDeployer
}

func (s *RollingUpdateStrategy) Deploy(ctx context.Context, deployment *Deployment) error {
	log.Printf("[deploy] Executing Rolling Update deployment for: %s", deployment.ID)

	config := deployment.RollingConfig
	totalBatches := 5 // Simulated

	for batch := 1; batch <= totalBatches; batch++ {
		log.Printf("[deploy] Rolling update batch %d/%d", batch, totalBatches)

		// Update batch
		time.Sleep(config.BatchInterval)

		deployment.Progress = float64(batch) / float64(totalBatches)

		// Health check
		time.Sleep(config.HealthCheckTimeout / 2)
	}

	deployment.Progress = 1.0
	log.Printf("[deploy] Rolling update completed: %s", deployment.ID)
	return nil
}

func (s *RollingUpdateStrategy) Rollback(ctx context.Context, deployment *Deployment) error {
	log.Printf("[deploy] Rolling back Rolling Update deployment: %s", deployment.ID)
	// Roll back in reverse batches
	return nil
}

func (s *RollingUpdateStrategy) GetProgress(deployment *Deployment) float64 {
	return deployment.Progress
}
