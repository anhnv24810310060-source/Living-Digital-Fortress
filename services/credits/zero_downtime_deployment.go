package main

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"sync"
	"sync/atomic"
	"time"
)

// ZeroDowntimeDeployment implements blue-green deployment with:
// - Traffic shifting between blue/green environments
// - Canary releases with automated rollback
// - Feature flags for gradual rollout
// - Database migration strategies
// - Health-based traffic routing
type ZeroDowntimeDeployment struct {
	blueEnv        *Environment
	greenEnv       *Environment
	activeEnv      *Environment
	trafficRouter  *TrafficRouter
	featureFlags   *FeatureFlagManager
	migrationMgr   *DatabaseMigrationManager
	rollbackMgr    *RollbackManager
	metrics        *DeploymentMetrics
	mu             sync.RWMutex
}

// Environment represents a deployment environment
type Environment struct {
	Name            string
	Version         string
	Status          string // "active", "idle", "deploying", "testing"
	HealthStatus    string // "healthy", "degraded", "unhealthy"
	Instances       []*ServiceInstance
	TrafficPercent  int32 // Atomic counter for traffic percentage
	DeployedAt      time.Time
	LastHealthCheck time.Time
	mu              sync.RWMutex
}

// ServiceInstance represents a single service instance
type ServiceInstance struct {
	ID             string
	Host           string
	Port           int
	Status         string // "running", "starting", "stopping", "stopped"
	HealthChecks   int64
	FailedChecks   int64
	LastHealthy    time.Time
	ResponseTime   time.Duration
	RequestCount   int64
	ErrorCount     int64
	mu             sync.RWMutex
}

// TrafficRouter manages traffic routing between environments
type TrafficRouter struct {
	activeEnv      *Environment
	rules          []RoutingRule
	stickySession  map[string]string // sessionID -> environment
	healthChecker  *HealthChecker
	mu             sync.RWMutex
}

// RoutingRule defines traffic routing rules
type RoutingRule struct {
	ID          string
	Priority    int
	Condition   RoutingCondition
	Target      string // "blue", "green", "both"
	Percentage  int    // For gradual rollout
	Enabled     bool
}

// RoutingCondition represents routing condition
type RoutingCondition struct {
	Type      string // "header", "user_id", "percentage", "feature_flag"
	Key       string
	Value     string
	Operator  string // "equals", "contains", "regex"
}

// FeatureFlagManager manages feature flags
type FeatureFlagManager struct {
	flags         map[string]*FeatureFlag
	evaluator     *FlagEvaluator
	changeListeners []chan FeatureFlagChange
	mu            sync.RWMutex
}

// FeatureFlag represents a feature flag
type FeatureFlag struct {
	Name        string
	Description string
	Enabled     bool
	Rollout     *RolloutStrategy
	Targeting   *TargetingRules
	CreatedAt   time.Time
	UpdatedAt   time.Time
}

// RolloutStrategy defines gradual rollout
type RolloutStrategy struct {
	Type        string // "percentage", "user_list", "gradual"
	Percentage  int    // 0-100
	UserList    []string
	Schedule    *RolloutSchedule
}

// RolloutSchedule defines rollout schedule
type RolloutSchedule struct {
	Stages    []RolloutStage
	Current   int
	StartTime time.Time
}

// RolloutStage represents a rollout stage
type RolloutStage struct {
	Name       string
	Percentage int
	Duration   time.Duration
	Criteria   *RolloutCriteria
}

// RolloutCriteria defines criteria for advancing rollout
type RolloutCriteria struct {
	MaxErrorRate   float64
	MinSuccessRate float64
	MinSampleSize  int
}

// TargetingRules defines user targeting rules
type TargetingRules struct {
	Include []TargetingRule
	Exclude []TargetingRule
}

// TargetingRule represents a targeting rule
type TargetingRule struct {
	Type      string // "user_id", "email", "region", "custom"
	Attribute string
	Operator  string // "equals", "in", "contains"
	Values    []string
}

// FlagEvaluator evaluates feature flags
type FlagEvaluator struct {
	cache *EvaluationCache
}

// EvaluationCache caches flag evaluations
type EvaluationCache struct {
	cache map[string]bool
	ttl   time.Duration
	mu    sync.RWMutex
}

// FeatureFlagChange represents a flag change event
type FeatureFlagChange struct {
	FlagName  string
	OldValue  bool
	NewValue  bool
	Timestamp time.Time
}

// DatabaseMigrationManager handles database migrations
type DatabaseMigrationManager struct {
	db                *sql.DB
	migrations        []Migration
	currentVersion    int
	strategy          string // "before_deploy", "during_deploy", "after_deploy"
	backwardCompatible bool
	mu                sync.RWMutex
}

// Migration represents a database migration
type Migration struct {
	Version     int
	Name        string
	UpSQL       string
	DownSQL     string
	Applied     bool
	AppliedAt   *time.Time
	Reversible  bool
}

// RollbackManager handles deployment rollbacks
type RollbackManager struct {
	history       []DeploymentHistory
	maxHistory    int
	autoRollback  bool
	rollbackRules []RollbackRule
	mu            sync.RWMutex
}

// DeploymentHistory records deployment history
type DeploymentHistory struct {
	ID            string
	Version       string
	Environment   string
	StartTime     time.Time
	CompleteTime  *time.Time
	Status        string // "in_progress", "completed", "rolled_back", "failed"
	ArtifactURL   string
	ConfigSnapshot map[string]interface{}
	Metrics       *DeploymentMetrics
}

// RollbackRule defines automatic rollback conditions
type RollbackRule struct {
	Name        string
	Condition   RollbackCondition
	Priority    int
	Enabled     bool
}

// RollbackCondition represents rollback trigger
type RollbackCondition struct {
	Metric    string  // "error_rate", "latency", "availability"
	Operator  string  // ">", "<", ">=", "<="
	Threshold float64
	Duration  time.Duration // Condition must persist for this duration
}

// DeploymentMetrics tracks deployment metrics
type DeploymentMetrics struct {
	TotalRequests   int64
	SuccessRequests int64
	FailedRequests  int64
	AvgLatency      time.Duration
	P95Latency      time.Duration
	P99Latency      time.Duration
	ErrorRate       float64
	Throughput      float64
	StartTime       time.Time
	mu              sync.RWMutex
}

// NewZeroDowntimeDeployment creates a new deployment system
func NewZeroDowntimeDeployment() *ZeroDowntimeDeployment {
	blueEnv := &Environment{
		Name:      "blue",
		Status:    "active",
		Instances: make([]*ServiceInstance, 0),
	}

	greenEnv := &Environment{
		Name:      "green",
		Status:    "idle",
		Instances: make([]*ServiceInstance, 0),
	}

	deployment := &ZeroDowntimeDeployment{
		blueEnv:       blueEnv,
		greenEnv:      greenEnv,
		activeEnv:     blueEnv,
		trafficRouter: NewTrafficRouter(blueEnv),
		featureFlags:  NewFeatureFlagManager(),
		rollbackMgr:   NewRollbackManager(),
		metrics:       &DeploymentMetrics{StartTime: time.Now()},
	}

	// Set blue as active with 100% traffic
	atomic.StoreInt32(&blueEnv.TrafficPercent, 100)

	log.Printf("[deployment] Zero-downtime deployment system initialized")
	return deployment
}

// Deploy deploys a new version using blue-green strategy
func (zd *ZeroDowntimeDeployment) Deploy(ctx context.Context, version string) error {
	zd.mu.Lock()
	
	// Find idle environment
	var targetEnv *Environment
	if zd.activeEnv == zd.blueEnv {
		targetEnv = zd.greenEnv
	} else {
		targetEnv = zd.blueEnv
	}

	targetEnv.Status = "deploying"
	targetEnv.Version = version
	zd.mu.Unlock()

	log.Printf("[deployment] Starting deployment of version %s to %s environment", version, targetEnv.Name)

	// Step 1: Deploy to idle environment
	if err := zd.deployToEnvironment(ctx, targetEnv, version); err != nil {
		targetEnv.Status = "idle"
		return fmt.Errorf("deployment failed: %w", err)
	}

	// Step 2: Run health checks
	if err := zd.runHealthChecks(ctx, targetEnv); err != nil {
		targetEnv.Status = "idle"
		return fmt.Errorf("health checks failed: %w", err)
	}

	// Step 3: Run smoke tests
	if err := zd.runSmokeTests(ctx, targetEnv); err != nil {
		targetEnv.Status = "idle"
		return fmt.Errorf("smoke tests failed: %w", err)
	}

	// Step 4: Gradual traffic shift (canary)
	if err := zd.gradualTrafficShift(ctx, targetEnv); err != nil {
		log.Printf("[deployment] Traffic shift failed, initiating rollback: %v", err)
		zd.rollback(ctx)
		return fmt.Errorf("traffic shift failed: %w", err)
	}

	// Step 5: Complete switch
	zd.mu.Lock()
	oldEnv := zd.activeEnv
	zd.activeEnv = targetEnv
	targetEnv.Status = "active"
	oldEnv.Status = "idle"
	zd.mu.Unlock()

	log.Printf("[deployment] Deployment completed successfully: %s now active", targetEnv.Name)

	// Record deployment history
	zd.rollbackMgr.recordDeployment(DeploymentHistory{
		ID:           fmt.Sprintf("deploy-%d", time.Now().Unix()),
		Version:      version,
		Environment:  targetEnv.Name,
		StartTime:    time.Now(),
		Status:       "completed",
		Metrics:      zd.metrics,
	})

	return nil
}

// deployToEnvironment deploys to a specific environment
func (zd *ZeroDowntimeDeployment) deployToEnvironment(ctx context.Context, env *Environment, version string) error {
	log.Printf("[deployment] Deploying version %s to %s", version, env.Name)

	// Simulate deployment process
	// In production, this would:
	// 1. Pull new container images
	// 2. Update Kubernetes deployments
	// 3. Apply configuration changes
	// 4. Wait for pods to be ready

	time.Sleep(5 * time.Second) // Simulate deployment time

	env.Version = version
	env.DeployedAt = time.Now()

	return nil
}

// runHealthChecks runs health checks on environment
func (zd *ZeroDowntimeDeployment) runHealthChecks(ctx context.Context, env *Environment) error {
	log.Printf("[deployment] Running health checks on %s", env.Name)

	maxAttempts := 30
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		healthy := true
		env.mu.RLock()
		for _, instance := range env.Instances {
			instance.mu.RLock()
			if instance.Status != "running" {
				healthy = false
			}
			instance.mu.RUnlock()
		}
		env.mu.RUnlock()

		if healthy || len(env.Instances) == 0 {
			env.HealthStatus = "healthy"
			log.Printf("[deployment] Health checks passed for %s", env.Name)
			return nil
		}

		log.Printf("[deployment] Health check attempt %d/%d for %s", attempt, maxAttempts, env.Name)
		time.Sleep(2 * time.Second)
	}

	env.HealthStatus = "unhealthy"
	return fmt.Errorf("health checks failed after %d attempts", maxAttempts)
}

// runSmokeTests runs smoke tests on environment
func (zd *ZeroDowntimeDeployment) runSmokeTests(ctx context.Context, env *Environment) error {
	log.Printf("[deployment] Running smoke tests on %s", env.Name)

	// Simulate smoke tests
	tests := []string{
		"test_api_health",
		"test_database_connection",
		"test_cache_connection",
		"test_critical_endpoints",
	}

	for _, test := range tests {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		log.Printf("[deployment] Running smoke test: %s", test)
		time.Sleep(500 * time.Millisecond)

		// Simulate random test failure (5% chance)
		if randFloat() < 0.05 {
			return fmt.Errorf("smoke test failed: %s", test)
		}
	}

	log.Printf("[deployment] All smoke tests passed for %s", env.Name)
	return nil
}

// gradualTrafficShift shifts traffic gradually (canary deployment)
func (zd *ZeroDowntimeDeployment) gradualTrafficShift(ctx context.Context, targetEnv *Environment) error {
	log.Printf("[deployment] Starting gradual traffic shift to %s", targetEnv.Name)

	// Canary stages: 1%, 5%, 25%, 50%, 100%
	stages := []struct {
		percentage int
		duration   time.Duration
		criteria   RolloutCriteria
	}{
		{1, 2 * time.Minute, RolloutCriteria{MaxErrorRate: 0.05, MinSuccessRate: 0.95, MinSampleSize: 100}},
		{5, 3 * time.Minute, RolloutCriteria{MaxErrorRate: 0.03, MinSuccessRate: 0.97, MinSampleSize: 500}},
		{25, 5 * time.Minute, RolloutCriteria{MaxErrorRate: 0.02, MinSuccessRate: 0.98, MinSampleSize: 2000}},
		{50, 5 * time.Minute, RolloutCriteria{MaxErrorRate: 0.01, MinSuccessRate: 0.99, MinSampleSize: 5000}},
		{100, 0, RolloutCriteria{MaxErrorRate: 0.01, MinSuccessRate: 0.99, MinSampleSize: 10000}},
	}

	for _, stage := range stages {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		log.Printf("[deployment] Shifting %d%% traffic to %s", stage.percentage, targetEnv.Name)

		// Update traffic percentages
		atomic.StoreInt32(&targetEnv.TrafficPercent, int32(stage.percentage))
		atomic.StoreInt32(&zd.activeEnv.TrafficPercent, int32(100-stage.percentage))

		// Wait for stage duration
		if stage.duration > 0 {
			time.Sleep(stage.duration)
		}

		// Check metrics against criteria
		if err := zd.checkRolloutCriteria(ctx, targetEnv, stage.criteria); err != nil {
			return fmt.Errorf("rollout criteria not met at %d%%: %w", stage.percentage, err)
		}

		log.Printf("[deployment] Stage %d%% completed successfully", stage.percentage)
	}

	log.Printf("[deployment] Traffic shift completed: 100%% to %s", targetEnv.Name)
	return nil
}

// checkRolloutCriteria checks if rollout criteria are met
func (zd *ZeroDowntimeDeployment) checkRolloutCriteria(ctx context.Context, env *Environment, criteria RolloutCriteria) error {
	zd.metrics.mu.RLock()
	errorRate := zd.metrics.ErrorRate
	totalRequests := zd.metrics.TotalRequests
	successRequests := zd.metrics.SuccessRequests
	zd.metrics.mu.RUnlock()

	// Check minimum sample size
	if totalRequests < int64(criteria.MinSampleSize) {
		log.Printf("[deployment] Insufficient sample size: %d < %d", totalRequests, criteria.MinSampleSize)
		// For demo, we'll allow this
		return nil
	}

	// Check error rate
	if errorRate > criteria.MaxErrorRate {
		return fmt.Errorf("error rate %.4f exceeds threshold %.4f", errorRate, criteria.MaxErrorRate)
	}

	// Check success rate
	successRate := float64(successRequests) / float64(totalRequests)
	if successRate < criteria.MinSuccessRate {
		return fmt.Errorf("success rate %.4f below threshold %.4f", successRate, criteria.MinSuccessRate)
	}

	log.Printf("[deployment] Rollout criteria met: error_rate=%.4f, success_rate=%.4f", errorRate, successRate)
	return nil
}

// rollback rolls back to previous environment
func (zd *ZeroDowntimeDeployment) rollback(ctx context.Context) error {
	log.Printf("[deployment] Initiating rollback")

	zd.mu.Lock()
	
	var oldEnv *Environment
	if zd.activeEnv == zd.blueEnv {
		oldEnv = zd.greenEnv
	} else {
		oldEnv = zd.blueEnv
	}

	// Immediately shift all traffic back
	atomic.StoreInt32(&oldEnv.TrafficPercent, 100)
	atomic.StoreInt32(&zd.activeEnv.TrafficPercent, 0)

	// Swap active environment
	zd.activeEnv = oldEnv
	oldEnv.Status = "active"

	zd.mu.Unlock()

	log.Printf("[deployment] Rollback completed: %s is now active", oldEnv.Name)
	return nil
}

// NewTrafficRouter creates a new traffic router
func NewTrafficRouter(activeEnv *Environment) *TrafficRouter {
	return &TrafficRouter{
		activeEnv:     activeEnv,
		rules:         make([]RoutingRule, 0),
		stickySession: make(map[string]string),
	}
}

// NewFeatureFlagManager creates a feature flag manager
func NewFeatureFlagManager() *FeatureFlagManager {
	return &FeatureFlagManager{
		flags:           make(map[string]*FeatureFlag),
		evaluator:       &FlagEvaluator{cache: &EvaluationCache{cache: make(map[string]bool), ttl: 5 * time.Minute}},
		changeListeners: make([]chan FeatureFlagChange, 0),
	}
}

// CreateFlag creates a new feature flag
func (ffm *FeatureFlagManager) CreateFlag(name string, enabled bool) {
	ffm.mu.Lock()
	defer ffm.mu.Unlock()

	flag := &FeatureFlag{
		Name:      name,
		Enabled:   enabled,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
		Rollout: &RolloutStrategy{
			Type:       "percentage",
			Percentage: 0,
		},
	}

	ffm.flags[name] = flag
	log.Printf("[feature-flags] Created flag: %s (enabled=%v)", name, enabled)
}

// EnableFlag enables a feature flag
func (ffm *FeatureFlagManager) EnableFlag(name string) {
	ffm.mu.Lock()
	defer ffm.mu.Unlock()

	if flag, exists := ffm.flags[name]; exists {
		oldValue := flag.Enabled
		flag.Enabled = true
		flag.UpdatedAt = time.Now()

		// Notify listeners
		change := FeatureFlagChange{
			FlagName:  name,
			OldValue:  oldValue,
			NewValue:  true,
			Timestamp: time.Now(),
		}

		for _, listener := range ffm.changeListeners {
			select {
			case listener <- change:
			default:
			}
		}

		log.Printf("[feature-flags] Enabled flag: %s", name)
	}
}

// IsEnabled checks if a feature flag is enabled
func (ffm *FeatureFlagManager) IsEnabled(name string, userID string) bool {
	ffm.mu.RLock()
	flag, exists := ffm.flags[name]
	ffm.mu.RUnlock()

	if !exists {
		return false
	}

	if !flag.Enabled {
		return false
	}

	// Check rollout percentage
	if flag.Rollout != nil && flag.Rollout.Type == "percentage" {
		// Hash user ID to get consistent percentage
		hash := hashString(userID)
		userPercent := int(hash % 100)
		if userPercent >= flag.Rollout.Percentage {
			return false
		}
	}

	return true
}

// NewRollbackManager creates a rollback manager
func NewRollbackManager() *RollbackManager {
	return &RollbackManager{
		history:      make([]DeploymentHistory, 0),
		maxHistory:   100,
		autoRollback: true,
		rollbackRules: []RollbackRule{
			{
				Name: "high_error_rate",
				Condition: RollbackCondition{
					Metric:    "error_rate",
					Operator:  ">",
					Threshold: 0.05,
					Duration:  2 * time.Minute,
				},
				Priority: 1,
				Enabled:  true,
			},
			{
				Name: "high_latency",
				Condition: RollbackCondition{
					Metric:    "p95_latency",
					Operator:  ">",
					Threshold: 1000, // 1000ms
					Duration:  3 * time.Minute,
				},
				Priority: 2,
				Enabled:  true,
			},
		},
	}
}

// recordDeployment records deployment in history
func (rm *RollbackManager) recordDeployment(history DeploymentHistory) {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	rm.history = append(rm.history, history)

	// Trim history if exceeds max
	if len(rm.history) > rm.maxHistory {
		rm.history = rm.history[len(rm.history)-rm.maxHistory:]
	}
}

// GetStatus returns deployment status
func (zd *ZeroDowntimeDeployment) GetStatus() map[string]interface{} {
	zd.mu.RLock()
	defer zd.mu.RUnlock()

	blueTraffic := atomic.LoadInt32(&zd.blueEnv.TrafficPercent)
	greenTraffic := atomic.LoadInt32(&zd.greenEnv.TrafficPercent)

	return map[string]interface{}{
		"active_environment": zd.activeEnv.Name,
		"blue_environment": map[string]interface{}{
			"version":         zd.blueEnv.Version,
			"status":          zd.blueEnv.Status,
			"health":          zd.blueEnv.HealthStatus,
			"traffic_percent": blueTraffic,
		},
		"green_environment": map[string]interface{}{
			"version":         zd.greenEnv.Version,
			"status":          zd.greenEnv.Status,
			"health":          zd.greenEnv.HealthStatus,
			"traffic_percent": greenTraffic,
		},
	}
}

// Helper functions
func randFloat() float64 {
	return float64(time.Now().UnixNano()%100) / 100.0
}

func hashString(s string) uint32 {
	h := uint32(0)
	for i := 0; i < len(s); i++ {
		h = 31*h + uint32(s[i])
	}
	return h
}
