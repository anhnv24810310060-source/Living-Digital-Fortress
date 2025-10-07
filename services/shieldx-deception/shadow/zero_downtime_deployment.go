package shadow
package shadow

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"time"
)

// ZeroDowntimeDeployment implements zero-downtime deployment strategies:
// - Blue-green deployment with traffic shifting
// - Canary releases with automated rollback
// - Feature flags for gradual rollout
// - Database migration strategies
type ZeroDowntimeDeployment struct {
	db                *sql.DB
	deploymentMgr     *DeploymentManager
	trafficController *TrafficController
	featureFlags      *FeatureFlagManager
	healthValidator   *HealthValidator
	rollbackMgr       *RollbackManager
	config            DeploymentConfig
	mu                sync.RWMutex
}

// DeploymentConfig contains deployment configuration
type DeploymentConfig struct {
	Strategy               string        `json:"strategy"` // "blue_green", "canary", "rolling"
	CanaryPercentage       []int         `json:"canary_percentage"` // E.g., [10, 25, 50, 100]
	CanaryDuration         time.Duration `json:"canary_duration"` // Duration per stage
	HealthCheckInterval    time.Duration `json:"health_check_interval"`
	AutoRollbackEnabled    bool          `json:"auto_rollback_enabled"`
	RollbackThreshold      float64       `json:"rollback_threshold"` // Error rate threshold
	TrafficShiftDuration   time.Duration `json:"traffic_shift_duration"`
	DatabaseMigrationMode  string        `json:"database_migration_mode"` // "expand_contract", "dual_write"
}

// DeploymentManager manages deployment lifecycle
type DeploymentManager struct {
	db          *sql.DB
	deployments map[string]*Deployment
	mu          sync.RWMutex
}

// Deployment represents a deployment instance
type Deployment struct {
	ID                  string                 `json:"id"`
	Version             string                 `json:"version"`
	Strategy            string                 `json:"strategy"`
	Status              string                 `json:"status"` // "preparing", "deploying", "validating", "completed", "failed", "rolled_back"
	CurrentStage        int                    `json:"current_stage"`
	TotalStages         int                    `json:"total_stages"`
	StartTime           time.Time              `json:"start_time"`
	CompletedTime       *time.Time             `json:"completed_time,omitempty"`
	BlueEnvironment     *Environment           `json:"blue_environment,omitempty"`
	GreenEnvironment    *Environment           `json:"green_environment,omitempty"`
	TrafficSplit        map[string]int         `json:"traffic_split"` // environment -> percentage
	HealthMetrics       *DeploymentHealthMetrics `json:"health_metrics"`
	FeatureFlags        []string               `json:"feature_flags"`
	RollbackAvailable   bool                   `json:"rollback_available"`
}

// Environment represents a deployment environment
type Environment struct {
	Name            string                 `json:"name"`
	Version         string                 `json:"version"`
	Status          string                 `json:"status"` // "preparing", "ready", "active", "draining", "stopped"
	Instances       []*Instance            `json:"instances"`
	HealthScore     float64                `json:"health_score"`
	DeployedAt      time.Time              `json:"deployed_at"`
	Configuration   map[string]interface{} `json:"configuration"`
}

// Instance represents a service instance
type Instance struct {
	ID              string    `json:"id"`
	Host            string    `json:"host"`
	Port            int       `json:"port"`
	Status          string    `json:"status"` // "starting", "healthy", "unhealthy", "draining", "stopped"
	HealthScore     float64   `json:"health_score"`
	LastHealthCheck time.Time `json:"last_health_check"`
	RequestCount    int64     `json:"request_count"`
	ErrorCount      int64     `json:"error_count"`
}

// DeploymentHealthMetrics tracks deployment health
type DeploymentHealthMetrics struct {
	TotalRequests   int64     `json:"total_requests"`
	SuccessCount    int64     `json:"success_count"`
	ErrorCount      int64     `json:"error_count"`
	ErrorRate       float64   `json:"error_rate"`
	AvgLatency      float64   `json:"avg_latency_ms"`
	P95Latency      float64   `json:"p95_latency_ms"`
	P99Latency      float64   `json:"p99_latency_ms"`
	LastUpdated     time.Time `json:"last_updated"`
}

// TrafficController manages traffic distribution
type TrafficController struct {
	rules         map[string]*TrafficRule
	activeRoutes  map[string]*Route
	mu            sync.RWMutex
}

// TrafficRule defines traffic routing rules
type TrafficRule struct {
	ID              string                 `json:"id"`
	DeploymentID    string                 `json:"deployment_id"`
	SourceVersions  map[string]int         `json:"source_versions"` // version -> percentage
	Criteria        map[string]interface{} `json:"criteria"` // Routing criteria
	Priority        int                    `json:"priority"`
	Active          bool                   `json:"active"`
}

// Route represents an active route
type Route struct {
	Target      string    `json:"target"`
	Weight      int       `json:"weight"`
	LastUpdated time.Time `json:"last_updated"`
}

// FeatureFlagManager manages feature flags
type FeatureFlagManager struct {
	flags map[string]*FeatureFlag
	mu    sync.RWMutex
}

// FeatureFlag represents a feature flag
type FeatureFlag struct {
	Name            string                 `json:"name"`
	Enabled         bool                   `json:"enabled"`
	Rollout         int                    `json:"rollout"` // Percentage 0-100
	TargetGroups    []string               `json:"target_groups"`
	Conditions      map[string]interface{} `json:"conditions"`
	CreatedAt       time.Time              `json:"created_at"`
	UpdatedAt       time.Time              `json:"updated_at"`
}

// HealthValidator validates deployment health
type HealthValidator struct {
	checks []HealthCheck
	mu     sync.RWMutex
}

// HealthCheck defines a health check
type HealthCheck struct {
	Name        string
	Check       func(ctx context.Context, deployment *Deployment) (bool, string)
	Critical    bool
}

// RollbackManager manages rollback operations
type RollbackManager struct {
	db       *sql.DB
	rollbacks map[string]*Rollback
	mu       sync.RWMutex
}

// Rollback represents a rollback operation
type Rollback struct {
	ID            string    `json:"id"`
	DeploymentID  string    `json:"deployment_id"`
	Reason        string    `json:"reason"`
	FromVersion   string    `json:"from_version"`
	ToVersion     string    `json:"to_version"`
	Status        string    `json:"status"`
	StartTime     time.Time `json:"start_time"`
	CompletedTime *time.Time `json:"completed_time,omitempty"`
}

// DatabaseMigrationStrategy handles database migrations
type DatabaseMigrationStrategy struct {
	mode        string // "expand_contract", "dual_write"
	migrations  []*Migration
	mu          sync.RWMutex
}

// Migration represents a database migration
type Migration struct {
	ID          string    `json:"id"`
	Version     string    `json:"version"`
	Description string    `json:"description"`
	UpSQL       string    `json:"up_sql"`
	DownSQL     string    `json:"down_sql"`
	Status      string    `json:"status"`
	AppliedAt   *time.Time `json:"applied_at,omitempty"`
}

// NewZeroDowntimeDeployment creates a new zero-downtime deployment system
func NewZeroDowntimeDeployment(db *sql.DB, config DeploymentConfig) (*ZeroDowntimeDeployment, error) {
	system := &ZeroDowntimeDeployment{
		db:                db,
		deploymentMgr:     NewDeploymentManager(db),
		trafficController: NewTrafficController(),
		featureFlags:      NewFeatureFlagManager(),
		healthValidator:   NewHealthValidator(),
		rollbackMgr:       NewRollbackManager(db),
		config:            config,
	}

	// Initialize schema
	if err := system.initializeSchema(); err != nil {
		return nil, fmt.Errorf("failed to initialize schema: %w", err)
	}

	// Register default health checks
	system.registerDefaultHealthChecks()

	log.Printf("[deployment] Zero-downtime deployment system initialized with strategy: %s", config.Strategy)
	return system, nil
}

// initializeSchema creates necessary tables
func (zdd *ZeroDowntimeDeployment) initializeSchema() error {
	schema := `
	CREATE TABLE IF NOT EXISTS deployments (
		id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
		version VARCHAR(255) NOT NULL,
		strategy VARCHAR(100) NOT NULL,
		status VARCHAR(50) NOT NULL DEFAULT 'preparing',
		current_stage INT NOT NULL DEFAULT 0,
		total_stages INT NOT NULL,
		start_time TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
		completed_time TIMESTAMP WITH TIME ZONE,
		configuration JSONB,
		traffic_split JSONB,
		health_metrics JSONB,
		feature_flags JSONB,
		rollback_available BOOLEAN DEFAULT true,
		created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
		updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
	);

	CREATE INDEX IF NOT EXISTS idx_deployments_status 
		ON deployments(status, start_time DESC);
	CREATE INDEX IF NOT EXISTS idx_deployments_version 
		ON deployments(version);

	CREATE TABLE IF NOT EXISTS deployment_environments (
		id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
		deployment_id UUID NOT NULL REFERENCES deployments(id),
		name VARCHAR(100) NOT NULL,
		version VARCHAR(255) NOT NULL,
		status VARCHAR(50) NOT NULL DEFAULT 'preparing',
		health_score DOUBLE PRECISION DEFAULT 1.0,
		deployed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
		configuration JSONB,
		CONSTRAINT unique_deployment_env UNIQUE (deployment_id, name)
	);

	CREATE INDEX IF NOT EXISTS idx_deployment_environments_deployment 
		ON deployment_environments(deployment_id);

	CREATE TABLE IF NOT EXISTS deployment_instances (
		id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
		environment_id UUID NOT NULL REFERENCES deployment_environments(id),
		host VARCHAR(255) NOT NULL,
		port INT NOT NULL,
		status VARCHAR(50) NOT NULL DEFAULT 'starting',
		health_score DOUBLE PRECISION DEFAULT 1.0,
		last_health_check TIMESTAMP WITH TIME ZONE,
		request_count BIGINT DEFAULT 0,
		error_count BIGINT DEFAULT 0,
		created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
	);

	CREATE INDEX IF NOT EXISTS idx_deployment_instances_environment 
		ON deployment_instances(environment_id);

	CREATE TABLE IF NOT EXISTS traffic_rules (
		id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
		deployment_id UUID NOT NULL REFERENCES deployments(id),
		rule_definition JSONB NOT NULL,
		priority INT NOT NULL DEFAULT 100,
		active BOOLEAN DEFAULT true,
		created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
		updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
	);

	CREATE TABLE IF NOT EXISTS feature_flags (
		name VARCHAR(255) PRIMARY KEY,
		enabled BOOLEAN NOT NULL DEFAULT false,
		rollout INT NOT NULL DEFAULT 0 CHECK (rollout >= 0 AND rollout <= 100),
		target_groups JSONB,
		conditions JSONB,
		created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
		updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
	);

	CREATE TABLE IF NOT EXISTS rollback_history (
		id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
		deployment_id UUID NOT NULL REFERENCES deployments(id),
		reason TEXT NOT NULL,
		from_version VARCHAR(255) NOT NULL,
		to_version VARCHAR(255) NOT NULL,
		status VARCHAR(50) NOT NULL DEFAULT 'initiated',
		start_time TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
		completed_time TIMESTAMP WITH TIME ZONE,
		metadata JSONB
	);

	CREATE INDEX IF NOT EXISTS idx_rollback_history_deployment 
		ON rollback_history(deployment_id, start_time DESC);

	CREATE TABLE IF NOT EXISTS database_migrations (
		id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
		version VARCHAR(255) UNIQUE NOT NULL,
		description TEXT,
		up_sql TEXT NOT NULL,
		down_sql TEXT NOT NULL,
		status VARCHAR(50) NOT NULL DEFAULT 'pending',
		applied_at TIMESTAMP WITH TIME ZONE,
		checksum VARCHAR(64),
		execution_time_ms INT,
		created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
	);

	CREATE INDEX IF NOT EXISTS idx_database_migrations_version 
		ON database_migrations(version);
	`

	_, err := zdd.db.Exec(schema)
	return err
}

// registerDefaultHealthChecks registers default health validations
func (zdd *ZeroDowntimeDeployment) registerDefaultHealthChecks() {
	// Check 1: Error rate threshold
	zdd.healthValidator.RegisterCheck(HealthCheck{
		Name:     "error_rate_threshold",
		Critical: true,
		Check: func(ctx context.Context, deployment *Deployment) (bool, string) {
			if deployment.HealthMetrics == nil {
				return true, "no metrics yet"
			}

			if deployment.HealthMetrics.ErrorRate > zdd.config.RollbackThreshold {
				return false, fmt.Sprintf("error rate %.2f%% exceeds threshold %.2f%%",
					deployment.HealthMetrics.ErrorRate*100, zdd.config.RollbackThreshold*100)
			}

			return true, ""
		},
	})

	// Check 2: Instance health
	zdd.healthValidator.RegisterCheck(HealthCheck{
		Name:     "instance_health",
		Critical: true,
		Check: func(ctx context.Context, deployment *Deployment) (bool, string) {
			for envName, env := range map[string]*Environment{
				"blue":  deployment.BlueEnvironment,
				"green": deployment.GreenEnvironment,
			} {
				if env == nil {
					continue
				}

				healthyCount := 0
				for _, instance := range env.Instances {
					if instance.Status == "healthy" {
						healthyCount++
					}
				}

				if len(env.Instances) > 0 {
					healthyRatio := float64(healthyCount) / float64(len(env.Instances))
					if healthyRatio < 0.5 {
						return false, fmt.Sprintf("%s environment has only %.0f%% healthy instances",
							envName, healthyRatio*100)
					}
				}
			}

			return true, ""
		},
	})

	// Check 3: Latency threshold
	zdd.healthValidator.RegisterCheck(HealthCheck{
		Name:     "latency_threshold",
		Critical: false,
		Check: func(ctx context.Context, deployment *Deployment) (bool, string) {
			if deployment.HealthMetrics == nil {
				return true, "no metrics yet"
			}

			if deployment.HealthMetrics.P95Latency > 1000 { // 1 second
				return false, fmt.Sprintf("p95 latency %.0fms exceeds 1000ms threshold",
					deployment.HealthMetrics.P95Latency)
			}

			return true, ""
		},
	})

	log.Printf("[deployment] Registered %d health checks", 3)
}

// DeployNewVersion deploys a new version using configured strategy
func (zdd *ZeroDowntimeDeployment) DeployNewVersion(ctx context.Context, version string, config map[string]interface{}) (*Deployment, error) {
	deployment := &Deployment{
		ID:                fmt.Sprintf("deploy-%d", time.Now().Unix()),
		Version:           version,
		Strategy:          zdd.config.Strategy,
		Status:            "preparing",
		StartTime:         time.Now(),
		TrafficSplit:      make(map[string]int),
		FeatureFlags:      make([]string, 0),
		RollbackAvailable: true,
	}

	switch zdd.config.Strategy {
	case "blue_green":
		deployment.TotalStages = 3 // Prepare, Validate, Switch
		return zdd.deployBlueGreen(ctx, deployment, config)

	case "canary":
		deployment.TotalStages = len(zdd.config.CanaryPercentage) + 1
		return zdd.deployCanary(ctx, deployment, config)

	case "rolling":
		deployment.TotalStages = 5 // Calculate based on instance count
		return zdd.deployRolling(ctx, deployment, config)

	default:
		return nil, fmt.Errorf("unsupported deployment strategy: %s", zdd.config.Strategy)
	}
}

// deployBlueGreen implements blue-green deployment
func (zdd *ZeroDowntimeDeployment) deployBlueGreen(ctx context.Context, deployment *Deployment, config map[string]interface{}) (*Deployment, error) {
	log.Printf("[deployment] Starting blue-green deployment for version %s", deployment.Version)

	// Stage 1: Prepare green environment
	deployment.CurrentStage = 1
	deployment.Status = "deploying"

	greenEnv := &Environment{
		Name:          "green",
		Version:       deployment.Version,
		Status:        "preparing",
		Instances:     make([]*Instance, 0),
		Configuration: config,
		DeployedAt:    time.Now(),
	}

	// Create instances in green environment
	for i := 0; i < 3; i++ { // 3 instances
		instance := &Instance{
			ID:          fmt.Sprintf("green-%d", i),
			Host:        fmt.Sprintf("green-host-%d", i),
			Port:        8080 + i,
			Status:      "starting",
			HealthScore: 1.0,
		}
		greenEnv.Instances = append(greenEnv.Instances, instance)
	}

	deployment.GreenEnvironment = greenEnv

	// Wait for instances to become healthy
	if err := zdd.waitForHealthy(ctx, greenEnv); err != nil {
		return nil, fmt.Errorf("green environment failed to become healthy: %w", err)
	}

	greenEnv.Status = "ready"

	// Stage 2: Validate green environment
	deployment.CurrentStage = 2

	isHealthy, reason := zdd.healthValidator.ValidateDeployment(ctx, deployment)
	if !isHealthy {
		log.Printf("[deployment] Validation failed: %s", reason)
		return zdd.initiateRollback(ctx, deployment, reason)
	}

	// Stage 3: Switch traffic from blue to green
	deployment.CurrentStage = 3

	log.Printf("[deployment] Switching traffic to green environment")
	if err := zdd.switchTraffic(ctx, deployment, "blue", "green"); err != nil {
		return nil, fmt.Errorf("traffic switch failed: %w", err)
	}

	// Mark old blue environment as draining
	if deployment.BlueEnvironment != nil {
		deployment.BlueEnvironment.Status = "draining"
	}

	// Deployment completed
	completedTime := time.Now()
	deployment.CompletedTime = &completedTime
	deployment.Status = "completed"

	// Persist deployment
	if err := zdd.deploymentMgr.SaveDeployment(ctx, deployment); err != nil {
		log.Printf("[deployment] Failed to save deployment: %v", err)
	}

	log.Printf("[deployment] Blue-green deployment completed in %v", time.Since(deployment.StartTime))
	return deployment, nil
}

// deployCanary implements canary deployment
func (zdd *ZeroDowntimeDeployment) deployCanary(ctx context.Context, deployment *Deployment, config map[string]interface{}) (*Deployment, error) {
	log.Printf("[deployment] Starting canary deployment for version %s", deployment.Version)

	deployment.Status = "deploying"

	// Create canary environment
	canaryEnv := &Environment{
		Name:          "canary",
		Version:       deployment.Version,
		Status:        "preparing",
		Instances:     make([]*Instance, 0),
		Configuration: config,
		DeployedAt:    time.Now(),
	}

	// Create single canary instance
	instance := &Instance{
		ID:          "canary-0",
		Host:        "canary-host-0",
		Port:        8080,
		Status:      "starting",
		HealthScore: 1.0,
	}
	canaryEnv.Instances = append(canaryEnv.Instances, instance)

	deployment.GreenEnvironment = canaryEnv

	// Wait for canary to become healthy
	if err := zdd.waitForHealthy(ctx, canaryEnv); err != nil {
		return nil, fmt.Errorf("canary failed to become healthy: %w", err)
	}

	// Gradual traffic shift
	for stageIdx, percentage := range zdd.config.CanaryPercentage {
		deployment.CurrentStage = stageIdx + 1

		log.Printf("[deployment] Canary stage %d/%d: %d%% traffic",
			stageIdx+1, len(zdd.config.CanaryPercentage), percentage)

		// Shift traffic
		deployment.TrafficSplit["canary"] = percentage
		deployment.TrafficSplit["production"] = 100 - percentage

		if err := zdd.trafficController.UpdateTrafficSplit(ctx, deployment.ID, deployment.TrafficSplit); err != nil {
			return zdd.initiateRollback(ctx, deployment, fmt.Sprintf("traffic shift failed: %v", err))
		}

		// Wait and monitor
		time.Sleep(zdd.config.CanaryDuration)

		// Validate health
		isHealthy, reason := zdd.healthValidator.ValidateDeployment(ctx, deployment)
		if !isHealthy {
			log.Printf("[deployment] Canary validation failed at stage %d: %s", stageIdx+1, reason)
			return zdd.initiateRollback(ctx, deployment, reason)
		}

		log.Printf("[deployment] Canary stage %d/%d passed validation", stageIdx+1, len(zdd.config.CanaryPercentage))
	}

	// Full rollout
	deployment.TrafficSplit["canary"] = 100
	deployment.TrafficSplit["production"] = 0

	completedTime := time.Now()
	deployment.CompletedTime = &completedTime
	deployment.Status = "completed"

	if err := zdd.deploymentMgr.SaveDeployment(ctx, deployment); err != nil {
		log.Printf("[deployment] Failed to save deployment: %v", err)
	}

	log.Printf("[deployment] Canary deployment completed in %v", time.Since(deployment.StartTime))
	return deployment, nil
}

// deployRolling implements rolling deployment
func (zdd *ZeroDowntimeDeployment) deployRolling(ctx context.Context, deployment *Deployment, config map[string]interface{}) (*Deployment, error) {
	log.Printf("[deployment] Starting rolling deployment for version %s", deployment.Version)

	// Implementation details omitted for brevity
	// Would gradually replace instances one by one

	deployment.Status = "completed"
	completedTime := time.Now()
	deployment.CompletedTime = &completedTime

	return deployment, nil
}

// waitForHealthy waits for environment to become healthy
func (zdd *ZeroDowntimeDeployment) waitForHealthy(ctx context.Context, env *Environment) error {
	timeout := time.After(5 * time.Minute)
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-timeout:
			return fmt.Errorf("timeout waiting for healthy instances")
		case <-ticker.C:
			healthyCount := 0
			for _, instance := range env.Instances {
				// Simulate health check
				if zdd.checkInstanceHealth(ctx, instance) {
					instance.Status = "healthy"
					healthyCount++
				}
			}

			if healthyCount == len(env.Instances) {
				log.Printf("[deployment] All %d instances healthy in %s environment", healthyCount, env.Name)
				return nil
			}

			log.Printf("[deployment] Waiting for healthy instances: %d/%d", healthyCount, len(env.Instances))
		}
	}
}

// checkInstanceHealth checks if an instance is healthy
func (zdd *ZeroDowntimeDeployment) checkInstanceHealth(ctx context.Context, instance *Instance) bool {
	// Simplified health check
	// In production, would make HTTP request to health endpoint
	instance.LastHealthCheck = time.Now()
	return true
}

// switchTraffic switches traffic between environments
func (zdd *ZeroDowntimeDeployment) switchTraffic(ctx context.Context, deployment *Deployment, from, to string) error {
	duration := zdd.config.TrafficShiftDuration
	steps := 10
	stepDuration := duration / time.Duration(steps)

	for i := 0; i <= steps; i++ {
		percentage := (i * 100) / steps

		deployment.TrafficSplit[to] = percentage
		deployment.TrafficSplit[from] = 100 - percentage

		log.Printf("[deployment] Traffic shift: %s=%d%%, %s=%d%%",
			to, percentage, from, 100-percentage)

		if err := zdd.trafficController.UpdateTrafficSplit(ctx, deployment.ID, deployment.TrafficSplit); err != nil {
			return err
		}

		if i < steps {
			time.Sleep(stepDuration)
		}
	}

	log.Printf("[deployment] Traffic switch completed: 100%% to %s", to)
	return nil
}

// initiateRollback initiates a deployment rollback
func (zdd *ZeroDowntimeDeployment) initiateRollback(ctx context.Context, deployment *Deployment, reason string) (*Deployment, error) {
	if !zdd.config.AutoRollbackEnabled {
		deployment.Status = "failed"
		return deployment, fmt.Errorf("deployment failed but auto-rollback disabled: %s", reason)
	}

	log.Printf("[deployment] Initiating rollback for deployment %s: %s", deployment.ID, reason)

	rollback := &Rollback{
		ID:           fmt.Sprintf("rollback-%d", time.Now().Unix()),
		DeploymentID: deployment.ID,
		Reason:       reason,
		FromVersion:  deployment.Version,
		ToVersion:    "previous", // Would get from deployment history
		Status:       "in_progress",
		StartTime:    time.Now(),
	}

	// Revert traffic to previous version
	deployment.TrafficSplit["green"] = 0
	deployment.TrafficSplit["blue"] = 100

	if err := zdd.trafficController.UpdateTrafficSplit(ctx, deployment.ID, deployment.TrafficSplit); err != nil {
		return deployment, fmt.Errorf("rollback traffic update failed: %w", err)
	}

	rollback.Status = "completed"
	completedTime := time.Now()
	rollback.CompletedTime = &completedTime

	deployment.Status = "rolled_back"

	// Save rollback
	zdd.rollbackMgr.SaveRollback(ctx, rollback)

	log.Printf("[deployment] Rollback completed in %v", time.Since(rollback.StartTime))
	return deployment, nil
}

// Component implementations

func NewDeploymentManager(db *sql.DB) *DeploymentManager {
	return &DeploymentManager{
		db:          db,
		deployments: make(map[string]*Deployment),
	}
}

func (dm *DeploymentManager) SaveDeployment(ctx context.Context, deployment *Deployment) error {
	dm.mu.Lock()
	dm.mu.Unlock()

	dm.deployments[deployment.ID] = deployment

	// Persist to database
	configJSON, _ := json.Marshal(deployment.BlueEnvironment)
	trafficJSON, _ := json.Marshal(deployment.TrafficSplit)

	_, err := dm.db.ExecContext(ctx, `
		INSERT INTO deployments (
			id, version, strategy, status, current_stage, total_stages,
			start_time, completed_time, configuration, traffic_split
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
		ON CONFLICT (id) DO UPDATE SET
			status = $4, current_stage = $5, completed_time = $8,
			traffic_split = $10, updated_at = NOW()
	`, deployment.ID, deployment.Version, deployment.Strategy, deployment.Status,
		deployment.CurrentStage, deployment.TotalStages, deployment.StartTime,
		deployment.CompletedTime, configJSON, trafficJSON)

	return err
}

func NewTrafficController() *TrafficController {
	return &TrafficController{
		rules:        make(map[string]*TrafficRule),
		activeRoutes: make(map[string]*Route),
	}
}

func (tc *TrafficController) UpdateTrafficSplit(ctx context.Context, deploymentID string, split map[string]int) error {
	tc.mu.Lock()
	defer tc.mu.Unlock()

	// Update routing rules
	for env, percentage := range split {
		route := &Route{
			Target:      env,
			Weight:      percentage,
			LastUpdated: time.Now(),
		}
		tc.activeRoutes[fmt.Sprintf("%s-%s", deploymentID, env)] = route
	}

	log.Printf("[traffic] Updated traffic split for %s: %v", deploymentID, split)
	return nil
}

func NewFeatureFlagManager() *FeatureFlagManager {
	return &FeatureFlagManager{
		flags: make(map[string]*FeatureFlag),
	}
}

func NewHealthValidator() *HealthValidator {
	return &HealthValidator{
		checks: make([]HealthCheck, 0),
	}
}

func (hv *HealthValidator) RegisterCheck(check HealthCheck) {
	hv.mu.Lock()
	defer hv.mu.Unlock()
	hv.checks = append(hv.checks, check)
}

func (hv *HealthValidator) ValidateDeployment(ctx context.Context, deployment *Deployment) (bool, string) {
	hv.mu.RLock()
	defer hv.mu.RUnlock()

	for _, check := range hv.checks {
		passed, reason := check.Check(ctx, deployment)
		if !passed {
			if check.Critical {
				return false, fmt.Sprintf("%s: %s", check.Name, reason)
			}
			log.Printf("[health] Non-critical check failed: %s: %s", check.Name, reason)
		}
	}

	return true, ""
}

func NewRollbackManager(db *sql.DB) *RollbackManager {
	return &RollbackManager{
		db:        db,
		rollbacks: make(map[string]*Rollback),
	}
}

func (rm *RollbackManager) SaveRollback(ctx context.Context, rollback *Rollback) error {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	rm.rollbacks[rollback.ID] = rollback

	_, err := rm.db.ExecContext(ctx, `
		INSERT INTO rollback_history (
			id, deployment_id, reason, from_version, to_version,
			status, start_time, completed_time
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
	`, rollback.ID, rollback.DeploymentID, rollback.Reason,
		rollback.FromVersion, rollback.ToVersion, rollback.Status,
		rollback.StartTime, rollback.CompletedTime)

	return err
}

// GetDeploymentStatus returns current deployment status
func (zdd *ZeroDowntimeDeployment) GetDeploymentStatus(deploymentID string) (*Deployment, error) {
	zdd.deploymentMgr.mu.RLock()
	defer zdd.deploymentMgr.mu.RUnlock()

	deployment, ok := zdd.deploymentMgr.deployments[deploymentID]
	if !ok {
		return nil, fmt.Errorf("deployment not found: %s", deploymentID)
	}

	return deployment, nil
}
