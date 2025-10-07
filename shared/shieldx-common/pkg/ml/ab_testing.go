package ml

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/redis/go-redis/v9"
)

// ABTestManager manages A/B testing experiments for ML models
type ABTestManager struct {
	redisClient *redis.Client
	mu          sync.RWMutex
	experiments map[string]*Experiment
}

// Experiment represents an A/B test configuration
type Experiment struct {
	ID               string                 `json:"id"`
	Name             string                 `json:"name"`
	Description      string                 `json:"description"`
	Status           ExperimentStatus       `json:"status"`
	Models           []ModelVariant         `json:"models"`
	TrafficSplit     map[string]float64     `json:"traffic_split"` // modelID -> percentage
	StartDate        time.Time              `json:"start_date"`
	EndDate          time.Time              `json:"end_date"`
	TargetMetric     string                 `json:"target_metric"`
	MinimumSamples   int                    `json:"minimum_samples"`
	Metrics          map[string]*MetricData `json:"metrics"`
	WinnerModelID    string                 `json:"winner_model_id,omitempty"`
	CreatedAt        time.Time              `json:"created_at"`
	UpdatedAt        time.Time              `json:"updated_at"`
	StickyAssignment bool                   `json:"sticky_assignment"` // Consistent user assignment
}

// ModelVariant represents a model in an A/B test
type ModelVariant struct {
	ModelID     string  `json:"model_id"`
	Name        string  `json:"name"`
	Version     string  `json:"version"`
	IsControl   bool    `json:"is_control"`
	TrafficPct  float64 `json:"traffic_pct"`
	Description string  `json:"description"`
}

// MetricData stores metrics for a model variant
type MetricData struct {
	ModelID       string               `json:"model_id"`
	SampleCount   int                  `json:"sample_count"`
	Metrics       map[string]float64   `json:"metrics"`
	LastUpdated   time.Time            `json:"last_updated"`
	Distributions map[string][]float64 `json:"distributions"` // For statistical testing
}

// ExperimentStatus represents the state of an experiment
type ExperimentStatus string

const (
	ExperimentStatusDraft    ExperimentStatus = "draft"
	ExperimentStatusRunning  ExperimentStatus = "running"
	ExperimentStatusPaused   ExperimentStatus = "paused"
	ExperimentStatusComplete ExperimentStatus = "complete"
	ExperimentStatusArchived ExperimentStatus = "archived"
)

// Assignment represents a user-to-model assignment
type Assignment struct {
	ExperimentID string    `json:"experiment_id"`
	UserID       string    `json:"user_id"`
	ModelID      string    `json:"model_id"`
	AssignedAt   time.Time `json:"assigned_at"`
}

// NewABTestManager creates a new A/B testing manager
func NewABTestManager(redisClient *redis.Client) (*ABTestManager, error) {
	manager := &ABTestManager{
		redisClient: redisClient,
		experiments: make(map[string]*Experiment),
	}

	// Load active experiments from Redis
	if err := manager.loadExperiments(context.Background()); err != nil {
		return nil, fmt.Errorf("failed to load experiments: %w", err)
	}

	return manager, nil
}

// Prometheus metrics
var (
	abAssignments = prometheus.NewCounterVec(
		prometheus.CounterOpts{Namespace: "ml", Subsystem: "ab", Name: "assignments_total", Help: "Total user-to-model assignments."},
		[]string{"experiment_id", "model_id"},
	)
	abMetricsRecorded = prometheus.NewCounterVec(
		prometheus.CounterOpts{Namespace: "ml", Subsystem: "ab", Name: "metrics_recorded_total", Help: "Total metrics recorded for variants."},
		[]string{"experiment_id", "model_id", "metric"},
	)
	abExperimentsStarted = prometheus.NewCounterVec(
		prometheus.CounterOpts{Namespace: "ml", Subsystem: "ab", Name: "experiments_started_total", Help: "Experiments started."},
		[]string{"experiment_id"},
	)
	abExperimentsCompleted = prometheus.NewCounterVec(
		prometheus.CounterOpts{Namespace: "ml", Subsystem: "ab", Name: "experiments_completed_total", Help: "Experiments completed."},
		[]string{"experiment_id"},
	)
)

func init() {
	_ = prometheus.Register(abAssignments)
	_ = prometheus.Register(abMetricsRecorded)
	_ = prometheus.Register(abExperimentsStarted)
	_ = prometheus.Register(abExperimentsCompleted)
}

// CreateExperiment creates a new A/B test experiment
func (ab *ABTestManager) CreateExperiment(ctx context.Context, experiment *Experiment) error {
	ab.mu.Lock()
	defer ab.mu.Unlock()

	// Validate traffic split
	if err := validateTrafficSplit(experiment.TrafficSplit); err != nil {
		return fmt.Errorf("invalid traffic split: %w", err)
	}

	// Set timestamps
	now := time.Now()
	experiment.CreatedAt = now
	experiment.UpdatedAt = now
	experiment.Status = ExperimentStatusDraft

	// Initialize metrics
	experiment.Metrics = make(map[string]*MetricData)
	for _, variant := range experiment.Models {
		experiment.Metrics[variant.ModelID] = &MetricData{
			ModelID:       variant.ModelID,
			SampleCount:   0,
			Metrics:       make(map[string]float64),
			Distributions: make(map[string][]float64),
			LastUpdated:   now,
		}
	}

	// Store in memory and Redis
	ab.experiments[experiment.ID] = experiment
	return ab.persistExperiment(ctx, experiment)
}

// GetModelAssignment assigns a user to a model variant
func (ab *ABTestManager) GetModelAssignment(ctx context.Context, experimentID, userID string) (string, error) {
	ab.mu.RLock()
	experiment, exists := ab.experiments[experimentID]
	ab.mu.RUnlock()

	if !exists {
		return "", fmt.Errorf("experiment not found: %s", experimentID)
	}

	if experiment.Status != ExperimentStatusRunning {
		return "", fmt.Errorf("experiment not running: %s", experiment.Status)
	}

	// Check for sticky assignment
	if experiment.StickyAssignment {
		if assignment, err := ab.getStickyAssignment(ctx, experimentID, userID); err == nil {
			return assignment.ModelID, nil
		}
	}

	// Assign based on traffic split
	modelID := ab.assignModel(experiment, userID)

	// Store assignment if sticky
	if experiment.StickyAssignment {
		assignment := &Assignment{
			ExperimentID: experimentID,
			UserID:       userID,
			ModelID:      modelID,
			AssignedAt:   time.Now(),
		}
		ab.storeStickyAssignment(ctx, assignment)
	}
	// Metrics
	abAssignments.WithLabelValues(experimentID, modelID).Inc()

	return modelID, nil
}

// RecordMetric records a metric value for a model variant
func (ab *ABTestManager) RecordMetric(ctx context.Context, experimentID, modelID, metricName string, value float64) error {
	ab.mu.Lock()
	defer ab.mu.Unlock()

	experiment, exists := ab.experiments[experimentID]
	if !exists {
		return fmt.Errorf("experiment not found: %s", experimentID)
	}

	metricData, exists := experiment.Metrics[modelID]
	if !exists {
		return fmt.Errorf("model not found in experiment: %s", modelID)
	}

	// Update metrics
	metricData.SampleCount++

	// Running average
	currentValue := metricData.Metrics[metricName]
	n := float64(metricData.SampleCount)
	newValue := (currentValue*(n-1) + value) / n
	metricData.Metrics[metricName] = newValue

	// Store individual values for distribution (limited to last 1000)
	distribution := metricData.Distributions[metricName]
	distribution = append(distribution, value)
	if len(distribution) > 1000 {
		distribution = distribution[len(distribution)-1000:]
	}
	metricData.Distributions[metricName] = distribution

	metricData.LastUpdated = time.Now()
	experiment.UpdatedAt = time.Now()

	// Persist to Redis
	abMetricsRecorded.WithLabelValues(experimentID, modelID, metricName).Inc()
	return ab.persistExperiment(ctx, experiment)
}

// StartExperiment starts an experiment
func (ab *ABTestManager) StartExperiment(ctx context.Context, experimentID string) error {
	ab.mu.Lock()
	defer ab.mu.Unlock()

	experiment, exists := ab.experiments[experimentID]
	if !exists {
		return fmt.Errorf("experiment not found: %s", experimentID)
	}

	if experiment.Status != ExperimentStatusDraft && experiment.Status != ExperimentStatusPaused {
		return fmt.Errorf("cannot start experiment in status: %s", experiment.Status)
	}

	experiment.Status = ExperimentStatusRunning
	if experiment.StartDate.IsZero() {
		experiment.StartDate = time.Now()
	}
	experiment.UpdatedAt = time.Now()

	if err := ab.persistExperiment(ctx, experiment); err != nil {
		return err
	}
	abExperimentsStarted.WithLabelValues(experimentID).Inc()
	return nil
}

// StopExperiment stops an experiment
func (ab *ABTestManager) StopExperiment(ctx context.Context, experimentID string) error {
	ab.mu.Lock()
	defer ab.mu.Unlock()

	experiment, exists := ab.experiments[experimentID]
	if !exists {
		return fmt.Errorf("experiment not found: %s", experimentID)
	}

	experiment.Status = ExperimentStatusComplete
	experiment.EndDate = time.Now()
	experiment.UpdatedAt = time.Now()

	// Determine winner based on target metric
	winner := ab.determineWinner(experiment)
	experiment.WinnerModelID = winner

	if err := ab.persistExperiment(ctx, experiment); err != nil {
		return err
	}
	abExperimentsCompleted.WithLabelValues(experimentID).Inc()
	return nil
}

// GetExperimentResults returns the current results of an experiment
func (ab *ABTestManager) GetExperimentResults(ctx context.Context, experimentID string) (*Experiment, error) {
	ab.mu.RLock()
	defer ab.mu.RUnlock()

	experiment, exists := ab.experiments[experimentID]
	if !exists {
		return nil, fmt.Errorf("experiment not found: %s", experimentID)
	}

	return experiment, nil
}

// ListExperiments returns all experiments
func (ab *ABTestManager) ListExperiments(status ExperimentStatus) []*Experiment {
	ab.mu.RLock()
	defer ab.mu.RUnlock()

	var experiments []*Experiment
	for _, exp := range ab.experiments {
		if status == "" || exp.Status == status {
			experiments = append(experiments, exp)
		}
	}

	return experiments
}

// Helper functions

func (ab *ABTestManager) assignModel(experiment *Experiment, userID string) string {
	// Hash user ID for consistent assignment
	hash := sha256.Sum256([]byte(experiment.ID + userID))
	hashInt := binary.BigEndian.Uint64(hash[:8])

	// Convert to percentage (0-100)
	percentage := float64(hashInt%10000) / 100.0

	// Assign based on traffic split
	cumulative := 0.0
	for _, variant := range experiment.Models {
		cumulative += variant.TrafficPct
		if percentage < cumulative {
			return variant.ModelID
		}
	}

	// Fallback to control
	for _, variant := range experiment.Models {
		if variant.IsControl {
			return variant.ModelID
		}
	}

	return experiment.Models[0].ModelID
}

func (ab *ABTestManager) getStickyAssignment(ctx context.Context, experimentID, userID string) (*Assignment, error) {
	key := fmt.Sprintf("ab:assignment:%s:%s", experimentID, userID)
	data, err := ab.redisClient.Get(ctx, key).Bytes()
	if err != nil {
		return nil, err
	}

	var assignment Assignment
	if err := json.Unmarshal(data, &assignment); err != nil {
		return nil, err
	}

	return &assignment, nil
}

func (ab *ABTestManager) storeStickyAssignment(ctx context.Context, assignment *Assignment) error {
	key := fmt.Sprintf("ab:assignment:%s:%s", assignment.ExperimentID, assignment.UserID)
	data, err := json.Marshal(assignment)
	if err != nil {
		return err
	}

	// Store with TTL of 30 days
	return ab.redisClient.Set(ctx, key, data, 30*24*time.Hour).Err()
}

func (ab *ABTestManager) persistExperiment(ctx context.Context, experiment *Experiment) error {
	key := fmt.Sprintf("ab:experiment:%s", experiment.ID)
	data, err := json.Marshal(experiment)
	if err != nil {
		return fmt.Errorf("failed to marshal experiment: %w", err)
	}

	return ab.redisClient.Set(ctx, key, data, 0).Err()
}

func (ab *ABTestManager) loadExperiments(ctx context.Context) error {
	keys, err := ab.redisClient.Keys(ctx, "ab:experiment:*").Result()
	if err != nil {
		return err
	}

	for _, key := range keys {
		data, err := ab.redisClient.Get(ctx, key).Bytes()
		if err != nil {
			continue
		}

		var experiment Experiment
		if err := json.Unmarshal(data, &experiment); err != nil {
			continue
		}

		ab.experiments[experiment.ID] = &experiment
	}

	return nil
}

func (ab *ABTestManager) determineWinner(experiment *Experiment) string {
	if experiment.TargetMetric == "" {
		return ""
	}

	var bestModelID string
	var bestValue float64

	for modelID, metrics := range experiment.Metrics {
		if metrics.SampleCount < experiment.MinimumSamples {
			continue
		}

		value, exists := metrics.Metrics[experiment.TargetMetric]
		if !exists {
			continue
		}

		if bestModelID == "" || value > bestValue {
			bestModelID = modelID
			bestValue = value
		}
	}

	return bestModelID
}

func validateTrafficSplit(split map[string]float64) error {
	total := 0.0
	for _, pct := range split {
		if pct < 0 || pct > 100 {
			return fmt.Errorf("invalid percentage: %.2f", pct)
		}
		total += pct
	}

	if total < 99.9 || total > 100.1 {
		return fmt.Errorf("traffic split must sum to 100%%, got %.2f%%", total)
	}

	return nil
}
