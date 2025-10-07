package ml

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"
)

// EnhancedModelRegistry provides production-ready model versioning with A/B testing
type EnhancedModelRegistry struct {
	// Model storage
	models        map[string]*EnhancedModelVersion
	activeModel   string
	previousModel string
	mu            sync.RWMutex
	
	// Storage configuration
	storagePath   string
	maxVersions   int
	
	// A/B Testing
	abTestConfig  *ABTestConfig
	abMu          sync.RWMutex
	
	// Performance tracking
	performanceMetrics map[string]*ModelMetrics
	metricsMu          sync.RWMutex
	
	// Rollback history
	rollbackHistory []RollbackEvent
	
	// Auto-save
	autoSave      bool
	saveTicker    *time.Ticker
}

// EnhancedModelVersion represents a versioned ML model with full metadata
type EnhancedModelVersion struct {
	Version     string                 `json:"version"`
	Name        string                 `json:"name"`
	Algorithm   string                 `json:"algorithm"`
	CreatedAt   time.Time              `json:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at"`
	Status      string                 `json:"status"` // "active", "retired", "testing"
	
	// Model artifacts
	ModelData   []byte                 `json:"-"` // Serialized model
	Checksum    string                 `json:"checksum"`
	SizeBytes   int64                  `json:"size_bytes"`
	
	// Metadata
	TrainedOn   int                    `json:"trained_samples"`
	Accuracy    float64                `json:"accuracy"`
	Precision   float64                `json:"precision"`
	Recall      float64                `json:"recall"`
	F1Score     float64                `json:"f1_score"`
	
	// Configuration
	Hyperparams map[string]interface{} `json:"hyperparameters"`
	Features    []string               `json:"features"`
	
	// Deployment info
	DeployedAt  time.Time              `json:"deployed_at,omitempty"`
	DeployedBy  string                 `json:"deployed_by"`
	Environment string                 `json:"environment"` // "dev", "staging", "production"
	
	// Performance benchmarks
	InferenceTimeMs float64            `json:"inference_time_ms"`
	MemoryUsageMB   float64            `json:"memory_usage_mb"`
}

// ModelMetrics tracks real-time model performance
type ModelMetrics struct {
	Version         string
	TotalPredictions uint64
	TotalCorrect     uint64
	TotalIncorrect   uint64
	AvgLatencyMs     float64
	P95LatencyMs     float64
	P99LatencyMs     float64
	ErrorRate        float64
	LastUpdated      time.Time
	
	// Latency histogram
	latencies       []float64
	latencyMu       sync.Mutex
}

// ABTestConfig defines A/B testing parameters
type ABTestConfig struct {
	Enabled       bool      `json:"enabled"`
	ModelA        string    `json:"model_a"`        // Control version
	ModelB        string    `json:"model_b"`        // Variant version
	TrafficSplit  float64   `json:"traffic_split"`  // % to model B (0.0-1.0)
	StartedAt     time.Time `json:"started_at"`
	Duration      time.Duration `json:"duration"`
	MinSamples    int       `json:"min_samples"`    // Minimum samples before decision
	
	// Success criteria
	MinAccuracyGain  float64 `json:"min_accuracy_gain"`  // Minimum % improvement
	MaxLatencyMs     float64 `json:"max_latency_ms"`     // Maximum allowed latency
	MaxErrorRate     float64 `json:"max_error_rate"`     // Maximum allowed error rate
}

// RollbackEvent records model rollback history
type RollbackEvent struct {
	Timestamp    time.Time
	FromVersion  string
	ToVersion    string
	Reason       string
	TriggeredBy  string
	Automatic    bool
}

// ABTestResult contains A/B test evaluation results
type ABTestResult struct {
	ModelA         string
	ModelB         string
	StartedAt      time.Time
	Duration       time.Duration
	Status         string // "insufficient_samples", "model_a_wins", "model_b_wins", "inconclusive"
	Recommendation string // "continue", "promote_model_b", "keep_model_a"
	Winner         string
	
	AccuracyA      float64
	AccuracyB      float64
	AccuracyGain   float64
	LatencyA       float64
	LatencyB       float64
	ErrorRateA     float64
	ErrorRateB     float64
}

// NewEnhancedModelRegistry creates a production-ready model registry
func NewEnhancedModelRegistry(storagePath string, maxVersions int) (*EnhancedModelRegistry, error) {
	if maxVersions <= 0 {
		maxVersions = 10
	}
	
	// Create storage directory
	if err := os.MkdirAll(storagePath, 0755); err != nil {
		return nil, fmt.Errorf("failed to create storage: %w", err)
	}
	
	mr := &EnhancedModelRegistry{
		models:             make(map[string]*EnhancedModelVersion),
		storagePath:        storagePath,
		maxVersions:        maxVersions,
		performanceMetrics: make(map[string]*ModelMetrics),
		rollbackHistory:    make([]RollbackEvent, 0),
		autoSave:           true,
	}
	
	// Load existing models
	if err := mr.loadModels(); err != nil {
		return nil, fmt.Errorf("failed to load models: %w", err)
	}
	
	// Start auto-save routine
	mr.saveTicker = time.NewTicker(5 * time.Minute)
	go mr.autoSaveLoop()
	
	return mr, nil
}

// Stop gracefully shuts down the registry
func (mr *EnhancedModelRegistry) Stop() {
	if mr.saveTicker != nil {
		mr.saveTicker.Stop()
	}
	mr.saveMetadata()
}

// RegisterModel registers a new model version
func (mr *EnhancedModelRegistry) RegisterModel(model *EnhancedModelVersion) error {
	if model.Version == "" {
		model.Version = generateVersion()
	}
	
	// Calculate checksum
	model.Checksum = calculateChecksum(model.ModelData)
	model.SizeBytes = int64(len(model.ModelData))
	model.CreatedAt = time.Now()
	model.UpdatedAt = time.Now()
	model.Status = "registered"
	
	mr.mu.Lock()
	defer mr.mu.Unlock()
	
	// Check if version exists
	if _, exists := mr.models[model.Version]; exists {
		return fmt.Errorf("model version %s already exists", model.Version)
	}
	
	// Save model to disk
	if err := mr.saveModelToDisk(model); err != nil {
		return fmt.Errorf("failed to save model: %w", err)
	}
	
	mr.models[model.Version] = model
	
	// Initialize metrics
	mr.metricsMu.Lock()
	mr.performanceMetrics[model.Version] = &ModelMetrics{
		Version:     model.Version,
		LastUpdated: time.Now(),
		latencies:   make([]float64, 0, 1000),
	}
	mr.metricsMu.Unlock()
	
	// Enforce max versions limit
	mr.cleanupOldVersions()
	
	return nil
}

// ActivateModel activates a specific model version
func (mr *EnhancedModelRegistry) ActivateModel(version string, deployedBy string) error {
	mr.mu.Lock()
	defer mr.mu.Unlock()
	
	model, exists := mr.models[version]
	if !exists {
		return fmt.Errorf("model version %s not found", version)
	}
	
	// Save previous active model for potential rollback
	if mr.activeModel != "" {
		mr.previousModel = mr.activeModel
		if prevModel, ok := mr.models[mr.activeModel]; ok {
			prevModel.Status = "retired"
		}
	}
	
	// Activate new model
	model.Status = "active"
	model.DeployedAt = time.Now()
	model.DeployedBy = deployedBy
	model.Environment = "production"
	mr.activeModel = version
	
	return mr.saveMetadata()
}

// GetActiveModel returns the currently active model
func (mr *EnhancedModelRegistry) GetActiveModel() (*EnhancedModelVersion, error) {
	mr.mu.RLock()
	defer mr.mu.RUnlock()
	
	if mr.activeModel == "" {
		return nil, fmt.Errorf("no active model")
	}
	
	model, exists := mr.models[mr.activeModel]
	if !exists {
		return nil, fmt.Errorf("active model not found")
	}
	
	return model, nil
}

// Rollback rolls back to previous model version
func (mr *EnhancedModelRegistry) Rollback(reason string, triggeredBy string) error {
	mr.mu.Lock()
	defer mr.mu.Unlock()
	
	if mr.previousModel == "" {
		return fmt.Errorf("no previous model to rollback to")
	}
	
	fromVersion := mr.activeModel
	toVersion := mr.previousModel
	
	// Deactivate current model
	if currentModel, ok := mr.models[mr.activeModel]; ok {
		currentModel.Status = "retired"
	}
	
	// Activate previous model
	if prevModel, ok := mr.models[mr.previousModel]; ok {
		prevModel.Status = "active"
		prevModel.DeployedAt = time.Now()
		mr.activeModel = mr.previousModel
		mr.previousModel = ""
	}
	
	// Record rollback event
	event := RollbackEvent{
		Timestamp:   time.Now(),
		FromVersion: fromVersion,
		ToVersion:   toVersion,
		Reason:      reason,
		TriggeredBy: triggeredBy,
		Automatic:   false,
	}
	mr.rollbackHistory = append(mr.rollbackHistory, event)
	
	return mr.saveMetadata()
}

// StartABTest starts A/B testing between two models
func (mr *EnhancedModelRegistry) StartABTest(config *ABTestConfig) error {
	mr.abMu.Lock()
	defer mr.abMu.Unlock()
	
	// Validate models exist
	mr.mu.RLock()
	_, existsA := mr.models[config.ModelA]
	_, existsB := mr.models[config.ModelB]
	mr.mu.RUnlock()
	
	if !existsA {
		return fmt.Errorf("model A (%s) not found", config.ModelA)
	}
	if !existsB {
		return fmt.Errorf("model B (%s) not found", config.ModelB)
	}
	
	// Validate traffic split
	if config.TrafficSplit < 0 || config.TrafficSplit > 1 {
		return fmt.Errorf("invalid traffic split: %f (must be 0.0-1.0)", config.TrafficSplit)
	}
	
	config.Enabled = true
	config.StartedAt = time.Now()
	mr.abTestConfig = config
	
	// Update model status
	mr.mu.Lock()
	if modelB, ok := mr.models[config.ModelB]; ok {
		modelB.Status = "testing"
	}
	mr.mu.Unlock()
	
	return nil
}

// StopABTest stops the current A/B test
func (mr *EnhancedModelRegistry) StopABTest() error {
	mr.abMu.Lock()
	defer mr.abMu.Unlock()
	
	if mr.abTestConfig == nil {
		return fmt.Errorf("no active A/B test")
	}
	
	mr.abTestConfig.Enabled = false
	return nil
}

// EvaluateABTest evaluates A/B test results and recommends a winner
func (mr *EnhancedModelRegistry) EvaluateABTest() (*ABTestResult, error) {
	mr.abMu.RLock()
	config := mr.abTestConfig
	mr.abMu.RUnlock()
	
	if config == nil || !config.Enabled {
		return nil, fmt.Errorf("no active A/B test")
	}
	
	// Get metrics for both models
	mr.metricsMu.RLock()
	metricsA := mr.performanceMetrics[config.ModelA]
	metricsB := mr.performanceMetrics[config.ModelB]
	mr.metricsMu.RUnlock()
	
	result := &ABTestResult{
		ModelA:    config.ModelA,
		ModelB:    config.ModelB,
		StartedAt: config.StartedAt,
		Duration:  time.Since(config.StartedAt),
	}
	
	// Check if we have enough samples
	if metricsB.TotalPredictions < uint64(config.MinSamples) {
		result.Status = "insufficient_samples"
		result.Recommendation = "continue"
		return result, nil
	}
	
	// Calculate metrics
	accuracyA := float64(metricsA.TotalCorrect) / float64(metricsA.TotalPredictions)
	accuracyB := float64(metricsB.TotalCorrect) / float64(metricsB.TotalPredictions)
	accuracyGain := (accuracyB - accuracyA) / accuracyA
	
	result.AccuracyA = accuracyA
	result.AccuracyB = accuracyB
	result.AccuracyGain = accuracyGain
	result.LatencyA = metricsA.AvgLatencyMs
	result.LatencyB = metricsB.AvgLatencyMs
	result.ErrorRateA = metricsA.ErrorRate
	result.ErrorRateB = metricsB.ErrorRate
	
	// Decision logic
	if accuracyGain >= config.MinAccuracyGain && 
	   metricsB.AvgLatencyMs <= config.MaxLatencyMs &&
	   metricsB.ErrorRate <= config.MaxErrorRate {
		result.Status = "model_b_wins"
		result.Recommendation = "promote_model_b"
		result.Winner = config.ModelB
	} else if accuracyGain < -0.05 || 
	          metricsB.AvgLatencyMs > config.MaxLatencyMs ||
	          metricsB.ErrorRate > config.MaxErrorRate {
		result.Status = "model_a_wins"
		result.Recommendation = "keep_model_a"
		result.Winner = config.ModelA
	} else {
		result.Status = "inconclusive"
		result.Recommendation = "continue"
	}
	
	return result, nil
}

// RecordPrediction records a prediction for metrics tracking
func (mr *EnhancedModelRegistry) RecordPrediction(version string, latencyMs float64, correct bool) {
	mr.metricsMu.Lock()
	defer mr.metricsMu.Unlock()
	
	metrics, exists := mr.performanceMetrics[version]
	if !exists {
		metrics = &ModelMetrics{
			Version:   version,
			latencies: make([]float64, 0, 1000),
		}
		mr.performanceMetrics[version] = metrics
	}
	
	metrics.TotalPredictions++
	if correct {
		metrics.TotalCorrect++
	} else {
		metrics.TotalIncorrect++
	}
	
	// Update latency metrics
	metrics.latencyMu.Lock()
	metrics.latencies = append(metrics.latencies, latencyMs)
	if len(metrics.latencies) > 10000 {
		// Keep only recent 10K samples
		metrics.latencies = metrics.latencies[1000:]
	}
	metrics.latencyMu.Unlock()
	
	// Recalculate aggregates
	metrics.AvgLatencyMs = average(metrics.latencies)
	metrics.P95LatencyMs = percentile(metrics.latencies, 0.95)
	metrics.P99LatencyMs = percentile(metrics.latencies, 0.99)
	metrics.ErrorRate = float64(metrics.TotalIncorrect) / float64(metrics.TotalPredictions)
	metrics.LastUpdated = time.Now()
}

// ListModels returns all registered models
func (mr *EnhancedModelRegistry) ListModels() []*EnhancedModelVersion {
	mr.mu.RLock()
	defer mr.mu.RUnlock()
	
	models := make([]*EnhancedModelVersion, 0, len(mr.models))
	for _, model := range mr.models {
		models = append(models, model)
	}
	
	// Sort by creation time (newest first)
	sort.Slice(models, func(i, j int) bool {
		return models[i].CreatedAt.After(models[j].CreatedAt)
	})
	
	return models
}

// GetModelMetrics returns performance metrics for a model
func (mr *EnhancedModelRegistry) GetModelMetrics(version string) (*ModelMetrics, error) {
	mr.metricsMu.RLock()
	defer mr.metricsMu.RUnlock()
	
	metrics, exists := mr.performanceMetrics[version]
	if !exists {
		return nil, fmt.Errorf("metrics not found for version %s", version)
	}
	
	return metrics, nil
}

// Private helper methods

func (mr *EnhancedModelRegistry) saveModelToDisk(model *EnhancedModelVersion) error {
	filename := filepath.Join(mr.storagePath, fmt.Sprintf("%s.model", model.Version))
	return ioutil.WriteFile(filename, model.ModelData, 0644)
}

func (mr *EnhancedModelRegistry) loadModels() error {
	// Load metadata first
	metadataFile := filepath.Join(mr.storagePath, "registry_metadata.json")
	if _, err := os.Stat(metadataFile); os.IsNotExist(err) {
		return nil // No existing models
	}
	
	data, err := ioutil.ReadFile(metadataFile)
	if err != nil {
		return err
	}
	
	var metadata struct {
		Models          map[string]*EnhancedModelVersion `json:"models"`
		ActiveModel     string                           `json:"active_model"`
		PreviousModel   string                           `json:"previous_model"`
		RollbackHistory []RollbackEvent                  `json:"rollback_history"`
	}
	
	if err := json.Unmarshal(data, &metadata); err != nil {
		return err
	}
	
	mr.models = metadata.Models
	mr.activeModel = metadata.ActiveModel
	mr.previousModel = metadata.PreviousModel
	mr.rollbackHistory = metadata.RollbackHistory
	
	// Load model data from disk
	for version, model := range mr.models {
		filename := filepath.Join(mr.storagePath, fmt.Sprintf("%s.model", version))
		modelData, err := ioutil.ReadFile(filename)
		if err == nil {
			model.ModelData = modelData
		}
	}
	
	return nil
}

func (mr *EnhancedModelRegistry) saveMetadata() error {
	metadataFile := filepath.Join(mr.storagePath, "registry_metadata.json")
	
	metadata := struct {
		Models          map[string]*EnhancedModelVersion `json:"models"`
		ActiveModel     string                           `json:"active_model"`
		PreviousModel   string                           `json:"previous_model"`
		RollbackHistory []RollbackEvent                  `json:"rollback_history"`
		LastSaved       time.Time                        `json:"last_saved"`
	}{
		Models:          mr.models,
		ActiveModel:     mr.activeModel,
		PreviousModel:   mr.previousModel,
		RollbackHistory: mr.rollbackHistory,
		LastSaved:       time.Now(),
	}
	
	data, err := json.MarshalIndent(metadata, "", "  ")
	if err != nil {
		return err
	}
	
	return ioutil.WriteFile(metadataFile, data, 0644)
}

func (mr *EnhancedModelRegistry) autoSaveLoop() {
	for range mr.saveTicker.C {
		if mr.autoSave {
			mr.mu.RLock()
			mr.saveMetadata()
			mr.mu.RUnlock()
		}
	}
}

func (mr *EnhancedModelRegistry) cleanupOldVersions() {
	if len(mr.models) <= mr.maxVersions {
		return
	}
	
	// Sort by creation time
	versions := make([]string, 0, len(mr.models))
	for v := range mr.models {
		versions = append(versions, v)
	}
	
	sort.Slice(versions, func(i, j int) bool {
		return mr.models[versions[i]].CreatedAt.Before(mr.models[versions[j]].CreatedAt)
	})
	
	// Delete oldest versions (except active and previous)
	toDelete := len(mr.models) - mr.maxVersions
	for i := 0; i < toDelete; i++ {
		version := versions[i]
		if version != mr.activeModel && version != mr.previousModel {
			// Delete from disk
			filename := filepath.Join(mr.storagePath, fmt.Sprintf("%s.model", version))
			os.Remove(filename)
			
			// Delete from memory
			delete(mr.models, version)
			delete(mr.performanceMetrics, version)
		}
	}
}

// Helper functions

func generateVersion() string {
	return fmt.Sprintf("v%d", time.Now().Unix())
}

func calculateChecksum(data []byte) string {
	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:])
}

func average(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range vals {
		sum += v
	}
	return sum / float64(len(vals))
}

func percentile(vals []float64, p float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	
	sorted := make([]float64, len(vals))
	copy(sorted, vals)
	sort.Float64s(sorted)
	
	idx := int(float64(len(sorted)) * p)
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	
	return sorted[idx]
}
