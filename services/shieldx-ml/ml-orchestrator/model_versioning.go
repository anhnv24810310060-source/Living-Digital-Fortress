//go:build experimental_model_versioning
// +build experimental_model_versioning

package main

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

// ModelVersion represents a versioned ML model with metadata
type ModelVersion struct {
	ID              string                 `json:"id"`
	Version         string                 `json:"version"`
	Name            string                 `json:"name"`
	Algorithm       string                 `json:"algorithm"`
	CreatedAt       time.Time              `json:"created_at"`
	Hash            string                 `json:"hash"` // SHA256 of model data
	Size            int64                  `json:"size"`
	Metrics         map[string]float64     `json:"metrics"` // accuracy, precision, recall, etc.
	Hyperparameters map[string]interface{} `json:"hyperparameters"`
	Status          string                 `json:"status"` // active, deprecated, rollback
	Tags            []string               `json:"tags"`
	ParentVersion   string                 `json:"parent_version,omitempty"`
	DataPath        string                 `json:"data_path"` // Path to serialized model
}

// ModelRegistry manages model versions with automatic backup and rollback
type ModelRegistry struct {
	models         map[string]*ModelVersion // version -> model
	activeVersions map[string]string        // model_name -> active_version
	storagePath    string
	maxVersions    int
	mu             sync.RWMutex
}

// RollbackPolicy defines how to rollback on failures
type RollbackPolicy struct {
	MaxFailureRate   float64       // Trigger rollback if failure rate exceeds this
	EvaluationWindow time.Duration // Time window for failure evaluation
	MinRequests      int           // Minimum requests before considering rollback
}

// CanaryDeployment manages gradual model rollout
type CanaryDeployment struct {
	NewVersion        string
	OldVersion        string
	TrafficSplit      float64 // 0.0 to 1.0 - fraction to new version
	StartTime         time.Time
	MetricsNew        *DeploymentMetrics
	MetricsOld        *DeploymentMetrics
	PromotionCriteria PromotionCriteria
	Status            string // active, promoted, rolled_back
	mu                sync.RWMutex
}

type DeploymentMetrics struct {
	Requests    int
	Errors      int
	AvgLatency  float64
	P95Latency  float64
	P99Latency  float64
	Accuracy    float64
	LastUpdated time.Time
	latencies   []float64
	mu          sync.Mutex
}

type PromotionCriteria struct {
	MinRequests      int
	MaxErrorRate     float64
	MaxLatencyP95    float64
	MinAccuracy      float64
	RequiredDuration time.Duration
}

// NewModelRegistry creates a new registry with persistence
func NewModelRegistry(storagePath string, maxVersions int) (*ModelRegistry, error) {
	if err := os.MkdirAll(storagePath, 0755); err != nil {
		return nil, fmt.Errorf("failed to create storage path: %w", err)
	}

	registry := &ModelRegistry{
		models:         make(map[string]*ModelVersion),
		activeVersions: make(map[string]string),
		storagePath:    storagePath,
		maxVersions:    maxVersions,
	}

	// Load existing models from disk
	if err := registry.loadFromDisk(); err != nil {
		return nil, fmt.Errorf("failed to load models: %w", err)
	}

	return registry, nil
}

// RegisterModel adds a new model version to the registry
func (mr *ModelRegistry) RegisterModel(name, algorithm string, data []byte, hyperparams map[string]interface{}) (*ModelVersion, error) {
	mr.mu.Lock()
	defer mr.mu.Unlock()

	version := generateVersion()
	hash := calculateHash(data)

	// Save model data to disk
	dataPath := filepath.Join(mr.storagePath, fmt.Sprintf("%s_%s.model", name, version))
	if err := ioutil.WriteFile(dataPath, data, 0644); err != nil {
		return nil, fmt.Errorf("failed to save model data: %w", err)
	}

	model := &ModelVersion{
		ID:              generateID(),
		Version:         version,
		Name:            name,
		Algorithm:       algorithm,
		CreatedAt:       time.Now(),
		Hash:            hash,
		Size:            int64(len(data)),
		Metrics:         make(map[string]float64),
		Hyperparameters: hyperparams,
		Status:          "registered",
		Tags:            []string{},
		DataPath:        dataPath,
	}

	// Check if there's an active version - make it parent
	if activeVer, exists := mr.activeVersions[name]; exists {
		model.ParentVersion = activeVer
	}

	mr.models[version] = model

	// Save metadata
	if err := mr.saveMetadata(model); err != nil {
		return nil, fmt.Errorf("failed to save metadata: %w", err)
	}

	// Enforce max versions limit
	mr.enforceVersionLimit(name)

	return model, nil
}

// ActivateVersion sets a model version as active
func (mr *ModelRegistry) ActivateVersion(version string) error {
	mr.mu.Lock()
	defer mr.mu.Unlock()

	model, exists := mr.models[version]
	if !exists {
		return fmt.Errorf("version not found: %s", version)
	}

	// Deactivate current active version
	if currentActive, ok := mr.activeVersions[model.Name]; ok {
		if currentModel, exists := mr.models[currentActive]; exists {
			currentModel.Status = "deprecated"
			mr.saveMetadata(currentModel)
		}
	}

	model.Status = "active"
	mr.activeVersions[model.Name] = version

	return mr.saveMetadata(model)
}

// GetActiveVersion returns the currently active version for a model
func (mr *ModelRegistry) GetActiveVersion(name string) (*ModelVersion, error) {
	mr.mu.RLock()
	defer mr.mu.RUnlock()

	activeVer, exists := mr.activeVersions[name]
	if !exists {
		return nil, fmt.Errorf("no active version for model: %s", name)
	}

	model, exists := mr.models[activeVer]
	if !exists {
		return nil, fmt.Errorf("active version not found: %s", activeVer)
	}

	return model, nil
}

// LoadModelData loads the serialized model data
func (mr *ModelRegistry) LoadModelData(version string) ([]byte, error) {
	mr.mu.RLock()
	model, exists := mr.models[version]
	mr.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("version not found: %s", version)
	}

	return ioutil.ReadFile(model.DataPath)
}

// Rollback reverts to the previous version
func (mr *ModelRegistry) Rollback(name string) error {
	mr.mu.Lock()
	defer mr.mu.Unlock()

	currentActive, exists := mr.activeVersions[name]
	if !exists {
		return fmt.Errorf("no active version for: %s", name)
	}

	currentModel, exists := mr.models[currentActive]
	if !exists {
		return fmt.Errorf("current version not found")
	}

	if currentModel.ParentVersion == "" {
		return fmt.Errorf("no parent version to rollback to")
	}

	parentModel, exists := mr.models[currentModel.ParentVersion]
	if !exists {
		return fmt.Errorf("parent version not found: %s", currentModel.ParentVersion)
	}

	// Perform rollback
	currentModel.Status = "rollback"
	parentModel.Status = "active"
	mr.activeVersions[name] = parentModel.Version

	mr.saveMetadata(currentModel)
	return mr.saveMetadata(parentModel)
}

// ListVersions returns all versions for a model, sorted by creation time
func (mr *ModelRegistry) ListVersions(name string) []*ModelVersion {
	mr.mu.RLock()
	defer mr.mu.RUnlock()

	versions := make([]*ModelVersion, 0)
	for _, model := range mr.models {
		if model.Name == name {
			versions = append(versions, model)
		}
	}

	sort.Slice(versions, func(i, j int) bool {
		return versions[i].CreatedAt.After(versions[j].CreatedAt)
	})

	return versions
}

// UpdateMetrics updates performance metrics for a model version
func (mr *ModelRegistry) UpdateMetrics(version string, metrics map[string]float64) error {
	mr.mu.Lock()
	defer mr.mu.Unlock()

	model, exists := mr.models[version]
	if !exists {
		return fmt.Errorf("version not found: %s", version)
	}

	for k, v := range metrics {
		model.Metrics[k] = v
	}

	return mr.saveMetadata(model)
}

// CompareVersions returns a comparison between two model versions
func (mr *ModelRegistry) CompareVersions(version1, version2 string) (map[string]interface{}, error) {
	mr.mu.RLock()
	defer mr.mu.RUnlock()

	m1, exists := mr.models[version1]
	if !exists {
		return nil, fmt.Errorf("version1 not found: %s", version1)
	}

	m2, exists := mr.models[version2]
	if !exists {
		return nil, fmt.Errorf("version2 not found: %s", version2)
	}

	comparison := map[string]interface{}{
		"version1": map[string]interface{}{
			"version":    m1.Version,
			"created_at": m1.CreatedAt,
			"metrics":    m1.Metrics,
			"status":     m1.Status,
		},
		"version2": map[string]interface{}{
			"version":    m2.Version,
			"created_at": m2.CreatedAt,
			"metrics":    m2.Metrics,
			"status":     m2.Status,
		},
		"metric_deltas": calculateMetricDeltas(m1.Metrics, m2.Metrics),
	}

	return comparison, nil
}

// === Canary Deployment ===

func NewCanaryDeployment(oldVersion, newVersion string, criteria PromotionCriteria) *CanaryDeployment {
	return &CanaryDeployment{
		NewVersion:        newVersion,
		OldVersion:        oldVersion,
		TrafficSplit:      0.05, // Start with 5% traffic
		StartTime:         time.Now(),
		MetricsNew:        NewDeploymentMetrics(),
		MetricsOld:        NewDeploymentMetrics(),
		PromotionCriteria: criteria,
		Status:            "active",
	}
}

func NewDeploymentMetrics() *DeploymentMetrics {
	return &DeploymentMetrics{
		latencies:   make([]float64, 0, 1000),
		LastUpdated: time.Now(),
	}
}

// RecordRequest records metrics for a request
func (dm *DeploymentMetrics) RecordRequest(latency float64, isError bool, accuracy float64) {
	dm.mu.Lock()
	defer dm.mu.Unlock()

	dm.Requests++
	if isError {
		dm.Errors++
	}

	dm.latencies = append(dm.latencies, latency)

	// Keep only recent 1000 latencies
	if len(dm.latencies) > 1000 {
		dm.latencies = dm.latencies[len(dm.latencies)-1000:]
	}

	// Recalculate percentiles
	if len(dm.latencies) > 0 {
		sorted := make([]float64, len(dm.latencies))
		copy(sorted, dm.latencies)
		sort.Float64s(sorted)

		dm.AvgLatency = average(sorted)
		dm.P95Latency = percentile(sorted, 0.95)
		dm.P99Latency = percentile(sorted, 0.99)
	}

	// Update accuracy (exponential moving average)
	alpha := 0.1
	dm.Accuracy = alpha*accuracy + (1-alpha)*dm.Accuracy

	dm.LastUpdated = time.Now()
}

// ShouldPromote evaluates if the canary should be promoted
func (cd *CanaryDeployment) ShouldPromote() (bool, string) {
	cd.mu.RLock()
	defer cd.mu.RUnlock()

	criteria := cd.PromotionCriteria

	// Check minimum requests
	if cd.MetricsNew.Requests < criteria.MinRequests {
		return false, fmt.Sprintf("insufficient_requests: %d/%d", cd.MetricsNew.Requests, criteria.MinRequests)
	}

	// Check duration
	if time.Since(cd.StartTime) < criteria.RequiredDuration {
		return false, "insufficient_duration"
	}

	// Check error rate
	errorRate := float64(cd.MetricsNew.Errors) / float64(cd.MetricsNew.Requests)
	if errorRate > criteria.MaxErrorRate {
		return false, fmt.Sprintf("high_error_rate: %.2f%%", errorRate*100)
	}

	// Check latency
	if cd.MetricsNew.P95Latency > criteria.MaxLatencyP95 {
		return false, fmt.Sprintf("high_latency: %.2fms", cd.MetricsNew.P95Latency)
	}

	// Check accuracy
	if cd.MetricsNew.Accuracy < criteria.MinAccuracy {
		return false, fmt.Sprintf("low_accuracy: %.2f", cd.MetricsNew.Accuracy)
	}

	// Compare with old version
	if cd.MetricsOld.Requests > criteria.MinRequests {
		oldErrorRate := float64(cd.MetricsOld.Errors) / float64(cd.MetricsOld.Requests)
		if errorRate > oldErrorRate*1.2 { // 20% worse
			return false, "error_rate_regression"
		}

		if cd.MetricsNew.P95Latency > cd.MetricsOld.P95Latency*1.3 { // 30% slower
			return false, "latency_regression"
		}
	}

	return true, "all_criteria_met"
}

// ShouldRollback determines if the canary should be rolled back
func (cd *CanaryDeployment) ShouldRollback() (bool, string) {
	cd.mu.RLock()
	defer cd.mu.RUnlock()

	if cd.MetricsNew.Requests < 10 {
		return false, ""
	}

	// Critical error rate
	errorRate := float64(cd.MetricsNew.Errors) / float64(cd.MetricsNew.Requests)
	if errorRate > 0.5 {
		return true, "critical_error_rate"
	}

	// Critical latency
	if cd.MetricsNew.P95Latency > 5000 { // 5 seconds
		return true, "critical_latency"
	}

	// Compare with old version if available
	if cd.MetricsOld.Requests > 10 {
		oldErrorRate := float64(cd.MetricsOld.Errors) / float64(cd.MetricsOld.Requests)
		if errorRate > oldErrorRate*3.0 { // 3x worse
			return true, "error_rate_spike"
		}
	}

	return false, ""
}

// IncreaseTraffic gradually increases traffic to the new version
func (cd *CanaryDeployment) IncreaseTraffic(amount float64) {
	cd.mu.Lock()
	defer cd.mu.Unlock()

	cd.TrafficSplit += amount
	if cd.TrafficSplit > 1.0 {
		cd.TrafficSplit = 1.0
	}
}

// === Helper Functions ===

func (mr *ModelRegistry) loadFromDisk() error {
	metadataDir := filepath.Join(mr.storagePath, "metadata")
	if err := os.MkdirAll(metadataDir, 0755); err != nil {
		return err
	}

	files, err := ioutil.ReadDir(metadataDir)
	if err != nil {
		return err
	}

	for _, file := range files {
		if filepath.Ext(file.Name()) == ".json" {
			data, err := ioutil.ReadFile(filepath.Join(metadataDir, file.Name()))
			if err != nil {
				continue
			}

			var model ModelVersion
			if err := json.Unmarshal(data, &model); err != nil {
				continue
			}

			mr.models[model.Version] = &model

			if model.Status == "active" {
				mr.activeVersions[model.Name] = model.Version
			}
		}
	}

	return nil
}

func (mr *ModelRegistry) saveMetadata(model *ModelVersion) error {
	metadataDir := filepath.Join(mr.storagePath, "metadata")
	os.MkdirAll(metadataDir, 0755)

	data, err := json.MarshalIndent(model, "", "  ")
	if err != nil {
		return err
	}

	filename := filepath.Join(metadataDir, fmt.Sprintf("%s.json", model.Version))
	return ioutil.WriteFile(filename, data, 0644)
}

func (mr *ModelRegistry) enforceVersionLimit(name string) {
	versions := mr.ListVersions(name)

	if len(versions) <= mr.maxVersions {
		return
	}

	// Keep active and recent versions, delete old non-active
	toDelete := len(versions) - mr.maxVersions
	deleted := 0

	for _, v := range versions {
		if deleted >= toDelete {
			break
		}

		if v.Status != "active" && v.Status != "rollback" {
			// Delete model data
			os.Remove(v.DataPath)

			// Delete metadata
			metadataPath := filepath.Join(mr.storagePath, "metadata", fmt.Sprintf("%s.json", v.Version))
			os.Remove(metadataPath)

			// Remove from registry
			delete(mr.models, v.Version)
			deleted++
		}
	}
}

func generateVersion() string {
	return fmt.Sprintf("v%d", time.Now().Unix())
}

func generateID() string {
	return fmt.Sprintf("model-%d", time.Now().UnixNano())
}

func calculateHash(data []byte) string {
	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:])
}

func calculateMetricDeltas(m1, m2 map[string]float64) map[string]float64 {
	deltas := make(map[string]float64)

	for key, val1 := range m1 {
		if val2, exists := m2[key]; exists {
			deltas[key] = val2 - val1
		}
	}

	return deltas
}

func average(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(float64(len(sorted)-1) * p)
	return sorted[idx]
}
