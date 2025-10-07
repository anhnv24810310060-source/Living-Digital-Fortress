package ml

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/redis/go-redis/v9"
)

// ModelRegistry manages ML model versions and metadata
type ModelRegistry struct {
	storageDir  string
	redisClient *redis.Client
	mu          sync.RWMutex
	models      map[string]*ModelMetadata
}

// ModelMetadata stores information about a trained model
type ModelMetadata struct {
	ModelID      string                 `json:"model_id"`
	Name         string                 `json:"name"`
	Version      string                 `json:"version"`
	Algorithm    string                 `json:"algorithm"`
	Framework    string                 `json:"framework"`
	Status       ModelStatus            `json:"status"`
	Metrics      map[string]float64     `json:"metrics"`
	Parameters   map[string]interface{} `json:"parameters"`
	Tags         []string               `json:"tags"`
	FilePath     string                 `json:"file_path"`
	FileHash     string                 `json:"file_hash"`
	FileSize     int64                  `json:"file_size"`
	TrainingData TrainingMetadata       `json:"training_data"`
	CreatedAt    time.Time              `json:"created_at"`
	UpdatedAt    time.Time              `json:"updated_at"`
	CreatedBy    string                 `json:"created_by"`
	Description  string                 `json:"description"`
}

// TrainingMetadata stores training job information
type TrainingMetadata struct {
	JobID           string    `json:"job_id"`
	DatasetVersion  string    `json:"dataset_version"`
	SampleCount     int       `json:"sample_count"`
	FeatureCount    int       `json:"feature_count"`
	TrainingTime    int64     `json:"training_time_ms"`
	ValidationScore float64   `json:"validation_score"`
	StartedAt       time.Time `json:"started_at"`
	CompletedAt     time.Time `json:"completed_at"`
}

// ModelStatus represents the lifecycle state of a model
type ModelStatus string

const (
	ModelStatusDraft      ModelStatus = "draft"
	ModelStatusTesting    ModelStatus = "testing"
	ModelStatusStaging    ModelStatus = "staging"
	ModelStatusProduction ModelStatus = "production"
	ModelStatusArchived   ModelStatus = "archived"
)

// NewModelRegistry creates a new model registry
func NewModelRegistry(storageDir string, redisClient *redis.Client) (*ModelRegistry, error) {
	if err := os.MkdirAll(storageDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create storage directory: %w", err)
	}

	registry := &ModelRegistry{
		storageDir:  storageDir,
		redisClient: redisClient,
		models:      make(map[string]*ModelMetadata),
	}

	// Load existing models from storage
	if err := registry.loadModels(); err != nil {
		return nil, fmt.Errorf("failed to load existing models: %w", err)
	}

	return registry, nil
}

// Prometheus metrics
var (
	mrModelsRegistered = prometheus.NewCounter(
		prometheus.CounterOpts{
			Namespace: "ml",
			Subsystem: "registry",
			Name:      "models_registered_total",
			Help:      "Total number of models registered in the registry.",
		},
	)

	mrModelPromotions = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: "ml",
			Subsystem: "registry",
			Name:      "model_promotions_total",
			Help:      "Total number of model status promotions.",
		},
		[]string{"from", "to"},
	)

	mrModelFileSize = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: "ml",
			Subsystem: "registry",
			Name:      "model_file_size_bytes",
			Help:      "Size of the model file in bytes.",
		},
		[]string{"model_id"},
	)
)

func init() {
	// Safe register; ignore duplicate registration in case of multiple imports
	_ = prometheus.Register(mrModelsRegistered)
	_ = prometheus.Register(mrModelPromotions)
	_ = prometheus.Register(mrModelFileSize)
}

// RegisterModel registers a new model in the registry
func (mr *ModelRegistry) RegisterModel(ctx context.Context, metadata *ModelMetadata, modelFile io.Reader) error {
	mr.mu.Lock()
	defer mr.mu.Unlock()

	// Generate model ID if not provided
	if metadata.ModelID == "" {
		metadata.ModelID = generateModelID(metadata.Name, metadata.Version)
	}

	// Set timestamps
	now := time.Now()
	metadata.CreatedAt = now
	metadata.UpdatedAt = now
	metadata.Status = ModelStatusDraft

	// Save model file
	filePath := filepath.Join(mr.storageDir, metadata.ModelID+".model")
	file, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("failed to create model file: %w", err)
	}
	defer file.Close()

	// Calculate hash while copying
	hash := sha256.New()
	multiWriter := io.MultiWriter(file, hash)

	size, err := io.Copy(multiWriter, modelFile)
	if err != nil {
		return fmt.Errorf("failed to save model file: %w", err)
	}

	metadata.FilePath = filePath
	metadata.FileHash = hex.EncodeToString(hash.Sum(nil))
	metadata.FileSize = size

	// Metrics
	mrModelsRegistered.Inc()
	mrModelFileSize.WithLabelValues(metadata.ModelID).Set(float64(size))

	// Store metadata
	mr.models[metadata.ModelID] = metadata

	// Persist to Redis for distributed access
	if err := mr.persistMetadata(ctx, metadata); err != nil {
		return fmt.Errorf("failed to persist metadata: %w", err)
	}

	// Save metadata to disk
	if err := mr.saveMetadataToDisk(metadata); err != nil {
		return fmt.Errorf("failed to save metadata to disk: %w", err)
	}

	return nil
}

// GetModel retrieves model metadata by ID
func (mr *ModelRegistry) GetModel(ctx context.Context, modelID string) (*ModelMetadata, error) {
	mr.mu.RLock()
	metadata, exists := mr.models[modelID]
	mr.mu.RUnlock()

	if exists {
		return metadata, nil
	}

	// Try to load from Redis
	return mr.loadFromRedis(ctx, modelID)
}

// ListModels returns all registered models
func (mr *ModelRegistry) ListModels(status ModelStatus) []*ModelMetadata {
	mr.mu.RLock()
	defer mr.mu.RUnlock()

	var models []*ModelMetadata
	for _, metadata := range mr.models {
		if status == "" || metadata.Status == status {
			models = append(models, metadata)
		}
	}

	return models
}

// PromoteModel promotes a model to a new status
func (mr *ModelRegistry) PromoteModel(ctx context.Context, modelID string, newStatus ModelStatus) error {
	mr.mu.Lock()
	defer mr.mu.Unlock()

	metadata, exists := mr.models[modelID]
	if !exists {
		return fmt.Errorf("model not found: %s", modelID)
	}

	// Validate status transition
	if !isValidStatusTransition(metadata.Status, newStatus) {
		return fmt.Errorf("invalid status transition: %s -> %s", metadata.Status, newStatus)
	}

	from := string(metadata.Status)
	metadata.Status = newStatus
	metadata.UpdatedAt = time.Now()

	// Persist changes
	if err := mr.persistMetadata(ctx, metadata); err != nil {
		return fmt.Errorf("failed to persist metadata: %w", err)
	}

	// Metrics
	mrModelPromotions.WithLabelValues(from, string(newStatus)).Inc()

	return mr.saveMetadataToDisk(metadata)
}

// GetModelFile returns a reader for the model file
func (mr *ModelRegistry) GetModelFile(modelID string) (io.ReadCloser, error) {
	mr.mu.RLock()
	metadata, exists := mr.models[modelID]
	mr.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("model not found: %s", modelID)
	}

	file, err := os.Open(metadata.FilePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open model file: %w", err)
	}

	return file, nil
}

// DeleteModel removes a model from the registry
func (mr *ModelRegistry) DeleteModel(ctx context.Context, modelID string) error {
	mr.mu.Lock()
	defer mr.mu.Unlock()

	metadata, exists := mr.models[modelID]
	if !exists {
		return fmt.Errorf("model not found: %s", modelID)
	}

	// Archive instead of delete if in production
	if metadata.Status == ModelStatusProduction {
		metadata.Status = ModelStatusArchived
		metadata.UpdatedAt = time.Now()
		return mr.persistMetadata(ctx, metadata)
	}

	// Delete model file
	if err := os.Remove(metadata.FilePath); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to delete model file: %w", err)
	}

	// Delete metadata file
	metadataPath := filepath.Join(mr.storageDir, metadata.ModelID+".json")
	if err := os.Remove(metadataPath); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to delete metadata file: %w", err)
	}

	// Remove from Redis
	if err := mr.redisClient.Del(ctx, redisModelKey(modelID)).Err(); err != nil {
		return fmt.Errorf("failed to delete from Redis: %w", err)
	}

	// Remove from memory
	delete(mr.models, modelID)

	return nil
}

// Helper functions

func (mr *ModelRegistry) persistMetadata(ctx context.Context, metadata *ModelMetadata) error {
	data, err := json.Marshal(metadata)
	if err != nil {
		return fmt.Errorf("failed to marshal metadata: %w", err)
	}

	key := redisModelKey(metadata.ModelID)
	if err := mr.redisClient.Set(ctx, key, data, 0).Err(); err != nil {
		return fmt.Errorf("failed to set Redis key: %w", err)
	}

	return nil
}

func (mr *ModelRegistry) loadFromRedis(ctx context.Context, modelID string) (*ModelMetadata, error) {
	key := redisModelKey(modelID)
	data, err := mr.redisClient.Get(ctx, key).Bytes()
	if err != nil {
		return nil, fmt.Errorf("model not found in Redis: %w", err)
	}

	var metadata ModelMetadata
	if err := json.Unmarshal(data, &metadata); err != nil {
		return nil, fmt.Errorf("failed to unmarshal metadata: %w", err)
	}

	// Cache in memory
	mr.mu.Lock()
	mr.models[modelID] = &metadata
	mr.mu.Unlock()

	return &metadata, nil
}

func (mr *ModelRegistry) saveMetadataToDisk(metadata *ModelMetadata) error {
	metadataPath := filepath.Join(mr.storageDir, metadata.ModelID+".json")
	data, err := json.MarshalIndent(metadata, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal metadata: %w", err)
	}

	if err := os.WriteFile(metadataPath, data, 0644); err != nil {
		return fmt.Errorf("failed to write metadata file: %w", err)
	}

	return nil
}

func (mr *ModelRegistry) loadModels() error {
	files, err := os.ReadDir(mr.storageDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("failed to read storage directory: %w", err)
	}

	for _, file := range files {
		if filepath.Ext(file.Name()) != ".json" {
			continue
		}

		metadataPath := filepath.Join(mr.storageDir, file.Name())
		data, err := os.ReadFile(metadataPath)
		if err != nil {
			continue
		}

		var metadata ModelMetadata
		if err := json.Unmarshal(data, &metadata); err != nil {
			continue
		}

		mr.models[metadata.ModelID] = &metadata
	}

	return nil
}

func generateModelID(name, version string) string {
	hash := sha256.Sum256([]byte(name + version + time.Now().String()))
	return hex.EncodeToString(hash[:])[:16]
}

func redisModelKey(modelID string) string {
	return fmt.Sprintf("ml:model:%s", modelID)
}

func isValidStatusTransition(from, to ModelStatus) bool {
	validTransitions := map[ModelStatus][]ModelStatus{
		ModelStatusDraft:      {ModelStatusTesting, ModelStatusArchived},
		ModelStatusTesting:    {ModelStatusStaging, ModelStatusDraft, ModelStatusArchived},
		ModelStatusStaging:    {ModelStatusProduction, ModelStatusTesting, ModelStatusArchived},
		ModelStatusProduction: {ModelStatusArchived},
		ModelStatusArchived:   {ModelStatusTesting},
	}

	allowed, exists := validTransitions[from]
	if !exists {
		return false
	}

	for _, status := range allowed {
		if status == to {
			return true
		}
	}

	return false
}
