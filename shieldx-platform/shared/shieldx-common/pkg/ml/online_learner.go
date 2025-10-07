package ml

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/redis/go-redis/v9"
)

// OnlineLearner manages online/incremental learning pipelines
type OnlineLearner struct {
	redisClient     *redis.Client
	modelRegistry   *ModelRegistry
	mu              sync.RWMutex
	pipelines       map[string]*LearningPipeline
	updateThreshold int // Number of samples before triggering update
}

// Prometheus metrics
var (
	olUpdates = prometheus.NewCounterVec(
		prometheus.CounterOpts{Namespace: "ml", Subsystem: "online", Name: "updates_total", Help: "Total number of model updates per pipeline."},
		[]string{"pipeline_id"},
	)
	olSamples = prometheus.NewCounterVec(
		prometheus.CounterOpts{Namespace: "ml", Subsystem: "online", Name: "samples_total", Help: "Total samples ingested per pipeline."},
		[]string{"pipeline_id"},
	)
)

func init() {
	_ = prometheus.Register(olUpdates)
	_ = prometheus.Register(olSamples)
}

// LearningPipeline represents an online learning configuration
type LearningPipeline struct {
	PipelineID      string                 `json:"pipeline_id"`
	Name            string                 `json:"name"`
	ModelID         string                 `json:"model_id"`
	Status          PipelineStatus         `json:"status"`
	Strategy        LearningStrategy       `json:"strategy"`
	UpdateFrequency time.Duration          `json:"update_frequency"`
	BatchSize       int                    `json:"batch_size"`
	MinSamples      int                    `json:"min_samples"`
	MaxBufferSize   int                    `json:"max_buffer_size"`
	LearningRate    float64                `json:"learning_rate"`
	DataBuffer      []TrainingSample       `json:"-"` // Not persisted
	LastUpdate      time.Time              `json:"last_update"`
	UpdateCount     int                    `json:"update_count"`
	Metrics         *PipelineMetrics       `json:"metrics"`
	Config          map[string]interface{} `json:"config"`
	CreatedAt       time.Time              `json:"created_at"`
	UpdatedAt       time.Time              `json:"updated_at"`
}

// TrainingSample represents a single training sample
type TrainingSample struct {
	SampleID  string                 `json:"sample_id"`
	Features  map[string]float64     `json:"features"`
	Label     float64                `json:"label"`
	Weight    float64                `json:"weight"`
	Metadata  map[string]interface{} `json:"metadata"`
	Timestamp time.Time              `json:"timestamp"`
}

// PipelineMetrics tracks pipeline performance
type PipelineMetrics struct {
	TotalSamples     int       `json:"total_samples"`
	ProcessedSamples int       `json:"processed_samples"`
	UpdateCount      int       `json:"update_count"`
	AverageLatencyMs float64   `json:"average_latency_ms"`
	LastAccuracy     float64   `json:"last_accuracy"`
	AccuracyHistory  []float64 `json:"accuracy_history"`
	ErrorRate        float64   `json:"error_rate"`
	ThroughputPerSec float64   `json:"throughput_per_sec"`
	LastMetricUpdate time.Time `json:"last_metric_update"`
}

// PipelineStatus represents the state of a learning pipeline
type PipelineStatus string

const (
	PipelineStatusActive   PipelineStatus = "active"
	PipelineStatusPaused   PipelineStatus = "paused"
	PipelineStatusTraining PipelineStatus = "training"
	PipelineStatusError    PipelineStatus = "error"
	PipelineStatusArchived PipelineStatus = "archived"
)

// LearningStrategy defines the online learning approach
type LearningStrategy string

const (
	StrategyIncremental   LearningStrategy = "incremental"    // Update on each sample
	StrategyMiniBatch     LearningStrategy = "mini_batch"     // Update on batches
	StrategyPeriodicBatch LearningStrategy = "periodic_batch" // Update periodically
	StrategyAdaptive      LearningStrategy = "adaptive"       // Dynamic based on drift
)

// UpdateResult contains the result of a model update
type UpdateResult struct {
	UpdateID         string             `json:"update_id"`
	PipelineID       string             `json:"pipeline_id"`
	OldModelID       string             `json:"old_model_id"`
	NewModelID       string             `json:"new_model_id"`
	SamplesUsed      int                `json:"samples_used"`
	TrainingTimeMs   int64              `json:"training_time_ms"`
	ImprovementScore float64            `json:"improvement_score"`
	Metrics          map[string]float64 `json:"metrics"`
	Success          bool               `json:"success"`
	Error            string             `json:"error,omitempty"`
	Timestamp        time.Time          `json:"timestamp"`
}

// NewOnlineLearner creates a new online learning manager
func NewOnlineLearner(redisClient *redis.Client, modelRegistry *ModelRegistry) (*OnlineLearner, error) {
	learner := &OnlineLearner{
		redisClient:     redisClient,
		modelRegistry:   modelRegistry,
		pipelines:       make(map[string]*LearningPipeline),
		updateThreshold: 100,
	}

	// Load existing pipelines
	if err := learner.loadPipelines(context.Background()); err != nil {
		return nil, fmt.Errorf("failed to load pipelines: %w", err)
	}

	return learner, nil
}

// CreatePipeline creates a new online learning pipeline
func (ol *OnlineLearner) CreatePipeline(ctx context.Context, pipeline *LearningPipeline) error {
	ol.mu.Lock()
	defer ol.mu.Unlock()

	// Initialize pipeline
	now := time.Now()
	pipeline.CreatedAt = now
	pipeline.UpdatedAt = now
	pipeline.Status = PipelineStatusActive
	pipeline.DataBuffer = make([]TrainingSample, 0, pipeline.MaxBufferSize)

	if pipeline.Metrics == nil {
		pipeline.Metrics = &PipelineMetrics{
			AccuracyHistory: make([]float64, 0),
		}
	}

	// Validate model exists
	if _, err := ol.modelRegistry.GetModel(ctx, pipeline.ModelID); err != nil {
		return fmt.Errorf("model not found: %w", err)
	}

	// Store pipeline
	ol.pipelines[pipeline.PipelineID] = pipeline

	return ol.persistPipeline(ctx, pipeline)
}

// AddSample adds a new training sample to the pipeline
func (ol *OnlineLearner) AddSample(ctx context.Context, pipelineID string, sample TrainingSample) error {
	ol.mu.Lock()
	defer ol.mu.Unlock()

	pipeline, exists := ol.pipelines[pipelineID]
	if !exists {
		return fmt.Errorf("pipeline not found: %s", pipelineID)
	}

	if pipeline.Status != PipelineStatusActive {
		return fmt.Errorf("pipeline not active: %s", pipeline.Status)
	}

	// Set timestamp if not provided
	if sample.Timestamp.IsZero() {
		sample.Timestamp = time.Now()
	}

	// Add to buffer
	pipeline.DataBuffer = append(pipeline.DataBuffer, sample)
	pipeline.Metrics.TotalSamples++
	olSamples.WithLabelValues(pipelineID).Inc()

	// Check if update is needed
	shouldUpdate := ol.shouldTriggerUpdate(pipeline)

	if shouldUpdate {
		// Trigger async update
		go ol.triggerModelUpdate(ctx, pipelineID)
	}

	return nil
}

// AddBatch adds multiple training samples
func (ol *OnlineLearner) AddBatch(ctx context.Context, pipelineID string, samples []TrainingSample) error {
	for _, sample := range samples {
		if err := ol.AddSample(ctx, pipelineID, sample); err != nil {
			return fmt.Errorf("failed to add sample: %w", err)
		}
	}
	return nil
}

// TriggerUpdate manually triggers a model update
func (ol *OnlineLearner) TriggerUpdate(ctx context.Context, pipelineID string) error {
	return ol.triggerModelUpdate(ctx, pipelineID)
}

func (ol *OnlineLearner) triggerModelUpdate(ctx context.Context, pipelineID string) error {
	ol.mu.Lock()
	pipeline, exists := ol.pipelines[pipelineID]
	if !exists {
		ol.mu.Unlock()
		return fmt.Errorf("pipeline not found: %s", pipelineID)
	}

	if len(pipeline.DataBuffer) < pipeline.MinSamples {
		ol.mu.Unlock()
		return fmt.Errorf("insufficient samples: %d < %d", len(pipeline.DataBuffer), pipeline.MinSamples)
	}

	// Mark as training
	pipeline.Status = PipelineStatusTraining
	ol.mu.Unlock()

	startTime := time.Now()

	// Perform model update (placeholder - actual implementation depends on ML framework)
	result := &UpdateResult{
		UpdateID:    fmt.Sprintf("update_%s_%d", pipelineID, time.Now().Unix()),
		PipelineID:  pipelineID,
		OldModelID:  pipeline.ModelID,
		SamplesUsed: len(pipeline.DataBuffer),
		Timestamp:   time.Now(),
	}

	// Simulate training
	// In production, this would call actual training code
	newModelID, metrics, err := ol.performIncrementalTraining(ctx, pipeline)

	trainingTime := time.Since(startTime).Milliseconds()
	result.TrainingTimeMs = trainingTime

	if err != nil {
		result.Success = false
		result.Error = err.Error()

		ol.mu.Lock()
		pipeline.Status = PipelineStatusError
		ol.mu.Unlock()

		return ol.storeUpdateResult(ctx, result)
	}

	result.NewModelID = newModelID
	result.Metrics = metrics
	result.Success = true

	// Update pipeline state
	ol.mu.Lock()
	pipeline.ModelID = newModelID
	pipeline.LastUpdate = time.Now()
	pipeline.UpdateCount++
	pipeline.Status = PipelineStatusActive

	// Clear buffer or keep recent samples based on strategy
	if pipeline.Strategy == StrategyIncremental {
		// Keep last 20% of samples for context
		keepSize := len(pipeline.DataBuffer) / 5
		if keepSize > 0 {
			pipeline.DataBuffer = pipeline.DataBuffer[len(pipeline.DataBuffer)-keepSize:]
		} else {
			pipeline.DataBuffer = make([]TrainingSample, 0, pipeline.MaxBufferSize)
		}
	} else {
		pipeline.DataBuffer = make([]TrainingSample, 0, pipeline.MaxBufferSize)
	}

	// Update metrics
	if accuracy, ok := metrics["accuracy"]; ok {
		pipeline.Metrics.LastAccuracy = accuracy
		pipeline.Metrics.AccuracyHistory = append(pipeline.Metrics.AccuracyHistory, accuracy)

		// Keep only last 100 accuracy values
		if len(pipeline.Metrics.AccuracyHistory) > 100 {
			pipeline.Metrics.AccuracyHistory = pipeline.Metrics.AccuracyHistory[len(pipeline.Metrics.AccuracyHistory)-100:]
		}
	}
	pipeline.Metrics.UpdateCount++
	pipeline.Metrics.ProcessedSamples += result.SamplesUsed
	pipeline.Metrics.LastMetricUpdate = time.Now()

	ol.mu.Unlock()

	// Persist pipeline state
	if err := ol.persistPipeline(ctx, pipeline); err != nil {
		return fmt.Errorf("failed to persist pipeline: %w", err)
	}

	// Store update result
	olUpdates.WithLabelValues(pipelineID).Inc()
	return ol.storeUpdateResult(ctx, result)
}

func (ol *OnlineLearner) performIncrementalTraining(ctx context.Context, pipeline *LearningPipeline) (string, map[string]float64, error) {
	// This is a placeholder for actual training logic
	// In production, this would:
	// 1. Load current model
	// 2. Perform incremental training with new samples
	// 3. Evaluate performance
	// 4. Save new model to registry
	// 5. Return new model ID and metrics

	// For now, return mock data
	newModelID := fmt.Sprintf("%s_v%d", pipeline.ModelID, pipeline.UpdateCount+1)
	metrics := map[string]float64{
		"accuracy":  0.92 + (float64(pipeline.UpdateCount) * 0.001), // Simulated improvement
		"precision": 0.90,
		"recall":    0.88,
		"f1_score":  0.89,
	}

	return newModelID, metrics, nil
}

func (ol *OnlineLearner) shouldTriggerUpdate(pipeline *LearningPipeline) bool {
	switch pipeline.Strategy {
	case StrategyIncremental:
		return len(pipeline.DataBuffer) >= pipeline.MinSamples

	case StrategyMiniBatch:
		return len(pipeline.DataBuffer) >= pipeline.BatchSize

	case StrategyPeriodicBatch:
		return time.Since(pipeline.LastUpdate) >= pipeline.UpdateFrequency &&
			len(pipeline.DataBuffer) >= pipeline.MinSamples

	case StrategyAdaptive:
		// Trigger if buffer is full or accuracy is dropping
		bufferFull := len(pipeline.DataBuffer) >= pipeline.MaxBufferSize
		accuracyDropping := ol.isAccuracyDropping(pipeline)
		return bufferFull || accuracyDropping

	default:
		return false
	}
}

func (ol *OnlineLearner) isAccuracyDropping(pipeline *LearningPipeline) bool {
	history := pipeline.Metrics.AccuracyHistory
	if len(history) < 5 {
		return false
	}

	// Check if last 3 values are consistently lower than previous average
	recent := history[len(history)-3:]
	previous := history[:len(history)-3]

	recentAvg := 0.0
	for _, v := range recent {
		recentAvg += v
	}
	recentAvg /= float64(len(recent))

	previousAvg := 0.0
	for _, v := range previous {
		previousAvg += v
	}
	previousAvg /= float64(len(previous))

	// Accuracy drop threshold: 5%
	return (previousAvg - recentAvg) > 0.05
}

// GetPipelineStatus returns the current status of a pipeline
func (ol *OnlineLearner) GetPipelineStatus(ctx context.Context, pipelineID string) (*LearningPipeline, error) {
	ol.mu.RLock()
	defer ol.mu.RUnlock()

	pipeline, exists := ol.pipelines[pipelineID]
	if !exists {
		return nil, fmt.Errorf("pipeline not found: %s", pipelineID)
	}

	return pipeline, nil
}

// ListPipelines returns all learning pipelines
func (ol *OnlineLearner) ListPipelines(status PipelineStatus) []*LearningPipeline {
	ol.mu.RLock()
	defer ol.mu.RUnlock()

	var pipelines []*LearningPipeline
	for _, pipeline := range ol.pipelines {
		if status == "" || pipeline.Status == status {
			pipelines = append(pipelines, pipeline)
		}
	}

	return pipelines
}

// PausePipeline pauses a learning pipeline
func (ol *OnlineLearner) PausePipeline(ctx context.Context, pipelineID string) error {
	ol.mu.Lock()
	defer ol.mu.Unlock()

	pipeline, exists := ol.pipelines[pipelineID]
	if !exists {
		return fmt.Errorf("pipeline not found: %s", pipelineID)
	}

	pipeline.Status = PipelineStatusPaused
	return ol.persistPipeline(ctx, pipeline)
}

// ResumePipeline resumes a paused pipeline
func (ol *OnlineLearner) ResumePipeline(ctx context.Context, pipelineID string) error {
	ol.mu.Lock()
	defer ol.mu.Unlock()

	pipeline, exists := ol.pipelines[pipelineID]
	if !exists {
		return fmt.Errorf("pipeline not found: %s", pipelineID)
	}

	pipeline.Status = PipelineStatusActive
	return ol.persistPipeline(ctx, pipeline)
}

// Helper functions

func (ol *OnlineLearner) persistPipeline(ctx context.Context, pipeline *LearningPipeline) error {
	key := fmt.Sprintf("ml:pipeline:%s", pipeline.PipelineID)
	data, err := json.Marshal(pipeline)
	if err != nil {
		return fmt.Errorf("failed to marshal pipeline: %w", err)
	}

	return ol.redisClient.Set(ctx, key, data, 0).Err()
}

func (ol *OnlineLearner) loadPipelines(ctx context.Context) error {
	keys, err := ol.redisClient.Keys(ctx, "ml:pipeline:*").Result()
	if err != nil {
		return err
	}

	for _, key := range keys {
		data, err := ol.redisClient.Get(ctx, key).Bytes()
		if err != nil {
			continue
		}

		var pipeline LearningPipeline
		if err := json.Unmarshal(data, &pipeline); err != nil {
			continue
		}

		// Initialize buffer
		pipeline.DataBuffer = make([]TrainingSample, 0, pipeline.MaxBufferSize)

		ol.pipelines[pipeline.PipelineID] = &pipeline
	}

	return nil
}

func (ol *OnlineLearner) storeUpdateResult(ctx context.Context, result *UpdateResult) error {
	key := fmt.Sprintf("ml:update:%s", result.UpdateID)
	data, err := json.Marshal(result)
	if err != nil {
		return fmt.Errorf("failed to marshal update result: %w", err)
	}

	// Store for 30 days
	return ol.redisClient.Set(ctx, key, data, 30*24*time.Hour).Err()
}
