# ML Package

This package provides production-grade ML management capabilities:

- Model registry with lifecycle and integrity checks
- A/B testing framework with sticky assignments
- Feature drift detection (KS + PSI)
- Online learning pipelines with multiple strategies
- Prometheus metrics for observability

Integrate by wiring Redis client and, if needed, exposing Prometheus metrics endpoint from your service.# ML Pipeline - Model Management

Production-grade ML pipeline with comprehensive model management capabilities.

## Overview

This package provides a complete ML lifecycle management system including:
- **Model Registry**: Version control and metadata management for ML models
- **A/B Testing**: Experiment framework for model comparison and gradual rollout
- **Drift Detection**: Automated monitoring for feature and prediction drift
- **Online Learning**: Incremental learning pipelines for continuous model improvement

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     ML Pipeline Manager                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │    Model     │  │   A/B Test   │  │    Drift     │      │
│  │   Registry   │  │   Manager    │  │   Detector   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           Online Learning Pipeline                   │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
    Redis Cache         Model Storage        PostgreSQL
```

## Components

### 1. Model Registry

Manages ML model versions with full metadata tracking:

**Features:**
- Model versioning and lifecycle management (draft → testing → staging → production)
- SHA256 hash verification for model integrity
- Distributed storage with Redis + local filesystem
- Model promotion workflow with status transitions
- Comprehensive metadata (metrics, parameters, training info)

**Usage:**
```go
registry := ml.NewModelRegistry("/var/ml/models", redisClient)

// Register new model
metadata := &ml.ModelMetadata{
    Name:        "threat-detector",
    Version:     "v1.2.0",
    Algorithm:   "RandomForest",
    Framework:   "scikit-learn",
    Description: "Security threat detection model",
}
err := registry.RegisterModel(ctx, metadata, modelFile)

// Promote to production
err = registry.PromoteModel(ctx, modelID, ml.ModelStatusProduction)

// Retrieve model
model, err := registry.GetModel(ctx, modelID)
```

### 2. A/B Testing Framework

Enables controlled experiments for model comparison:

**Features:**
- Traffic splitting with configurable percentages
- Sticky user assignments (consistent experience)
- Real-time metric collection
- Statistical winner determination
- Multiple variants support

**Usage:**
```go
abTest := ml.NewABTestManager(redisClient)

// Create experiment
experiment := &ml.Experiment{
    ID:           "threat-model-v2-test",
    Name:         "Threat Model V2 Comparison",
    TargetMetric: "accuracy",
    Models: []ml.ModelVariant{
        {ModelID: "v1", IsControl: true, TrafficPct: 50.0},
        {ModelID: "v2", IsControl: false, TrafficPct: 50.0},
    },
    StickyAssignment: true,
}
err := abTest.CreateExperiment(ctx, experiment)

// Start experiment
err = abTest.StartExperiment(ctx, experiment.ID)

// Get model assignment for user
modelID, err := abTest.GetModelAssignment(ctx, experimentID, userID)

// Record metrics
err = abTest.RecordMetric(ctx, experimentID, modelID, "accuracy", 0.95)

// Get results
results, err := abTest.GetExperimentResults(ctx, experimentID)
```

### 3. Drift Detection

Monitors for data distribution changes that affect model performance:

**Features:**
- Feature drift detection using KS-test and PSI
- Prediction drift monitoring
- Automatic baseline comparison
- Configurable alert thresholds
- Severity levels (low, medium, high, critical)
- Actionable recommendations

**Metrics:**
- Kolmogorov-Smirnov (KS) statistic
- Population Stability Index (PSI)
- Distribution histograms

**Usage:**
```go
detector := ml.NewDriftDetector(redisClient, 0.05, 1000)

// Set baseline from training data
baseline := map[string][]float64{
    "request_size":  trainingData["request_size"],
    "response_time": trainingData["response_time"],
}
err := detector.SetBaseline(ctx, baseline)

// Record production features
err = detector.RecordFeature(ctx, "request_size", 1024.0)

// Monitor continuously
go detector.MonitorAllFeatures(ctx)

// Set alert callback
detector.SetAlertCallback(func(alert ml.DriftAlert) {
    log.Printf("Drift detected: %s (severity: %s)", alert.Message, alert.Severity)
    // Trigger retraining pipeline
})

// Get drift reports
alerts, err := detector.GetDriftReports(ctx, time.Now().Add(-24*time.Hour))
```

### 4. Online Learning Pipeline

Enables continuous model improvement with production data:

**Features:**
- Multiple learning strategies:
  - **Incremental**: Update on each sample
  - **Mini-batch**: Update on small batches
  - **Periodic**: Update on schedule
  - **Adaptive**: Dynamic updates based on performance
- Sample buffering with configurable size
- Automatic retraining triggers
- Performance tracking
- Graceful degradation on errors

**Usage:**
```go
learner := ml.NewOnlineLearner(redisClient, modelRegistry)

// Create pipeline
pipeline := &ml.LearningPipeline{
    PipelineID:      "threat-online-v1",
    Name:            "Threat Detection Online Learning",
    ModelID:         baseModelID,
    Strategy:        ml.StrategyAdaptive,
    UpdateFrequency: 1 * time.Hour,
    BatchSize:       100,
    MinSamples:      50,
    MaxBufferSize:   1000,
    LearningRate:    0.01,
}
err := learner.CreatePipeline(ctx, pipeline)

// Add training samples
sample := ml.TrainingSample{
    SampleID: "sample_123",
    Features: map[string]float64{
        "request_size":  1024.0,
        "response_time": 150.0,
    },
    Label:  1.0, // Threat detected
    Weight: 1.0,
}
err = learner.AddSample(ctx, pipelineID, sample)

// Manual trigger
err = learner.TriggerUpdate(ctx, pipelineID)

// Monitor status
status, err := learner.GetPipelineStatus(ctx, pipelineID)
```

## Integration with Existing System

### MLflow Integration (Planned)

```go
// Export model to MLflow
mlflowClient := mlflow.NewClient(mlflowURL)
err := mlflowClient.LogModel(
    experimentName: "threat-detection",
    runName:        metadata.Version,
    modelPath:      metadata.FilePath,
    metrics:        metadata.Metrics,
    parameters:     metadata.Parameters,
)
```

### Service Integration

The ML pipeline integrates with:
- **Feature Store** (`ml-service/feature_store.py`): Provides training data
- **ML Orchestrator** (`services/ml-orchestrator`): Coordinates inference
- **Guardian Sandbox**: Provides behavioral features
- **Deception Layer**: Provides attacker behavior data

## Configuration

### Environment Variables

```bash
# Model Registry
ML_MODEL_STORAGE_DIR=/var/ml/models
ML_REDIS_URL=redis://localhost:6379

# A/B Testing
ML_AB_STICKY_ASSIGNMENT=true
ML_AB_MIN_SAMPLES=1000

# Drift Detection
ML_DRIFT_THRESHOLD=0.05
ML_DRIFT_WINDOW_SIZE=1000
ML_DRIFT_CHECK_INTERVAL=300

# Online Learning
ML_ONLINE_STRATEGY=adaptive
ML_ONLINE_BATCH_SIZE=100
ML_ONLINE_MAX_BUFFER=1000
```

### Redis Keys

- Model Registry: `ml:model:{model_id}`
- A/B Tests: `ab:experiment:{exp_id}`, `ab:assignment:{exp_id}:{user_id}`
- Drift Detection: `drift:stats:{type}:{feature}`, `drift:alert:{alert_id}`
- Online Learning: `ml:pipeline:{pipeline_id}`, `ml:update:{update_id}`

## Metrics & Monitoring

### Prometheus Metrics

```
# Model Registry
ml_models_total{status="production|staging|testing"}
ml_model_file_size_bytes{model_id}
ml_model_promotion_total{from_status, to_status}

# A/B Testing
ml_experiment_active_total
ml_experiment_traffic{experiment_id, model_id}
ml_experiment_metric{experiment_id, model_id, metric_name}

# Drift Detection
ml_drift_alerts_total{severity}
ml_drift_score{feature_name, type}

# Online Learning
ml_pipeline_updates_total{pipeline_id}
ml_pipeline_samples_processed{pipeline_id}
ml_pipeline_accuracy{pipeline_id}
```

## Best Practices

### Model Lifecycle

1. **Development**: Create model with status `draft`
2. **Testing**: Promote to `testing`, run offline evaluation
3. **Staging**: Promote to `staging`, run A/B test with 10% traffic
4. **Production**: After validation, promote to `production`
5. **Retirement**: Archive old models instead of deleting

### A/B Testing

- Start with small traffic splits (5-10%)
- Run for minimum sample size (1000+ per variant)
- Monitor both primary and secondary metrics
- Use sticky assignment for consistent UX
- Document winner selection criteria

### Drift Detection

- Set baseline from representative training data
- Use multiple detection methods (KS + PSI)
- Define clear action thresholds per severity
- Automate retraining pipelines
- Keep drift history for analysis

### Online Learning

- Start with conservative strategies (periodic batch)
- Monitor accuracy trends closely
- Implement rollback mechanisms
- Validate updated models before deployment
- Balance exploration vs exploitation

## Security Considerations

- Model files are verified with SHA256 hashes
- Access control via Redis ACLs
- Audit trail for all model promotions
- Encrypted model storage recommended
- Validation of all training samples

## Performance

- Model metadata cached in Redis for fast access
- Lazy loading of model files
- Distributed architecture supports horizontal scaling
- Background processing for expensive operations
- Configurable batch sizes for optimal throughput

## Troubleshooting

### Model Registration Fails
- Check storage directory permissions
- Verify Redis connectivity
- Ensure sufficient disk space

### Drift Alerts Too Frequent
- Increase drift threshold
- Expand window size
- Review baseline statistics

### Online Learning Not Updating
- Check buffer size and sample count
- Verify strategy configuration
- Review pipeline status for errors

## Future Enhancements

- [ ] Integration with Kubeflow for distributed training
- [ ] AutoML for hyperparameter optimization
- [ ] Model explainability (SHAP/LIME)
- [ ] Multi-armed bandit algorithms
- [ ] Federated learning support
- [ ] Model compression and quantization
- [ ] GPU acceleration for training

## References

- [MLflow Documentation](https://mlflow.org/)
- [Kubeflow Documentation](https://www.kubeflow.org/)
- [Drift Detection Methods](https://arxiv.org/abs/2004.03045)
- [Online Learning Strategies](https://www.jmlr.org/papers/volume12/dredze11a/dredze11a.pdf)
