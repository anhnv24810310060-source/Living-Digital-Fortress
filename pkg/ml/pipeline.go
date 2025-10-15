package ml

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// PipelineStage represents a stage in the ML pipeline
type PipelineStage interface {
	Name() string
	Execute(ctx context.Context, input interface{}) (interface{}, error)
	Validate() error
}

// Pipeline represents an ML pipeline
type Pipeline struct {
	mu sync.RWMutex
	
	name        string
	stages      []PipelineStage
	metadata    map[string]interface{}
	
	// Execution tracking
	runs        []*PipelineRun
	lastRun     *PipelineRun
	
	// Configuration
	maxRetries  int
	timeout     time.Duration
	parallel    bool
}

// PipelineRun represents a single execution of the pipeline
type PipelineRun struct {
	ID          string
	StartTime   time.Time
	EndTime     time.Time
	Status      string // "running", "completed", "failed", "timeout"
	Error       error
	
	StageResults map[string]*StageResult
	Metrics      map[string]float64
}

// StageResult represents the result of a stage execution
type StageResult struct {
	StageName   string
	StartTime   time.Time
	EndTime     time.Time
	Duration    time.Duration
	Status      string
	Error       error
	Output      interface{}
	Metrics     map[string]float64
}

// DataLoadStage loads data for the pipeline
type DataLoadStage struct {
	name       string
	source     string
	batchSize  int
	validation bool
}

// FeatureEngStage performs feature engineering
type FeatureEngStage struct {
	name        string
	extractors  []interface{} // Feature extractors
	scalers     []interface{} // Scalers
	encoders    []interface{} // Encoders
}

// ModelTrainStage trains an ML model
type ModelTrainStage struct {
	name         string
	model        interface{}
	hyperparams  map[string]interface{}
	crossVal     bool
	kFolds       int
}

// ModelEvalStage evaluates model performance
type ModelEvalStage struct {
	name      string
	metrics   []string
	testSize  float64
	threshold float64
}

// ModelDeployStage deploys the model
type ModelDeployStage struct {
	name        string
	target      string // "production", "staging", "canary"
	version     string
	rollback    bool
}

// PipelineConfig configures the pipeline
type PipelineConfig struct {
	Name       string
	MaxRetries int
	Timeout    time.Duration
	Parallel   bool
}

// NewPipeline creates a new ML pipeline
func NewPipeline(config PipelineConfig) *Pipeline {
	if config.MaxRetries <= 0 {
		config.MaxRetries = 3
	}
	if config.Timeout <= 0 {
		config.Timeout = 30 * time.Minute
	}
	
	return &Pipeline{
		name:       config.Name,
		stages:     make([]PipelineStage, 0),
		metadata:   make(map[string]interface{}),
		runs:       make([]*PipelineRun, 0),
		maxRetries: config.MaxRetries,
		timeout:    config.Timeout,
		parallel:   config.Parallel,
	}
}

// AddStage adds a stage to the pipeline
func (p *Pipeline) AddStage(stage PipelineStage) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	
	// Validate stage
	if err := stage.Validate(); err != nil {
		return fmt.Errorf("invalid stage %s: %w", stage.Name(), err)
	}
	
	p.stages = append(p.stages, stage)
	return nil
}

// Execute runs the pipeline
func (p *Pipeline) Execute(ctx context.Context, input interface{}) (*PipelineRun, error) {
	p.mu.Lock()
	runID := fmt.Sprintf("%s-%d", p.name, time.Now().Unix())
	run := &PipelineRun{
		ID:           runID,
		StartTime:    time.Now(),
		Status:       "running",
		StageResults: make(map[string]*StageResult),
		Metrics:      make(map[string]float64),
	}
	p.runs = append(p.runs, run)
	p.lastRun = run
	p.mu.Unlock()
	
	// Create timeout context
	ctx, cancel := context.WithTimeout(ctx, p.timeout)
	defer cancel()
	
	// Execute stages
	var err error
	var output interface{} = input
	
	for _, stage := range p.stages {
		select {
		case <-ctx.Done():
			run.Status = "timeout"
			run.Error = ctx.Err()
			run.EndTime = time.Now()
			return run, ctx.Err()
		default:
			output, err = p.executeStage(ctx, stage, output, run)
			if err != nil {
				run.Status = "failed"
				run.Error = err
				run.EndTime = time.Now()
				return run, err
			}
		}
	}
	
	run.Status = "completed"
	run.EndTime = time.Now()
	return run, nil
}

// executeStage executes a single stage with retry logic
func (p *Pipeline) executeStage(ctx context.Context, stage PipelineStage, input interface{}, run *PipelineRun) (interface{}, error) {
	result := &StageResult{
		StageName: stage.Name(),
		StartTime: time.Now(),
		Status:    "running",
		Metrics:   make(map[string]float64),
	}
	
	var output interface{}
	var err error
	
	// Retry logic
	for attempt := 0; attempt <= p.maxRetries; attempt++ {
		output, err = stage.Execute(ctx, input)
		
		if err == nil {
			result.Status = "completed"
			result.Output = output
			break
		}
		
		if attempt < p.maxRetries {
			time.Sleep(time.Duration(attempt+1) * time.Second)
		}
	}
	
	result.EndTime = time.Now()
	result.Duration = result.EndTime.Sub(result.StartTime)
	
	if err != nil {
		result.Status = "failed"
		result.Error = err
		run.StageResults[stage.Name()] = result
		return nil, fmt.Errorf("stage %s failed after %d retries: %w", stage.Name(), p.maxRetries, err)
	}
	
	run.StageResults[stage.Name()] = result
	return output, nil
}

// GetLastRun returns the last pipeline run
func (p *Pipeline) GetLastRun() *PipelineRun {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.lastRun
}

// GetRuns returns all pipeline runs
func (p *Pipeline) GetRuns() []*PipelineRun {
	p.mu.RLock()
	defer p.mu.RUnlock()
	
	runs := make([]*PipelineRun, len(p.runs))
	copy(runs, p.runs)
	return runs
}

// GetMetrics returns aggregated metrics from the last run
func (p *Pipeline) GetMetrics() map[string]float64 {
	p.mu.RLock()
	defer p.mu.RUnlock()
	
	if p.lastRun == nil {
		return make(map[string]float64)
	}
	
	metrics := make(map[string]float64)
	for k, v := range p.lastRun.Metrics {
		metrics[k] = v
	}
	return metrics
}

// Validate validates the entire pipeline
func (p *Pipeline) Validate() error {
	p.mu.RLock()
	defer p.mu.RUnlock()
	
	if len(p.stages) == 0 {
		return fmt.Errorf("pipeline has no stages")
	}
	
	for _, stage := range p.stages {
		if err := stage.Validate(); err != nil {
			return fmt.Errorf("stage %s validation failed: %w", stage.Name(), err)
		}
	}
	
	return nil
}

// DataLoadStage implementation

func NewDataLoadStage(name, source string, batchSize int) *DataLoadStage {
	return &DataLoadStage{
		name:       name,
		source:     source,
		batchSize:  batchSize,
		validation: true,
	}
}

func (s *DataLoadStage) Name() string {
	return s.name
}

func (s *DataLoadStage) Execute(ctx context.Context, input interface{}) (interface{}, error) {
	// Simulate data loading
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-time.After(100 * time.Millisecond):
		// Return mock data
		data := map[string]interface{}{
			"source":    s.source,
			"batchSize": s.batchSize,
			"records":   1000,
		}
		return data, nil
	}
}

func (s *DataLoadStage) Validate() error {
	if s.source == "" {
		return fmt.Errorf("data source not specified")
	}
	if s.batchSize <= 0 {
		return fmt.Errorf("invalid batch size: %d", s.batchSize)
	}
	return nil
}

// FeatureEngStage implementation

func NewFeatureEngStage(name string) *FeatureEngStage {
	return &FeatureEngStage{
		name:       name,
		extractors: make([]interface{}, 0),
		scalers:    make([]interface{}, 0),
		encoders:   make([]interface{}, 0),
	}
}

func (s *FeatureEngStage) Name() string {
	return s.name
}

func (s *FeatureEngStage) Execute(ctx context.Context, input interface{}) (interface{}, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-time.After(50 * time.Millisecond):
		// Mock feature engineering
		data, ok := input.(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("invalid input type")
		}
		
		data["features_extracted"] = true
		data["feature_count"] = 100
		return data, nil
	}
}

func (s *FeatureEngStage) Validate() error {
	return nil
}

func (s *FeatureEngStage) AddExtractor(extractor interface{}) {
	s.extractors = append(s.extractors, extractor)
}

// ModelTrainStage implementation

func NewModelTrainStage(name string, model interface{}) *ModelTrainStage {
	return &ModelTrainStage{
		name:        name,
		model:       model,
		hyperparams: make(map[string]interface{}),
		crossVal:    true,
		kFolds:      5,
	}
}

func (s *ModelTrainStage) Name() string {
	return s.name
}

func (s *ModelTrainStage) Execute(ctx context.Context, input interface{}) (interface{}, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-time.After(200 * time.Millisecond):
		// Mock training
		data, ok := input.(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("invalid input type")
		}
		
		data["model_trained"] = true
		data["accuracy"] = 0.95
		data["loss"] = 0.05
		return data, nil
	}
}

func (s *ModelTrainStage) Validate() error {
	if s.model == nil {
		return fmt.Errorf("model not specified")
	}
	if s.kFolds <= 0 {
		return fmt.Errorf("invalid k-folds: %d", s.kFolds)
	}
	return nil
}

func (s *ModelTrainStage) SetHyperParams(params map[string]interface{}) {
	s.hyperparams = params
}

// ModelEvalStage implementation

func NewModelEvalStage(name string, metrics []string) *ModelEvalStage {
	return &ModelEvalStage{
		name:      name,
		metrics:   metrics,
		testSize:  0.2,
		threshold: 0.8,
	}
}

func (s *ModelEvalStage) Name() string {
	return s.name
}

func (s *ModelEvalStage) Execute(ctx context.Context, input interface{}) (interface{}, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-time.After(100 * time.Millisecond):
		// Mock evaluation
		data, ok := input.(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("invalid input type")
		}
		
		data["evaluated"] = true
		data["test_accuracy"] = 0.93
		data["precision"] = 0.91
		data["recall"] = 0.94
		data["f1_score"] = 0.925
		
		// Check threshold
		if acc, ok := data["test_accuracy"].(float64); ok {
			if acc < s.threshold {
				return nil, fmt.Errorf("model accuracy %.2f below threshold %.2f", acc, s.threshold)
			}
		}
		
		return data, nil
	}
}

func (s *ModelEvalStage) Validate() error {
	if len(s.metrics) == 0 {
		return fmt.Errorf("no evaluation metrics specified")
	}
	if s.testSize <= 0 || s.testSize >= 1 {
		return fmt.Errorf("invalid test size: %f", s.testSize)
	}
	return nil
}

// ModelDeployStage implementation

func NewModelDeployStage(name, target, version string) *ModelDeployStage {
	return &ModelDeployStage{
		name:     name,
		target:   target,
		version:  version,
		rollback: true,
	}
}

func (s *ModelDeployStage) Name() string {
	return s.name
}

func (s *ModelDeployStage) Execute(ctx context.Context, input interface{}) (interface{}, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-time.After(150 * time.Millisecond):
		// Mock deployment
		data, ok := input.(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("invalid input type")
		}
		
		data["deployed"] = true
		data["deployment_target"] = s.target
		data["version"] = s.version
		data["deployment_time"] = time.Now().Unix()
		
		return data, nil
	}
}

func (s *ModelDeployStage) Validate() error {
	if s.target == "" {
		return fmt.Errorf("deployment target not specified")
	}
	if s.version == "" {
		return fmt.Errorf("version not specified")
	}
	return nil
}

// PipelineBuilder for fluent API
type PipelineBuilder struct {
	pipeline *Pipeline
	err      error
}

// NewPipelineBuilder creates a new pipeline builder
func NewPipelineBuilder(name string) *PipelineBuilder {
	return &PipelineBuilder{
		pipeline: NewPipeline(PipelineConfig{
			Name: name,
		}),
	}
}

func (pb *PipelineBuilder) WithTimeout(timeout time.Duration) *PipelineBuilder {
	pb.pipeline.timeout = timeout
	return pb
}

func (pb *PipelineBuilder) WithRetries(retries int) *PipelineBuilder {
	pb.pipeline.maxRetries = retries
	return pb
}

func (pb *PipelineBuilder) AddStage(stage PipelineStage) *PipelineBuilder {
	if pb.err == nil {
		pb.err = pb.pipeline.AddStage(stage)
	}
	return pb
}

func (pb *PipelineBuilder) Build() (*Pipeline, error) {
	if pb.err != nil {
		return nil, pb.err
	}
	
	if err := pb.pipeline.Validate(); err != nil {
		return nil, err
	}
	
	return pb.pipeline, nil
}
