package ml

import (
	"context"
	"testing"
	"time"
)

func TestNewPipeline(t *testing.T) {
	pipeline := NewPipeline(PipelineConfig{
		Name: "test-pipeline",
	})
	
	if pipeline == nil {
		t.Fatal("NewPipeline returned nil")
	}
	
	if pipeline.name != "test-pipeline" {
		t.Errorf("Pipeline name = %s, want test-pipeline", pipeline.name)
	}
	
	if pipeline.maxRetries != 3 {
		t.Errorf("Default maxRetries = %d, want 3", pipeline.maxRetries)
	}
}

func TestPipeline_AddStage(t *testing.T) {
	pipeline := NewPipeline(PipelineConfig{Name: "test"})
	
	stage := NewDataLoadStage("load", "db", 100)
	err := pipeline.AddStage(stage)
	
	if err != nil {
		t.Fatalf("AddStage failed: %v", err)
	}
	
	if len(pipeline.stages) != 1 {
		t.Errorf("Expected 1 stage, got %d", len(pipeline.stages))
	}
}

func TestPipeline_AddInvalidStage(t *testing.T) {
	pipeline := NewPipeline(PipelineConfig{Name: "test"})
	
	// Invalid stage (no source)
	stage := NewDataLoadStage("load", "", 100)
	err := pipeline.AddStage(stage)
	
	if err == nil {
		t.Error("Should fail for invalid stage")
	}
}

func TestPipeline_Execute(t *testing.T) {
	pipeline := NewPipeline(PipelineConfig{
		Name:    "test-pipeline",
		Timeout: 10 * time.Second,
	})
	
	// Add stages
	pipeline.AddStage(NewDataLoadStage("load", "db", 100))
	pipeline.AddStage(NewFeatureEngStage("features"))
	pipeline.AddStage(NewModelTrainStage("train", "model"))
	
	ctx := context.Background()
	run, err := pipeline.Execute(ctx, nil)
	
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}
	
	if run.Status != "completed" {
		t.Errorf("Run status = %s, want completed", run.Status)
	}
	
	if len(run.StageResults) != 3 {
		t.Errorf("Expected 3 stage results, got %d", len(run.StageResults))
	}
}

func TestPipeline_ExecuteTimeout(t *testing.T) {
	pipeline := NewPipeline(PipelineConfig{
		Name:       "test-pipeline",
		Timeout:    10 * time.Millisecond, // Very short timeout
		MaxRetries: 0,                     // No retries to speed up test
	})
	
	// Add slow stage
	pipeline.AddStage(NewModelTrainStage("train", "model"))
	
	ctx := context.Background()
	run, err := pipeline.Execute(ctx, nil)
	
	if err == nil {
		t.Error("Should timeout")
	}
	
	// Should be timeout or failed (depends on race condition)
	if run.Status != "timeout" && run.Status != "failed" {
		t.Errorf("Run status = %s, want timeout or failed", run.Status)
	}
}

func TestPipeline_GetLastRun(t *testing.T) {
	pipeline := NewPipeline(PipelineConfig{Name: "test"})
	pipeline.AddStage(NewDataLoadStage("load", "db", 100))
	
	if pipeline.GetLastRun() != nil {
		t.Error("GetLastRun should return nil before execution")
	}
	
	ctx := context.Background()
	pipeline.Execute(ctx, nil)
	
	lastRun := pipeline.GetLastRun()
	if lastRun == nil {
		t.Error("GetLastRun should return run after execution")
	}
}

func TestPipeline_GetRuns(t *testing.T) {
	pipeline := NewPipeline(PipelineConfig{Name: "test"})
	pipeline.AddStage(NewDataLoadStage("load", "db", 100))
	
	ctx := context.Background()
	
	// Execute multiple times
	pipeline.Execute(ctx, nil)
	pipeline.Execute(ctx, nil)
	
	runs := pipeline.GetRuns()
	if len(runs) != 2 {
		t.Errorf("Expected 2 runs, got %d", len(runs))
	}
}

func TestPipeline_Validate(t *testing.T) {
	pipeline := NewPipeline(PipelineConfig{Name: "test"})
	
	// Empty pipeline should fail
	err := pipeline.Validate()
	if err == nil {
		t.Error("Empty pipeline should fail validation")
	}
	
	// Add valid stage
	pipeline.AddStage(NewDataLoadStage("load", "db", 100))
	err = pipeline.Validate()
	if err != nil {
		t.Errorf("Valid pipeline should pass: %v", err)
	}
}

func TestDataLoadStage_Execute(t *testing.T) {
	stage := NewDataLoadStage("load", "database", 100)
	
	ctx := context.Background()
	output, err := stage.Execute(ctx, nil)
	
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}
	
	data, ok := output.(map[string]interface{})
	if !ok {
		t.Fatal("Output is not map")
	}
	
	if data["source"] != "database" {
		t.Errorf("Source = %v, want database", data["source"])
	}
}

func TestDataLoadStage_Validate(t *testing.T) {
	// Valid stage
	stage := NewDataLoadStage("load", "db", 100)
	if err := stage.Validate(); err != nil {
		t.Errorf("Valid stage failed: %v", err)
	}
	
	// Invalid: no source
	stage2 := NewDataLoadStage("load", "", 100)
	if err := stage2.Validate(); err == nil {
		t.Error("Should fail without source")
	}
	
	// Invalid: bad batch size
	stage3 := NewDataLoadStage("load", "db", 0)
	if err := stage3.Validate(); err == nil {
		t.Error("Should fail with zero batch size")
	}
}

func TestFeatureEngStage_Execute(t *testing.T) {
	stage := NewFeatureEngStage("features")
	
	input := map[string]interface{}{
		"records": 1000,
	}
	
	ctx := context.Background()
	output, err := stage.Execute(ctx, input)
	
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}
	
	data, ok := output.(map[string]interface{})
	if !ok {
		t.Fatal("Output is not map")
	}
	
	if !data["features_extracted"].(bool) {
		t.Error("Features should be extracted")
	}
}

func TestFeatureEngStage_AddExtractor(t *testing.T) {
	stage := NewFeatureEngStage("features")
	
	extractor := "mock-extractor"
	stage.AddExtractor(extractor)
	
	if len(stage.extractors) != 1 {
		t.Errorf("Expected 1 extractor, got %d", len(stage.extractors))
	}
}

func TestModelTrainStage_Execute(t *testing.T) {
	stage := NewModelTrainStage("train", "mock-model")
	
	input := map[string]interface{}{
		"features_extracted": true,
	}
	
	ctx := context.Background()
	output, err := stage.Execute(ctx, input)
	
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}
	
	data, ok := output.(map[string]interface{})
	if !ok {
		t.Fatal("Output is not map")
	}
	
	if !data["model_trained"].(bool) {
		t.Error("Model should be trained")
	}
	
	if data["accuracy"].(float64) != 0.95 {
		t.Errorf("Accuracy = %f, want 0.95", data["accuracy"])
	}
}

func TestModelTrainStage_Validate(t *testing.T) {
	// Valid stage
	stage := NewModelTrainStage("train", "model")
	if err := stage.Validate(); err != nil {
		t.Errorf("Valid stage failed: %v", err)
	}
	
	// Invalid: no model
	stage2 := NewModelTrainStage("train", nil)
	if err := stage2.Validate(); err == nil {
		t.Error("Should fail without model")
	}
	
	// Invalid: bad k-folds
	stage3 := NewModelTrainStage("train", "model")
	stage3.kFolds = 0
	if err := stage3.Validate(); err == nil {
		t.Error("Should fail with zero k-folds")
	}
}

func TestModelTrainStage_SetHyperParams(t *testing.T) {
	stage := NewModelTrainStage("train", "model")
	
	params := map[string]interface{}{
		"learning_rate": 0.01,
		"epochs":        100,
	}
	
	stage.SetHyperParams(params)
	
	if len(stage.hyperparams) != 2 {
		t.Errorf("Expected 2 hyperparams, got %d", len(stage.hyperparams))
	}
}

func TestModelEvalStage_Execute(t *testing.T) {
	stage := NewModelEvalStage("eval", []string{"accuracy", "precision"})
	
	input := map[string]interface{}{
		"model_trained": true,
	}
	
	ctx := context.Background()
	output, err := stage.Execute(ctx, input)
	
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}
	
	data, ok := output.(map[string]interface{})
	if !ok {
		t.Fatal("Output is not map")
	}
	
	if !data["evaluated"].(bool) {
		t.Error("Model should be evaluated")
	}
}

func TestModelEvalStage_ThresholdFail(t *testing.T) {
	stage := NewModelEvalStage("eval", []string{"accuracy"})
	stage.threshold = 0.99 // Very high threshold
	
	input := map[string]interface{}{
		"model_trained": true,
	}
	
	ctx := context.Background()
	_, err := stage.Execute(ctx, input)
	
	if err == nil {
		t.Error("Should fail with high threshold")
	}
}

func TestModelEvalStage_Validate(t *testing.T) {
	// Valid stage
	stage := NewModelEvalStage("eval", []string{"accuracy"})
	if err := stage.Validate(); err != nil {
		t.Errorf("Valid stage failed: %v", err)
	}
	
	// Invalid: no metrics
	stage2 := NewModelEvalStage("eval", []string{})
	if err := stage2.Validate(); err == nil {
		t.Error("Should fail without metrics")
	}
	
	// Invalid: bad test size
	stage3 := NewModelEvalStage("eval", []string{"accuracy"})
	stage3.testSize = 1.5
	if err := stage3.Validate(); err == nil {
		t.Error("Should fail with invalid test size")
	}
}

func TestModelDeployStage_Execute(t *testing.T) {
	stage := NewModelDeployStage("deploy", "production", "v1.0.0")
	
	input := map[string]interface{}{
		"evaluated": true,
	}
	
	ctx := context.Background()
	output, err := stage.Execute(ctx, input)
	
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}
	
	data, ok := output.(map[string]interface{})
	if !ok {
		t.Fatal("Output is not map")
	}
	
	if !data["deployed"].(bool) {
		t.Error("Model should be deployed")
	}
	
	if data["deployment_target"] != "production" {
		t.Errorf("Target = %v, want production", data["deployment_target"])
	}
}

func TestModelDeployStage_Validate(t *testing.T) {
	// Valid stage
	stage := NewModelDeployStage("deploy", "production", "v1.0")
	if err := stage.Validate(); err != nil {
		t.Errorf("Valid stage failed: %v", err)
	}
	
	// Invalid: no target
	stage2 := NewModelDeployStage("deploy", "", "v1.0")
	if err := stage2.Validate(); err == nil {
		t.Error("Should fail without target")
	}
	
	// Invalid: no version
	stage3 := NewModelDeployStage("deploy", "production", "")
	if err := stage3.Validate(); err == nil {
		t.Error("Should fail without version")
	}
}

func TestPipelineBuilder(t *testing.T) {
	builder := NewPipelineBuilder("test-pipeline")
	
	pipeline, err := builder.
		WithTimeout(30 * time.Second).
		WithRetries(5).
		AddStage(NewDataLoadStage("load", "db", 100)).
		AddStage(NewFeatureEngStage("features")).
		AddStage(NewModelTrainStage("train", "model")).
		Build()
	
	if err != nil {
		t.Fatalf("Build failed: %v", err)
	}
	
	if pipeline.timeout != 30*time.Second {
		t.Errorf("Timeout = %v, want 30s", pipeline.timeout)
	}
	
	if pipeline.maxRetries != 5 {
		t.Errorf("MaxRetries = %d, want 5", pipeline.maxRetries)
	}
	
	if len(pipeline.stages) != 3 {
		t.Errorf("Expected 3 stages, got %d", len(pipeline.stages))
	}
}

func TestPipelineBuilder_InvalidStage(t *testing.T) {
	builder := NewPipelineBuilder("test")
	
	_, err := builder.
		AddStage(NewDataLoadStage("load", "", 100)). // Invalid
		Build()
	
	if err == nil {
		t.Error("Should fail with invalid stage")
	}
}

func TestPipelineBuilder_EmptyPipeline(t *testing.T) {
	builder := NewPipelineBuilder("test")
	
	_, err := builder.Build()
	
	if err == nil {
		t.Error("Should fail with empty pipeline")
	}
}

func TestPipeline_FullWorkflow(t *testing.T) {
	// Create complete ML pipeline
	pipeline, err := NewPipelineBuilder("ml-workflow").
		WithTimeout(5 * time.Second).
		WithRetries(2).
		AddStage(NewDataLoadStage("load", "database", 1000)).
		AddStage(NewFeatureEngStage("feature-extraction")).
		AddStage(NewModelTrainStage("training", "isolation-forest")).
		AddStage(NewModelEvalStage("evaluation", []string{"accuracy", "f1"})).
		AddStage(NewModelDeployStage("deployment", "staging", "v1.0.0")).
		Build()
	
	if err != nil {
		t.Fatalf("Pipeline build failed: %v", err)
	}
	
	// Execute pipeline
	ctx := context.Background()
	run, err := pipeline.Execute(ctx, nil)
	
	if err != nil {
		t.Fatalf("Pipeline execution failed: %v", err)
	}
	
	if run.Status != "completed" {
		t.Errorf("Run status = %s, want completed", run.Status)
	}
	
	// Verify all stages executed
	expectedStages := []string{
		"load",
		"feature-extraction",
		"training",
		"evaluation",
		"deployment",
	}
	
	for _, stageName := range expectedStages {
		if result, ok := run.StageResults[stageName]; !ok {
			t.Errorf("Stage %s not executed", stageName)
		} else if result.Status != "completed" {
			t.Errorf("Stage %s status = %s, want completed", stageName, result.Status)
		}
	}
	
	// Check pipeline metrics
	metrics := pipeline.GetMetrics()
	if metrics == nil {
		t.Error("GetMetrics should return metrics")
	}
}

func TestStageResult_Duration(t *testing.T) {
	stage := NewDataLoadStage("load", "db", 100)
	
	ctx := context.Background()
	start := time.Now()
	stage.Execute(ctx, nil)
	duration := time.Since(start)
	
	if duration < 50*time.Millisecond {
		t.Errorf("Duration too short: %v", duration)
	}
}

func TestPipeline_ConcurrentExecution(t *testing.T) {
	pipeline := NewPipeline(PipelineConfig{Name: "concurrent-test"})
	pipeline.AddStage(NewDataLoadStage("load", "db", 100))
	
	ctx := context.Background()
	
	// Execute multiple times concurrently
	done := make(chan bool, 3)
	
	for i := 0; i < 3; i++ {
		go func() {
			_, err := pipeline.Execute(ctx, nil)
			if err != nil {
				t.Errorf("Concurrent execution failed: %v", err)
			}
			done <- true
		}()
	}
	
	// Wait for all executions
	for i := 0; i < 3; i++ {
		<-done
	}
	
	runs := pipeline.GetRuns()
	if len(runs) != 3 {
		t.Errorf("Expected 3 runs, got %d", len(runs))
	}
}

func BenchmarkPipeline_Execute(b *testing.B) {
	pipeline, _ := NewPipelineBuilder("benchmark").
		AddStage(NewDataLoadStage("load", "db", 100)).
		AddStage(NewFeatureEngStage("features")).
		AddStage(NewModelTrainStage("train", "model")).
		Build()
	
	ctx := context.Background()
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := pipeline.Execute(ctx, nil)
		if err != nil {
			b.Fatalf("Execute failed: %v", err)
		}
	}
}

func BenchmarkPipelineBuilder(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := NewPipelineBuilder("bench").
			AddStage(NewDataLoadStage("load", "db", 100)).
			AddStage(NewFeatureEngStage("features")).
			Build()
		
		if err != nil {
			b.Fatalf("Build failed: %v", err)
		}
	}
}
