package ml

import (
	"fmt"
	"math"
	"testing"
	"time"
)

func TestNewAutoMLEngine(t *testing.T) {
	searchSpace := []HyperparameterSpace{
		{Name: "learning_rate", Type: "float", Min: 0.001, Max: 0.1, LogScale: true},
		{Name: "batch_size", Type: "int", Min: 16, Max: 128},
	}
	
	config := AutoMLConfig{
		Strategy:  RandomSearch,
		MaxTrials: 10,
		Timeout:   1 * time.Minute,
		NWorkers:  2,
	}
	
	engine := NewAutoMLEngine(searchSpace, config)
	if engine == nil {
		t.Fatal("NewAutoMLEngine returned nil")
	}
	
	if engine.strategy != RandomSearch {
		t.Errorf("Expected strategy=random, got %s", engine.strategy)
	}
	if engine.maxTrials != 10 {
		t.Errorf("Expected maxTrials=10, got %d", engine.maxTrials)
	}
}

func TestAutoML_RandomSearch(t *testing.T) {
	searchSpace := []HyperparameterSpace{
		{Name: "x", Type: "float", Min: -5.0, Max: 5.0},
		{Name: "y", Type: "float", Min: -5.0, Max: 5.0},
	}
	
	config := AutoMLConfig{
		Strategy:  RandomSearch,
		MaxTrials: 20,
		NWorkers:  2,
	}
	
	engine := NewAutoMLEngine(searchSpace, config)
	
	// Objective: minimize (x-2)^2 + (y+1)^2 (minimum at x=2, y=-1)
	objective := func(config HyperparameterConfig) (float64, error) {
		x := config["x"].(float64)
		y := config["y"].(float64)
		
		// Minimize distance from (2, -1), convert to maximization
		dist := math.Sqrt((x-2)*(x-2) + (y+1)*(y+1))
		score := 10.0 - dist // Higher score is better
		
		return score, nil
	}
	
	bestConfig, bestScore, err := engine.Optimize(objective)
	if err != nil {
		t.Fatalf("Optimize failed: %v", err)
	}
	
	if bestConfig == nil {
		t.Fatal("Expected best config, got nil")
	}
	
	x := (*bestConfig)["x"].(float64)
	y := (*bestConfig)["y"].(float64)
	
	// Should find something close to (2, -1)
	if math.Abs(x-2) > 3 || math.Abs(y+1) > 3 {
		t.Logf("Warning: Found x=%.2f, y=%.2f, expected near (2, -1)", x, y)
	}
	
	if bestScore < 5.0 {
		t.Logf("Warning: Best score %.2f is low", bestScore)
	}
	
	stats := engine.GetStats()
	if stats["total_trials"].(int) == 0 {
		t.Error("Should have completed trials")
	}
}

func TestAutoML_GridSearch(t *testing.T) {
	searchSpace := []HyperparameterSpace{
		{Name: "param1", Type: "int", Min: 1, Max: 3},
		{Name: "param2", Type: "categorical", Options: []string{"a", "b"}},
	}
	
	config := AutoMLConfig{
		Strategy:  GridSearch,
		MaxTrials: 100,
		NWorkers:  2,
	}
	
	engine := NewAutoMLEngine(searchSpace, config)
	
	// Simple objective
	objective := func(config HyperparameterConfig) (float64, error) {
		param1 := config["param1"].(int)
		param2 := config["param2"].(string)
		
		score := float64(param1)
		if param2 == "b" {
			score += 5.0
		}
		
		return score, nil
	}
	
	bestConfig, bestScore, err := engine.Optimize(objective)
	if err != nil {
		t.Fatalf("Optimize failed: %v", err)
	}
	
	if bestConfig == nil {
		t.Fatal("Expected best config, got nil")
	}
	
	// Best should be param1=3, param2="b" with score 8.0
	param1 := (*bestConfig)["param1"].(int)
	param2 := (*bestConfig)["param2"].(string)
	
	if param1 != 3 || param2 != "b" {
		t.Errorf("Expected param1=3, param2=b, got param1=%d, param2=%s", param1, param2)
	}
	
	if math.Abs(bestScore-8.0) > 0.1 {
		t.Errorf("Expected score=8.0, got %.2f", bestScore)
	}
}

func TestAutoML_BayesianSearch(t *testing.T) {
	searchSpace := []HyperparameterSpace{
		{Name: "x", Type: "float", Min: -5.0, Max: 5.0},
		{Name: "y", Type: "float", Min: -5.0, Max: 5.0},
	}
	
	config := AutoMLConfig{
		Strategy:  BayesianSearch,
		MaxTrials: 30,
		NWorkers:  1, // Sequential for Bayesian
	}
	
	engine := NewAutoMLEngine(searchSpace, config)
	
	// Objective: minimize (x-1)^2 + (y-1)^2
	objective := func(config HyperparameterConfig) (float64, error) {
		x := config["x"].(float64)
		y := config["y"].(float64)
		
		dist := math.Sqrt((x-1)*(x-1) + (y-1)*(y-1))
		score := 10.0 - dist
		
		return score, nil
	}
	
	bestConfig, bestScore, err := engine.Optimize(objective)
	if err != nil {
		t.Fatalf("Optimize failed: %v", err)
	}
	
	if bestConfig == nil {
		t.Fatal("Expected best config, got nil")
	}
	
	x := (*bestConfig)["x"].(float64)
	y := (*bestConfig)["y"].(float64)
	
	// Bayesian should converge better than random
	if math.Abs(x-1) > 2 || math.Abs(y-1) > 2 {
		t.Logf("Warning: Bayesian found x=%.2f, y=%.2f, expected near (1, 1)", x, y)
	}
	
	if bestScore < 7.0 {
		t.Logf("Warning: Best score %.2f could be better", bestScore)
	}
}

func TestAutoML_IntParameters(t *testing.T) {
	searchSpace := []HyperparameterSpace{
		{Name: "n_estimators", Type: "int", Min: 10, Max: 100},
		{Name: "max_depth", Type: "int", Min: 3, Max: 10},
	}
	
	config := AutoMLConfig{
		Strategy:  RandomSearch,
		MaxTrials: 10,
		NWorkers:  2,
	}
	
	engine := NewAutoMLEngine(searchSpace, config)
	
	objective := func(config HyperparameterConfig) (float64, error) {
		nEst := config["n_estimators"].(int)
		maxDepth := config["max_depth"].(int)
		
		// Dummy score
		score := float64(nEst)/10.0 + float64(maxDepth)
		return score, nil
	}
	
	bestConfig, bestScore, err := engine.Optimize(objective)
	if err != nil {
		t.Fatalf("Optimize failed: %v", err)
	}
	
	if bestConfig == nil {
		t.Fatal("Expected best config, got nil")
	}
	
	if bestScore <= 0 {
		t.Errorf("Expected positive score, got %.2f", bestScore)
	}
}

func TestAutoML_CategoricalParameters(t *testing.T) {
	searchSpace := []HyperparameterSpace{
		{Name: "algorithm", Type: "categorical", Options: []string{"lof", "isolation_forest", "one_class_svm"}},
		{Name: "kernel", Type: "categorical", Options: []string{"rbf", "linear", "poly"}},
	}
	
	config := AutoMLConfig{
		Strategy:  RandomSearch,
		MaxTrials: 15,
		NWorkers:  2,
	}
	
	engine := NewAutoMLEngine(searchSpace, config)
	
	objective := func(config HyperparameterConfig) (float64, error) {
		algo := config["algorithm"].(string)
		kernel := config["kernel"].(string)
		
		score := 1.0
		if algo == "one_class_svm" {
			score += 5.0
		}
		if kernel == "rbf" {
			score += 3.0
		}
		
		return score, nil
	}
	
	bestConfig, bestScore, err := engine.Optimize(objective)
	if err != nil {
		t.Fatalf("Optimize failed: %v", err)
	}
	
	algo := (*bestConfig)["algorithm"].(string)
	kernel := (*bestConfig)["kernel"].(string)
	
	// Best should be one_class_svm + rbf = 9.0
	if algo != "one_class_svm" || kernel != "rbf" {
		t.Logf("Warning: Expected one_class_svm+rbf, got %s+%s", algo, kernel)
	}
	
	if bestScore < 8.0 {
		t.Logf("Warning: Best score %.2f, expected 9.0", bestScore)
	}
}

func TestAutoML_LogScaleParameter(t *testing.T) {
	searchSpace := []HyperparameterSpace{
		{Name: "learning_rate", Type: "float", Min: 0.0001, Max: 1.0, LogScale: true},
	}
	
	config := AutoMLConfig{
		Strategy:  RandomSearch,
		MaxTrials: 20,
		NWorkers:  1,
	}
	
	engine := NewAutoMLEngine(searchSpace, config)
	
	objective := func(config HyperparameterConfig) (float64, error) {
		lr := config["learning_rate"].(float64)
		
		// Optimal around 0.01
		score := 10.0 - math.Abs(math.Log10(lr)+2.0)
		return score, nil
	}
	
	bestConfig, bestScore, err := engine.Optimize(objective)
	if err != nil {
		t.Fatalf("Optimize failed: %v", err)
	}
	
	lr := (*bestConfig)["learning_rate"].(float64)
	
	if lr < 0.0001 || lr > 1.0 {
		t.Errorf("Learning rate %.6f out of bounds", lr)
	}
	
	if bestScore <= 0 {
		t.Errorf("Expected positive score, got %.2f", bestScore)
	}
}

func TestAutoML_ErrorHandling(t *testing.T) {
	searchSpace := []HyperparameterSpace{
		{Name: "param", Type: "float", Min: 0.0, Max: 10.0},
	}
	
	config := AutoMLConfig{
		Strategy:  RandomSearch,
		MaxTrials: 10,
		NWorkers:  2,
	}
	
	engine := NewAutoMLEngine(searchSpace, config)
	
	// Objective that sometimes fails
	objective := func(config HyperparameterConfig) (float64, error) {
		param := config["param"].(float64)
		
		if param < 2.0 {
			return 0, fmt.Errorf("param too small")
		}
		
		return param, nil
	}
	
	bestConfig, _, err := engine.Optimize(objective)
	
	// Should still find some successful trials
	if err != nil {
		t.Fatalf("Optimize failed: %v", err)
	}
	
	if bestConfig == nil {
		t.Fatal("Expected best config despite some errors")
	}
	
	stats := engine.GetStats()
	successRate := float64(stats["successful_trials"].(int)) / float64(stats["total_trials"].(int))
	
	if successRate < 0.1 {
		t.Logf("Warning: Low success rate: %.1f%%", successRate*100)
	}
}

func TestAutoML_Timeout(t *testing.T) {
	searchSpace := []HyperparameterSpace{
		{Name: "param", Type: "float", Min: 0.0, Max: 10.0},
	}
	
	config := AutoMLConfig{
		Strategy:  RandomSearch,
		MaxTrials: 1000, // High number
		Timeout:   100 * time.Millisecond, // Short timeout
		NWorkers:  2,
	}
	
	engine := NewAutoMLEngine(searchSpace, config)
	
	objective := func(config HyperparameterConfig) (float64, error) {
		time.Sleep(10 * time.Millisecond) // Slow evaluation
		return config["param"].(float64), nil
	}
	
	start := time.Now()
	_, _, err := engine.Optimize(objective)
	duration := time.Since(start)
	
	if err != nil {
		t.Fatalf("Optimize failed: %v", err)
	}
	
	// Should respect timeout (with some margin)
	if duration > 500*time.Millisecond {
		t.Errorf("Timeout not respected: took %v", duration)
	}
	
	stats := engine.GetStats()
	trials := stats["total_trials"].(int)
	
	// Should have completed less than 1000 trials
	if trials >= 1000 {
		t.Errorf("Should have stopped early, completed %d trials", trials)
	}
}

func TestAutoML_GetTrials(t *testing.T) {
	searchSpace := []HyperparameterSpace{
		{Name: "param", Type: "float", Min: 0.0, Max: 10.0},
	}
	
	config := AutoMLConfig{
		Strategy:  RandomSearch,
		MaxTrials: 5,
		NWorkers:  1,
	}
	
	engine := NewAutoMLEngine(searchSpace, config)
	
	objective := func(config HyperparameterConfig) (float64, error) {
		return config["param"].(float64), nil
	}
	
	engine.Optimize(objective)
	
	trials := engine.GetTrials()
	
	if len(trials) != 5 {
		t.Errorf("Expected 5 trials, got %d", len(trials))
	}
	
	// Check trials are recorded correctly
	for _, trial := range trials {
		if trial.Config == nil {
			t.Error("Trial config should not be nil")
		}
		if trial.Error != nil {
			t.Errorf("Trial error should be nil, got: %v", trial.Error)
		}
	}
}

func TestAutoML_DefaultConfig(t *testing.T) {
	searchSpace := []HyperparameterSpace{
		{Name: "param", Type: "float", Min: 0.0, Max: 10.0},
	}
	
	// Empty config should use defaults
	engine := NewAutoMLEngine(searchSpace, AutoMLConfig{})
	
	if engine.maxTrials != 100 {
		t.Errorf("Expected default maxTrials=100, got %d", engine.maxTrials)
	}
	if engine.timeout != 1*time.Hour {
		t.Errorf("Expected default timeout=1h, got %v", engine.timeout)
	}
	if engine.nWorkers != 4 {
		t.Errorf("Expected default nWorkers=4, got %d", engine.nWorkers)
	}
	if engine.strategy != RandomSearch {
		t.Errorf("Expected default strategy=random, got %s", engine.strategy)
	}
}

func BenchmarkAutoML_RandomSearch(b *testing.B) {
	searchSpace := []HyperparameterSpace{
		{Name: "x", Type: "float", Min: -10.0, Max: 10.0},
		{Name: "y", Type: "float", Min: -10.0, Max: 10.0},
		{Name: "z", Type: "float", Min: -10.0, Max: 10.0},
	}
	
	objective := func(config HyperparameterConfig) (float64, error) {
		x := config["x"].(float64)
		y := config["y"].(float64)
		z := config["z"].(float64)
		
		score := -(x*x + y*y + z*z)
		return score, nil
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		config := AutoMLConfig{
			Strategy:  RandomSearch,
			MaxTrials: 50,
			NWorkers:  4,
		}
		engine := NewAutoMLEngine(searchSpace, config)
		engine.Optimize(objective)
	}
}

func BenchmarkAutoML_GridSearch(b *testing.B) {
	searchSpace := []HyperparameterSpace{
		{Name: "x", Type: "int", Min: 1, Max: 5},
		{Name: "y", Type: "int", Min: 1, Max: 5},
	}
	
	objective := func(config HyperparameterConfig) (float64, error) {
		x := config["x"].(int)
		y := config["y"].(int)
		
		score := -float64(x*x + y*y)
		return score, nil
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		config := AutoMLConfig{
			Strategy:  GridSearch,
			MaxTrials: 100,
			NWorkers:  4,
		}
		engine := NewAutoMLEngine(searchSpace, config)
		engine.Optimize(objective)
	}
}
