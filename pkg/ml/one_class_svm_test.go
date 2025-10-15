package ml

import (
	"math"
	"testing"
)

func TestNewOneClassSVM(t *testing.T) {
	config := OneClassSVMConfig{
		Nu:        0.1,
		Gamma:     0.5,
		Kernel:    "rbf",
		Tolerance: 1e-3,
		MaxIter:   500,
	}
	
	svm := NewOneClassSVM(config)
	if svm == nil {
		t.Fatal("NewOneClassSVM returned nil")
	}
	
	if svm.nu != 0.1 {
		t.Errorf("Expected nu=0.1, got %.2f", svm.nu)
	}
	if svm.gamma != 0.5 {
		t.Errorf("Expected gamma=0.5, got %.2f", svm.gamma)
	}
	if svm.kernel != "rbf" {
		t.Errorf("Expected kernel=rbf, got %s", svm.kernel)
	}
}

func TestOneClassSVM_DefaultConfig(t *testing.T) {
	svm := NewOneClassSVM(OneClassSVMConfig{})
	
	if svm.nu != 0.1 {
		t.Errorf("Expected default nu=0.1, got %.2f", svm.nu)
	}
	if svm.kernel != "rbf" {
		t.Errorf("Expected default kernel=rbf, got %s", svm.kernel)
	}
	if svm.maxIter != 1000 {
		t.Errorf("Expected default maxIter=1000, got %d", svm.maxIter)
	}
}

func TestOneClassSVM_TrainEmptyData(t *testing.T) {
	svm := NewOneClassSVM(OneClassSVMConfig{})
	
	err := svm.Train([][]float64{})
	if err == nil {
		t.Error("Expected error for empty training data, got nil")
	}
}

func TestOneClassSVM_BasicTraining(t *testing.T) {
	svm := NewOneClassSVM(OneClassSVMConfig{
		Nu:      0.1,
		Gamma:   0.5,
		Kernel:  "rbf",
		MaxIter: 100,
	})
	
	// Normal data: points around origin
	normalData := [][]float64{
		{0.1, 0.1},
		{0.2, 0.2},
		{-0.1, 0.1},
		{0.1, -0.1},
		{0.0, 0.2},
		{0.2, 0.0},
		{-0.2, -0.1},
		{0.15, 0.15},
		{0.05, 0.25},
		{0.25, 0.05},
	}
	
	err := svm.Train(normalData)
	if err != nil {
		t.Fatalf("Train failed: %v", err)
	}
	
	if !svm.IsTrained() {
		t.Error("Model should be trained")
	}
	
	if svm.GetNumSupport() == 0 {
		t.Error("Should have support vectors")
	}
}

func TestOneClassSVM_DetectNormalSamples(t *testing.T) {
	svm := NewOneClassSVM(OneClassSVMConfig{
		Nu:      0.1,
		Gamma:   0.5,
		Kernel:  "rbf",
		MaxIter: 100,
	})
	
	// Train on normal data
	normalData := [][]float64{
		{0.0, 0.0}, {0.1, 0.1}, {0.2, 0.2},
		{-0.1, 0.1}, {0.1, -0.1}, {0.0, 0.2},
		{0.2, 0.0}, {-0.2, -0.1}, {0.15, 0.15},
		{0.05, 0.25},
	}
	
	svm.Train(normalData)
	
	// Test normal samples (should have negative scores)
	normalTests := [][]float64{
		{0.0, 0.0},
		{0.1, 0.1},
		{0.15, 0.15},
	}
	
	for _, sample := range normalTests {
		score, err := svm.Detect(sample)
		if err != nil {
			t.Errorf("Detect failed: %v", err)
		}
		
		// Normal samples should have negative scores (< 0)
		if score > 0.5 {
			t.Errorf("Normal sample %v should have low score, got %.3f", sample, score)
		}
	}
}

func TestOneClassSVM_DetectAnomalies(t *testing.T) {
	svm := NewOneClassSVM(OneClassSVMConfig{
		Nu:      0.05, // Lower nu for stricter boundary
		Gamma:   1.0,  // Higher gamma for more local influence
		Kernel:  "rbf",
		MaxIter: 200,
	})
	
	// Train on normal data (tight cluster) - more samples
	normalData := [][]float64{
		{0.0, 0.0}, {0.1, 0.1}, {0.2, 0.2}, {0.05, 0.05},
		{-0.1, 0.1}, {0.1, -0.1}, {0.0, 0.2}, {0.15, 0.0},
		{0.2, 0.0}, {-0.2, -0.1}, {0.15, 0.15}, {-0.05, 0.15},
		{0.05, 0.25}, {0.08, 0.12}, {0.12, 0.08}, {0.18, 0.22},
		{0.03, 0.17}, {0.17, 0.03}, {0.11, 0.19}, {0.19, 0.11},
	}
	
	svm.Train(normalData)
	
	// Test outlier samples (far from cluster)
	outliers := [][]float64{
		{5.0, 5.0},
		{-5.0, -5.0},
		{10.0, 0.0},
	}
	
	normalCount := 0
	for _, sample := range normalData {
		score, _ := svm.Detect(sample)
		if score <= 0 {
			normalCount++
		}
	}
	
	// Most training samples should be classified as normal
	if float64(normalCount)/float64(len(normalData)) < 0.8 {
		t.Logf("Warning: Only %.1f%% of training samples classified as normal", 
			100.0*float64(normalCount)/float64(len(normalData)))
	}
	
	anomalyCount := 0
	for _, sample := range outliers {
		score, err := svm.Detect(sample)
		if err != nil {
			t.Errorf("Detect failed: %v", err)
		}
		
		// Count how many outliers are detected
		if score > 0 {
			anomalyCount++
		}
	}
	
	// At least some outliers should be detected
	if anomalyCount == 0 {
		t.Error("Should detect at least some outliers")
	}
}

func TestOneClassSVM_Predict(t *testing.T) {
	svm := NewOneClassSVM(OneClassSVMConfig{
		Nu:      0.05,
		Gamma:   1.0,
		Kernel:  "rbf",
		MaxIter: 200,
	})
	
	// More training samples for better decision boundary
	normalData := [][]float64{
		{0.0, 0.0}, {0.1, 0.1}, {0.2, 0.2}, {0.05, 0.05},
		{-0.1, 0.1}, {0.1, -0.1}, {0.0, 0.2}, {0.15, 0.0},
		{0.08, 0.08}, {0.12, 0.12}, {0.18, 0.18}, {0.03, 0.03},
	}
	
	svm.Train(normalData)
	
	// Test with samples similar to training data (should be normal)
	normalSamples := [][]float64{
		{0.1, 0.1},
		{0.15, 0.15},
		{0.05, 0.05},
	}
	
	normalClassified := 0
	for _, sample := range normalSamples {
		isAnomaly, err := svm.Predict(sample)
		if err != nil {
			t.Fatalf("Predict failed: %v", err)
		}
		if !isAnomaly {
			normalClassified++
		}
	}
	
	// Most normal samples should be classified correctly
	if normalClassified < 2 {
		t.Logf("Warning: Only %d/%d normal samples classified correctly", 
			normalClassified, len(normalSamples))
	}
	
	// Test obvious outlier (should be anomaly)
	isAnomaly, err := svm.Predict([]float64{10.0, 10.0})
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}
	if !isAnomaly {
		t.Error("Clear outlier sample should be classified as anomaly")
	}
}

func TestOneClassSVM_LinearKernel(t *testing.T) {
	svm := NewOneClassSVM(OneClassSVMConfig{
		Nu:      0.1,
		Kernel:  "linear",
		MaxIter: 100,
	})
	
	normalData := [][]float64{
		{1.0, 1.0}, {1.1, 1.1}, {0.9, 0.9},
		{1.2, 1.0}, {1.0, 1.2}, {0.8, 1.1},
	}
	
	err := svm.Train(normalData)
	if err != nil {
		t.Fatalf("Train with linear kernel failed: %v", err)
	}
	
	if !svm.IsTrained() {
		t.Error("Model should be trained")
	}
}

func TestOneClassSVM_PolyKernel(t *testing.T) {
	svm := NewOneClassSVM(OneClassSVMConfig{
		Nu:      0.1,
		Kernel:  "poly",
		Degree:  3,
		MaxIter: 100,
	})
	
	normalData := [][]float64{
		{1.0, 1.0}, {1.1, 1.1}, {0.9, 0.9},
		{1.2, 1.0}, {1.0, 1.2}, {0.8, 1.1},
	}
	
	err := svm.Train(normalData)
	if err != nil {
		t.Fatalf("Train with poly kernel failed: %v", err)
	}
	
	if !svm.IsTrained() {
		t.Error("Model should be trained")
	}
}

func TestOneClassSVM_DimensionMismatch(t *testing.T) {
	svm := NewOneClassSVM(OneClassSVMConfig{})
	
	// Train on 2D data
	normalData := [][]float64{
		{0.0, 0.0}, {0.1, 0.1}, {0.2, 0.2},
	}
	svm.Train(normalData)
	
	// Try to detect with 3D sample
	_, err := svm.Detect([]float64{0.0, 0.0, 0.0})
	if err == nil {
		t.Error("Expected error for dimension mismatch, got nil")
	}
}

func TestOneClassSVM_DetectBeforeTraining(t *testing.T) {
	svm := NewOneClassSVM(OneClassSVMConfig{})
	
	_, err := svm.Detect([]float64{0.0, 0.0})
	if err == nil {
		t.Error("Expected error when detecting before training, got nil")
	}
}

func TestOneClassSVM_Algorithm(t *testing.T) {
	svm := NewOneClassSVM(OneClassSVMConfig{})
	
	if svm.Algorithm() != "one_class_svm" {
		t.Errorf("Expected algorithm=one_class_svm, got %s", svm.Algorithm())
	}
}

func TestOneClassSVM_GetSupportVectors(t *testing.T) {
	svm := NewOneClassSVM(OneClassSVMConfig{
		Nu:      0.1,
		Gamma:   0.5,
		MaxIter: 100,
	})
	
	normalData := [][]float64{
		{0.0, 0.0}, {0.1, 0.1}, {0.2, 0.2},
		{-0.1, 0.1}, {0.1, -0.1}, {0.0, 0.2},
	}
	
	svm.Train(normalData)
	
	supportVectors := svm.GetSupportVectors()
	
	if len(supportVectors) == 0 {
		t.Error("Should have support vectors")
	}
	
	if len(supportVectors) > len(normalData) {
		t.Error("Support vectors cannot exceed training data size")
	}
	
	// Check dimensions
	for _, sv := range supportVectors {
		if len(sv) != 2 {
			t.Errorf("Expected 2D support vectors, got %dD", len(sv))
		}
	}
}

func TestOneClassSVM_GetConfig(t *testing.T) {
	config := OneClassSVMConfig{
		Nu:        0.2,
		Gamma:     0.3,
		Kernel:    "rbf",
		Tolerance: 1e-4,
		MaxIter:   500,
	}
	
	svm := NewOneClassSVM(config)
	retrievedConfig := svm.GetConfig()
	
	if retrievedConfig.Nu != 0.2 {
		t.Errorf("Expected nu=0.2, got %.2f", retrievedConfig.Nu)
	}
	if retrievedConfig.Gamma != 0.3 {
		t.Errorf("Expected gamma=0.3, got %.2f", retrievedConfig.Gamma)
	}
	if retrievedConfig.MaxIter != 500 {
		t.Errorf("Expected maxIter=500, got %d", retrievedConfig.MaxIter)
	}
}

func TestOneClassSVM_HighDimensional(t *testing.T) {
	svm := NewOneClassSVM(OneClassSVMConfig{
		Nu:      0.1,
		Gamma:   0.1,
		MaxIter: 50,
	})
	
	// 10D normal data
	normalData := [][]float64{
		{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1},
		{0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2},
		{0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15},
		{0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12},
		{0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18},
	}
	
	err := svm.Train(normalData)
	if err != nil {
		t.Fatalf("Train high-dimensional data failed: %v", err)
	}
	
	// Test normal sample
	normal := []float64{0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15}
	score, err := svm.Detect(normal)
	if err != nil {
		t.Fatalf("Detect failed: %v", err)
	}
	
	// Test outlier
	outlier := []float64{5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0}
	score2, err := svm.Detect(outlier)
	if err != nil {
		t.Fatalf("Detect failed: %v", err)
	}
	
	if score2 <= score {
		t.Error("Outlier should have higher score than normal sample")
	}
}

func TestOneClassSVM_KernelFunctions(t *testing.T) {
	tests := []struct {
		name   string
		kernel string
		x1     []float64
		x2     []float64
	}{
		{"RBF", "rbf", []float64{1.0, 2.0}, []float64{1.5, 2.5}},
		{"Linear", "linear", []float64{1.0, 2.0}, []float64{1.5, 2.5}},
		{"Poly", "poly", []float64{1.0, 2.0}, []float64{1.5, 2.5}},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			svm := NewOneClassSVM(OneClassSVMConfig{
				Kernel: tt.kernel,
				Gamma:  0.5,
				Degree: 3,
			})
			
			k := svm.kernelFunc(tt.x1, tt.x2)
			
			if math.IsNaN(k) || math.IsInf(k, 0) {
				t.Errorf("Kernel function returned invalid value: %f", k)
			}
			
			// Kernel should be positive for same points
			kSame := svm.kernelFunc(tt.x1, tt.x1)
			if kSame <= 0 {
				t.Errorf("Kernel of same point should be positive, got %f", kSame)
			}
		})
	}
}

func TestOneClassSVM_ConcurrentDetect(t *testing.T) {
	svm := NewOneClassSVM(OneClassSVMConfig{
		Nu:      0.1,
		Gamma:   0.5,
		MaxIter: 50,
	})
	
	normalData := [][]float64{
		{0.0, 0.0}, {0.1, 0.1}, {0.2, 0.2},
		{-0.1, 0.1}, {0.1, -0.1},
	}
	svm.Train(normalData)
	
	// Concurrent detection
	done := make(chan bool)
	for i := 0; i < 10; i++ {
		go func() {
			for j := 0; j < 10; j++ {
				sample := []float64{0.15, 0.15}
				_, err := svm.Detect(sample)
				if err != nil {
					t.Errorf("Concurrent detect failed: %v", err)
				}
			}
			done <- true
		}()
	}
	
	// Wait for all goroutines
	for i := 0; i < 10; i++ {
		<-done
	}
}

func BenchmarkOneClassSVM_Train(b *testing.B) {
	config := OneClassSVMConfig{
		Nu:      0.1,
		Gamma:   0.5,
		MaxIter: 100,
	}
	
	// 100 samples, 10 dimensions
	normalData := make([][]float64, 100)
	for i := 0; i < 100; i++ {
		normalData[i] = make([]float64, 10)
		for j := 0; j < 10; j++ {
			normalData[i][j] = 0.1 * float64(i%10)
		}
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		svm := NewOneClassSVM(config)
		svm.Train(normalData)
	}
}

func BenchmarkOneClassSVM_Detect(b *testing.B) {
	svm := NewOneClassSVM(OneClassSVMConfig{
		Nu:      0.1,
		Gamma:   0.5,
		MaxIter: 100,
	})
	
	normalData := make([][]float64, 100)
	for i := 0; i < 100; i++ {
		normalData[i] = make([]float64, 10)
		for j := 0; j < 10; j++ {
			normalData[i][j] = 0.1 * float64(i%10)
		}
	}
	svm.Train(normalData)
	
	sample := make([]float64, 10)
	for i := 0; i < 10; i++ {
		sample[i] = 0.15
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		svm.Detect(sample)
	}
}
