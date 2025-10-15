package ml

import (
	"math"
	"testing"
)

func TestNewLOFDetector(t *testing.T) {
	detector := NewLOFDetector(5, 1.5)
	if detector == nil {
		t.Fatal("NewLOFDetector returned nil")
	}
	if detector.k != 5 {
		t.Errorf("Expected k=5, got k=%d", detector.k)
	}
	if detector.threshold != 1.5 {
		t.Errorf("Expected threshold=1.5, got threshold=%f", detector.threshold)
	}
	if detector.trained {
		t.Error("Detector should not be trained initially")
	}
}

func TestLOFDetector_Train(t *testing.T) {
	detector := NewLOFDetector(3, 1.5)

	tests := []struct {
		name      string
		data      [][]float64
		expectErr bool
	}{
		{
			name:      "empty data",
			data:      [][]float64{},
			expectErr: true,
		},
		{
			name: "insufficient data",
			data: [][]float64{
				{1.0, 1.0},
				{2.0, 2.0},
			},
			expectErr: true,
		},
		{
			name: "valid data",
			data: [][]float64{
				{1.0, 1.0},
				{1.1, 1.0},
				{0.9, 1.1},
				{1.0, 0.9},
				{1.2, 1.1},
			},
			expectErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := detector.Train(tt.data)
			if (err != nil) != tt.expectErr {
				t.Errorf("Train() error = %v, expectErr %v", err, tt.expectErr)
			}
			if !tt.expectErr && !detector.trained {
				t.Error("Detector should be trained after successful Train()")
			}
		})
	}
}

func TestLOFDetector_Detect_NotTrained(t *testing.T) {
	detector := NewLOFDetector(5, 1.5)
	point := []float64{1.0, 1.0}

	isAnomaly, score := detector.Detect(point)
	if isAnomaly {
		t.Error("Untrained detector should not detect anomalies")
	}
	if score != 0.0 {
		t.Errorf("Expected score=0.0 for untrained detector, got %f", score)
	}
}

func TestLOFDetector_Detect_NormalPoint(t *testing.T) {
	detector := NewLOFDetector(3, 1.5)

	// Create training data with normal cluster
	trainData := [][]float64{
		{1.0, 1.0},
		{1.1, 1.0},
		{0.9, 1.1},
		{1.0, 0.9},
		{1.2, 1.1},
		{0.8, 0.9},
		{1.1, 1.2},
		{0.9, 0.8},
	}

	err := detector.Train(trainData)
	if err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	// Test point within cluster
	normalPoint := []float64{1.05, 1.05}
	isAnomaly, score := detector.Detect(normalPoint)

	if isAnomaly {
		t.Errorf("Normal point detected as anomaly with score %f", score)
	}
}

func TestLOFDetector_Detect_Outlier(t *testing.T) {
	detector := NewLOFDetector(3, 1.5)

	// Create training data with tight cluster
	trainData := [][]float64{
		{1.0, 1.0},
		{1.1, 1.0},
		{0.9, 1.1},
		{1.0, 0.9},
		{1.2, 1.1},
		{0.8, 0.9},
		{1.1, 1.2},
		{0.9, 0.8},
	}

	err := detector.Train(trainData)
	if err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	// Test outlier far from cluster
	outlier := []float64{10.0, 10.0}
	isAnomaly, score := detector.Detect(outlier)

	if !isAnomaly {
		t.Errorf("Outlier not detected as anomaly, score: %f", score)
	}
	if score <= 0 {
		t.Errorf("Expected positive score for outlier, got %f", score)
	}
}

func TestLOFDetector_Detect_MultipleOutliers(t *testing.T) {
	detector := NewLOFDetector(5, 1.5)

	// Create training data with cluster around origin
	trainData := make([][]float64, 0)
	for i := 0; i < 20; i++ {
		trainData = append(trainData, []float64{
			float64(i%5) * 0.1,
			float64(i/5) * 0.1,
		})
	}

	err := detector.Train(trainData)
	if err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	tests := []struct {
		name      string
		point     []float64
		wantAnomaly bool
	}{
		{
			name:      "normal point in cluster",
			point:     []float64{0.15, 0.15},
			wantAnomaly: false,
		},
		{
			name:      "outlier far away",
			point:     []float64{5.0, 5.0},
			wantAnomaly: true,
		},
		{
			name:      "moderate outlier",
			point:     []float64{1.0, 1.0},
			wantAnomaly: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			isAnomaly, score := detector.Detect(tt.point)
			if isAnomaly != tt.wantAnomaly {
				t.Errorf("Detect(%v) = %v (score=%f), want %v",
					tt.point, isAnomaly, score, tt.wantAnomaly)
			}
		})
	}
}

func TestLOFDetector_Algorithm(t *testing.T) {
	detector := NewLOFDetector(5, 1.5)
	if detector.Algorithm() != "local-outlier-factor" {
		t.Errorf("Expected algorithm name 'local-outlier-factor', got '%s'", detector.Algorithm())
	}
}

func TestEuclideanDistance(t *testing.T) {
	tests := []struct {
		name string
		a    []float64
		b    []float64
		want float64
	}{
		{
			name: "same point",
			a:    []float64{1.0, 1.0},
			b:    []float64{1.0, 1.0},
			want: 0.0,
		},
		{
			name: "unit distance",
			a:    []float64{0.0, 0.0},
			b:    []float64{1.0, 0.0},
			want: 1.0,
		},
		{
			name: "diagonal distance",
			a:    []float64{0.0, 0.0},
			b:    []float64{3.0, 4.0},
			want: 5.0,
		},
		{
			name: "different dimensions",
			a:    []float64{1.0, 1.0},
			b:    []float64{1.0},
			want: math.Inf(1),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := euclideanDistance(tt.a, tt.b)
			if math.IsInf(tt.want, 1) {
				if !math.IsInf(got, 1) {
					t.Errorf("euclideanDistance() = %v, want Inf", got)
				}
			} else if math.Abs(got-tt.want) > 1e-10 {
				t.Errorf("euclideanDistance() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestEqualPoints(t *testing.T) {
	tests := []struct {
		name string
		a    []float64
		b    []float64
		want bool
	}{
		{
			name: "equal points",
			a:    []float64{1.0, 2.0, 3.0},
			b:    []float64{1.0, 2.0, 3.0},
			want: true,
		},
		{
			name: "different points",
			a:    []float64{1.0, 2.0},
			b:    []float64{1.0, 2.1},
			want: false,
		},
		{
			name: "different length",
			a:    []float64{1.0, 2.0},
			b:    []float64{1.0, 2.0, 3.0},
			want: false,
		},
		{
			name: "within tolerance",
			a:    []float64{1.0, 2.0},
			b:    []float64{1.0, 2.0 + 1e-11},
			want: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := equalPoints(tt.a, tt.b); got != tt.want {
				t.Errorf("equalPoints() = %v, want %v", got, tt.want)
			}
		})
	}
}

// Benchmark tests
func BenchmarkLOFDetector_Train(b *testing.B) {
	detector := NewLOFDetector(5, 1.5)
	data := generateRandomData(1000, 10)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = detector.Train(data)
	}
}

func BenchmarkLOFDetector_Detect(b *testing.B) {
	detector := NewLOFDetector(5, 1.5)
	trainData := generateRandomData(1000, 10)
	_ = detector.Train(trainData)

	testPoint := make([]float64, 10)
	for i := range testPoint {
		testPoint[i] = 0.5
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = detector.Detect(testPoint)
	}
}

// Helper function to generate random data for benchmarks
func generateRandomData(n, dim int) [][]float64 {
	data := make([][]float64, n)
	for i := 0; i < n; i++ {
		data[i] = make([]float64, dim)
		for j := 0; j < dim; j++ {
			data[i][j] = float64(i*j) / float64(n)
		}
	}
	return data
}

// Test concurrent access
func TestLOFDetector_ConcurrentAccess(t *testing.T) {
	detector := NewLOFDetector(5, 1.5)

	trainData := [][]float64{
		{1.0, 1.0}, {1.1, 1.0}, {0.9, 1.1},
		{1.0, 0.9}, {1.2, 1.1}, {0.8, 0.9},
	}

	err := detector.Train(trainData)
	if err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	// Test concurrent detections
	done := make(chan bool)
	for i := 0; i < 10; i++ {
		go func() {
			for j := 0; j < 100; j++ {
				point := []float64{1.0, 1.0}
				_, _ = detector.Detect(point)
			}
			done <- true
		}()
	}

	// Wait for all goroutines
	for i := 0; i < 10; i++ {
		<-done
	}
}
