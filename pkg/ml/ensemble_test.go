package ml

import (
	"testing"
)

// Mock anomaly detector for testing
type mockDetector struct {
	algorithm    string
	prediction   bool
	score        float64
	trainCalled  bool
	detectCalled bool
}

func (m *mockDetector) Train(data [][]float64) error {
	m.trainCalled = true
	return nil
}

func (m *mockDetector) Detect(point []float64) (bool, float64) {
	m.detectCalled = true
	return m.prediction, m.score
}

func (m *mockDetector) Algorithm() string {
	return m.algorithm
}

func TestNewEnsembleDetector(t *testing.T) {
	ensemble := NewEnsembleDetector(VotingMajority)
	if ensemble == nil {
		t.Fatal("NewEnsembleDetector returned nil")
	}
	if ensemble.strategy != VotingMajority {
		t.Errorf("Expected strategy %s, got %s", VotingMajority, ensemble.strategy)
	}
	if len(ensemble.models) != 0 {
		t.Error("Expected empty models slice")
	}
}

func TestEnsembleDetector_AddModel(t *testing.T) {
	ensemble := NewEnsembleDetector(VotingMajority)

	tests := []struct {
		name      string
		model     AnomalyModel
		weight    float64
		expectErr bool
	}{
		{
			name:      "nil model",
			model:     nil,
			weight:    1.0,
			expectErr: true,
		},
		{
			name:      "zero weight",
			model:     &mockDetector{},
			weight:    0.0,
			expectErr: true,
		},
		{
			name:      "negative weight",
			model:     &mockDetector{},
			weight:    -1.0,
			expectErr: true,
		},
		{
			name:      "valid model",
			model:     &mockDetector{algorithm: "test"},
			weight:    1.0,
			expectErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ensemble.AddModel(tt.model, tt.weight)
			if (err != nil) != tt.expectErr {
				t.Errorf("AddModel() error = %v, expectErr %v", err, tt.expectErr)
			}
		})
	}
}

func TestEnsembleDetector_Train(t *testing.T) {
	t.Run("empty ensemble", func(t *testing.T) {
		ensemble := NewEnsembleDetector(VotingMajority)
		err := ensemble.Train([][]float64{{1.0}})
		if err == nil {
			t.Error("Expected error for empty ensemble")
		}
	})

	t.Run("trains all models", func(t *testing.T) {
		ensemble := NewEnsembleDetector(VotingMajority)
		mock1 := &mockDetector{algorithm: "model1"}
		mock2 := &mockDetector{algorithm: "model2"}

		ensemble.AddModel(mock1, 1.0)
		ensemble.AddModel(mock2, 1.0)

		data := [][]float64{{1.0, 1.0}}
		err := ensemble.Train(data)
		if err != nil {
			t.Errorf("Train() failed: %v", err)
		}

		if !mock1.trainCalled {
			t.Error("Model 1 Train() not called")
		}
		if !mock2.trainCalled {
			t.Error("Model 2 Train() not called")
		}
		if !ensemble.trained {
			t.Error("Ensemble should be marked as trained")
		}
	})
}

func TestEnsembleDetector_Detect_MajorityVoting(t *testing.T) {
	tests := []struct {
		name          string
		predictions   []bool
		scores        []float64
		wantAnomaly   bool
		checkScore    bool
		minScore      float64
		maxScore      float64
	}{
		{
			name:        "all agree anomaly",
			predictions: []bool{true, true, true},
			scores:      []float64{0.8, 0.9, 0.7},
			wantAnomaly: true,
		},
		{
			name:        "majority anomaly",
			predictions: []bool{true, true, false},
			scores:      []float64{0.8, 0.9, 0.2},
			wantAnomaly: true,
		},
		{
			name:        "majority normal",
			predictions: []bool{false, false, true},
			scores:      []float64{0.2, 0.1, 0.8},
			wantAnomaly: false,
		},
		{
			name:        "all agree normal",
			predictions: []bool{false, false, false},
			scores:      []float64{0.1, 0.2, 0.15},
			wantAnomaly: false,
		},
		{
			name:        "tie - should be normal (not majority)",
			predictions: []bool{true, false},
			scores:      []float64{0.8, 0.2},
			wantAnomaly: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ensemble := NewEnsembleDetector(VotingMajority)

			// Add mock models
			for i := range tt.predictions {
				mock := &mockDetector{
					algorithm:  "test",
					prediction: tt.predictions[i],
					score:      tt.scores[i],
				}
				ensemble.AddModel(mock, 1.0)
			}

			// Train ensemble
			ensemble.Train([][]float64{{1.0}})

			// Detect
			isAnomaly, score := ensemble.Detect([]float64{1.0})

			if isAnomaly != tt.wantAnomaly {
				t.Errorf("Detect() isAnomaly = %v, want %v", isAnomaly, tt.wantAnomaly)
			}

			if tt.checkScore {
				if score < tt.minScore || score > tt.maxScore {
					t.Errorf("Detect() score = %v, want between %v and %v",
						score, tt.minScore, tt.maxScore)
				}
			}
		})
	}
}

func TestEnsembleDetector_Detect_WeightedVoting(t *testing.T) {
	ensemble := NewEnsembleDetector(VotingWeighted)

	// Add models with different weights
	// Model 1: strong anomaly detector (high weight)
	mock1 := &mockDetector{
		algorithm:  "strong",
		prediction: true,
		score:      0.9,
	}
	ensemble.AddModel(mock1, 3.0) // Higher weight

	// Model 2: weak detector (low weight)
	mock2 := &mockDetector{
		algorithm:  "weak",
		prediction: false,
		score:      0.3,
	}
	ensemble.AddModel(mock2, 1.0) // Lower weight

	ensemble.Train([][]float64{{1.0}})

	isAnomaly, score := ensemble.Detect([]float64{1.0})

	// Should be anomaly because strong detector has higher weight
	if !isAnomaly {
		t.Error("Expected anomaly with weighted voting (strong model has higher weight)")
	}
	if score <= 0 {
		t.Errorf("Expected positive score, got %f", score)
	}
}

func TestEnsembleDetector_Detect_AverageVoting(t *testing.T) {
	ensemble := NewEnsembleDetector(VotingAverage)

	mock1 := &mockDetector{prediction: true, score: 0.8}
	mock2 := &mockDetector{prediction: false, score: 0.3}
	mock3 := &mockDetector{prediction: true, score: 0.7}

	ensemble.AddModel(mock1, 1.0)
	ensemble.AddModel(mock2, 1.0)
	ensemble.AddModel(mock3, 1.0)

	ensemble.Train([][]float64{{1.0}})

	isAnomaly, score := ensemble.Detect([]float64{1.0})

	expectedAvg := (0.8 + 0.3 + 0.7) / 3.0
	if score < expectedAvg-0.01 || score > expectedAvg+0.01 {
		t.Errorf("Expected average score %f, got %f", expectedAvg, score)
	}

	// Average > 0.5, should be anomaly
	if !isAnomaly {
		t.Error("Expected anomaly with average score > 0.5")
	}
}

func TestEnsembleDetector_Detect_MaxVoting(t *testing.T) {
	ensemble := NewEnsembleDetector(VotingMax)

	mock1 := &mockDetector{prediction: false, score: 0.3}
	mock2 := &mockDetector{prediction: true, score: 0.9}  // Highest
	mock3 := &mockDetector{prediction: false, score: 0.4}

	ensemble.AddModel(mock1, 1.0)
	ensemble.AddModel(mock2, 1.0)
	ensemble.AddModel(mock3, 1.0)

	ensemble.Train([][]float64{{1.0}})

	isAnomaly, score := ensemble.Detect([]float64{1.0})

	if score != 0.9 {
		t.Errorf("Expected max score 0.9, got %f", score)
	}
	if !isAnomaly {
		t.Error("Expected anomaly (at least one model detected it)")
	}
}

func TestEnsembleDetector_Detect_MinVoting(t *testing.T) {
	ensemble := NewEnsembleDetector(VotingMin)

	mock1 := &mockDetector{prediction: true, score: 0.8}
	mock2 := &mockDetector{prediction: true, score: 0.6}  // Lowest
	mock3 := &mockDetector{prediction: true, score: 0.9}

	ensemble.AddModel(mock1, 1.0)
	ensemble.AddModel(mock2, 1.0)
	ensemble.AddModel(mock3, 1.0)

	ensemble.Train([][]float64{{1.0}})

	isAnomaly, score := ensemble.Detect([]float64{1.0})

	if score != 0.6 {
		t.Errorf("Expected min score 0.6, got %f", score)
	}
	if !isAnomaly {
		t.Error("Expected anomaly (all models detected it)")
	}
}

func TestEnsembleDetector_Algorithm(t *testing.T) {
	tests := []struct {
		name      string
		strategy  EnsembleStrategy
		numModels int
		want      string
	}{
		{
			name:      "empty ensemble",
			strategy:  VotingMajority,
			numModels: 0,
			want:      "ensemble-empty",
		},
		{
			name:      "majority with 3 models",
			strategy:  VotingMajority,
			numModels: 3,
			want:      "ensemble-voting_majority-3-models",
		},
		{
			name:      "weighted with 2 models",
			strategy:  VotingWeighted,
			numModels: 2,
			want:      "ensemble-voting_weighted-2-models",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ensemble := NewEnsembleDetector(tt.strategy)

			for i := 0; i < tt.numModels; i++ {
				ensemble.AddModel(&mockDetector{}, 1.0)
			}

			got := ensemble.Algorithm()
			if got != tt.want {
				t.Errorf("Algorithm() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestEnsembleDetector_GetModels(t *testing.T) {
	ensemble := NewEnsembleDetector(VotingMajority)

	ensemble.AddModel(&mockDetector{algorithm: "model1"}, 1.0)
	ensemble.AddModel(&mockDetector{algorithm: "model2"}, 1.0)
	ensemble.AddModel(&mockDetector{algorithm: "model3"}, 1.0)

	models := ensemble.GetModels()

	if len(models) != 3 {
		t.Errorf("Expected 3 models, got %d", len(models))
	}

	expected := []string{"model1", "model2", "model3"}
	for i, model := range models {
		if model != expected[i] {
			t.Errorf("Model %d: got %s, want %s", i, model, expected[i])
		}
	}
}

func TestEnsembleDetector_SetGetStrategy(t *testing.T) {
	ensemble := NewEnsembleDetector(VotingMajority)

	if ensemble.GetStrategy() != VotingMajority {
		t.Errorf("Initial strategy = %v, want %v", ensemble.GetStrategy(), VotingMajority)
	}

	ensemble.SetStrategy(VotingWeighted)

	if ensemble.GetStrategy() != VotingWeighted {
		t.Errorf("After SetStrategy = %v, want %v", ensemble.GetStrategy(), VotingWeighted)
	}
}

func TestEnsembleDetector_RealModels(t *testing.T) {
	// Test with real LOF and IsolationForest detectors
	ensemble := NewEnsembleDetector(VotingMajority)

	// Create training data with normal cluster
	trainData := [][]float64{
		{1.0, 1.0}, {1.1, 1.0}, {0.9, 1.1},
		{1.0, 0.9}, {1.2, 1.1}, {0.8, 0.9},
	}

	// Add LOF detector
	lof := NewLOFDetector(3, 1.5)
	ensemble.AddModel(lof, 1.0)

	// Add Isolation Forest detector
	iforest := NewIsolationForest(10, 256)
	ensemble.AddModel(iforest, 1.0)

	// Train ensemble
	err := ensemble.Train(trainData)
	if err != nil {
		t.Fatalf("Train failed: %v", err)
	}

	// Test normal point
	normalPoint := []float64{1.05, 1.05}
	isAnomaly, _ := ensemble.Detect(normalPoint)
	if isAnomaly {
		t.Error("Normal point detected as anomaly by ensemble")
	}

	// Test outlier
	outlier := []float64{10.0, 10.0}
	isAnomaly, _ = ensemble.Detect(outlier)
	if !isAnomaly {
		t.Error("Outlier not detected as anomaly by ensemble")
	}
}

func TestEnsembleDetector_ConcurrentAccess(t *testing.T) {
	ensemble := NewEnsembleDetector(VotingMajority)

	mock1 := &mockDetector{prediction: true, score: 0.8}
	mock2 := &mockDetector{prediction: false, score: 0.3}

	ensemble.AddModel(mock1, 1.0)
	ensemble.AddModel(mock2, 1.0)

	ensemble.Train([][]float64{{1.0}})

	// Test concurrent detections
	done := make(chan bool)
	for i := 0; i < 10; i++ {
		go func() {
			for j := 0; j < 100; j++ {
				ensemble.Detect([]float64{1.0})
			}
			done <- true
		}()
	}

	// Wait for all goroutines
	for i := 0; i < 10; i++ {
		<-done
	}
}

func BenchmarkEnsembleDetector_Detect(b *testing.B) {
	ensemble := NewEnsembleDetector(VotingMajority)

	// Add multiple models
	for i := 0; i < 5; i++ {
		mock := &mockDetector{
			prediction: i%2 == 0,
			score:      float64(i) / 10.0,
		}
		ensemble.AddModel(mock, 1.0)
	}

	ensemble.Train([][]float64{{1.0}})
	point := []float64{1.0}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ensemble.Detect(point)
	}
}
