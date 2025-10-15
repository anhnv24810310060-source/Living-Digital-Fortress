package ml

import (
	"math"
	"testing"
	"time"
)

func TestTimeSeriesExtractor_BasicStatistics(t *testing.T) {
	extractor := NewTimeSeriesExtractor(TimeSeriesConfig{
		WindowSize: 5,
	})

	ts := &TimeSeries{
		Points: []TimeSeriesPoint{
			{Timestamp: time.Now(), Value: 1.0},
			{Timestamp: time.Now(), Value: 2.0},
			{Timestamp: time.Now(), Value: 3.0},
			{Timestamp: time.Now(), Value: 4.0},
			{Timestamp: time.Now(), Value: 5.0},
		},
	}

	features, err := extractor.ExtractFeatures(ts)
	if err != nil {
		t.Fatalf("ExtractFeatures failed: %v", err)
	}

	// Check mean
	if math.Abs(features.Mean-3.0) > 0.01 {
		t.Errorf("Mean = %f, want 3.0", features.Mean)
	}

	// Check median
	if math.Abs(features.Median-3.0) > 0.01 {
		t.Errorf("Median = %f, want 3.0", features.Median)
	}

	// Check min/max
	if features.Min != 1.0 {
		t.Errorf("Min = %f, want 1.0", features.Min)
	}
	if features.Max != 5.0 {
		t.Errorf("Max = %f, want 5.0", features.Max)
	}

	// Check range
	if features.Range != 4.0 {
		t.Errorf("Range = %f, want 4.0", features.Range)
	}
}

func TestTimeSeriesExtractor_Trend(t *testing.T) {
	extractor := NewTimeSeriesExtractor(TimeSeriesConfig{})

	// Upward trend
	ts := &TimeSeries{
		Points: []TimeSeriesPoint{
			{Timestamp: time.Now(), Value: 1.0},
			{Timestamp: time.Now(), Value: 2.0},
			{Timestamp: time.Now(), Value: 3.0},
			{Timestamp: time.Now(), Value: 4.0},
			{Timestamp: time.Now(), Value: 5.0},
		},
	}

	features, err := extractor.ExtractFeatures(ts)
	if err != nil {
		t.Fatalf("ExtractFeatures failed: %v", err)
	}

	// Trend should be positive
	if features.Trend <= 0 {
		t.Errorf("Trend = %f, want positive", features.Trend)
	}

	// Downward trend
	ts.Points = []TimeSeriesPoint{
		{Timestamp: time.Now(), Value: 5.0},
		{Timestamp: time.Now(), Value: 4.0},
		{Timestamp: time.Now(), Value: 3.0},
		{Timestamp: time.Now(), Value: 2.0},
		{Timestamp: time.Now(), Value: 1.0},
	}

	features, err = extractor.ExtractFeatures(ts)
	if err != nil {
		t.Fatalf("ExtractFeatures failed: %v", err)
	}

	// Trend should be negative
	if features.Trend >= 0 {
		t.Errorf("Trend = %f, want negative", features.Trend)
	}
}

func TestTimeSeriesExtractor_AutoCorrelation(t *testing.T) {
	extractor := NewTimeSeriesExtractor(TimeSeriesConfig{
		LagPeriods: []int{1, 2},
	})

	// Perfect positive autocorrelation: repeating pattern
	ts := &TimeSeries{
		Points: []TimeSeriesPoint{
			{Timestamp: time.Now(), Value: 1.0},
			{Timestamp: time.Now(), Value: 2.0},
			{Timestamp: time.Now(), Value: 1.0},
			{Timestamp: time.Now(), Value: 2.0},
			{Timestamp: time.Now(), Value: 1.0},
			{Timestamp: time.Now(), Value: 2.0},
		},
	}

	features, err := extractor.ExtractFeatures(ts)
	if err != nil {
		t.Fatalf("ExtractFeatures failed: %v", err)
	}

	// AutoCorr1 should be negative (alternating pattern)
	if features.AutoCorr1 >= 0 {
		t.Errorf("AutoCorr1 = %f, want negative for alternating pattern", features.AutoCorr1)
	}
}

func TestTimeSeriesExtractor_Peaks(t *testing.T) {
	extractor := NewTimeSeriesExtractor(TimeSeriesConfig{})

	ts := &TimeSeries{
		Points: []TimeSeriesPoint{
			{Timestamp: time.Now(), Value: 1.0},
			{Timestamp: time.Now(), Value: 3.0}, // Peak
			{Timestamp: time.Now(), Value: 2.0},
			{Timestamp: time.Now(), Value: 4.0}, // Peak
			{Timestamp: time.Now(), Value: 1.0},
		},
	}

	features, err := extractor.ExtractFeatures(ts)
	if err != nil {
		t.Fatalf("ExtractFeatures failed: %v", err)
	}

	if features.NumPeaks != 2 {
		t.Errorf("NumPeaks = %d, want 2", features.NumPeaks)
	}
}

func TestTimeSeriesExtractor_Valleys(t *testing.T) {
	extractor := NewTimeSeriesExtractor(TimeSeriesConfig{})

	ts := &TimeSeries{
		Points: []TimeSeriesPoint{
			{Timestamp: time.Now(), Value: 5.0},
			{Timestamp: time.Now(), Value: 2.0}, // Valley
			{Timestamp: time.Now(), Value: 4.0},
			{Timestamp: time.Now(), Value: 1.0}, // Valley
			{Timestamp: time.Now(), Value: 3.0},
		},
	}

	features, err := extractor.ExtractFeatures(ts)
	if err != nil {
		t.Fatalf("ExtractFeatures failed: %v", err)
	}

	if features.NumValleys != 2 {
		t.Errorf("NumValleys = %d, want 2", features.NumValleys)
	}
}

func TestTimeSeriesExtractor_ZeroCrossings(t *testing.T) {
	extractor := NewTimeSeriesExtractor(TimeSeriesConfig{})

	ts := &TimeSeries{
		Points: []TimeSeriesPoint{
			{Timestamp: time.Now(), Value: -1.0},
			{Timestamp: time.Now(), Value: 1.0},  // Crossing
			{Timestamp: time.Now(), Value: 2.0},
			{Timestamp: time.Now(), Value: -1.0}, // Crossing
			{Timestamp: time.Now(), Value: 1.0},  // Crossing
		},
	}

	features, err := extractor.ExtractFeatures(ts)
	if err != nil {
		t.Fatalf("ExtractFeatures failed: %v", err)
	}

	// Should have 3 crossings around mean (0.4)
	if features.ZeroCrossings < 2 {
		t.Errorf("ZeroCrossings = %d, want at least 2", features.ZeroCrossings)
	}
}

func TestTimeSeriesExtractor_Distribution(t *testing.T) {
	extractor := NewTimeSeriesExtractor(TimeSeriesConfig{})

	// Right-skewed distribution
	ts := &TimeSeries{
		Points: []TimeSeriesPoint{
			{Timestamp: time.Now(), Value: 1.0},
			{Timestamp: time.Now(), Value: 1.0},
			{Timestamp: time.Now(), Value: 1.0},
			{Timestamp: time.Now(), Value: 2.0},
			{Timestamp: time.Now(), Value: 10.0}, // Outlier
		},
	}

	features, err := extractor.ExtractFeatures(ts)
	if err != nil {
		t.Fatalf("ExtractFeatures failed: %v", err)
	}

	// Skewness should be positive
	if features.Skewness <= 0 {
		t.Errorf("Skewness = %f, want positive for right-skewed", features.Skewness)
	}

	// IQR should be reasonable
	if features.IQR < 0 {
		t.Errorf("IQR = %f, want non-negative", features.IQR)
	}
}

func TestTimeSeriesExtractor_Changes(t *testing.T) {
	extractor := NewTimeSeriesExtractor(TimeSeriesConfig{})

	ts := &TimeSeries{
		Points: []TimeSeriesPoint{
			{Timestamp: time.Now(), Value: 1.0},
			{Timestamp: time.Now(), Value: 3.0}, // +2
			{Timestamp: time.Now(), Value: 2.0}, // -1
			{Timestamp: time.Now(), Value: 5.0}, // +3
		},
	}

	features, err := extractor.ExtractFeatures(ts)
	if err != nil {
		t.Fatalf("ExtractFeatures failed: %v", err)
	}

	// AbsChanges = |2| + |-1| + |3| = 6
	if math.Abs(features.AbsChanges-6.0) > 0.01 {
		t.Errorf("AbsChanges = %f, want 6.0", features.AbsChanges)
	}

	// ChangeRate = 6 / 4 = 1.5
	if math.Abs(features.ChangeRate-1.5) > 0.01 {
		t.Errorf("ChangeRate = %f, want 1.5", features.ChangeRate)
	}
}

func TestTimeSeriesExtractor_RollingMean(t *testing.T) {
	extractor := NewTimeSeriesExtractor(TimeSeriesConfig{
		WindowSize: 3,
	})

	ts := &TimeSeries{
		Points: []TimeSeriesPoint{
			{Timestamp: time.Now(), Value: 1.0},
			{Timestamp: time.Now(), Value: 2.0},
			{Timestamp: time.Now(), Value: 3.0},
			{Timestamp: time.Now(), Value: 4.0},
			{Timestamp: time.Now(), Value: 5.0},
		},
	}

	features, err := extractor.ExtractFeatures(ts)
	if err != nil {
		t.Fatalf("ExtractFeatures failed: %v", err)
	}

	// Rolling mean with window=3: [2.0, 3.0, 4.0]
	expected := []float64{2.0, 3.0, 4.0}
	if len(features.RollingMean) != len(expected) {
		t.Fatalf("RollingMean length = %d, want %d", len(features.RollingMean), len(expected))
	}

	for i, exp := range expected {
		if math.Abs(features.RollingMean[i]-exp) > 0.01 {
			t.Errorf("RollingMean[%d] = %f, want %f", i, features.RollingMean[i], exp)
		}
	}
}

func TestTimeSeriesExtractor_EWMA(t *testing.T) {
	extractor := NewTimeSeriesExtractor(TimeSeriesConfig{})

	ts := &TimeSeries{
		Points: []TimeSeriesPoint{
			{Timestamp: time.Now(), Value: 10.0},
			{Timestamp: time.Now(), Value: 20.0},
			{Timestamp: time.Now(), Value: 30.0},
		},
	}

	features, err := extractor.ExtractFeatures(ts)
	if err != nil {
		t.Fatalf("ExtractFeatures failed: %v", err)
	}

	// EWMA should smooth the values
	if len(features.EWMA) != 3 {
		t.Fatalf("EWMA length = %d, want 3", len(features.EWMA))
	}

	// First value should be the same
	if features.EWMA[0] != 10.0 {
		t.Errorf("EWMA[0] = %f, want 10.0", features.EWMA[0])
	}

	// Values should be increasing
	if features.EWMA[1] <= features.EWMA[0] {
		t.Errorf("EWMA not increasing")
	}
	if features.EWMA[2] <= features.EWMA[1] {
		t.Errorf("EWMA not increasing")
	}
}

func TestTimeSeriesExtractor_Entropy(t *testing.T) {
	extractor := NewTimeSeriesExtractor(TimeSeriesConfig{})

	// Uniform distribution (high entropy)
	ts := &TimeSeries{
		Points: []TimeSeriesPoint{
			{Timestamp: time.Now(), Value: 1.0},
			{Timestamp: time.Now(), Value: 2.0},
			{Timestamp: time.Now(), Value: 3.0},
			{Timestamp: time.Now(), Value: 4.0},
			{Timestamp: time.Now(), Value: 5.0},
		},
	}

	features, err := extractor.ExtractFeatures(ts)
	if err != nil {
		t.Fatalf("ExtractFeatures failed: %v", err)
	}

	// Entropy should be positive
	if features.Entropy <= 0 {
		t.Errorf("Entropy = %f, want positive", features.Entropy)
	}

	// Constant series (low entropy)
	ts.Points = []TimeSeriesPoint{
		{Timestamp: time.Now(), Value: 1.0},
		{Timestamp: time.Now(), Value: 1.0},
		{Timestamp: time.Now(), Value: 1.0},
	}

	features2, err := extractor.ExtractFeatures(ts)
	if err != nil {
		t.Fatalf("ExtractFeatures failed: %v", err)
	}

	// Constant should have zero entropy
	if features2.Entropy != 0 {
		t.Errorf("Entropy for constant series = %f, want 0", features2.Entropy)
	}
}

func TestTimeSeriesExtractor_Complexity(t *testing.T) {
	extractor := NewTimeSeriesExtractor(TimeSeriesConfig{})

	// Complex pattern
	ts := &TimeSeries{
		Points: []TimeSeriesPoint{
			{Timestamp: time.Now(), Value: 1.0},
			{Timestamp: time.Now(), Value: 5.0},
			{Timestamp: time.Now(), Value: 2.0},
			{Timestamp: time.Now(), Value: 8.0},
			{Timestamp: time.Now(), Value: 3.0},
		},
	}

	features, err := extractor.ExtractFeatures(ts)
	if err != nil {
		t.Fatalf("ExtractFeatures failed: %v", err)
	}

	// Complexity should be positive
	if features.Complexity <= 0 {
		t.Errorf("Complexity = %f, want positive", features.Complexity)
	}
}

func TestTimeSeriesExtractor_ToVector(t *testing.T) {
	extractor := NewTimeSeriesExtractor(TimeSeriesConfig{})

	ts := &TimeSeries{
		Points: []TimeSeriesPoint{
			{Timestamp: time.Now(), Value: 1.0},
			{Timestamp: time.Now(), Value: 2.0},
			{Timestamp: time.Now(), Value: 3.0},
		},
	}

	features, err := extractor.ExtractFeatures(ts)
	if err != nil {
		t.Fatalf("ExtractFeatures failed: %v", err)
	}

	vector := features.ToVector()
	
	// Should have 26 features
	if len(vector) != 26 {
		t.Errorf("Vector length = %d, want 26", len(vector))
	}

	// All values should be finite
	for i, v := range vector {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("Vector[%d] = %f, want finite value", i, v)
		}
	}
}

func TestTimeSeriesExtractor_EmptySeries(t *testing.T) {
	extractor := NewTimeSeriesExtractor(TimeSeriesConfig{})

	ts := &TimeSeries{
		Points: []TimeSeriesPoint{},
	}

	_, err := extractor.ExtractFeatures(ts)
	if err == nil {
		t.Error("ExtractFeatures should fail for empty series")
	}
}

func TestTimeSeriesExtractor_SinglePoint(t *testing.T) {
	extractor := NewTimeSeriesExtractor(TimeSeriesConfig{})

	ts := &TimeSeries{
		Points: []TimeSeriesPoint{
			{Timestamp: time.Now(), Value: 5.0},
		},
	}

	features, err := extractor.ExtractFeatures(ts)
	if err != nil {
		t.Fatalf("ExtractFeatures failed: %v", err)
	}

	// Should handle single point gracefully
	if features.Mean != 5.0 {
		t.Errorf("Mean = %f, want 5.0", features.Mean)
	}
	if features.Std != 0 {
		t.Errorf("Std = %f, want 0", features.Std)
	}
}

func TestTimeSeriesExtractor_SeasonalPattern(t *testing.T) {
	extractor := NewTimeSeriesExtractor(TimeSeriesConfig{
		SeasonalPeriod: 4,
	})

	// Seasonal pattern: [1,2,3,4,1,2,3,4,1,2,3,4]
	points := make([]TimeSeriesPoint, 12)
	for i := 0; i < 12; i++ {
		points[i] = TimeSeriesPoint{
			Timestamp: time.Now(),
			Value:     float64((i % 4) + 1),
		}
	}

	ts := &TimeSeries{Points: points}

	features, err := extractor.ExtractFeatures(ts)
	if err != nil {
		t.Fatalf("ExtractFeatures failed: %v", err)
	}

	// Seasonality should be high (positive autocorrelation at lag=4)
	if features.Seasonality <= 0 {
		t.Errorf("Seasonality = %f, want positive for seasonal pattern", features.Seasonality)
	}
}

func TestTimeSeriesExtractor_LongestRun(t *testing.T) {
	extractor := NewTimeSeriesExtractor(TimeSeriesConfig{})

	// Increasing run of length 4 (indices 0-3)
	ts := &TimeSeries{
		Points: []TimeSeriesPoint{
			{Timestamp: time.Now(), Value: 1.0},
			{Timestamp: time.Now(), Value: 2.0},
			{Timestamp: time.Now(), Value: 3.0},
			{Timestamp: time.Now(), Value: 4.0},
			{Timestamp: time.Now(), Value: 5.0},
			{Timestamp: time.Now(), Value: 4.0},
		},
	}

	features, err := extractor.ExtractFeatures(ts)
	if err != nil {
		t.Fatalf("ExtractFeatures failed: %v", err)
	}

	// Longest increasing run: 1->2->3->4->5 = 5 values
	if features.LongestRun < 4 {
		t.Errorf("LongestRun = %d, want at least 4", features.LongestRun)
	}
}

func TestTimeSeriesExtractor_DefaultConfig(t *testing.T) {
	extractor := NewTimeSeriesExtractor(TimeSeriesConfig{})

	// Check defaults
	if extractor.windowSize != 10 {
		t.Errorf("Default windowSize = %d, want 10", extractor.windowSize)
	}
	if len(extractor.lagPeriods) != 2 {
		t.Errorf("Default lagPeriods length = %d, want 2", len(extractor.lagPeriods))
	}
	if extractor.trendMethod != "linear" {
		t.Errorf("Default trendMethod = %s, want linear", extractor.trendMethod)
	}
	if extractor.seasonalPeriod != 24 {
		t.Errorf("Default seasonalPeriod = %d, want 24", extractor.seasonalPeriod)
	}
}

func BenchmarkTimeSeriesExtractor_Extract(b *testing.B) {
	extractor := NewTimeSeriesExtractor(TimeSeriesConfig{})

	// Generate 1000 points
	points := make([]TimeSeriesPoint, 1000)
	for i := 0; i < 1000; i++ {
		points[i] = TimeSeriesPoint{
			Timestamp: time.Now(),
			Value:     float64(i%100) + math.Sin(float64(i)/10),
		}
	}

	ts := &TimeSeries{Points: points}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := extractor.ExtractFeatures(ts)
		if err != nil {
			b.Fatalf("ExtractFeatures failed: %v", err)
		}
	}
}

func BenchmarkTimeSeriesExtractor_RollingWindow(b *testing.B) {
	extractor := NewTimeSeriesExtractor(TimeSeriesConfig{
		WindowSize: 50,
	})

	points := make([]TimeSeriesPoint, 1000)
	for i := 0; i < 1000; i++ {
		points[i] = TimeSeriesPoint{
			Timestamp: time.Now(),
			Value:     float64(i),
		}
	}

	ts := &TimeSeries{Points: points}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := extractor.ExtractFeatures(ts)
		if err != nil {
			b.Fatalf("ExtractFeatures failed: %v", err)
		}
	}
}
