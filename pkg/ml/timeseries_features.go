package ml

import (
	"fmt"
	"math"
	"sync"
	"time"
)

// TimeSeriesFeatures represents extracted time-series features
type TimeSeriesFeatures struct {
	// Statistical features
	Mean              float64
	Median            float64
	Std               float64
	Min               float64
	Max               float64
	Range             float64
	
	// Distribution features
	Skewness          float64 // Measure of asymmetry
	Kurtosis          float64 // Measure of tailedness
	Percentile25      float64
	Percentile75      float64
	IQR               float64 // Interquartile range
	
	// Temporal features
	Trend             float64 // Linear trend coefficient
	Seasonality       float64 // Seasonal strength
	AutoCorr1         float64 // Lag-1 autocorrelation
	AutoCorr24        float64 // Lag-24 autocorrelation (daily pattern)
	
	// Change features
	DiffMean          float64 // Mean of first differences
	DiffStd           float64 // Std of first differences
	AbsChanges        float64 // Sum of absolute changes
	ChangeRate        float64 // Rate of change
	
	// Pattern features
	NumPeaks          int     // Number of local maxima
	NumValleys        int     // Number of local minima
	LongestRun        int     // Longest consecutive same-direction run
	ZeroCrossings     int     // Number of zero crossings
	
	// Complexity features
	Entropy           float64 // Shannon entropy
	Complexity        float64 // Lempel-Ziv complexity
	ApproxEntropy     float64 // Approximate entropy
	
	// Rolling window features
	RollingMean       []float64 // Moving average
	RollingStd        []float64 // Moving std
	EWMA              []float64 // Exponentially weighted moving average
}

// TimeSeriesPoint represents a single time-series data point
type TimeSeriesPoint struct {
	Timestamp time.Time
	Value     float64
	Tags      map[string]string
}

// TimeSeries represents a time series
type TimeSeries struct {
	Points    []TimeSeriesPoint
	StartTime time.Time
	EndTime   time.Time
	Interval  time.Duration
}

// TimeSeriesExtractor extracts features from time series data
type TimeSeriesExtractor struct {
	mu sync.RWMutex
	
	windowSize     int           // Window size for rolling features
	lagPeriods     []int         // Lag periods for autocorrelation
	trendMethod    string        // "linear", "polynomial"
	seasonalPeriod int           // Period for seasonality detection
}

// TimeSeriesConfig configures the extractor
type TimeSeriesConfig struct {
	WindowSize     int
	LagPeriods     []int
	TrendMethod    string
	SeasonalPeriod int
}

// NewTimeSeriesExtractor creates a new time-series feature extractor
func NewTimeSeriesExtractor(config TimeSeriesConfig) *TimeSeriesExtractor {
	if config.WindowSize <= 0 {
		config.WindowSize = 10
	}
	if len(config.LagPeriods) == 0 {
		config.LagPeriods = []int{1, 24}
	}
	if config.TrendMethod == "" {
		config.TrendMethod = "linear"
	}
	if config.SeasonalPeriod <= 0 {
		config.SeasonalPeriod = 24
	}

	return &TimeSeriesExtractor{
		windowSize:     config.WindowSize,
		lagPeriods:     config.LagPeriods,
		trendMethod:    config.TrendMethod,
		seasonalPeriod: config.SeasonalPeriod,
	}
}

// ExtractFeatures extracts features from a time series
func (tse *TimeSeriesExtractor) ExtractFeatures(ts *TimeSeries) (*TimeSeriesFeatures, error) {
	if len(ts.Points) == 0 {
		return nil, fmt.Errorf("empty time series")
	}

	values := make([]float64, len(ts.Points))
	for i, p := range ts.Points {
		values[i] = p.Value
	}

	features := &TimeSeriesFeatures{}

	// Statistical features
	features.Mean = tse.calculateMean(values)
	features.Median = tse.calculateMedian(values)
	features.Std = tse.calculateStd(values, features.Mean)
	features.Min = tse.calculateMin(values)
	features.Max = tse.calculateMax(values)
	features.Range = features.Max - features.Min

	// Distribution features
	features.Skewness = tse.calculateSkewness(values, features.Mean, features.Std)
	features.Kurtosis = tse.calculateKurtosis(values, features.Mean, features.Std)
	features.Percentile25 = tse.calculatePercentile(values, 0.25)
	features.Percentile75 = tse.calculatePercentile(values, 0.75)
	features.IQR = features.Percentile75 - features.Percentile25

	// Temporal features
	features.Trend = tse.calculateTrend(values)
	features.Seasonality = tse.calculateSeasonality(values)
	features.AutoCorr1 = tse.calculateAutoCorr(values, 1)
	if len(values) > 24 {
		features.AutoCorr24 = tse.calculateAutoCorr(values, 24)
	}

	// Change features
	diffs := tse.calculateDifferences(values)
	if len(diffs) > 0 {
		features.DiffMean = tse.calculateMean(diffs)
		features.DiffStd = tse.calculateStd(diffs, features.DiffMean)
		features.AbsChanges = tse.sumAbsolute(diffs)
		features.ChangeRate = features.AbsChanges / float64(len(values))
	}

	// Pattern features
	features.NumPeaks = tse.countPeaks(values)
	features.NumValleys = tse.countValleys(values)
	features.LongestRun = tse.longestRun(values)
	features.ZeroCrossings = tse.countZeroCrossings(values, features.Mean)

	// Complexity features
	features.Entropy = tse.calculateEntropy(values)
	features.Complexity = tse.calculateComplexity(values)
	features.ApproxEntropy = tse.calculateApproxEntropy(values)

	// Rolling window features
	features.RollingMean = tse.calculateRollingMean(values, tse.windowSize)
	features.RollingStd = tse.calculateRollingStd(values, tse.windowSize)
	features.EWMA = tse.calculateEWMA(values, 0.3)

	return features, nil
}

// Statistical calculations

func (tse *TimeSeriesExtractor) calculateMean(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

func (tse *TimeSeriesExtractor) calculateMedian(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	
	sorted := make([]float64, len(values))
	copy(sorted, values)
	
	// Simple bubble sort for median
	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[j] < sorted[i] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}
	
	mid := len(sorted) / 2
	if len(sorted)%2 == 0 {
		return (sorted[mid-1] + sorted[mid]) / 2
	}
	return sorted[mid]
}

func (tse *TimeSeriesExtractor) calculateStd(values []float64, mean float64) float64 {
	if len(values) == 0 {
		return 0
	}
	
	sumSq := 0.0
	for _, v := range values {
		diff := v - mean
		sumSq += diff * diff
	}
	return math.Sqrt(sumSq / float64(len(values)))
}

func (tse *TimeSeriesExtractor) calculateMin(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	min := values[0]
	for _, v := range values {
		if v < min {
			min = v
		}
	}
	return min
}

func (tse *TimeSeriesExtractor) calculateMax(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	max := values[0]
	for _, v := range values {
		if v > max {
			max = v
		}
	}
	return max
}

func (tse *TimeSeriesExtractor) calculatePercentile(values []float64, p float64) float64 {
	if len(values) == 0 {
		return 0
	}
	
	sorted := make([]float64, len(values))
	copy(sorted, values)
	
	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[j] < sorted[i] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}
	
	idx := int(float64(len(sorted)-1) * p)
	return sorted[idx]
}

// Distribution features

func (tse *TimeSeriesExtractor) calculateSkewness(values []float64, mean, std float64) float64 {
	if len(values) == 0 || std == 0 {
		return 0
	}
	
	sum := 0.0
	for _, v := range values {
		z := (v - mean) / std
		sum += z * z * z
	}
	return sum / float64(len(values))
}

func (tse *TimeSeriesExtractor) calculateKurtosis(values []float64, mean, std float64) float64 {
	if len(values) == 0 || std == 0 {
		return 0
	}
	
	sum := 0.0
	for _, v := range values {
		z := (v - mean) / std
		sum += z * z * z * z
	}
	return sum/float64(len(values)) - 3.0 // Excess kurtosis
}

// Temporal features

func (tse *TimeSeriesExtractor) calculateTrend(values []float64) float64 {
	if len(values) < 2 {
		return 0
	}
	
	// Linear regression slope
	n := float64(len(values))
	sumX := 0.0
	sumY := 0.0
	sumXY := 0.0
	sumXX := 0.0
	
	for i, y := range values {
		x := float64(i)
		sumX += x
		sumY += y
		sumXY += x * y
		sumXX += x * x
	}
	
	slope := (n*sumXY - sumX*sumY) / (n*sumXX - sumX*sumX)
	return slope
}

func (tse *TimeSeriesExtractor) calculateSeasonality(values []float64) float64 {
	if len(values) < tse.seasonalPeriod*2 {
		return 0
	}
	
	// Simplified: calculate autocorrelation at seasonal lag
	return tse.calculateAutoCorr(values, tse.seasonalPeriod)
}

func (tse *TimeSeriesExtractor) calculateAutoCorr(values []float64, lag int) float64 {
	if len(values) <= lag {
		return 0
	}
	
	mean := tse.calculateMean(values)
	
	// Calculate covariance at lag
	cov := 0.0
	for i := 0; i < len(values)-lag; i++ {
		cov += (values[i] - mean) * (values[i+lag] - mean)
	}
	
	// Calculate variance
	variance := 0.0
	for _, v := range values {
		variance += (v - mean) * (v - mean)
	}
	
	if variance == 0 {
		return 0
	}
	
	return cov / variance
}

// Change features

func (tse *TimeSeriesExtractor) calculateDifferences(values []float64) []float64 {
	if len(values) < 2 {
		return []float64{}
	}
	
	diffs := make([]float64, len(values)-1)
	for i := 1; i < len(values); i++ {
		diffs[i-1] = values[i] - values[i-1]
	}
	return diffs
}

func (tse *TimeSeriesExtractor) sumAbsolute(values []float64) float64 {
	sum := 0.0
	for _, v := range values {
		sum += math.Abs(v)
	}
	return sum
}

// Pattern features

func (tse *TimeSeriesExtractor) countPeaks(values []float64) int {
	if len(values) < 3 {
		return 0
	}
	
	count := 0
	for i := 1; i < len(values)-1; i++ {
		if values[i] > values[i-1] && values[i] > values[i+1] {
			count++
		}
	}
	return count
}

func (tse *TimeSeriesExtractor) countValleys(values []float64) int {
	if len(values) < 3 {
		return 0
	}
	
	count := 0
	for i := 1; i < len(values)-1; i++ {
		if values[i] < values[i-1] && values[i] < values[i+1] {
			count++
		}
	}
	return count
}

func (tse *TimeSeriesExtractor) longestRun(values []float64) int {
	if len(values) < 2 {
		return len(values)
	}
	
	maxRun := 1
	currentRun := 1
	increasing := values[1] > values[0]
	
	for i := 2; i < len(values); i++ {
		if (increasing && values[i] > values[i-1]) || 
		   (!increasing && values[i] < values[i-1]) {
			currentRun++
		} else {
			if currentRun > maxRun {
				maxRun = currentRun
			}
			currentRun = 1
			increasing = values[i] > values[i-1]
		}
	}
	
	if currentRun > maxRun {
		maxRun = currentRun
	}
	
	return maxRun
}

func (tse *TimeSeriesExtractor) countZeroCrossings(values []float64, mean float64) int {
	if len(values) < 2 {
		return 0
	}
	
	count := 0
	prevSign := values[0] > mean
	
	for i := 1; i < len(values); i++ {
		currentSign := values[i] > mean
		if prevSign != currentSign {
			count++
		}
		prevSign = currentSign
	}
	
	return count
}

// Complexity features

func (tse *TimeSeriesExtractor) calculateEntropy(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	
	// Discretize values into bins
	nBins := 10
	min := tse.calculateMin(values)
	max := tse.calculateMax(values)
	
	if max == min {
		return 0
	}
	
	binWidth := (max - min) / float64(nBins)
	counts := make([]int, nBins)
	
	for _, v := range values {
		bin := int((v - min) / binWidth)
		if bin >= nBins {
			bin = nBins - 1
		}
		counts[bin]++
	}
	
	// Calculate entropy
	entropy := 0.0
	total := float64(len(values))
	for _, count := range counts {
		if count > 0 {
			p := float64(count) / total
			entropy -= p * math.Log2(p)
		}
	}
	
	return entropy
}

func (tse *TimeSeriesExtractor) calculateComplexity(values []float64) float64 {
	// Simplified Lempel-Ziv complexity
	if len(values) < 2 {
		return 0
	}
	
	// Convert to binary string based on median
	median := tse.calculateMedian(values)
	binary := make([]int, len(values))
	for i, v := range values {
		if v > median {
			binary[i] = 1
		}
	}
	
	// Count unique substrings
	unique := make(map[string]bool)
	for i := 0; i < len(binary); i++ {
		for j := i + 1; j <= len(binary); j++ {
			substr := fmt.Sprintf("%v", binary[i:j])
			unique[substr] = true
		}
	}
	
	return float64(len(unique)) / float64(len(values))
}

func (tse *TimeSeriesExtractor) calculateApproxEntropy(values []float64) float64 {
	if len(values) < 2 {
		return 0
	}
	
	m := 2 // Pattern length
	r := 0.2 * tse.calculateStd(values, tse.calculateMean(values)) // Tolerance
	
	if r == 0 {
		return 0
	}
	
	phi := func(m int) float64 {
		patterns := make(map[string]int)
		
		for i := 0; i <= len(values)-m; i++ {
			pattern := ""
			for j := 0; j < m; j++ {
				pattern += fmt.Sprintf("%.2f,", values[i+j])
			}
			patterns[pattern]++
		}
		
		sum := 0.0
		n := float64(len(values) - m + 1)
		for _, count := range patterns {
			p := float64(count) / n
			sum += p * math.Log(p)
		}
		
		return sum / n
	}
	
	return phi(m) - phi(m+1)
}

// Rolling window features

func (tse *TimeSeriesExtractor) calculateRollingMean(values []float64, window int) []float64 {
	if len(values) < window {
		return []float64{}
	}
	
	rolling := make([]float64, len(values)-window+1)
	for i := 0; i <= len(values)-window; i++ {
		sum := 0.0
		for j := 0; j < window; j++ {
			sum += values[i+j]
		}
		rolling[i] = sum / float64(window)
	}
	
	return rolling
}

func (tse *TimeSeriesExtractor) calculateRollingStd(values []float64, window int) []float64 {
	if len(values) < window {
		return []float64{}
	}
	
	rolling := make([]float64, len(values)-window+1)
	for i := 0; i <= len(values)-window; i++ {
		mean := 0.0
		for j := 0; j < window; j++ {
			mean += values[i+j]
		}
		mean /= float64(window)
		
		variance := 0.0
		for j := 0; j < window; j++ {
			diff := values[i+j] - mean
			variance += diff * diff
		}
		rolling[i] = math.Sqrt(variance / float64(window))
	}
	
	return rolling
}

func (tse *TimeSeriesExtractor) calculateEWMA(values []float64, alpha float64) []float64 {
	if len(values) == 0 {
		return []float64{}
	}
	
	ewma := make([]float64, len(values))
	ewma[0] = values[0]
	
	for i := 1; i < len(values); i++ {
		ewma[i] = alpha*values[i] + (1-alpha)*ewma[i-1]
	}
	
	return ewma
}

// ToVector converts time-series features to vector
func (tsf *TimeSeriesFeatures) ToVector() []float64 {
	return []float64{
		tsf.Mean,
		tsf.Median,
		tsf.Std,
		tsf.Min,
		tsf.Max,
		tsf.Range,
		tsf.Skewness,
		tsf.Kurtosis,
		tsf.Percentile25,
		tsf.Percentile75,
		tsf.IQR,
		tsf.Trend,
		tsf.Seasonality,
		tsf.AutoCorr1,
		tsf.AutoCorr24,
		tsf.DiffMean,
		tsf.DiffStd,
		tsf.AbsChanges,
		tsf.ChangeRate,
		float64(tsf.NumPeaks),
		float64(tsf.NumValleys),
		float64(tsf.LongestRun),
		float64(tsf.ZeroCrossings),
		tsf.Entropy,
		tsf.Complexity,
		tsf.ApproxEntropy,
	}
}
