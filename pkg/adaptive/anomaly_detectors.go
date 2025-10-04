// Package adaptive - Anomaly Detection Algorithms
package adaptive

import (
	"math"
	"sync"
)

// ZScoreDetector implements Z-score (standard score) anomaly detection
// Detects outliers based on standard deviations from the mean
type ZScoreDetector struct {
	mu       sync.RWMutex
	mean     float64
	stdDev   float64
	threshold float64 // Number of standard deviations for anomaly
	trained  bool
}

// NewZScoreDetector creates a Z-score detector
// threshold: number of std deviations (typically 2-3)
func NewZScoreDetector(threshold float64) *ZScoreDetector {
	return &ZScoreDetector{
		threshold: threshold,
	}
}

// Train computes mean and standard deviation from historical data
func (zsd *ZScoreDetector) Train(data []float64) error {
	if len(data) == 0 {
		return nil
	}
	
	zsd.mu.Lock()
	defer zsd.mu.Unlock()
	
	// Compute mean
	sum := 0.0
	for _, v := range data {
		sum += v
	}
	zsd.mean = sum / float64(len(data))
	
	// Compute standard deviation
	variance := 0.0
	for _, v := range data {
		diff := v - zsd.mean
		variance += diff * diff
	}
	variance /= float64(len(data))
	zsd.stdDev = math.Sqrt(variance)
	
	zsd.trained = true
	return nil
}

// Detect checks if a value is an anomaly
// Returns (isAnomaly, anomalyScore)
func (zsd *ZScoreDetector) Detect(value float64) (bool, float64) {
	zsd.mu.RLock()
	defer zsd.mu.RUnlock()
	
	if !zsd.trained || zsd.stdDev == 0 {
		return false, 0.0
	}
	
	// Calculate Z-score
	zScore := math.Abs((value - zsd.mean) / zsd.stdDev)
	
	// Normalize score to 0-1 range
	normalizedScore := math.Min(zScore/10.0, 1.0)
	
	isAnomaly := zScore > zsd.threshold
	return isAnomaly, normalizedScore
}

func (zsd *ZScoreDetector) Algorithm() string {
	return "z-score"
}

// IQRDetector implements Interquartile Range (IQR) anomaly detection
// More robust to outliers than Z-score
type IQRDetector struct {
	mu    sync.RWMutex
	q1    float64 // 25th percentile
	q3    float64 // 75th percentile
	iqr   float64 // q3 - q1
	k     float64 // IQR multiplier (typically 1.5)
	trained bool
}

func NewIQRDetector() *IQRDetector {
	return &IQRDetector{
		k: 1.5,
	}
}

func (iqr *IQRDetector) Train(data []float64) error {
	if len(data) < 4 {
		return nil
	}
	
	iqr.mu.Lock()
	defer iqr.mu.Unlock()
	
	// Sort data (simplified - real impl would use quickselect)
	sorted := make([]float64, len(data))
	copy(sorted, data)
	bubbleSort(sorted)
	
	// Compute quartiles
	n := len(sorted)
	iqr.q1 = sorted[n/4]
	iqr.q3 = sorted[3*n/4]
	iqr.iqr = iqr.q3 - iqr.q1
	
	iqr.trained = true
	return nil
}

func (iqr *IQRDetector) Detect(value float64) (bool, float64) {
	iqr.mu.RLock()
	defer iqr.mu.RUnlock()
	
	if !iqr.trained || iqr.iqr == 0 {
		return false, 0.0
	}
	
	lowerBound := iqr.q1 - iqr.k*iqr.iqr
	upperBound := iqr.q3 + iqr.k*iqr.iqr
	
	isAnomaly := value < lowerBound || value > upperBound
	
	// Score based on distance from bounds
	var score float64
	if value < lowerBound {
		score = (lowerBound - value) / (iqr.iqr * iqr.k)
	} else if value > upperBound {
		score = (value - upperBound) / (iqr.iqr * iqr.k)
	}
	score = math.Min(score, 1.0)
	
	return isAnomaly, score
}

func (iqr *IQRDetector) Algorithm() string {
	return "iqr"
}

// MADDetector implements Median Absolute Deviation
// Very robust to outliers
type MADDetector struct {
	mu       sync.RWMutex
	median   float64
	mad      float64
	threshold float64 // Number of MADs
	trained  bool
}

func NewMADDetector(threshold float64) *MADDetector {
	if threshold == 0 {
		threshold = 3.0
	}
	return &MADDetector{
		threshold: threshold,
	}
}

func (mad *MADDetector) Train(data []float64) error {
	if len(data) == 0 {
		return nil
	}
	
	mad.mu.Lock()
	defer mad.mu.Unlock()
	
	// Compute median
	sorted := make([]float64, len(data))
	copy(sorted, data)
	bubbleSort(sorted)
	
	n := len(sorted)
	if n%2 == 0 {
		mad.median = (sorted[n/2-1] + sorted[n/2]) / 2.0
	} else {
		mad.median = sorted[n/2]
	}
	
	// Compute absolute deviations
	deviations := make([]float64, n)
	for i, v := range sorted {
		deviations[i] = math.Abs(v - mad.median)
	}
	bubbleSort(deviations)
	
	// Median of absolute deviations
	if n%2 == 0 {
		mad.mad = (deviations[n/2-1] + deviations[n/2]) / 2.0
	} else {
		mad.mad = deviations[n/2]
	}
	
	mad.trained = true
	return nil
}

func (mad *MADDetector) Detect(value float64) (bool, float64) {
	mad.mu.RLock()
	defer mad.mu.RUnlock()
	
	if !mad.trained || mad.mad == 0 {
		return false, 0.0
	}
	
	// Modified Z-score using MAD
	modifiedZ := 0.6745 * math.Abs(value-mad.median) / mad.mad
	
	normalizedScore := math.Min(modifiedZ/10.0, 1.0)
	isAnomaly := modifiedZ > mad.threshold
	
	return isAnomaly, normalizedScore
}

func (mad *MADDetector) Algorithm() string {
	return "mad"
}

// EWMADetector implements Exponentially Weighted Moving Average
// Good for streaming data with concept drift
type EWMADetector struct {
	mu      sync.RWMutex
	ewma    float64
	ewmsd   float64 // Exponentially weighted moving std dev
	alpha   float64 // Smoothing factor (0-1)
	beta    float64 // Std dev smoothing factor
	threshold float64
	trained bool
}

func NewEWMADetector(alpha, threshold float64) *EWMADetector {
	if alpha == 0 {
		alpha = 0.3
	}
	if threshold == 0 {
		threshold = 3.0
	}
	return &EWMADetector{
		alpha:     alpha,
		beta:      alpha,
		threshold: threshold,
	}
}

func (ewma *EWMADetector) Train(data []float64) error {
	if len(data) == 0 {
		return nil
	}
	
	ewma.mu.Lock()
	defer ewma.mu.Unlock()
	
	// Initialize with first value
	ewma.ewma = data[0]
	ewma.ewmsd = 0.0
	
	// Update with subsequent values
	for _, v := range data[1:] {
		prevEwma := ewma.ewma
		ewma.ewma = ewma.alpha*v + (1-ewma.alpha)*ewma.ewma
		
		// Update std dev
		diff := v - prevEwma
		ewma.ewmsd = math.Sqrt(ewma.beta*diff*diff + (1-ewma.beta)*ewma.ewmsd*ewma.ewmsd)
	}
	
	ewma.trained = true
	return nil
}

func (ewma *EWMADetector) Detect(value float64) (bool, float64) {
	ewma.mu.Lock()
	defer ewma.mu.Unlock()
	
	if !ewma.trained || ewma.ewmsd == 0 {
		// Update EWMA even if not fully trained
		if ewma.ewma == 0 {
			ewma.ewma = value
		} else {
			ewma.ewma = ewma.alpha*value + (1-ewma.alpha)*ewma.ewma
		}
		return false, 0.0
	}
	
	// Check if value is anomalous
	zScore := math.Abs((value - ewma.ewma) / ewma.ewmsd)
	isAnomaly := zScore > ewma.threshold
	normalizedScore := math.Min(zScore/10.0, 1.0)
	
	// Update EWMA with new value
	prevEwma := ewma.ewma
	ewma.ewma = ewma.alpha*value + (1-ewma.alpha)*ewma.ewma
	
	diff := value - prevEwma
	ewma.ewmsd = math.Sqrt(ewma.beta*diff*diff + (1-ewma.beta)*ewma.ewmsd*ewma.ewmsd)
	
	return isAnomaly, normalizedScore
}

func (ewma *EWMADetector) Algorithm() string {
	return "ewma"
}

// Helper: simple bubble sort (for small datasets)
func bubbleSort(arr []float64) {
	n := len(arr)
	for i := 0; i < n-1; i++ {
		for j := 0; j < n-i-1; j++ {
			if arr[j] > arr[j+1] {
				arr[j], arr[j+1] = arr[j+1], arr[j]
			}
		}
	}
}
