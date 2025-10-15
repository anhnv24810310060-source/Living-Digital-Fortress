package ml

import (
	"fmt"
	"math"
	"sync"
	"time"
)

// DriftDetector detects concept drift and data drift
type DriftDetector struct {
	mu sync.RWMutex
	
	name           string
	method         string // "ks", "psi", "wasserstein", "adwin"
	threshold      float64
	windowSize     int
	
	// Reference distribution
	reference      []float64
	referenceMean  float64
	referenceStd   float64
	
	// Current window
	current        []float64
	
	// Drift tracking
	driftDetected  bool
	driftScore     float64
	lastCheckTime  time.Time
	driftHistory   []*DriftEvent
}

// DriftEvent represents a drift detection event
type DriftEvent struct {
	Timestamp   time.Time
	Method      string
	Score       float64
	Threshold   float64
	Detected    bool
	Severity    string // "low", "medium", "high"
	Description string
}

// ModelMonitor monitors model performance
type ModelMonitor struct {
	mu sync.RWMutex
	
	modelName      string
	modelVersion   string
	
	// Performance metrics
	accuracy       float64
	precision      float64
	recall         float64
	f1Score        float64
	
	// Metric history
	metricHistory  []*MetricSnapshot
	
	// Drift detectors
	inputDrift     *DriftDetector
	outputDrift    *DriftDetector
	performanceDrift *DriftDetector
	
	// Alerts
	alerts         []*Alert
	alertThreshold float64
}

// MetricSnapshot represents metrics at a point in time
type MetricSnapshot struct {
	Timestamp   time.Time
	Accuracy    float64
	Precision   float64
	Recall      float64
	F1Score     float64
	Predictions int
	Latency     time.Duration
}

// Alert represents a monitoring alert
type Alert struct {
	ID          string
	Timestamp   time.Time
	Severity    string // "info", "warning", "critical"
	Type        string // "drift", "performance", "latency"
	Message     string
	Metric      string
	Value       float64
	Threshold   float64
	Resolved    bool
}

// DriftConfig configures drift detection
type DriftConfig struct {
	Name       string
	Method     string
	Threshold  float64
	WindowSize int
}

// MonitorConfig configures model monitoring
type MonitorConfig struct {
	ModelName      string
	ModelVersion   string
	AlertThreshold float64
	DriftMethod    string
}

// NewDriftDetector creates a new drift detector
func NewDriftDetector(config DriftConfig) *DriftDetector {
	if config.Method == "" {
		config.Method = "ks" // Kolmogorov-Smirnov by default
	}
	if config.Threshold <= 0 {
		config.Threshold = 0.05
	}
	if config.WindowSize <= 0 {
		config.WindowSize = 1000
	}
	
	return &DriftDetector{
		name:         config.Name,
		method:       config.Method,
		threshold:    config.Threshold,
		windowSize:   config.WindowSize,
		reference:    make([]float64, 0),
		current:      make([]float64, 0),
		driftHistory: make([]*DriftEvent, 0),
	}
}

// SetReference sets the reference distribution
func (dd *DriftDetector) SetReference(data []float64) error {
	dd.mu.Lock()
	defer dd.mu.Unlock()
	
	if len(data) == 0 {
		return fmt.Errorf("reference data is empty")
	}
	
	dd.reference = make([]float64, len(data))
	copy(dd.reference, data)
	
	// Calculate statistics
	dd.referenceMean = dd.calculateMean(dd.reference)
	dd.referenceStd = dd.calculateStd(dd.reference, dd.referenceMean)
	
	return nil
}

// AddObservation adds a new observation
func (dd *DriftDetector) AddObservation(value float64) {
	dd.mu.Lock()
	defer dd.mu.Unlock()
	
	dd.current = append(dd.current, value)
	
	// Maintain window size
	if len(dd.current) > dd.windowSize {
		dd.current = dd.current[1:]
	}
}

// CheckDrift checks for drift
func (dd *DriftDetector) CheckDrift() (*DriftEvent, error) {
	dd.mu.Lock()
	defer dd.mu.Unlock()
	
	if len(dd.reference) == 0 {
		return nil, fmt.Errorf("reference distribution not set")
	}
	
	if len(dd.current) < 30 {
		return nil, fmt.Errorf("insufficient current data (need at least 30 samples)")
	}
	
	var score float64
	var err error
	
	switch dd.method {
	case "ks":
		score, err = dd.ksTest()
	case "psi":
		score, err = dd.psiTest()
	case "wasserstein":
		score, err = dd.wassersteinDistance()
	default:
		return nil, fmt.Errorf("unknown drift detection method: %s", dd.method)
	}
	
	if err != nil {
		return nil, err
	}
	
	detected := score > dd.threshold
	dd.driftDetected = detected
	dd.driftScore = score
	dd.lastCheckTime = time.Now()
	
	// Determine severity
	severity := "low"
	if score > dd.threshold*2 {
		severity = "high"
	} else if score > dd.threshold*1.5 {
		severity = "medium"
	}
	
	event := &DriftEvent{
		Timestamp:   time.Now(),
		Method:      dd.method,
		Score:       score,
		Threshold:   dd.threshold,
		Detected:    detected,
		Severity:    severity,
		Description: fmt.Sprintf("Drift score: %.4f (threshold: %.4f)", score, dd.threshold),
	}
	
	dd.driftHistory = append(dd.driftHistory, event)
	
	return event, nil
}

// Kolmogorov-Smirnov test
func (dd *DriftDetector) ksTest() (float64, error) {
	ref := make([]float64, len(dd.reference))
	copy(ref, dd.reference)
	
	curr := make([]float64, len(dd.current))
	copy(curr, dd.current)
	
	// Sort both distributions
	dd.sortSlice(ref)
	dd.sortSlice(curr)
	
	// Calculate empirical CDFs and find max difference
	maxDiff := 0.0
	i, j := 0, 0
	
	for i < len(ref) && j < len(curr) {
		cdfRef := float64(i+1) / float64(len(ref))
		cdfCurr := float64(j+1) / float64(len(curr))
		
		diff := math.Abs(cdfRef - cdfCurr)
		if diff > maxDiff {
			maxDiff = diff
		}
		
		if ref[i] < curr[j] {
			i++
		} else {
			j++
		}
	}
	
	return maxDiff, nil
}

// Population Stability Index
func (dd *DriftDetector) psiTest() (float64, error) {
	// Create bins
	nBins := 10
	minVal := math.Min(dd.calculateMin(dd.reference), dd.calculateMin(dd.current))
	maxVal := math.Max(dd.calculateMax(dd.reference), dd.calculateMax(dd.current))
	
	binWidth := (maxVal - minVal) / float64(nBins)
	if binWidth == 0 {
		return 0, nil
	}
	
	// Count frequencies in bins
	refBins := make([]int, nBins)
	currBins := make([]int, nBins)
	
	for _, v := range dd.reference {
		bin := int((v - minVal) / binWidth)
		if bin >= nBins {
			bin = nBins - 1
		}
		if bin < 0 {
			bin = 0
		}
		refBins[bin]++
	}
	
	for _, v := range dd.current {
		bin := int((v - minVal) / binWidth)
		if bin >= nBins {
			bin = nBins - 1
		}
		if bin < 0 {
			bin = 0
		}
		currBins[bin]++
	}
	
	// Calculate PSI
	psi := 0.0
	for i := 0; i < nBins; i++ {
		pRef := float64(refBins[i]) / float64(len(dd.reference))
		pCurr := float64(currBins[i]) / float64(len(dd.current))
		
		// Avoid log(0)
		if pRef == 0 {
			pRef = 0.0001
		}
		if pCurr == 0 {
			pCurr = 0.0001
		}
		
		psi += (pCurr - pRef) * math.Log(pCurr/pRef)
	}
	
	return psi, nil
}

// Wasserstein distance (Earth Mover's Distance)
func (dd *DriftDetector) wassersteinDistance() (float64, error) {
	ref := make([]float64, len(dd.reference))
	copy(ref, dd.reference)
	
	curr := make([]float64, len(dd.current))
	copy(curr, dd.current)
	
	// Sort both
	dd.sortSlice(ref)
	dd.sortSlice(curr)
	
	// Calculate Wasserstein-1 distance
	distance := 0.0
	i, j := 0, 0
	
	for i < len(ref) && j < len(curr) {
		distance += math.Abs(ref[i] - curr[j])
		i++
		j++
	}
	
	// Normalize by size
	n := math.Max(float64(len(ref)), float64(len(curr)))
	return distance / n, nil
}

// Helper functions

func (dd *DriftDetector) calculateMean(data []float64) float64 {
	if len(data) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range data {
		sum += v
	}
	return sum / float64(len(data))
}

func (dd *DriftDetector) calculateStd(data []float64, mean float64) float64 {
	if len(data) == 0 {
		return 0
	}
	sumSq := 0.0
	for _, v := range data {
		diff := v - mean
		sumSq += diff * diff
	}
	return math.Sqrt(sumSq / float64(len(data)))
}

func (dd *DriftDetector) calculateMin(data []float64) float64 {
	if len(data) == 0 {
		return 0
	}
	min := data[0]
	for _, v := range data {
		if v < min {
			min = v
		}
	}
	return min
}

func (dd *DriftDetector) calculateMax(data []float64) float64 {
	if len(data) == 0 {
		return 0
	}
	max := data[0]
	for _, v := range data {
		if v > max {
			max = v
		}
	}
	return max
}

func (dd *DriftDetector) sortSlice(data []float64) {
	// Simple bubble sort
	for i := 0; i < len(data); i++ {
		for j := i + 1; j < len(data); j++ {
			if data[j] < data[i] {
				data[i], data[j] = data[j], data[i]
			}
		}
	}
}

// GetDriftHistory returns drift detection history
func (dd *DriftDetector) GetDriftHistory() []*DriftEvent {
	dd.mu.RLock()
	defer dd.mu.RUnlock()
	
	history := make([]*DriftEvent, len(dd.driftHistory))
	copy(history, dd.driftHistory)
	return history
}

// ModelMonitor implementation

// NewModelMonitor creates a new model monitor
func NewModelMonitor(config MonitorConfig) *ModelMonitor {
	if config.AlertThreshold <= 0 {
		config.AlertThreshold = 0.05
	}
	if config.DriftMethod == "" {
		config.DriftMethod = "ks"
	}
	
	return &ModelMonitor{
		modelName:      config.ModelName,
		modelVersion:   config.ModelVersion,
		metricHistory:  make([]*MetricSnapshot, 0),
		alerts:         make([]*Alert, 0),
		alertThreshold: config.AlertThreshold,
		inputDrift:     NewDriftDetector(DriftConfig{Name: "input", Method: config.DriftMethod, Threshold: config.AlertThreshold}),
		outputDrift:    NewDriftDetector(DriftConfig{Name: "output", Method: config.DriftMethod, Threshold: config.AlertThreshold}),
		performanceDrift: NewDriftDetector(DriftConfig{Name: "performance", Method: config.DriftMethod, Threshold: config.AlertThreshold}),
	}
}

// RecordMetrics records model metrics
func (mm *ModelMonitor) RecordMetrics(accuracy, precision, recall, f1 float64, predictions int, latency time.Duration) {
	mm.mu.Lock()
	defer mm.mu.Unlock()
	
	snapshot := &MetricSnapshot{
		Timestamp:   time.Now(),
		Accuracy:    accuracy,
		Precision:   precision,
		Recall:      recall,
		F1Score:     f1,
		Predictions: predictions,
		Latency:     latency,
	}
	
	mm.metricHistory = append(mm.metricHistory, snapshot)
	mm.accuracy = accuracy
	mm.precision = precision
	mm.recall = recall
	mm.f1Score = f1
	
	// Check for performance degradation
	mm.checkPerformanceDegradation(snapshot)
}

// checkPerformanceDegradation checks if performance has degraded
func (mm *ModelMonitor) checkPerformanceDegradation(current *MetricSnapshot) {
	if len(mm.metricHistory) < 10 {
		return
	}
	
	// Calculate baseline from first 10 snapshots
	baselineAccuracy := 0.0
	for i := 0; i < 10; i++ {
		baselineAccuracy += mm.metricHistory[i].Accuracy
	}
	baselineAccuracy /= 10.0
	
	// Check if current is significantly lower
	degradation := baselineAccuracy - current.Accuracy
	
	if degradation > mm.alertThreshold {
		alert := &Alert{
			ID:        fmt.Sprintf("perf-%d", time.Now().Unix()),
			Timestamp: time.Now(),
			Severity:  "warning",
			Type:      "performance",
			Message:   fmt.Sprintf("Model accuracy dropped by %.2f%%", degradation*100),
			Metric:    "accuracy",
			Value:     current.Accuracy,
			Threshold: baselineAccuracy - mm.alertThreshold,
			Resolved:  false,
		}
		
		if degradation > mm.alertThreshold*2 {
			alert.Severity = "critical"
		}
		
		mm.alerts = append(mm.alerts, alert)
	}
}

// GetMetrics returns current metrics
func (mm *ModelMonitor) GetMetrics() map[string]float64 {
	mm.mu.RLock()
	defer mm.mu.RUnlock()
	
	return map[string]float64{
		"accuracy":  mm.accuracy,
		"precision": mm.precision,
		"recall":    mm.recall,
		"f1_score":  mm.f1Score,
	}
}

// GetAlerts returns active alerts
func (mm *ModelMonitor) GetAlerts() []*Alert {
	mm.mu.RLock()
	defer mm.mu.RUnlock()
	
	activeAlerts := make([]*Alert, 0)
	for _, alert := range mm.alerts {
		if !alert.Resolved {
			activeAlerts = append(activeAlerts, alert)
		}
	}
	return activeAlerts
}

// ResolveAlert resolves an alert
func (mm *ModelMonitor) ResolveAlert(alertID string) error {
	mm.mu.Lock()
	defer mm.mu.Unlock()
	
	for _, alert := range mm.alerts {
		if alert.ID == alertID {
			alert.Resolved = true
			return nil
		}
	}
	
	return fmt.Errorf("alert not found: %s", alertID)
}

// GetMetricHistory returns metric history
func (mm *ModelMonitor) GetMetricHistory() []*MetricSnapshot {
	mm.mu.RLock()
	defer mm.mu.RUnlock()
	
	history := make([]*MetricSnapshot, len(mm.metricHistory))
	copy(history, mm.metricHistory)
	return history
}

// GetInputDrift returns input drift detector
func (mm *ModelMonitor) GetInputDrift() *DriftDetector {
	return mm.inputDrift
}

// GetOutputDrift returns output drift detector
func (mm *ModelMonitor) GetOutputDrift() *DriftDetector {
	return mm.outputDrift
}

// GetPerformanceDrift returns performance drift detector
func (mm *ModelMonitor) GetPerformanceDrift() *DriftDetector {
	return mm.performanceDrift
}
