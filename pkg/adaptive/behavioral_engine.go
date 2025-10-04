// Package adaptive provides real-time behavioral analysis for traffic intelligence
// Uses streaming analytics and ML-based anomaly detection
package adaptive

import (
	"context"
	"math"
	"sync"
	"sync/atomic"
	"time"
)

// BehavioralEngine performs real-time traffic analysis with ML-based anomaly detection
type BehavioralEngine struct {
	mu sync.RWMutex
	
	// Time-series analysis
	metrics      map[string]*MetricTimeSeries
	patterns     map[string]*Pattern
	
	// Anomaly detection
	detectors    map[string]AnomalyDetector
	threshold    float64
	
	// Bot detection
	botDetector  *BotDetector
	
	// DDoS detection
	ddosDetector *DDoSDetector
	
	// Graph analysis (for relationship detection)
	graph        *TrafficGraph
	
	// Metrics
	requestsAnalyzed uint64
	anomaliesFound   uint64
	botsDetected     uint64
	ddosEventsCount  uint64
	
	// Config
	windowSize     time.Duration
	sensitivity    float64
}

// EngineConfig configures the behavioral engine
type EngineConfig struct {
	WindowSize     time.Duration
	Sensitivity    float64 // 0.0-1.0, higher = more sensitive
	EnableBotDetection  bool
	EnableDDoSDetection bool
	EnableGraphAnalysis bool
}

// MetricTimeSeries stores time-series data with seasonal decomposition support
type MetricTimeSeries struct {
	mu        sync.RWMutex
	datapoints []DataPoint
	capacity  int
	
	// Seasonal decomposition components
	trend      []float64
	seasonal   []float64
	residual   []float64
	
	// Statistics
	mean   float64
	stdDev float64
}

type DataPoint struct {
	Timestamp time.Time
	Value     float64
	Labels    map[string]string
}

// Pattern represents a detected traffic pattern
type Pattern struct {
	Type        PatternType
	Confidence  float64
	FirstSeen   time.Time
	LastSeen    time.Time
	Occurrences uint64
	Attributes  map[string]interface{}
}

type PatternType string

const (
	PatternBotTraffic         PatternType = "bot_traffic"
	PatternDDoS               PatternType = "ddos"
	PatternDataExfiltration   PatternType = "data_exfiltration"
	PatternCredentialStuffing PatternType = "credential_stuffing"
	PatternScanningBehavior   PatternType = "scanning"
	PatternRateLimitEvasion   PatternType = "rate_limit_evasion"
)

// AnomalyDetector defines the interface for anomaly detection algorithms
type AnomalyDetector interface {
	Train(data []float64) error
	Detect(value float64) (isAnomaly bool, score float64)
	Algorithm() string
}

// NewBehavioralEngine creates a new behavioral analysis engine
func NewBehavioralEngine(cfg EngineConfig) *BehavioralEngine {
	if cfg.WindowSize == 0 {
		cfg.WindowSize = 5 * time.Minute
	}
	if cfg.Sensitivity == 0 {
		cfg.Sensitivity = 0.7
	}
	
	eng := &BehavioralEngine{
		metrics:    make(map[string]*MetricTimeSeries),
		patterns:   make(map[string]*Pattern),
		detectors:  make(map[string]AnomalyDetector),
		threshold:  cfg.Sensitivity,
		windowSize: cfg.WindowSize,
		sensitivity: cfg.Sensitivity,
	}
	
	// Initialize sub-detectors
	if cfg.EnableBotDetection {
		eng.botDetector = NewBotDetector()
	}
	if cfg.EnableDDoSDetection {
		eng.ddosDetector = NewDDoSDetector(cfg.WindowSize)
	}
	if cfg.EnableGraphAnalysis {
		eng.graph = NewTrafficGraph()
	}
	
	// Start background analyzers
	go eng.continuousAnalysis()
	
	return eng
}

// RecordRequest records a request for analysis
func (e *BehavioralEngine) RecordRequest(req *Request) *AnalysisResult {
	atomic.AddUint64(&e.requestsAnalyzed, 1)
	
	result := &AnalysisResult{
		Timestamp:  time.Now(),
		RequestID:  req.ID,
		IsAnomaly:  false,
		Score:      0.0,
		Patterns:   []PatternType{},
		Attributes: make(map[string]interface{}),
	}
	
	// Bot detection
	if e.botDetector != nil {
		if isBot, confidence := e.botDetector.Detect(req); isBot {
			atomic.AddUint64(&e.botsDetected, 1)
			result.IsAnomaly = true
			result.Score = math.Max(result.Score, confidence)
			result.Patterns = append(result.Patterns, PatternBotTraffic)
			result.Attributes["bot_confidence"] = confidence
		}
	}
	
	// DDoS detection
	if e.ddosDetector != nil {
		if e.ddosDetector.RecordAndCheck(req) {
			atomic.AddUint64(&e.ddosEventsCount, 1)
			result.IsAnomaly = true
			result.Score = 1.0 // DDoS is critical
			result.Patterns = append(result.Patterns, PatternDDoS)
		}
	}
	
	// Update metrics
	e.updateMetrics(req)
	
	// Graph analysis (relationship detection)
	if e.graph != nil {
		e.graph.RecordInteraction(req.SourceIP, req.TargetEndpoint)
		if suspicious := e.graph.DetectSuspiciousPatterns(req.SourceIP); suspicious {
			result.IsAnomaly = true
			result.Score = math.Max(result.Score, 0.8)
			result.Patterns = append(result.Patterns, PatternScanningBehavior)
		}
	}
	
	// Check for credential stuffing
	if e.detectCredentialStuffing(req) {
		result.IsAnomaly = true
		result.Score = math.Max(result.Score, 0.9)
		result.Patterns = append(result.Patterns, PatternCredentialStuffing)
	}
	
	// Data exfiltration detection (large payload size)
	if req.ResponseSize > 10*1024*1024 { // > 10MB
		result.IsAnomaly = true
		result.Score = math.Max(result.Score, 0.7)
		result.Patterns = append(result.Patterns, PatternDataExfiltration)
		result.Attributes["response_size_mb"] = float64(req.ResponseSize) / (1024 * 1024)
	}
	
	if result.IsAnomaly {
		atomic.AddUint64(&e.anomaliesFound, 1)
	}
	
	return result
}

// updateMetrics updates time-series metrics
func (e *BehavioralEngine) updateMetrics(req *Request) {
	e.mu.Lock()
	defer e.mu.Unlock()
	
	// Requests per second
	key := "rps:" + req.SourceIP
	ts := e.getOrCreateTimeSeries(key, 1000)
	ts.AddPoint(DataPoint{
		Timestamp: time.Now(),
		Value:     1.0,
		Labels:    map[string]string{"ip": req.SourceIP},
	})
	
	// Response time
	if req.Duration > 0 {
		key = "latency:" + req.TargetEndpoint
		ts = e.getOrCreateTimeSeries(key, 1000)
		ts.AddPoint(DataPoint{
			Timestamp: time.Now(),
			Value:     float64(req.Duration.Milliseconds()),
			Labels:    map[string]string{"endpoint": req.TargetEndpoint},
		})
	}
	
	// Payload size
	if req.PayloadSize > 0 {
		key = "payload:" + req.SourceIP
		ts = e.getOrCreateTimeSeries(key, 1000)
		ts.AddPoint(DataPoint{
			Timestamp: time.Now(),
			Value:     float64(req.PayloadSize),
			Labels:    map[string]string{"ip": req.SourceIP},
		})
	}
}

func (e *BehavioralEngine) getOrCreateTimeSeries(key string, capacity int) *MetricTimeSeries {
	if ts, ok := e.metrics[key]; ok {
		return ts
	}
	ts := &MetricTimeSeries{
		datapoints: make([]DataPoint, 0, capacity),
		capacity:   capacity,
	}
	e.metrics[key] = ts
	return ts
}

// detectCredentialStuffing uses heuristics to detect credential stuffing attacks
func (e *BehavioralEngine) detectCredentialStuffing(req *Request) bool {
	// Check for rapid login attempts from same IP
	if req.Path == "/login" || req.Path == "/auth" {
		e.mu.RLock()
		key := "login:" + req.SourceIP
		ts, ok := e.metrics[key]
		e.mu.RUnlock()
		
		if ok {
			ts.mu.RLock()
			// Count requests in last 60 seconds
			cutoff := time.Now().Add(-60 * time.Second)
			count := 0
			for _, dp := range ts.datapoints {
				if dp.Timestamp.After(cutoff) {
					count++
				}
			}
			ts.mu.RUnlock()
			
			// More than 20 login attempts in 60s is suspicious
			if count > 20 {
				return true
			}
		}
	}
	
	return false
}

// continuousAnalysis runs background time-series analysis
func (e *BehavioralEngine) continuousAnalysis() {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()
	
	for range ticker.C {
		e.mu.RLock()
		for key, ts := range e.metrics {
			// Perform seasonal decomposition
			ts.Decompose()
			
			// Train anomaly detector
			if _, ok := e.detectors[key]; !ok {
				e.detectors[key] = NewZScoreDetector(3.0) // 3 sigma
			}
			
			ts.mu.RLock()
			values := make([]float64, len(ts.datapoints))
			for i, dp := range ts.datapoints {
				values[i] = dp.Value
			}
			ts.mu.RUnlock()
			
			if len(values) > 10 {
				e.detectors[key].Train(values)
			}
		}
		e.mu.RUnlock()
	}
}

// AddPoint adds a data point to the time series
func (ts *MetricTimeSeries) AddPoint(dp DataPoint) {
	ts.mu.Lock()
	defer ts.mu.Unlock()
	
	// Maintain fixed capacity (ring buffer)
	if len(ts.datapoints) >= ts.capacity {
		ts.datapoints = ts.datapoints[1:]
	}
	ts.datapoints = append(ts.datapoints, dp)
	
	// Update statistics
	ts.updateStats()
}

// updateStats computes mean and standard deviation
func (ts *MetricTimeSeries) updateStats() {
	if len(ts.datapoints) == 0 {
		return
	}
	
	// Mean
	sum := 0.0
	for _, dp := range ts.datapoints {
		sum += dp.Value
	}
	ts.mean = sum / float64(len(ts.datapoints))
	
	// Standard deviation
	variance := 0.0
	for _, dp := range ts.datapoints {
		diff := dp.Value - ts.mean
		variance += diff * diff
	}
	ts.stdDev = math.Sqrt(variance / float64(len(ts.datapoints)))
}

// Decompose performs seasonal decomposition (simplified STL decomposition)
func (ts *MetricTimeSeries) Decompose() {
	ts.mu.Lock()
	defer ts.mu.Unlock()
	
	n := len(ts.datapoints)
	if n < 10 {
		return
	}
	
	// Simple moving average for trend
	ts.trend = make([]float64, n)
	windowSize := 5
	for i := 0; i < n; i++ {
		start := i - windowSize/2
		end := i + windowSize/2 + 1
		if start < 0 {
			start = 0
		}
		if end > n {
			end = n
		}
		
		sum := 0.0
		count := 0
		for j := start; j < end; j++ {
			sum += ts.datapoints[j].Value
			count++
		}
		ts.trend[i] = sum / float64(count)
	}
	
	// Detrended = original - trend
	detrended := make([]float64, n)
	for i := 0; i < n; i++ {
		detrended[i] = ts.datapoints[i].Value - ts.trend[i]
	}
	
	// Seasonal component (simplified - daily pattern)
	period := 24 // Assuming hourly data
	if period > n {
		period = n
	}
	ts.seasonal = make([]float64, n)
	for i := 0; i < n; i++ {
		ts.seasonal[i] = detrended[i%period]
	}
	
	// Residual = detrended - seasonal
	ts.residual = make([]float64, n)
	for i := 0; i < n; i++ {
		ts.residual[i] = detrended[i] - ts.seasonal[i]
	}
}

// Request represents an analyzed HTTP request
type Request struct {
	ID             string
	Timestamp      time.Time
	SourceIP       string
	TargetEndpoint string
	Path           string
	Method         string
	UserAgent      string
	Duration       time.Duration
	StatusCode     int
	PayloadSize    int64
	ResponseSize   int64
	Headers        map[string]string
}

// AnalysisResult contains the result of behavioral analysis
type AnalysisResult struct {
	Timestamp  time.Time
	RequestID  string
	IsAnomaly  bool
	Score      float64 // 0.0-1.0, higher = more anomalous
	Patterns   []PatternType
	Attributes map[string]interface{}
}

// Metrics returns engine metrics
func (e *BehavioralEngine) Metrics() map[string]uint64 {
	return map[string]uint64{
		"requests_analyzed":   atomic.LoadUint64(&e.requestsAnalyzed),
		"anomalies_found":     atomic.LoadUint64(&e.anomaliesFound),
		"bots_detected":       atomic.LoadUint64(&e.botsDetected),
		"ddos_events":         atomic.LoadUint64(&e.ddosEventsCount),
		"metrics_tracked":     uint64(len(e.metrics)),
		"patterns_learned":    uint64(len(e.patterns)),
	}
}
