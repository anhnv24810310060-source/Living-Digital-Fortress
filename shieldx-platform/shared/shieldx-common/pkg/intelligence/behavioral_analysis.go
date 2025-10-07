// Package intelligence implements AI-powered traffic analysis
// Real-time behavioral analysis for anomaly detection
package intelligence

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"sync"
	"sync/atomic"
	"time"
)

// BehavioralAnalysisEngine analyzes traffic patterns in real-time
type BehavioralAnalysisEngine struct {
	config    *AnalysisConfig
	profiles  sync.Map // tenant -> *BehaviorProfile
	alertChan chan *ThreatAlert
	mu        sync.RWMutex
	running   atomic.Bool

	// Metrics
	eventsProcessed      atomic.Uint64
	anomaliesDetected    atomic.Uint64
	botTrafficDetected   atomic.Uint64
	ddosDetected         atomic.Uint64
	exfiltrationDetected atomic.Uint64
}

// AnalysisConfig holds configuration for behavioral analysis
type AnalysisConfig struct {
	// Detection thresholds
	BotDetectionThreshold  float64 // 0.995 = 99.5% confidence
	DDoSDetectionThreshold float64
	ExfiltrationThreshold  float64
	AnomalyThreshold       float64 // z-score threshold

	// Time windows
	ShortTermWindow  time.Duration // 5 minutes
	MediumTermWindow time.Duration // 1 hour
	LongTermWindow   time.Duration // 24 hours

	// Feature extraction
	EnableRequestPatterns bool
	EnableTimingAnalysis  bool
	EnablePayloadAnalysis bool
	EnableGraphAnalysis   bool

	// Performance
	MaxProfilesInMemory int
	ProfileTTL          time.Duration
}

// BehaviorProfile stores learned behavior for a tenant/user/IP
type BehaviorProfile struct {
	ID         string
	Type       string // "tenant", "user", "ip", "endpoint"
	CreatedAt  time.Time
	LastUpdate time.Time

	// Time-series statistics
	RequestRates *TimeSeriesStats
	ErrorRates   *TimeSeriesStats
	LatencyStats *TimeSeriesStats
	PayloadSizes *TimeSeriesStats

	// Pattern features
	EndpointSequences [][]string     // common endpoint access patterns
	TimeOfDayPattern  [24]float64    // request distribution by hour
	UserAgentEntropy  float64        // entropy of user agents
	GeoLocations      map[string]int // country -> count

	// Graph features (for relationship analysis)
	ConnectedEntities map[string]float64 // entity -> edge weight

	mu sync.RWMutex
}

// TimeSeriesStats holds statistical data for time series analysis
type TimeSeriesStats struct {
	Values     []float64
	Timestamps []int64
	Mean       float64
	StdDev     float64
	Trend      float64   // positive = increasing, negative = decreasing
	Seasonal   []float64 // seasonal decomposition
	MaxSize    int
	mu         sync.RWMutex
}

// ThreatAlert represents a detected threat
type ThreatAlert struct {
	Timestamp   time.Time
	Severity    string // "critical", "high", "medium", "low"
	Type        string // "bot", "ddos", "exfiltration", "anomaly"
	ProfileID   string
	ProfileType string
	Confidence  float64 // 0.0 to 1.0
	Features    map[string]interface{}
	Details     string
	Recommended string // recommended action
}

// TrafficEvent represents a single traffic event for analysis
type TrafficEvent struct {
	Timestamp   time.Time
	TenantID    string
	UserID      string
	SourceIP    string
	ClientIP    string
	Endpoint    string
	Path        string
	Method      string
	StatusCode  int
	Latency     time.Duration
	PayloadSize int64
	UserAgent   string
	Country     string
	Headers     map[string]string
	TLSVersion  string
	CipherSuite string
}

// NewBehavioralAnalysisEngine creates a new analysis engine
func NewBehavioralAnalysisEngine(cfg *AnalysisConfig) *BehavioralAnalysisEngine {
	if cfg == nil {
		cfg = DefaultAnalysisConfig()
	}
	return &BehavioralAnalysisEngine{
		config:    cfg,
		alertChan: make(chan *ThreatAlert, 1000),
	}
}

// DefaultAnalysisConfig returns default configuration
func DefaultAnalysisConfig() *AnalysisConfig {
	return &AnalysisConfig{
		BotDetectionThreshold:  0.995,
		DDoSDetectionThreshold: 0.98,
		ExfiltrationThreshold:  0.95,
		AnomalyThreshold:       3.0, // 3 standard deviations
		ShortTermWindow:        5 * time.Minute,
		MediumTermWindow:       1 * time.Hour,
		LongTermWindow:         24 * time.Hour,
		EnableRequestPatterns:  true,
		EnableTimingAnalysis:   true,
		EnablePayloadAnalysis:  true,
		EnableGraphAnalysis:    true,
		MaxProfilesInMemory:    10000,
		ProfileTTL:             7 * 24 * time.Hour,
	}
}

// Start begins the analysis engine
func (e *BehavioralAnalysisEngine) Start(ctx context.Context) error {
	if !e.running.CompareAndSwap(false, true) {
		return fmt.Errorf("already running")
	}

	log.Printf("[intelligence] started behavioral analysis engine")

	// Start background tasks
	go e.cleanupProfiles(ctx)
	go e.updateModels(ctx)

	return nil
}

// ProcessEvent processes a single traffic event
func (e *BehavioralAnalysisEngine) ProcessEvent(event *TrafficEvent) {
	e.eventsProcessed.Add(1)

	// Update multiple profiles (tenant, user, IP)
	e.updateProfile("tenant:"+event.TenantID, event)
	if event.UserID != "" {
		e.updateProfile("user:"+event.UserID, event)
	}
	e.updateProfile("ip:"+event.SourceIP, event)
	e.updateProfile("endpoint:"+event.Endpoint, event)

	// Run detection algorithms
	e.detectBot(event)
	e.detectDDoS(event)
	e.detectExfiltration(event)
	e.detectAnomaly(event)
}

// updateProfile updates or creates a behavior profile
func (e *BehavioralAnalysisEngine) updateProfile(profileID string, event *TrafficEvent) {
	var profile *BehaviorProfile

	if val, ok := e.profiles.Load(profileID); ok {
		profile = val.(*BehaviorProfile)
	} else {
		profile = e.createProfile(profileID, event)
		e.profiles.Store(profileID, profile)
	}

	profile.mu.Lock()
	defer profile.mu.Unlock()

	profile.LastUpdate = time.Now()

	// Update time series
	latencyMs := float64(event.Latency.Milliseconds())
	payloadKB := float64(event.PayloadSize) / 1024.0
	errorRate := 0.0
	if event.StatusCode >= 400 {
		errorRate = 1.0
	}

	profile.RequestRates.Add(1.0, event.Timestamp.Unix())
	profile.ErrorRates.Add(errorRate, event.Timestamp.Unix())
	profile.LatencyStats.Add(latencyMs, event.Timestamp.Unix())
	profile.PayloadSizes.Add(payloadKB, event.Timestamp.Unix())

	// Update time-of-day pattern
	hour := event.Timestamp.Hour()
	profile.TimeOfDayPattern[hour] += 1.0

	// Update geolocation distribution
	if event.Country != "" {
		profile.GeoLocations[event.Country]++
	}

	// Calculate user agent entropy (simplified)
	if event.UserAgent != "" {
		profile.UserAgentEntropy = e.calculateEntropy([]string{event.UserAgent})
	}
}

// createProfile creates a new behavior profile
func (e *BehavioralAnalysisEngine) createProfile(profileID string, event *TrafficEvent) *BehaviorProfile {
	profileType := "unknown"
	if len(profileID) > 0 {
		if profileID[0:2] == "te" {
			profileType = "tenant"
		} else if profileID[0:2] == "us" {
			profileType = "user"
		} else if profileID[0:2] == "ip" {
			profileType = "ip"
		} else if profileID[0:2] == "en" {
			profileType = "endpoint"
		}
	}

	return &BehaviorProfile{
		ID:                profileID,
		Type:              profileType,
		CreatedAt:         time.Now(),
		LastUpdate:        time.Now(),
		RequestRates:      NewTimeSeriesStats(1000),
		ErrorRates:        NewTimeSeriesStats(1000),
		LatencyStats:      NewTimeSeriesStats(1000),
		PayloadSizes:      NewTimeSeriesStats(1000),
		GeoLocations:      make(map[string]int),
		ConnectedEntities: make(map[string]float64),
	}
}

// detectBot detects bot traffic patterns
func (e *BehavioralAnalysisEngine) detectBot(event *TrafficEvent) {
	profileID := "ip:" + event.SourceIP
	val, ok := e.profiles.Load(profileID)
	if !ok {
		return
	}
	profile := val.(*BehaviorProfile)

	profile.mu.RLock()
	defer profile.mu.RUnlock()

	// Bot detection features
	score := 0.0

	// 1. Request rate abnormally high
	if profile.RequestRates.Mean > 100.0 { // >100 req/min
		score += 0.3
	}

	// 2. User agent entropy extremely low (same UA every time)
	if profile.UserAgentEntropy < 0.1 {
		score += 0.25
	}

	// 3. No time-of-day pattern (24/7 uniform)
	variance := e.calculateVariance(profile.TimeOfDayPattern[:])
	if variance < 0.01 {
		score += 0.2
	}

	// 4. Error rate high (scanning behavior)
	if profile.ErrorRates.Mean > 0.5 {
		score += 0.15
	}

	// 5. Latency too consistent (not human-like)
	if profile.LatencyStats.StdDev < 5.0 { // <5ms variation
		score += 0.1
	}

	if score >= e.config.BotDetectionThreshold {
		e.botTrafficDetected.Add(1)
		e.raiseAlert(&ThreatAlert{
			Timestamp:   time.Now(),
			Severity:    "high",
			Type:        "bot",
			ProfileID:   profileID,
			ProfileType: "ip",
			Confidence:  score,
			Details:     fmt.Sprintf("Bot-like traffic detected from %s", event.SourceIP),
			Recommended: "block-ip",
		})
	}
}

// detectDDoS detects DDoS attack patterns
func (e *BehavioralAnalysisEngine) detectDDoS(event *TrafficEvent) {
	profileID := "endpoint:" + event.Endpoint
	val, ok := e.profiles.Load(profileID)
	if !ok {
		return
	}
	profile := val.(*BehaviorProfile)

	profile.mu.RLock()
	defer profile.mu.RUnlock()

	// DDoS detection: sudden spike in requests
	if len(profile.RequestRates.Values) < 10 {
		return
	}

	// Calculate rate of change in last 1 minute vs baseline
	recentRate := e.calculateRecentRate(profile.RequestRates, 60) // last 60 seconds
	baselineRate := profile.RequestRates.Mean

	if baselineRate > 0 && recentRate/baselineRate > 10.0 { // 10x spike
		e.ddosDetected.Add(1)
		e.raiseAlert(&ThreatAlert{
			Timestamp:   time.Now(),
			Severity:    "critical",
			Type:        "ddos",
			ProfileID:   profileID,
			ProfileType: "endpoint",
			Confidence:  0.98,
			Features: map[string]interface{}{
				"recentRate":   recentRate,
				"baselineRate": baselineRate,
				"multiplier":   recentRate / baselineRate,
			},
			Details:     fmt.Sprintf("DDoS detected on %s: %.0fx traffic spike", event.Endpoint, recentRate/baselineRate),
			Recommended: "enable-rate-limit",
		})
	}
}

// detectExfiltration detects data exfiltration attempts
func (e *BehavioralAnalysisEngine) detectExfiltration(event *TrafficEvent) {
	profileID := "user:" + event.UserID
	if event.UserID == "" {
		return
	}

	val, ok := e.profiles.Load(profileID)
	if !ok {
		return
	}
	profile := val.(*BehaviorProfile)

	profile.mu.RLock()
	defer profile.mu.RUnlock()

	// Exfiltration detection: abnormal data transfer
	recentPayload := e.calculateRecentRate(profile.PayloadSizes, 300) // last 5 min
	baselinePayload := profile.PayloadSizes.Mean

	if baselinePayload > 0 && recentPayload/baselinePayload > 20.0 { // 20x data volume
		e.exfiltrationDetected.Add(1)
		e.raiseAlert(&ThreatAlert{
			Timestamp:   time.Now(),
			Severity:    "critical",
			Type:        "exfiltration",
			ProfileID:   profileID,
			ProfileType: "user",
			Confidence:  0.95,
			Features: map[string]interface{}{
				"recentPayloadKB":   recentPayload,
				"baselinePayloadKB": baselinePayload,
			},
			Details:     fmt.Sprintf("Possible data exfiltration by user %s", event.UserID),
			Recommended: "suspend-user",
		})
	}
}

// detectAnomaly detects general anomalies using z-score
func (e *BehavioralAnalysisEngine) detectAnomaly(event *TrafficEvent) {
	profileID := "tenant:" + event.TenantID
	val, ok := e.profiles.Load(profileID)
	if !ok {
		return
	}
	profile := val.(*BehaviorProfile)

	profile.mu.RLock()
	defer profile.mu.RUnlock()

	// Check latency anomaly
	latencyMs := float64(event.Latency.Milliseconds())
	if profile.LatencyStats.StdDev > 0 {
		zScore := (latencyMs - profile.LatencyStats.Mean) / profile.LatencyStats.StdDev
		if math.Abs(zScore) > e.config.AnomalyThreshold {
			e.anomaliesDetected.Add(1)
			e.raiseAlert(&ThreatAlert{
				Timestamp:   time.Now(),
				Severity:    "medium",
				Type:        "anomaly",
				ProfileID:   profileID,
				ProfileType: "tenant",
				Confidence:  0.85,
				Features: map[string]interface{}{
					"metric": "latency",
					"value":  latencyMs,
					"mean":   profile.LatencyStats.Mean,
					"stddev": profile.LatencyStats.StdDev,
					"zscore": zScore,
				},
				Details:     fmt.Sprintf("Latency anomaly: %.1fms (%.1f std devs)", latencyMs, zScore),
				Recommended: "investigate",
			})
		}
	}
}

// raiseAlert sends an alert to the alert channel
func (e *BehavioralAnalysisEngine) raiseAlert(alert *ThreatAlert) {
	select {
	case e.alertChan <- alert:
	default:
		log.Printf("[intelligence] alert channel full, dropping %s alert", alert.Type)
	}
}

// Alerts returns the alert channel
func (e *BehavioralAnalysisEngine) Alerts() <-chan *ThreatAlert {
	return e.alertChan
}

// Stats returns engine statistics
func (e *BehavioralAnalysisEngine) Stats() map[string]interface{} {
	profileCount := 0
	e.profiles.Range(func(k, v interface{}) bool {
		profileCount++
		return true
	})

	return map[string]interface{}{
		"eventsProcessed":      e.eventsProcessed.Load(),
		"anomaliesDetected":    e.anomaliesDetected.Load(),
		"botTrafficDetected":   e.botTrafficDetected.Load(),
		"ddosDetected":         e.ddosDetected.Load(),
		"exfiltrationDetected": e.exfiltrationDetected.Load(),
		"profilesInMemory":     profileCount,
	}
}

// cleanupProfiles removes stale profiles
func (e *BehavioralAnalysisEngine) cleanupProfiles(ctx context.Context) {
	ticker := time.NewTicker(10 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			cutoff := time.Now().Add(-e.config.ProfileTTL)
			e.profiles.Range(func(k, v interface{}) bool {
				profile := v.(*BehaviorProfile)
				profile.mu.RLock()
				lastUpdate := profile.LastUpdate
				profile.mu.RUnlock()

				if lastUpdate.Before(cutoff) {
					e.profiles.Delete(k)
				}
				return true
			})
		}
	}
}

// updateModels periodically updates statistical models
func (e *BehavioralAnalysisEngine) updateModels(ctx context.Context) {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			e.profiles.Range(func(k, v interface{}) bool {
				profile := v.(*BehaviorProfile)
				profile.mu.Lock()
				profile.RequestRates.UpdateStats()
				profile.ErrorRates.UpdateStats()
				profile.LatencyStats.UpdateStats()
				profile.PayloadSizes.UpdateStats()
				profile.mu.Unlock()
				return true
			})
		}
	}
}

// Helper functions

func NewTimeSeriesStats(maxSize int) *TimeSeriesStats {
	return &TimeSeriesStats{
		Values:     make([]float64, 0, maxSize),
		Timestamps: make([]int64, 0, maxSize),
		MaxSize:    maxSize,
	}
}

func (ts *TimeSeriesStats) Add(value float64, timestamp int64) {
	ts.mu.Lock()
	defer ts.mu.Unlock()

	ts.Values = append(ts.Values, value)
	ts.Timestamps = append(ts.Timestamps, timestamp)

	// Circular buffer
	if len(ts.Values) > ts.MaxSize {
		ts.Values = ts.Values[1:]
		ts.Timestamps = ts.Timestamps[1:]
	}
}

func (ts *TimeSeriesStats) UpdateStats() {
	ts.mu.Lock()
	defer ts.mu.Unlock()

	if len(ts.Values) == 0 {
		return
	}

	// Calculate mean
	sum := 0.0
	for _, v := range ts.Values {
		sum += v
	}
	ts.Mean = sum / float64(len(ts.Values))

	// Calculate standard deviation
	variance := 0.0
	for _, v := range ts.Values {
		diff := v - ts.Mean
		variance += diff * diff
	}
	variance /= float64(len(ts.Values))
	ts.StdDev = math.Sqrt(variance)

	// Calculate trend (simple linear regression slope)
	if len(ts.Values) >= 2 {
		n := float64(len(ts.Values))
		sumX := 0.0
		sumY := 0.0
		sumXY := 0.0
		sumX2 := 0.0

		for i, y := range ts.Values {
			x := float64(i)
			sumX += x
			sumY += y
			sumXY += x * y
			sumX2 += x * x
		}

		ts.Trend = (n*sumXY - sumX*sumY) / (n*sumX2 - sumX*sumX)
	}
}

func (e *BehavioralAnalysisEngine) calculateEntropy(values []string) float64 {
	if len(values) == 0 {
		return 0.0
	}

	freq := make(map[string]int)
	for _, v := range values {
		freq[v]++
	}

	entropy := 0.0
	total := float64(len(values))
	for _, count := range freq {
		p := float64(count) / total
		if p > 0 {
			entropy -= p * math.Log2(p)
		}
	}

	return entropy
}

func (e *BehavioralAnalysisEngine) calculateVariance(values []float64) float64 {
	if len(values) == 0 {
		return 0.0
	}

	mean := 0.0
	for _, v := range values {
		mean += v
	}
	mean /= float64(len(values))

	variance := 0.0
	for _, v := range values {
		diff := v - mean
		variance += diff * diff
	}
	variance /= float64(len(values))

	return variance
}

func (e *BehavioralAnalysisEngine) calculateRecentRate(ts *TimeSeriesStats, windowSeconds int64) float64 {
	ts.mu.RLock()
	defer ts.mu.RUnlock()

	if len(ts.Values) == 0 {
		return 0.0
	}

	cutoff := time.Now().Unix() - windowSeconds
	sum := 0.0
	count := 0

	for i := len(ts.Values) - 1; i >= 0; i-- {
		if ts.Timestamps[i] < cutoff {
			break
		}
		sum += ts.Values[i]
		count++
	}

	if count == 0 {
		return 0.0
	}

	return sum / float64(count)
}

// ExportProfile exports a profile for external analysis (e.g., ML training)
func (e *BehavioralAnalysisEngine) ExportProfile(profileID string) ([]byte, error) {
	val, ok := e.profiles.Load(profileID)
	if !ok {
		return nil, fmt.Errorf("profile not found")
	}

	profile := val.(*BehaviorProfile)
	profile.mu.RLock()
	defer profile.mu.RUnlock()

	return json.Marshal(profile)
}
