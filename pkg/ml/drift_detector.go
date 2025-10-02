package ml

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/redis/go-redis/v9"
)

// DriftDetector monitors for feature and prediction drift
type DriftDetector struct {
	redisClient      *redis.Client
	mu               sync.RWMutex
	baselineStats    map[string]*FeatureStats
	currentStats     map[string]*FeatureStats
	driftThreshold   float64
	windowSize       int
	checkIntervalSec int
	alertCallback    func(DriftAlert)
}

// Prometheus metrics
var (
	driftAlerts = prometheus.NewCounterVec(
		prometheus.CounterOpts{Namespace: "ml", Subsystem: "drift", Name: "alerts_total", Help: "Total number of drift alerts by severity."},
		[]string{"severity", "type", "feature"},
	)
	driftScoreGauge = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{Namespace: "ml", Subsystem: "drift", Name: "score", Help: "Latest drift score per feature."},
		[]string{"feature"},
	)
)

func init() {
	_ = prometheus.Register(driftAlerts)
	_ = prometheus.Register(driftScoreGauge)
}

// FeatureStats stores statistical information about a feature
type FeatureStats struct {
	FeatureName string    `json:"feature_name"`
	Mean        float64   `json:"mean"`
	Variance    float64   `json:"variance"`
	Min         float64   `json:"min"`
	Max         float64   `json:"max"`
	Count       int       `json:"count"`
	Histogram   []int     `json:"histogram"`
	Bins        []float64 `json:"bins"`
	UpdatedAt   time.Time `json:"updated_at"`
}

// DriftAlert represents a detected drift event
type DriftAlert struct {
	AlertID        string        `json:"alert_id"`
	Type           DriftType     `json:"type"`
	FeatureName    string        `json:"feature_name,omitempty"`
	Severity       DriftSeverity `json:"severity"`
	DriftScore     float64       `json:"drift_score"`
	Method         string        `json:"method"`
	BaselineStats  *FeatureStats `json:"baseline_stats"`
	CurrentStats   *FeatureStats `json:"current_stats"`
	DetectedAt     time.Time     `json:"detected_at"`
	Message        string        `json:"message"`
	Recommendation string        `json:"recommendation"`
}

// DriftType represents the type of drift detected
type DriftType string

const (
	DriftTypeFeature    DriftType = "feature"
	DriftTypePrediction DriftType = "prediction"
	DriftTypeConcept    DriftType = "concept"
)

// DriftSeverity indicates the severity of detected drift
type DriftSeverity string

const (
	DriftSeverityLow      DriftSeverity = "low"
	DriftSeverityMedium   DriftSeverity = "medium"
	DriftSeverityHigh     DriftSeverity = "high"
	DriftSeverityCritical DriftSeverity = "critical"
)

// NewDriftDetector creates a new drift detector
func NewDriftDetector(redisClient *redis.Client, driftThreshold float64, windowSize int) *DriftDetector {
	if driftThreshold == 0 {
		driftThreshold = 0.05 // Default 5% threshold
	}
	if windowSize == 0 {
		windowSize = 1000 // Default window of 1000 samples
	}

	detector := &DriftDetector{
		redisClient:      redisClient,
		baselineStats:    make(map[string]*FeatureStats),
		currentStats:     make(map[string]*FeatureStats),
		driftThreshold:   driftThreshold,
		windowSize:       windowSize,
		checkIntervalSec: 300, // Check every 5 minutes
	}

	return detector
}

// SetBaseline sets the baseline statistics for drift detection
func (dd *DriftDetector) SetBaseline(ctx context.Context, features map[string][]float64) error {
	dd.mu.Lock()
	defer dd.mu.Unlock()

	for name, values := range features {
		stats := calculateFeatureStats(name, values)
		dd.baselineStats[name] = stats

		// Persist to Redis
		if err := dd.persistStats(ctx, "baseline", name, stats); err != nil {
			return fmt.Errorf("failed to persist baseline for %s: %w", name, err)
		}
	}

	return nil
}

// RecordFeature records a new feature value for drift monitoring
func (dd *DriftDetector) RecordFeature(ctx context.Context, featureName string, value float64) error {
	dd.mu.Lock()
	defer dd.mu.Unlock()

	// Get or create current stats
	stats, exists := dd.currentStats[featureName]
	if !exists {
		stats = &FeatureStats{
			FeatureName: featureName,
			Min:         value,
			Max:         value,
			Count:       0,
			Histogram:   make([]int, 10), // 10 bins
		}
		dd.currentStats[featureName] = stats
	}

	// Update statistics incrementally
	stats.Count++

	// Update min/max
	if value < stats.Min {
		stats.Min = value
	}
	if value > stats.Max {
		stats.Max = value
	}

	// Update mean and variance (Welford's online algorithm)
	delta := value - stats.Mean
	stats.Mean += delta / float64(stats.Count)
	delta2 := value - stats.Mean
	stats.Variance += delta * delta2

	stats.UpdatedAt = time.Now()

	// Reset window if size exceeded
	if stats.Count >= dd.windowSize {
		if err := dd.checkDrift(ctx, featureName); err != nil {
			return fmt.Errorf("drift check failed: %w", err)
		}
		// Reset current stats after check
		dd.currentStats[featureName] = &FeatureStats{
			FeatureName: featureName,
			Min:         value,
			Max:         value,
			Count:       0,
			Histogram:   make([]int, 10),
		}
	}

	return nil
}

// CheckDrift performs drift detection for a specific feature
func (dd *DriftDetector) checkDrift(ctx context.Context, featureName string) error {
	baseline, hasBaseline := dd.baselineStats[featureName]
	current, hasCurrent := dd.currentStats[featureName]

	if !hasBaseline || !hasCurrent {
		return nil // Cannot check without both baseline and current
	}

	if current.Count < 100 {
		return nil // Not enough samples
	}

	// Calculate Kolmogorov-Smirnov test statistic
	ksStatistic := dd.calculateKSStatistic(baseline, current)

	// Calculate Population Stability Index (PSI)
	psi := dd.calculatePSI(baseline, current)

	// Determine drift
	driftDetected := ksStatistic > dd.driftThreshold || psi > 0.1

	if driftDetected {
		severity := dd.determineSeverity(ksStatistic, psi)

		alert := DriftAlert{
			AlertID:        fmt.Sprintf("drift_%s_%d", featureName, time.Now().Unix()),
			Type:           DriftTypeFeature,
			FeatureName:    featureName,
			Severity:       severity,
			DriftScore:     math.Max(ksStatistic, psi),
			Method:         "KS-Test + PSI",
			BaselineStats:  baseline,
			CurrentStats:   current,
			DetectedAt:     time.Now(),
			Message:        fmt.Sprintf("Feature drift detected: KS=%.4f, PSI=%.4f", ksStatistic, psi),
			Recommendation: dd.getRecommendation(severity),
		}

		// Store alert
		if err := dd.storeAlert(ctx, &alert); err != nil {
			return fmt.Errorf("failed to store alert: %w", err)
		}

		// Metrics
		driftAlerts.WithLabelValues(string(severity), string(DriftTypeFeature), featureName).Inc()
		driftScoreGauge.WithLabelValues(featureName).Set(math.Max(ksStatistic, psi))

		// Trigger callback
		if dd.alertCallback != nil {
			go dd.alertCallback(alert)
		}
	}

	return nil
}

// MonitorAllFeatures continuously monitors all features for drift
func (dd *DriftDetector) MonitorAllFeatures(ctx context.Context) {
	ticker := time.NewTicker(time.Duration(dd.checkIntervalSec) * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			dd.mu.RLock()
			features := make([]string, 0, len(dd.currentStats))
			for name := range dd.currentStats {
				features = append(features, name)
			}
			dd.mu.RUnlock()

			for _, feature := range features {
				if err := dd.checkDrift(ctx, feature); err != nil {
					// Log error but continue monitoring
					fmt.Printf("Drift check error for %s: %v\n", feature, err)
				}
			}
		}
	}
}

// GetDriftReports returns all drift alerts within a time range
func (dd *DriftDetector) GetDriftReports(ctx context.Context, since time.Time) ([]DriftAlert, error) {
	pattern := "drift:alert:*"
	keys, err := dd.redisClient.Keys(ctx, pattern).Result()
	if err != nil {
		return nil, fmt.Errorf("failed to get alert keys: %w", err)
	}

	var alerts []DriftAlert
	for _, key := range keys {
		data, err := dd.redisClient.Get(ctx, key).Bytes()
		if err != nil {
			continue
		}

		var alert DriftAlert
		if err := json.Unmarshal(data, &alert); err != nil {
			continue
		}

		if alert.DetectedAt.After(since) {
			alerts = append(alerts, alert)
		}
	}

	return alerts, nil
}

// SetAlertCallback sets a callback function for drift alerts
func (dd *DriftDetector) SetAlertCallback(callback func(DriftAlert)) {
	dd.alertCallback = callback
}

// Helper functions

func calculateFeatureStats(name string, values []float64) *FeatureStats {
	if len(values) == 0 {
		return &FeatureStats{FeatureName: name}
	}

	stats := &FeatureStats{
		FeatureName: name,
		Count:       len(values),
		Min:         values[0],
		Max:         values[0],
		UpdatedAt:   time.Now(),
	}

	// Calculate mean
	sum := 0.0
	for _, v := range values {
		sum += v
		if v < stats.Min {
			stats.Min = v
		}
		if v > stats.Max {
			stats.Max = v
		}
	}
	stats.Mean = sum / float64(len(values))

	// Calculate variance
	variance := 0.0
	for _, v := range values {
		diff := v - stats.Mean
		variance += diff * diff
	}
	stats.Variance = variance / float64(len(values))

	// Create histogram (10 bins)
	stats.Histogram = make([]int, 10)
	stats.Bins = make([]float64, 11)

	binWidth := (stats.Max - stats.Min) / 10.0
	for i := 0; i <= 10; i++ {
		stats.Bins[i] = stats.Min + float64(i)*binWidth
	}

	for _, v := range values {
		bin := int((v - stats.Min) / binWidth)
		if bin >= 10 {
			bin = 9
		}
		if bin < 0 {
			bin = 0
		}
		stats.Histogram[bin]++
	}

	return stats
}

func (dd *DriftDetector) calculateKSStatistic(baseline, current *FeatureStats) float64 {
	// Simplified KS statistic using histogram bins
	maxDiff := 0.0

	for i := 0; i < len(baseline.Histogram) && i < len(current.Histogram); i++ {
		baselinePct := float64(baseline.Histogram[i]) / float64(baseline.Count)
		currentPct := float64(current.Histogram[i]) / float64(current.Count)

		diff := math.Abs(baselinePct - currentPct)
		if diff > maxDiff {
			maxDiff = diff
		}
	}

	return maxDiff
}

func (dd *DriftDetector) calculatePSI(baseline, current *FeatureStats) float64 {
	// Population Stability Index
	psi := 0.0

	for i := 0; i < len(baseline.Histogram) && i < len(current.Histogram); i++ {
		baselinePct := float64(baseline.Histogram[i]) / float64(baseline.Count)
		currentPct := float64(current.Histogram[i]) / float64(current.Count)

		// Avoid log(0)
		if baselinePct < 0.0001 {
			baselinePct = 0.0001
		}
		if currentPct < 0.0001 {
			currentPct = 0.0001
		}

		psi += (currentPct - baselinePct) * math.Log(currentPct/baselinePct)
	}

	return psi
}

func (dd *DriftDetector) determineSeverity(ksStatistic, psi float64) DriftSeverity {
	maxScore := math.Max(ksStatistic, psi)

	if maxScore > 0.25 {
		return DriftSeverityCritical
	} else if maxScore > 0.15 {
		return DriftSeverityHigh
	} else if maxScore > 0.10 {
		return DriftSeverityMedium
	}
	return DriftSeverityLow
}

func (dd *DriftDetector) getRecommendation(severity DriftSeverity) string {
	switch severity {
	case DriftSeverityCritical:
		return "Immediate action required: Retrain model with recent data"
	case DriftSeverityHigh:
		return "Model retraining recommended within 24 hours"
	case DriftSeverityMedium:
		return "Monitor closely, consider retraining within a week"
	case DriftSeverityLow:
		return "Continue monitoring, no immediate action needed"
	default:
		return "No action required"
	}
}

func (dd *DriftDetector) persistStats(ctx context.Context, statsType, featureName string, stats *FeatureStats) error {
	key := fmt.Sprintf("drift:stats:%s:%s", statsType, featureName)
	data, err := json.Marshal(stats)
	if err != nil {
		return err
	}

	return dd.redisClient.Set(ctx, key, data, 7*24*time.Hour).Err()
}

func (dd *DriftDetector) storeAlert(ctx context.Context, alert *DriftAlert) error {
	key := fmt.Sprintf("drift:alert:%s", alert.AlertID)
	data, err := json.Marshal(alert)
	if err != nil {
		return err
	}

	// Store for 30 days
	return dd.redisClient.Set(ctx, key, data, 30*24*time.Hour).Err()
}
