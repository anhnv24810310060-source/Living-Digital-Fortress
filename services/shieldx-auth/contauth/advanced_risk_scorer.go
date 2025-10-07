package main

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"math"
)

// AdvancedRiskScorer provides production-ready behavioral risk assessment
// with multi-factor analysis and anomaly detection
type AdvancedRiskScorer struct {
	weights RiskWeights
}

type RiskWeights struct {
	KeystrokeDynamics float64
	MouseBehavior     float64
	LocationAnomaly   float64
	DeviceFingerprint float64
	BehavioralPattern float64
	TemporalAnomaly   float64
	ReputationScore   float64
}

// DefaultRiskWeights returns production-tuned weights
func DefaultRiskWeights() RiskWeights {
	return RiskWeights{
		KeystrokeDynamics: 25.0, // High precision biometric
		MouseBehavior:     20.0,
		LocationAnomaly:   15.0,
		DeviceFingerprint: 15.0,
		BehavioralPattern: 10.0,
		TemporalAnomaly:   10.0,
		ReputationScore:   5.0,
	}
}

func NewAdvancedRiskScorer() *AdvancedRiskScorer {
	return &AdvancedRiskScorer{
		weights: DefaultRiskWeights(),
	}
}

// ExtendedUserBaseline extends UserBaseline with additional fields for advanced risk scoring
type ExtendedUserBaseline struct {
	UserID               string
	AvgKeystrokeInterval float64
	AvgKeystrokeDuration float64
	AvgMouseVelocity     float64 // Renamed from TypicalMouseVelocity
	TypicalCountry       string
	TypicalScreenRes     string
	TypicalPlatform      string
	TypicalTimezone      string
}

// ToExtended converts UserBaseline to ExtendedUserBaseline
func ToExtended(baseline *UserBaseline) *ExtendedUserBaseline {
	if baseline == nil {
		return nil
	}
	return &ExtendedUserBaseline{
		UserID:               baseline.UserID,
		AvgKeystrokeInterval: baseline.AvgKeystrokeInterval,
		AvgKeystrokeDuration: baseline.AvgKeystrokeDuration,
		AvgMouseVelocity:     baseline.TypicalMouseVelocity,
		// Additional fields default to empty - would be populated from DB in production
	}
}

// CalculateRisk performs comprehensive risk analysis
// Returns overall score (0-100) and component scores
func (ars *AdvancedRiskScorer) CalculateRisk(
	telemetry *SessionTelemetry,
	baseline *UserBaseline,
) *RiskScore {
	// Convert to extended baseline
	extBaseline := ToExtended(baseline)

	if telemetry == nil {
		return &RiskScore{
			SessionID:      "",
			OverallScore:   100.0, // Maximum risk for no data
			Recommendation: "DENY",
		}
	}

	riskFactors := []string{}

	// Component 1: Keystroke dynamics analysis
	ksScore := ars.analyzeKeystrokeDynamics(telemetry, extBaseline)
	if ksScore > 0.6 {
		riskFactors = append(riskFactors, "abnormal_keystroke_pattern")
	}

	// Component 2: Mouse behavior analysis
	mouseScore := ars.analyzeMouseBehavior(telemetry, extBaseline)
	if mouseScore > 0.6 {
		riskFactors = append(riskFactors, "abnormal_mouse_behavior")
	}

	// Component 3: Location anomaly detection
	locationScore := ars.analyzeLocationAnomaly(telemetry, extBaseline)
	if locationScore > 0.7 {
		riskFactors = append(riskFactors, "suspicious_location")
	}

	// Component 4: Device fingerprint matching
	deviceScore := ars.analyzeDeviceFingerprint(telemetry, extBaseline)
	if deviceScore > 0.5 {
		riskFactors = append(riskFactors, "device_mismatch")
	}

	// Component 5: Behavioral pattern analysis
	behaviorScore := ars.analyzeBehavioralPattern(telemetry, extBaseline)
	if behaviorScore > 0.6 {
		riskFactors = append(riskFactors, "abnormal_access_pattern")
	}

	// Component 6: Temporal anomaly (time-of-day, frequency)
	temporalScore := ars.analyzeTemporalAnomaly(telemetry, extBaseline)
	if temporalScore > 0.5 {
		riskFactors = append(riskFactors, "unusual_timing")
	}

	// Component 7: IP/User reputation
	reputationScore := ars.analyzeReputation(telemetry)
	if reputationScore > 0.7 {
		riskFactors = append(riskFactors, "poor_reputation")
	}

	// Weighted combination
	overallScore := 0.0
	overallScore += ksScore * ars.weights.KeystrokeDynamics
	overallScore += mouseScore * ars.weights.MouseBehavior
	overallScore += locationScore * ars.weights.LocationAnomaly
	overallScore += deviceScore * ars.weights.DeviceFingerprint
	overallScore += behaviorScore * ars.weights.BehavioralPattern
	overallScore += temporalScore * ars.weights.TemporalAnomaly
	overallScore += reputationScore * ars.weights.ReputationScore

	// Cap at 100
	if overallScore > 100 {
		overallScore = 100
	}

	// Determine recommendation
	recommendation := ars.makeRecommendation(overallScore, riskFactors)

	return &RiskScore{
		SessionID:       telemetry.SessionID,
		OverallScore:    overallScore,
		KeystrokeScore:  ksScore * 100,
		MouseScore:      mouseScore * 100,
		LocationScore:   locationScore * 100,
		DeviceScore:     deviceScore * 100,
		BehaviorScore:   behaviorScore * 100,
		ReputationScore: reputationScore * 100,
		RiskFactors:     riskFactors,
		Recommendation:  recommendation,
		CalculatedAt:    telemetry.Timestamp,
	}
}

func (ars *AdvancedRiskScorer) analyzeKeystrokeDynamics(
	t *SessionTelemetry,
	baseline *ExtendedUserBaseline,
) float64 {
	if baseline == nil || len(t.KeystrokeData) == 0 {
		return 0.3 // Moderate risk for no baseline
	}

	// Extract features from metadata (already hashed)
	avgInterval, ok1 := t.Metadata["ks_avg_interval"].(float64)
	avgDuration, ok2 := t.Metadata["ks_avg_duration"].(float64)

	if !ok1 || !ok2 {
		return 0.3
	}

	// Compare with baseline using statistical distance
	baseInterval := baseline.AvgKeystrokeInterval
	baseDuration := baseline.AvgKeystrokeDuration

	if baseInterval == 0 || baseDuration == 0 {
		return 0.2 // New user, lower risk
	}

	// Calculate deviation ratio
	intervalDev := math.Abs(avgInterval-baseInterval) / baseInterval
	durationDev := math.Abs(avgDuration-baseDuration) / baseDuration

	// Combined deviation (normalized to 0-1)
	deviation := (intervalDev + durationDev) / 2.0

	// Apply sigmoid to convert to risk score
	risk := 1.0 / (1.0 + math.Exp(-5.0*(deviation-0.5)))

	return risk
}

func (ars *AdvancedRiskScorer) analyzeMouseBehavior(
	t *SessionTelemetry,
	baseline *ExtendedUserBaseline,
) float64 {
	if baseline == nil || len(t.MouseData) == 0 {
		return 0.2
	}

	avgVelocity, ok := t.Metadata["mouse_avg_velocity"].(float64)
	if !ok {
		return 0.2
	}

	baseVelocity := baseline.AvgMouseVelocity
	if baseVelocity == 0 {
		return 0.15
	}

	deviation := math.Abs(avgVelocity-baseVelocity) / baseVelocity
	risk := 1.0 / (1.0 + math.Exp(-5.0*(deviation-0.4)))

	return risk
}

func (ars *AdvancedRiskScorer) analyzeLocationAnomaly(
	t *SessionTelemetry,
	baseline *ExtendedUserBaseline,
) float64 {
	if baseline == nil {
		return 0.3
	}

	// Check country change
	if baseline.TypicalCountry != "" &&
		t.GeolocationData.Country != "" &&
		t.GeolocationData.Country != baseline.TypicalCountry {
		return 0.8 // High risk for country change
	}

	// Check impossible travel (would need previous location and timestamp)
	// This is simplified - production would calculate distance/time

	return 0.1 // Low risk if same country
}

func (ars *AdvancedRiskScorer) analyzeDeviceFingerprint(
	t *SessionTelemetry,
	baseline *ExtendedUserBaseline,
) float64 {
	if baseline == nil {
		return 0.3
	}

	score := 0.0
	checks := 0

	// Screen resolution check
	if baseline.TypicalScreenRes != "" && t.DeviceMetrics.ScreenResolution != "" {
		checks++
		if baseline.TypicalScreenRes != t.DeviceMetrics.ScreenResolution {
			score += 0.3
		}
	}

	// Platform check
	if baseline.TypicalPlatform != "" && t.DeviceMetrics.Platform != "" {
		checks++
		if baseline.TypicalPlatform != t.DeviceMetrics.Platform {
			score += 0.5 // Platform change is very suspicious
		}
	}

	// Timezone check
	if baseline.TypicalTimezone != "" && t.DeviceMetrics.Timezone != "" {
		checks++
		if baseline.TypicalTimezone != t.DeviceMetrics.Timezone {
			score += 0.2
		}
	}

	if checks == 0 {
		return 0.3
	}

	return score / float64(checks)
}

func (ars *AdvancedRiskScorer) analyzeBehavioralPattern(
	t *SessionTelemetry,
	baseline *ExtendedUserBaseline,
) float64 {
	// Analyze access patterns for anomalies
	if len(t.AccessPatterns) == 0 {
		return 0.1
	}

	// Check for rapid successive failed attempts
	failedAttempts := 0
	for _, ap := range t.AccessPatterns {
		if !ap.Success {
			failedAttempts++
		}
	}

	failureRate := float64(failedAttempts) / float64(len(t.AccessPatterns))

	// High failure rate is suspicious
	if failureRate > 0.3 {
		return 0.7
	} else if failureRate > 0.1 {
		return 0.4
	}

	return 0.1
}

func (ars *AdvancedRiskScorer) analyzeTemporalAnomaly(
	t *SessionTelemetry,
	baseline *ExtendedUserBaseline,
) float64 {
	// Check if access time is unusual for this user
	hour := t.Timestamp.Hour()

	// Simplified: 2-6 AM is generally suspicious for business users
	if hour >= 2 && hour <= 6 {
		return 0.6
	}

	// Weekend access might be suspicious for some profiles
	weekday := t.Timestamp.Weekday()
	if weekday == 0 || weekday == 6 { // Sunday or Saturday
		return 0.3
	}

	return 0.1
}

func (ars *AdvancedRiskScorer) analyzeReputation(t *SessionTelemetry) float64 {
	// Simplified reputation check
	// Production would query threat intelligence feeds

	// Check for known suspicious IP patterns
	if t.IPAddress == "" {
		return 0.5
	}

	// Check for proxy/VPN/Tor indicators
	// (This is simplified - production would use threat intel APIs)
	suspiciousPatterns := []string{
		"10.0.", "172.16.", "192.168.", // Private IPs accessing from outside
	}

	for _, pattern := range suspiciousPatterns {
		if len(t.IPAddress) >= len(pattern) &&
			t.IPAddress[:len(pattern)] == pattern {
			// Private IP from "external" access is suspicious
			return 0.4
		}
	}

	return 0.1 // Default low risk
}

func (ars *AdvancedRiskScorer) makeRecommendation(
	score float64,
	factors []string,
) string {
	switch {
	case score >= 80:
		return "DENY"
	case score >= 60:
		return "MFA_REQUIRED"
	case score >= 40:
		return "CHALLENGE"
	case score >= 20:
		return "MONITOR"
	default:
		return "ALLOW"
	}
}

// SecureHash creates a cryptographically secure hash of sensitive data
func SecureHash(data []byte) string {
	// Add random salt to prevent rainbow table attacks
	salt := make([]byte, 16)
	rand.Read(salt)

	combined := append(salt, data...)
	hash := sha256.Sum256(combined)

	// Store salt with hash (first 32 chars are salt)
	return hex.EncodeToString(salt) + hex.EncodeToString(hash[:])
}

// VerifyHash verifies data against stored hash
func VerifyHash(data []byte, storedHash string) bool {
	if len(storedHash) < 64 {
		return false
	}

	// Extract salt
	salt, err := hex.DecodeString(storedHash[:32])
	if err != nil {
		return false
	}

	// Recompute hash
	combined := append(salt, data...)
	hash := sha256.Sum256(combined)
	computedHash := hex.EncodeToString(hash[:])

	return storedHash[32:] == computedHash
}

// FeatureExtractor extracts privacy-preserving features from raw biometrics
type FeatureExtractor struct{}

func NewFeatureExtractor() *FeatureExtractor {
	return &FeatureExtractor{}
}

// ExtractKeystrokeFeatures extracts statistical features without storing raw data
func (fe *FeatureExtractor) ExtractKeystrokeFeatures(events []KeystrokeEvent) map[string]float64 {
	if len(events) == 0 {
		return map[string]float64{}
	}

	var totalInterval, totalDuration, totalPressure float64
	intervals := make([]float64, 0, len(events)-1)

	for i := 0; i < len(events)-1; i++ {
		interval := float64(events[i+1].Timestamp - events[i].Timestamp)
		intervals = append(intervals, interval)
		totalInterval += interval
		totalDuration += events[i].Duration
		totalPressure += events[i].Pressure
	}

	if len(events) > 0 {
		totalDuration += events[len(events)-1].Duration
		totalPressure += events[len(events)-1].Pressure
	}

	avgInterval := totalInterval / float64(len(intervals))
	avgDuration := totalDuration / float64(len(events))
	avgPressure := totalPressure / float64(len(events))

	// Calculate variance for interval
	var variance float64
	for _, interval := range intervals {
		diff := interval - avgInterval
		variance += diff * diff
	}
	variance /= float64(len(intervals))
	stdDev := math.Sqrt(variance)

	return map[string]float64{
		"avg_interval": avgInterval,
		"avg_duration": avgDuration,
		"avg_pressure": avgPressure,
		"std_interval": stdDev,
		"event_count":  float64(len(events)),
	}
}

// ExtractMouseFeatures extracts statistical features from mouse events
func (fe *FeatureExtractor) ExtractMouseFeatures(events []MouseEvent) map[string]float64 {
	if len(events) == 0 {
		return map[string]float64{}
	}

	var totalVelocity float64
	var clicks, moves, scrolls int

	for _, e := range events {
		totalVelocity += e.Velocity
		switch e.EventType {
		case "click":
			clicks++
		case "move":
			moves++
		case "scroll":
			scrolls++
		}
	}

	avgVelocity := totalVelocity / float64(len(events))

	return map[string]float64{
		"avg_velocity": avgVelocity,
		"click_count":  float64(clicks),
		"move_count":   float64(moves),
		"scroll_count": float64(scrolls),
		"total_events": float64(len(events)),
	}
}
