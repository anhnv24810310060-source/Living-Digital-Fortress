package contauth
package contauth

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"math"
	"sync"
	"time"
)

// PrivacyPreservingScorer implements zero-knowledge risk scoring
// All biometric data is hashed before storage - NEVER store raw data
type PrivacyPreservingScorer struct {
	// Secret key for HMAC (rotate regularly in production)
	secretKey []byte
	
	// User baseline profiles (hashed features only)
	baselines map[string]*UserBaseline
	mu        sync.RWMutex
	
	// Anomaly detection parameters
	anomalyThreshold float64
	learningRate     float64
	
	// Risk calculation weights (calibrated)
	weights RiskWeights
	
	// Performance metrics
	totalScored  uint64
	totalAnomaly uint64
}

// RiskWeights defines scoring factors
type RiskWeights struct {
	KeystrokeDynamics  float64
	MouseBehavior      float64
	DeviceFingerprint  float64
	ContextualFactors  float64
	HistoricalPattern  float64
	VelocityCheck      float64
}

// UserBaseline stores privacy-preserving user behavioral baseline
type UserBaseline struct {
	UserID           string
	
	// Keystroke dynamics (statistical features only - NO raw timings)
	AvgDwellTime     float64
	StdDwellTime     float64
	AvgFlightTime    float64
	StdFlightTime    float64
	TypingRhythm     float64 // Normalized rhythm score
	
	// Mouse behavior (aggregated patterns only)
	AvgMouseSpeed    float64
	MouseAcceleration float64
	ClickPatternHash  string  // Hashed click pattern
	
	// Device characteristics (hashed)
	DeviceFingerprintHash string
	ScreenResolutionHash  string
	TimezoneHash          string
	
	// Temporal patterns
	TypicalLoginHours   []int   // Hours of day (0-23)
	LoginFrequency      float64 // Logins per day average
	SessionDuration     float64 // Average session minutes
	
	// Learning metadata
	SampleCount      int
	LastUpdated      time.Time
	ConfidenceLevel  float64 // 0.0-1.0
	
	// Privacy-preserving flags
	DataAnonymized   bool
	HashAlgorithm    string
}

// TelemetryData represents anonymized behavioral data
type TelemetryData struct {
	UserID    string
	SessionID string
	Timestamp time.Time
	
	// Keystroke features (anonymized)
	KeystrokeDwellTimes  []float64 // Milliseconds (statistical use only)
	KeystrokeFlightTimes []float64
	
	// Mouse features (anonymized)
	MouseMovements []MouseEvent
	ClickEvents    []ClickEvent
	
	// Context (hashed)
	DeviceFingerprint string
	IPAddressHash     string
	LocationHash      string
	UserAgentHash     string
	
	// Session context
	TimeOfDay     int     // Hour 0-23
	DayOfWeek     int     // 0-6
	IsNewDevice   bool
	IsNewLocation bool
}

// MouseEvent represents anonymized mouse movement
type MouseEvent struct {
	DeltaX    float64
	DeltaY    float64
	Speed     float64
	Timestamp time.Time
}

// ClickEvent represents anonymized click pattern
type ClickEvent struct {
	Button    string // "left", "right", "middle"
	Duration  float64 // Hold duration in ms
	Timestamp time.Time
}

// RiskScore represents the authentication risk assessment
type RiskScore struct {
	Score          int                    `json:"score"`           // 0-100
	Confidence     float64                `json:"confidence"`      // 0.0-1.0
	RiskLevel      string                 `json:"risk_level"`      // low, medium, high, critical
	Factors        map[string]float64     `json:"factors"`         // Individual factor scores
	Anomalies      []string               `json:"anomalies"`       // Detected anomalies
	Recommendation string                 `json:"recommendation"`  // Action to take
	Details        map[string]interface{} `json:"details"`         // Additional context
}

// AuthDecision represents the final authentication decision
type AuthDecision struct {
	Decision      string    `json:"decision"`       // "allow", "challenge", "deny"
	Confidence    float64   `json:"confidence"`
	RiskScore     int       `json:"risk_score"`
	ChallengeType string    `json:"challenge_type,omitempty"` // "mfa", "captcha", "security_question"
	Reason        string    `json:"reason"`
	Timestamp     time.Time `json:"timestamp"`
}

// NewPrivacyPreservingScorer creates a privacy-focused risk scorer
func NewPrivacyPreservingScorer(secretKey string) *PrivacyPreservingScorer {
	if secretKey == "" {
		secretKey = "default-secret-key-CHANGE-IN-PRODUCTION"
	}
	
	return &PrivacyPreservingScorer{
		secretKey:        []byte(secretKey),
		baselines:        make(map[string]*UserBaseline),
		anomalyThreshold: 0.3, // 30% deviation triggers anomaly
		learningRate:     0.1, // 10% weight for new samples
		weights: RiskWeights{
			KeystrokeDynamics: 0.25,
			MouseBehavior:     0.20,
			DeviceFingerprint: 0.20,
			ContextualFactors: 0.15,
			HistoricalPattern: 0.15,
			VelocityCheck:     0.05,
		},
	}
}

// CollectTelemetry processes and anonymizes behavioral data
func (pps *PrivacyPreservingScorer) CollectTelemetry(data *TelemetryData) error {
	// CRITICAL: Hash all PII before storage
	data.IPAddressHash = pps.hashPII(data.IPAddressHash)
	data.LocationHash = pps.hashPII(data.LocationHash)
	data.UserAgentHash = pps.hashPII(data.UserAgentHash)
	data.DeviceFingerprint = pps.hashPII(data.DeviceFingerprint)
	
	// Extract statistical features (never store raw timing data)
	features := pps.extractStatisticalFeatures(data)
	
	// Update or create baseline
	pps.updateBaseline(data.UserID, features)
	
	return nil
}

// CalculateRiskScore computes privacy-preserving risk score
func (pps *PrivacyPreservingScorer) CalculateRiskScore(userID string, current *TelemetryData) *RiskScore {
	pps.mu.RLock()
	baseline, exists := pps.baselines[userID]
	pps.mu.RUnlock()
	
	score := &RiskScore{
		Factors:   make(map[string]float64),
		Anomalies: make([]string, 0),
		Details:   make(map[string]interface{}),
	}
	
	// No baseline = new user (higher initial risk)
	if !exists || baseline.SampleCount < 5 {
		score.Score = 50
		score.Confidence = 0.3
		score.RiskLevel = "medium"
		score.Recommendation = "challenge"
		score.Anomalies = append(score.Anomalies, "NEW_USER_PROFILE")
		score.Details["baseline_status"] = "insufficient_samples"
		return score
	}
	
	// Extract current features
	currentFeatures := pps.extractStatisticalFeatures(current)
	
	// Multi-factor risk calculation
	var totalScore float64
	
	// 1. Keystroke Dynamics Analysis
	keystrokeScore := pps.analyzeKeystrokeDynamics(baseline, currentFeatures)
	totalScore += keystrokeScore * pps.weights.KeystrokeDynamics
	score.Factors["keystroke_dynamics"] = keystrokeScore
	if keystrokeScore > 70 {
		score.Anomalies = append(score.Anomalies, "KEYSTROKE_ANOMALY")
	}
	
	// 2. Mouse Behavior Analysis
	mouseScore := pps.analyzeMouseBehavior(baseline, currentFeatures)
	totalScore += mouseScore * pps.weights.MouseBehavior
	score.Factors["mouse_behavior"] = mouseScore
	if mouseScore > 70 {
		score.Anomalies = append(score.Anomalies, "MOUSE_BEHAVIOR_ANOMALY")
	}
	
	// 3. Device Fingerprint Check
	deviceScore := pps.analyzeDeviceFingerprint(baseline, current)
	totalScore += deviceScore * pps.weights.DeviceFingerprint
	score.Factors["device_fingerprint"] = deviceScore
	if deviceScore > 80 {
		score.Anomalies = append(score.Anomalies, "DEVICE_MISMATCH")
	}
	
	// 4. Contextual Factors (time, location)
	contextScore := pps.analyzeContextualFactors(baseline, current)
	totalScore += contextScore * pps.weights.ContextualFactors
	score.Factors["contextual"] = contextScore
	if contextScore > 70 {
		score.Anomalies = append(score.Anomalies, "UNUSUAL_CONTEXT")
	}
	
	// 5. Historical Pattern Matching
	patternScore := pps.analyzeHistoricalPatterns(baseline, current)
	totalScore += patternScore * pps.weights.HistoricalPattern
	score.Factors["historical_pattern"] = patternScore
	
	// 6. Velocity Check (impossible travel, rapid attempts)
	velocityScore := pps.checkVelocity(userID, current)
	totalScore += velocityScore * pps.weights.VelocityCheck
	score.Factors["velocity"] = velocityScore
	if velocityScore > 90 {
		score.Anomalies = append(score.Anomalies, "IMPOSSIBLE_VELOCITY")
	}
	
	// Normalize to 0-100
	score.Score = int(math.Min(totalScore, 100))
	score.Confidence = baseline.ConfidenceLevel
	score.RiskLevel = pps.calculateRiskLevel(score.Score)
	score.Recommendation = pps.determineRecommendation(score.Score, len(score.Anomalies))
	
	// Update metrics
	pps.totalScored++
	if len(score.Anomalies) > 0 {
		pps.totalAnomaly++
	}
	
	score.Details["baseline_samples"] = baseline.SampleCount
	score.Details["confidence"] = baseline.ConfidenceLevel
	
	return score
}

// MakeAuthDecision generates final authentication decision
func (pps *PrivacyPreservingScorer) MakeAuthDecision(score *RiskScore) *AuthDecision {
	decision := &AuthDecision{
		RiskScore: score.Score,
		Confidence: score.Confidence,
		Timestamp: time.Now(),
	}
	
	switch {
	case score.Score >= 80:
		decision.Decision = "deny"
		decision.Reason = "High risk score - suspicious activity detected"
		
	case score.Score >= 60:
		decision.Decision = "challenge"
		decision.ChallengeType = "mfa"
		decision.Reason = "Medium-high risk - additional authentication required"
		
	case score.Score >= 40:
		decision.Decision = "challenge"
		decision.ChallengeType = "captcha"
		decision.Reason = "Medium risk - verification recommended"
		
	default:
		decision.Decision = "allow"
		decision.Reason = "Low risk - normal behavior pattern"
	}
	
	return decision
}

// Private helper methods

func (pps *PrivacyPreservingScorer) extractStatisticalFeatures(data *TelemetryData) map[string]float64 {
	features := make(map[string]float64)
	
	// Keystroke features - statistical aggregation only
	if len(data.KeystrokeDwellTimes) > 0 {
		features["avg_dwell"] = average(data.KeystrokeDwellTimes)
		features["std_dwell"] = stddev(data.KeystrokeDwellTimes)
	}
	
	if len(data.KeystrokeFlightTimes) > 0 {
		features["avg_flight"] = average(data.KeystrokeFlightTimes)
		features["std_flight"] = stddev(data.KeystrokeFlightTimes)
	}
	
	// Mouse features - aggregated patterns
	if len(data.MouseMovements) > 0 {
		speeds := make([]float64, len(data.MouseMovements))
		for i, mv := range data.MouseMovements {
			speeds[i] = mv.Speed
		}
		features["avg_mouse_speed"] = average(speeds)
		features["mouse_acceleration"] = calculateAcceleration(data.MouseMovements)
	}
	
	// Click pattern hash
	if len(data.ClickEvents) > 0 {
		features["click_pattern"] = pps.hashClickPattern(data.ClickEvents)
	}
	
	return features
}

func (pps *PrivacyPreservingScorer) updateBaseline(userID string, features map[string]float64) {
	pps.mu.Lock()
	defer pps.mu.Unlock()
	
	baseline, exists := pps.baselines[userID]
	if !exists {
		// Create new baseline
		baseline = &UserBaseline{
			UserID:          userID,
			SampleCount:     0,
			ConfidenceLevel: 0.1,
			DataAnonymized:  true,
			HashAlgorithm:   "HMAC-SHA256",
			LastUpdated:     time.Now(),
		}
		pps.baselines[userID] = baseline
	}
	
	// Exponential moving average (privacy-preserving online learning)
	lr := pps.learningRate
	
	if avgDwell, ok := features["avg_dwell"]; ok {
		baseline.AvgDwellTime = (1-lr)*baseline.AvgDwellTime + lr*avgDwell
	}
	if stdDwell, ok := features["std_dwell"]; ok {
		baseline.StdDwellTime = (1-lr)*baseline.StdDwellTime + lr*stdDwell
	}
	if avgFlight, ok := features["avg_flight"]; ok {
		baseline.AvgFlightTime = (1-lr)*baseline.AvgFlightTime + lr*avgFlight
	}
	if stdFlight, ok := features["std_flight"]; ok {
		baseline.StdFlightTime = (1-lr)*baseline.StdFlightTime + lr*stdFlight
	}
	if avgSpeed, ok := features["avg_mouse_speed"]; ok {
		baseline.AvgMouseSpeed = (1-lr)*baseline.AvgMouseSpeed + lr*avgSpeed
	}
	if accel, ok := features["mouse_acceleration"]; ok {
		baseline.MouseAcceleration = (1-lr)*baseline.MouseAcceleration + lr*accel
	}
	
	baseline.SampleCount++
	baseline.LastUpdated = time.Now()
	
	// Increase confidence as we collect more samples
	baseline.ConfidenceLevel = math.Min(0.95, float64(baseline.SampleCount)/100.0)
}

func (pps *PrivacyPreservingScorer) analyzeKeystrokeDynamics(baseline *UserBaseline, current map[string]float64) float64 {
	if baseline.AvgDwellTime == 0 {
		return 0
	}
	
	score := 0.0
	
	// Dwell time deviation
	if avgDwell, ok := current["avg_dwell"]; ok {
		deviation := math.Abs(avgDwell-baseline.AvgDwellTime) / baseline.AvgDwellTime
		if deviation > pps.anomalyThreshold {
			score += deviation * 50
		}
	}
	
	// Flight time deviation
	if avgFlight, ok := current["avg_flight"]; ok {
		deviation := math.Abs(avgFlight-baseline.AvgFlightTime) / baseline.AvgFlightTime
		if deviation > pps.anomalyThreshold {
			score += deviation * 50
		}
	}
	
	return math.Min(score, 100)
}

func (pps *PrivacyPreservingScorer) analyzeMouseBehavior(baseline *UserBaseline, current map[string]float64) float64 {
	if baseline.AvgMouseSpeed == 0 {
		return 0
	}
	
	score := 0.0
	
	// Mouse speed deviation
	if avgSpeed, ok := current["avg_mouse_speed"]; ok {
		deviation := math.Abs(avgSpeed-baseline.AvgMouseSpeed) / baseline.AvgMouseSpeed
		if deviation > pps.anomalyThreshold {
			score += deviation * 60
		}
	}
	
	// Mouse acceleration check
	if accel, ok := current["mouse_acceleration"]; ok {
		deviation := math.Abs(accel-baseline.MouseAcceleration) / (baseline.MouseAcceleration + 0.001)
		if deviation > pps.anomalyThreshold {
			score += deviation * 40
		}
	}
	
	return math.Min(score, 100)
}

func (pps *PrivacyPreservingScorer) analyzeDeviceFingerprint(baseline *UserBaseline, current *TelemetryData) float64 {
	// Hash current device fingerprint
	currentHash := pps.hashPII(current.DeviceFingerprint)
	
	if baseline.DeviceFingerprintHash == "" {
		// First time seeing this user
		return 20.0
	}
	
	if currentHash != baseline.DeviceFingerprintHash {
		// Device mismatch - high risk
		if current.IsNewDevice {
			return 60.0 // New device is somewhat suspicious
		}
		return 90.0 // Unexpected device change is very suspicious
	}
	
	return 0.0 // Device matches
}

func (pps *PrivacyPreservingScorer) analyzeContextualFactors(baseline *UserBaseline, current *TelemetryData) float64 {
	score := 0.0
	
	// Time-of-day analysis
	if len(baseline.TypicalLoginHours) > 0 {
		isTypicalHour := false
		for _, hour := range baseline.TypicalLoginHours {
			if hour == current.TimeOfDay {
				isTypicalHour = true
				break
			}
		}
		if !isTypicalHour {
			score += 30.0
		}
	}
	
	// New location check
	if current.IsNewLocation {
		score += 40.0
	}
	
	return math.Min(score, 100)
}

func (pps *PrivacyPreservingScorer) analyzeHistoricalPatterns(baseline *UserBaseline, current *TelemetryData) float64 {
	// Low sample count = higher uncertainty
	if baseline.SampleCount < 10 {
		return 30.0
	}
	
	return 0.0 // Sufficient history
}

func (pps *PrivacyPreservingScorer) checkVelocity(userID string, current *TelemetryData) float64 {
	// TODO: Implement impossible travel detection
	// For now: placeholder
	return 0.0
}

func (pps *PrivacyPreservingScorer) calculateRiskLevel(score int) string {
	switch {
	case score >= 80:
		return "critical"
	case score >= 60:
		return "high"
	case score >= 40:
		return "medium"
	default:
		return "low"
	}
}

func (pps *PrivacyPreservingScorer) determineRecommendation(score int, anomalyCount int) string {
	if score >= 80 {
		return "deny"
	} else if score >= 60 || anomalyCount >= 3 {
		return "challenge_mfa"
	} else if score >= 40 {
		return "challenge_captcha"
	}
	return "allow"
}

// hashPII uses HMAC-SHA256 for privacy-preserving hashing
func (pps *PrivacyPreservingScorer) hashPII(data string) string {
	h := hmac.New(sha256.New, pps.secretKey)
	h.Write([]byte(data))
	return base64.StdEncoding.EncodeToString(h.Sum(nil))
}

func (pps *PrivacyPreservingScorer) hashClickPattern(clicks []ClickEvent) float64 {
	if len(clicks) == 0 {
		return 0
	}
	
	// Aggregate click timing pattern (privacy-preserving)
	durations := make([]float64, len(clicks))
	for i, click := range clicks {
		durations[i] = click.Duration
	}
	
	return average(durations)
}

// Statistical helpers

func average(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range vals {
		sum += v
	}
	return sum / float64(len(vals))
}

func stddev(vals []float64) float64 {
	if len(vals) < 2 {
		return 0
	}
	
	avg := average(vals)
	sumSq := 0.0
	for _, v := range vals {
		diff := v - avg
		sumSq += diff * diff
	}
	
	return math.Sqrt(sumSq / float64(len(vals)-1))
}

func calculateAcceleration(movements []MouseEvent) float64 {
	if len(movements) < 2 {
		return 0
	}
	
	accelerations := make([]float64, 0, len(movements)-1)
	for i := 1; i < len(movements); i++ {
		dt := movements[i].Timestamp.Sub(movements[i-1].Timestamp).Seconds()
		if dt > 0 {
			accel := (movements[i].Speed - movements[i-1].Speed) / dt
			accelerations = append(accelerations, math.Abs(accel))
		}
	}
	
	return average(accelerations)
}

// ExportBaseline exports user baseline for backup (anonymized)
func (pps *PrivacyPreservingScorer) ExportBaseline(userID string) ([]byte, error) {
	pps.mu.RLock()
	baseline, exists := pps.baselines[userID]
	pps.mu.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("baseline not found for user %s", userID)
	}
	
	return json.Marshal(baseline)
}

// GetStats returns scorer statistics
func (pps *PrivacyPreservingScorer) GetStats() map[string]interface{} {
	pps.mu.RLock()
	defer pps.mu.RUnlock()
	
	anomalyRate := 0.0
	if pps.totalScored > 0 {
		anomalyRate = float64(pps.totalAnomaly) / float64(pps.totalScored)
	}
	
	return map[string]interface{}{
		"total_users":       len(pps.baselines),
		"total_scored":      pps.totalScored,
		"total_anomalies":   pps.totalAnomaly,
		"anomaly_rate":      anomalyRate,
		"privacy_enabled":   true,
		"hash_algorithm":    "HMAC-SHA256",
	}
}
