package main

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"sync"
	"time"
)

// BehavioralBiometricsEngine implements advanced behavioral analysis
// Phase 2: Keystroke dynamics + Mouse behavior + Device fingerprinting
//
// Architecture:
// - Multi-modal analysis (keyboard + mouse + touch)
// - Continuous authentication (session-based)
// - Anomaly detection using statistical models
// - Privacy-preserving (hashed features only, NEVER raw data)
//
// P0 Constraints (from Phân chia công việc.md):
// - ❌ KHÔNG store raw biometric data (chỉ hash/features)
// - ✅ PHẢI encrypt telemetry data at rest
// - ✅ PHẢI có rollback mechanism
// - ✅ PHẢI timeout processing

type BehavioralBiometricsEngine struct {
	// Privacy-preserving feature extraction
	hmacKey []byte

	// Behavioral models per user
	mu         sync.RWMutex
	userModels map[string]*BehavioralModel

	// Configuration
	config BiometricsConfig

	// Performance metrics
	analysisCount uint64
	avgLatencyMs  float64
}

// BiometricsConfig defines behavioral analysis parameters
type BiometricsConfig struct {
	// Keystroke dynamics
	TypingSpeedBuckets int     // Bucket size for k-anonymity
	DigrathThreshold   float64 // Digraph timing threshold (ms)
	DwellTimeWeight    float64 // Dwell time importance
	FlightTimeWeight   float64 // Flight time importance

	// Mouse behavior
	MouseVelocityBuckets int
	TrajectorySmoothing  int // Moving average window
	ClickPressureBuckets int

	// Device fingerprinting
	DeviceFingerprintSalt string
	FingerprintRotation   time.Duration

	// Risk thresholds
	SuspiciousThreshold   float64 // 0.0-1.0
	BlockThreshold        float64
	MinSamplesForBaseline int
}

// BehavioralModel represents user's unique behavioral baseline
type BehavioralModel struct {
	UserID      string
	CreatedAt   time.Time
	LastUpdated time.Time
	SampleCount int

	// Keystroke dynamics (privacy-preserved)
	AvgTypingSpeedWPM   float64
	StdDevTypingSpeed   float64
	AvgDwellTimeMs      float64
	AvgFlightTimeMs     float64
	DigraPhDistribution map[string]float64 // Hashed digraph timings

	// Mouse behavior
	AvgMouseVelocity    float64
	StdDevMouseVelocity float64
	AvgClickPressure    float64
	MouseTrajectoryHash string // Hashed trajectory patterns

	// Device characteristics
	KnownDeviceHashes  map[string]time.Time // Device -> last seen
	TypicalAccessHours []int                // Hours of day [0-23]

	// Trust score (adaptive)
	TrustScore float64 // 0.0-1.0
}

// KeystrokeTelemetry contains raw keystroke data (processed immediately, never stored)
type KeystrokeTelemetry struct {
	Timestamp  time.Time
	KeyCode    int     // Virtual key code (NOT the actual key for privacy)
	DwellTime  int     // Key press duration (ms)
	FlightTime int     // Time to next key (ms)
	Pressure   float64 // Key press force (0.0-1.0) if available
}

// MouseTelemetry contains raw mouse data (processed immediately, never stored)
type MouseTelemetry struct {
	Timestamp    time.Time
	X            int
	Y            int
	VelocityX    float64
	VelocityY    float64
	Acceleration float64
	ClickType    string // left/right/middle
	Pressure     float64
}

// DeviceTelemetry contains device characteristics
type DeviceTelemetry struct {
	UserAgent        string
	ScreenResolution string
	Timezone         string
	Language         string
	Plugins          []string
	Canvas           string // Canvas fingerprint hash
	WebGL            string // WebGL fingerprint hash
	AudioContext     string // Audio fingerprint hash
}

// BiometricFeatures contains extracted privacy-safe features
type BiometricFeatures struct {
	// Keystroke features (bucketed/hashed)
	TypingSpeedBucket int
	DwellTimeBucket   int
	FlightTimeBucket  int
	DigraPhHash       string

	// Mouse features (aggregated)
	MouseVelocityBucket int
	TrajectoryHash      string
	ClickPressureBucket int

	// Device features (hashed)
	DeviceFingerprintHash string
	IsNewDevice           bool
	IsUnusualTime         bool

	// Context
	SessionID string
	Timestamp time.Time
}

// RiskAssessment contains behavioral risk analysis result
type RiskAssessment struct {
	RiskScore      float64 // 0.0-1.0
	Confidence     float64 // 0.0-1.0
	Decision       string  // allow/challenge/deny
	RiskFactors    []string
	Recommendation string
	AnalyzedAt     time.Time
	LatencyMs      float64
}

// NewBehavioralBiometricsEngine creates engine with secure defaults
func NewBehavioralBiometricsEngine(hmacKey []byte) *BehavioralBiometricsEngine {
	if len(hmacKey) == 0 {
		// Generate secure default (in production: load from KMS/Vault)
		hmacKey = make([]byte, 32)
		copy(hmacKey, []byte("shieldx-biometrics-key-2024"))
	}

	return &BehavioralBiometricsEngine{
		hmacKey:    hmacKey,
		userModels: make(map[string]*BehavioralModel),
		config: BiometricsConfig{
			TypingSpeedBuckets:    20, // 20 WPM buckets
			DigrathThreshold:      50.0,
			DwellTimeWeight:       0.4,
			FlightTimeWeight:      0.6,
			MouseVelocityBuckets:  100, // 100 px/s buckets
			TrajectorySmoothing:   5,
			ClickPressureBuckets:  10,
			SuspiciousThreshold:   0.6,
			BlockThreshold:        0.85,
			MinSamplesForBaseline: 20,
		},
	}
}

// AnalyzeBehavior performs comprehensive behavioral risk assessment
// P0: Must NOT store raw keystroke/mouse data
func (bbe *BehavioralBiometricsEngine) AnalyzeBehavior(
	userID string,
	keystrokesRaw []*KeystrokeTelemetry,
	mouseRaw []*MouseTelemetry,
	deviceRaw *DeviceTelemetry,
) (*RiskAssessment, error) {

	startTime := time.Now()

	// Step 1: Extract privacy-safe features (immediate processing, no raw storage)
	features := bbe.extractFeatures(userID, keystrokesRaw, mouseRaw, deviceRaw)

	// Step 2: Get or create user behavioral model
	model := bbe.getUserModel(userID)

	// Step 3: Compute risk score
	riskScore := bbe.computeRiskScore(features, model)

	// Step 4: Make authentication decision
	decision, recommendation := bbe.makeDecision(riskScore, model.TrustScore)

	// Step 5: Update model if legitimate (incremental learning)
	if riskScore < bbe.config.SuspiciousThreshold {
		bbe.updateUserModel(userID, features)
	}

	assessment := &RiskAssessment{
		RiskScore:      riskScore,
		Confidence:     bbe.calculateConfidence(model),
		Decision:       decision,
		RiskFactors:    bbe.identifyRiskFactors(features, model),
		Recommendation: recommendation,
		AnalyzedAt:     time.Now(),
		LatencyMs:      time.Since(startTime).Seconds() * 1000,
	}

	return assessment, nil
}

// extractFeatures converts raw telemetry to privacy-safe features
// P0 Constraint: NEVER store raw data
func (bbe *BehavioralBiometricsEngine) extractFeatures(
	userID string,
	keystrokes []*KeystrokeTelemetry,
	mouse []*MouseTelemetry,
	device *DeviceTelemetry,
) *BiometricFeatures {

	features := &BiometricFeatures{
		Timestamp: time.Now(),
	}

	// Keystroke feature extraction
	if len(keystrokes) > 0 {
		avgDwell, avgFlight := bbe.analyzeKeystrokeDynamics(keystrokes)

		// Bucket typing speed for k-anonymity
		typingSpeedWPM := bbe.calculateTypingSpeed(keystrokes)
		features.TypingSpeedBucket = int(typingSpeedWPM / float64(bbe.config.TypingSpeedBuckets))

		// Bucket timings
		features.DwellTimeBucket = int(avgDwell / 10.0) // 10ms buckets
		features.FlightTimeBucket = int(avgFlight / 10.0)

		// Hash digraph patterns (NEVER store raw keystrokes)
		features.DigraPhHash = bbe.hashDigraphPattern(keystrokes)
	}

	// Mouse feature extraction
	if len(mouse) > 0 {
		avgVelocity := bbe.analyzeMouseBehavior(mouse)
		features.MouseVelocityBucket = int(avgVelocity / float64(bbe.config.MouseVelocityBuckets))
		features.TrajectoryHash = bbe.hashMouseTrajectory(mouse)

		if len(mouse) > 0 {
			avgPressure := 0.0
			for _, m := range mouse {
				avgPressure += m.Pressure
			}
			avgPressure /= float64(len(mouse))
			features.ClickPressureBucket = int(avgPressure * float64(bbe.config.ClickPressureBuckets))
		}
	}

	// Device fingerprint (hashed for privacy)
	if device != nil {
		features.DeviceFingerprintHash = bbe.hashDeviceFingerprint(device)
		features.IsUnusualTime = bbe.isUnusualAccessTime()
	}

	return features
}

// analyzeKeystrokeDynamics computes aggregate timing features
func (bbe *BehavioralBiometricsEngine) analyzeKeystrokeDynamics(keystrokes []*KeystrokeTelemetry) (avgDwell, avgFlight float64) {
	if len(keystrokes) == 0 {
		return 0, 0
	}

	totalDwell := 0.0
	totalFlight := 0.0

	for _, ks := range keystrokes {
		totalDwell += float64(ks.DwellTime)
		totalFlight += float64(ks.FlightTime)
	}

	avgDwell = totalDwell / float64(len(keystrokes))
	avgFlight = totalFlight / float64(len(keystrokes))

	return avgDwell, avgFlight
}

// calculateTypingSpeed estimates WPM (words per minute)
func (bbe *BehavioralBiometricsEngine) calculateTypingSpeed(keystrokes []*KeystrokeTelemetry) float64 {
	if len(keystrokes) < 2 {
		return 0
	}

	// Estimate based on average inter-keystroke interval
	totalTime := keystrokes[len(keystrokes)-1].Timestamp.Sub(keystrokes[0].Timestamp).Seconds()
	if totalTime == 0 {
		return 0
	}

	// Rough estimate: 5 chars per word
	charsPerSec := float64(len(keystrokes)) / totalTime
	wpm := (charsPerSec / 5.0) * 60.0

	return wpm
}

// hashDigraphPattern creates HMAC hash of keystroke timing patterns
// P0: NEVER store raw keystroke sequence
func (bbe *BehavioralBiometricsEngine) hashDigraphPattern(keystrokes []*KeystrokeTelemetry) string {
	if len(keystrokes) < 2 {
		return ""
	}

	// Extract timing features only (not actual keys)
	pattern := ""
	for i := 0; i < len(keystrokes)-1; i++ {
		// Quantize timings to 10ms buckets
		dwell := keystrokes[i].DwellTime / 10
		flight := keystrokes[i].FlightTime / 10
		pattern += fmt.Sprintf("%d:%d,", dwell, flight)
	}

	h := hmac.New(sha256.New, bbe.hmacKey)
	h.Write([]byte(pattern))
	return hex.EncodeToString(h.Sum(nil))
}

// analyzeMouseBehavior computes average velocity
func (bbe *BehavioralBiometricsEngine) analyzeMouseBehavior(mouse []*MouseTelemetry) float64 {
	if len(mouse) == 0 {
		return 0
	}

	totalVelocity := 0.0
	for _, m := range mouse {
		velocity := math.Sqrt(m.VelocityX*m.VelocityX + m.VelocityY*m.VelocityY)
		totalVelocity += velocity
	}

	return totalVelocity / float64(len(mouse))
}

// hashMouseTrajectory creates hash of mouse movement patterns
func (bbe *BehavioralBiometricsEngine) hashMouseTrajectory(mouse []*MouseTelemetry) string {
	if len(mouse) < 5 {
		return ""
	}

	// Extract movement direction changes (curvature features)
	directions := ""
	for i := 1; i < len(mouse); i++ {
		dx := mouse[i].X - mouse[i-1].X
		dy := mouse[i].Y - mouse[i-1].Y

		// Quantize to 8 directions (N, NE, E, SE, S, SW, W, NW)
		angle := math.Atan2(float64(dy), float64(dx)) * 180 / math.Pi
		dir := int((angle+22.5)/45.0) % 8
		directions += fmt.Sprintf("%d", dir)
	}

	h := hmac.New(sha256.New, bbe.hmacKey)
	h.Write([]byte(directions))
	return hex.EncodeToString(h.Sum(nil))
}

// hashDeviceFingerprint creates deterministic device hash
func (bbe *BehavioralBiometricsEngine) hashDeviceFingerprint(device *DeviceTelemetry) string {
	fingerprint := fmt.Sprintf("%s|%s|%s|%s|%s|%s|%s",
		device.UserAgent,
		device.ScreenResolution,
		device.Timezone,
		device.Language,
		device.Canvas,
		device.WebGL,
		device.AudioContext,
	)

	h := hmac.New(sha256.New, bbe.hmacKey)
	h.Write([]byte(fingerprint))
	h.Write([]byte(bbe.config.DeviceFingerprintSalt))

	return hex.EncodeToString(h.Sum(nil))
}

// isUnusualAccessTime checks if current time is outside typical hours
func (bbe *BehavioralBiometricsEngine) isUnusualAccessTime() bool {
	hour := time.Now().Hour()
	// Business hours: 8 AM - 6 PM
	return hour < 8 || hour > 18
}

// getUserModel retrieves or creates user behavioral model
func (bbe *BehavioralBiometricsEngine) getUserModel(userID string) *BehavioralModel {
	bbe.mu.RLock()
	model, exists := bbe.userModels[userID]
	bbe.mu.RUnlock()

	if exists {
		return model
	}

	// Create new model
	bbe.mu.Lock()
	defer bbe.mu.Unlock()

	model = &BehavioralModel{
		UserID:              userID,
		CreatedAt:           time.Now(),
		LastUpdated:         time.Now(),
		DigraPhDistribution: make(map[string]float64),
		KnownDeviceHashes:   make(map[string]time.Time),
		TypicalAccessHours:  []int{9, 10, 11, 12, 13, 14, 15, 16, 17},
		TrustScore:          1.0, // Initial trust
	}

	bbe.userModels[userID] = model
	return model
}

// computeRiskScore calculates behavioral risk (0.0-1.0)
func (bbe *BehavioralBiometricsEngine) computeRiskScore(features *BiometricFeatures, model *BehavioralModel) float64 {
	riskScore := 0.0

	// Insufficient baseline data = medium risk
	if model.SampleCount < bbe.config.MinSamplesForBaseline {
		return 0.5
	}

	// Factor 1: Typing speed deviation (20% weight)
	if model.StdDevTypingSpeed > 0 {
		speedDiff := math.Abs(float64(features.TypingSpeedBucket)*float64(bbe.config.TypingSpeedBuckets) - model.AvgTypingSpeedWPM)
		zScore := speedDiff / model.StdDevTypingSpeed
		if zScore > 2.0 { // >2 sigma deviation
			riskScore += 0.20
		}
	}

	// Factor 2: Mouse behavior deviation (20% weight)
	if model.StdDevMouseVelocity > 0 {
		velocityDiff := math.Abs(float64(features.MouseVelocityBucket)*float64(bbe.config.MouseVelocityBuckets) - model.AvgMouseVelocity)
		zScore := velocityDiff / model.StdDevMouseVelocity
		if zScore > 2.0 {
			riskScore += 0.20
		}
	}

	// Factor 3: Unknown device (30% weight)
	if _, known := model.KnownDeviceHashes[features.DeviceFingerprintHash]; !known {
		riskScore += 0.30
	}

	// Factor 4: Unusual access time (15% weight)
	if features.IsUnusualTime {
		riskScore += 0.15
	}

	// Factor 5: Digraph pattern mismatch (15% weight)
	if features.DigraPhHash != "" && model.DigraPhDistribution != nil {
		if _, exists := model.DigraPhDistribution[features.DigraPhHash]; !exists {
			riskScore += 0.15
		}
	}

	// Apply trust score adjustment
	riskScore = riskScore * (2.0 - model.TrustScore) // Low trust amplifies risk

	return math.Min(riskScore, 1.0)
}

// makeDecision determines authentication decision
func (bbe *BehavioralBiometricsEngine) makeDecision(riskScore, trustScore float64) (decision, recommendation string) {
	adjustedRisk := riskScore * (2.0 - trustScore)

	switch {
	case adjustedRisk >= bbe.config.BlockThreshold:
		return "deny", "block_and_require_password_reset"
	case adjustedRisk >= bbe.config.SuspiciousThreshold:
		return "challenge", "require_mfa_verification"
	default:
		return "allow", "proceed"
	}
}

// identifyRiskFactors extracts risk contributing factors
func (bbe *BehavioralBiometricsEngine) identifyRiskFactors(features *BiometricFeatures, model *BehavioralModel) []string {
	factors := []string{}

	if features.IsNewDevice {
		factors = append(factors, "new_device")
	}
	if features.IsUnusualTime {
		factors = append(factors, "unusual_access_time")
	}
	if model.SampleCount < bbe.config.MinSamplesForBaseline {
		factors = append(factors, "insufficient_baseline")
	}

	return factors
}

// calculateConfidence estimates confidence in risk assessment
func (bbe *BehavioralBiometricsEngine) calculateConfidence(model *BehavioralModel) float64 {
	if model.SampleCount == 0 {
		return 0.0
	}

	// Confidence increases with more samples (asymptotic to 1.0)
	confidence := 1.0 - math.Exp(-float64(model.SampleCount)/50.0)
	return confidence
}

// updateUserModel incrementally updates behavioral baseline
func (bbe *BehavioralBiometricsEngine) updateUserModel(userID string, features *BiometricFeatures) {
	bbe.mu.Lock()
	defer bbe.mu.Unlock()

	model := bbe.userModels[userID]
	if model == nil {
		return
	}

	// Exponential moving average (alpha = 0.1)
	alpha := 0.1

	// Update typing speed
	newTypingSpeed := float64(features.TypingSpeedBucket) * float64(bbe.config.TypingSpeedBuckets)
	if model.SampleCount > 0 {
		model.AvgTypingSpeedWPM = (1-alpha)*model.AvgTypingSpeedWPM + alpha*newTypingSpeed
	} else {
		model.AvgTypingSpeedWPM = newTypingSpeed
	}

	// Update mouse velocity
	newVelocity := float64(features.MouseVelocityBucket) * float64(bbe.config.MouseVelocityBuckets)
	if model.SampleCount > 0 {
		model.AvgMouseVelocity = (1-alpha)*model.AvgMouseVelocity + alpha*newVelocity
	} else {
		model.AvgMouseVelocity = newVelocity
	}

	// Add device to known devices
	model.KnownDeviceHashes[features.DeviceFingerprintHash] = time.Now()

	// Update digraph distribution
	if features.DigraPhHash != "" {
		model.DigraPhDistribution[features.DigraPhHash] = float64(time.Now().Unix()) / time.Hour.Seconds()
	}

	model.SampleCount++
	model.LastUpdated = time.Now()
}
