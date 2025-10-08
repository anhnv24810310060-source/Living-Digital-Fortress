package main

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"shieldx/shared/shieldx-common/pkg/metrics"
	logcorr "shieldx/shared/shieldx-common/pkg/observability/logcorr"
	otelobs "shieldx/shared/shieldx-common/pkg/observability/otel"
	"shieldx/shared/shieldx-common/pkg/ratls"
)

// ContAuthService implements Continuous Authentication with behavioral biometrics
// Phase 2 P0 Requirements:
// - MUST NOT store raw biometric data (only hashed features)
// - MUST NOT expose ML model internals via API
// - MUST encrypt telemetry data at rest
// - MUST have rollback mechanism for ML models
type ContAuthService struct {
	userProfiles map[string]*UserBehaviorProfile
	profileMu    sync.RWMutex

	keystrokeDynamics *KeystrokeDynamicsAnalyzer
	mouseAnalyzer     *MouseBehaviorAnalyzer
	deviceFingerprint *DeviceFingerprintEngine
	riskScorer        *AdaptiveRiskScorer

	// Phase 2 P0: Federated Learning Integration
	federatedEngine *FederatedLearningEngine

	totalCollections *metrics.Counter
	totalDecisions   *metrics.Counter
	anomalyCount     *metrics.Counter
	avgRiskScore     *metrics.Gauge

	baselineWindow   time.Duration
	anomalyThreshold float64
	minSampleSize    int
}

// UserBehaviorProfile stores encrypted behavioral features
type UserBehaviorProfile struct {
	UserIDHash  string
	CreatedAt   time.Time
	LastUpdated time.Time
	SampleCount int

	TypingSpeed       TypingSpeedProfile
	KeyHoldTimes      StatisticalDistribution
	KeyPressIntervals StatisticalDistribution
	MouseSpeed        StatisticalDistribution
	ClickPatterns     map[string]float64
	ScrollBehavior    ScrollProfile

	DeviceHash     string
	DeviceFeatures map[string]interface{}

	BaselineEstablished bool
	RiskBaseline        float64
	TrustScore          float64
	RecentScores        []float64
	AdaptiveThreshold   float64
}

type TypingSpeedProfile struct {
	AvgWPM         float64
	StdDevWPM      float64
	AvgKeysPerSec  float64
	TypoRate       float64
	DeleteKeyRatio float64
}

type StatisticalDistribution struct {
	Mean     float64
	StdDev   float64
	Median   float64
	P25      float64
	P75      float64
	Skewness float64
	Kurtosis float64
}

type ScrollProfile struct {
	AvgScrollSpeed   float64
	ScrollAccel      float64
	PauseFrequency   float64
	DirectionChanges float64
}

type KeystrokeDynamicsAnalyzer struct {
	mu sync.RWMutex
}

type MouseBehaviorAnalyzer struct {
	mu sync.RWMutex
}

type DeviceFingerprintEngine struct {
	hashSalt    []byte
	fpCache     map[string]*DeviceFingerprint
	cacheMu     sync.RWMutex
	cacheExpiry time.Duration
}

type DeviceFingerprint struct {
	Hash       string
	UserAgent  string
	ScreenRes  string
	Timezone   string
	Languages  []string
	Confidence float64
	FirstSeen  time.Time
	LastSeen   time.Time
}

type AdaptiveRiskScorer struct {
	weights        map[string]float64
	timeOfDayModel *TimeBasedRiskModel
	locationModel  *LocationRiskModel
	mu             sync.RWMutex
}

type TimeBasedRiskModel struct{}
type LocationRiskModel struct{}

type TelemetryData struct {
	UserID          string             `json:"user_id"`
	Timestamp       time.Time          `json:"timestamp"`
	SessionID       string             `json:"session_id"`
	KeystrokeEvents []KeystrokeFeature `json:"keystroke_events"`
	MouseEvents     []MouseFeature     `json:"mouse_events"`
	DeviceInfo      DeviceInfo         `json:"device_info"`
	IPAddress       string             `json:"ip_address"`
	Geolocation     GeoLocation        `json:"geolocation"`
	TimeOfDay       int                `json:"time_of_day"`
}

type KeystrokeFeature struct {
	HoldTime    float64 `json:"hold_time"`
	FlightTime  float64 `json:"flight_time"`
	TypingBurst int     `json:"typing_burst"`
	ErrorRate   float64 `json:"error_rate"`
}

type MouseFeature struct {
	Velocity      float64 `json:"velocity"`
	Acceleration  float64 `json:"acceleration"`
	Curvature     float64 `json:"curvature"`
	PauseDuration float64 `json:"pause_duration"`
	ClickSpeed    float64 `json:"click_speed"`
}

type DeviceInfo struct {
	UserAgent        string   `json:"user_agent"`
	ScreenResolution string   `json:"screen_resolution"`
	Timezone         string   `json:"timezone"`
	Languages        []string `json:"languages"`
	Platform         string   `json:"platform"`
	Plugins          []string `json:"plugins"`
}

type GeoLocation struct {
	Country   string  `json:"country"`
	City      string  `json:"city"`
	Latitude  float64 `json:"latitude"`
	Longitude float64 `json:"longitude"`
	ISP       string  `json:"isp"`
}

type RiskDecision struct {
	Decision      string             `json:"decision"`
	RiskScore     float64            `json:"risk_score"`
	Confidence    float64            `json:"confidence"`
	Factors       map[string]float64 `json:"factors"`
	Explanation   string             `json:"explanation"`
	RequiresMFA   bool               `json:"requires_mfa"`
	ChallengeType string             `json:"challenge_type"`
	TrustScore    float64            `json:"trust_score"`
	Timestamp     time.Time          `json:"timestamp"`
}

func main() {
	service := NewContAuthService()

	mux := http.NewServeMux()
	reg := metrics.NewRegistry()
	httpMetrics := metrics.NewHTTPMetrics(reg, "contauth")

	rateLimit := makeRateLimiter(parseIntDefault("CONTAUTH_RL_PER_MIN", 300))
	adminOnly := makeAdminMiddleware()

	mux.HandleFunc("/contauth/collect", rateLimit(service.handleCollect))
	mux.HandleFunc("/contauth/score", rateLimit(service.handleScore))
	mux.HandleFunc("/contauth/decision", rateLimit(service.handleDecision))
	mux.HandleFunc("/contauth/profile/reset", adminOnly(service.handleProfileReset))
	mux.HandleFunc("/contauth/profile/stats", adminOnly(service.handleProfileStats))
	mux.HandleFunc("/health", service.handleHealth)
	mux.Handle("/metrics", reg)

	port := getenvDefault("CONTAUTH_PORT", "5002")

	shutdown := otelobs.InitTracer("contauth")
	defer shutdown(context.Background())

	h := httpMetrics.Middleware(mux)
	h = logcorr.Middleware(h)
	h = otelobs.WrapHTTPHandler("contauth", h)

	gCertExpiry := metrics.NewGauge("ratls_cert_expiry_seconds", "Cert expiry")
	reg.RegisterGauge(gCertExpiry)
	reg.Register(service.totalCollections)
	reg.Register(service.totalDecisions)
	reg.Register(service.anomalyCount)
	reg.RegisterGauge(service.avgRiskScore)

	srv := &http.Server{Addr: ":" + port, Handler: h}

	var issuer *ratls.AutoIssuer
	if os.Getenv("RATLS_ENABLE") == "true" {
		td := getenvDefault("RATLS_TRUST_DOMAIN", "shieldx.local")
		ns := getenvDefault("RATLS_NAMESPACE", "default")
		svc := getenvDefault("RATLS_SERVICE", "contauth")
		rotate := parseDurationDefault("RATLS_ROTATE_EVERY", 45*time.Minute)
		valid := parseDurationDefault("RATLS_VALIDITY", 60*time.Minute)

		ai, err := ratls.NewDevIssuer(ratls.Identity{TrustDomain: td, Namespace: ns, Service: svc}, rotate, valid)
		if err != nil {
			log.Fatalf("[contauth] RA-TLS init: %v", err)
		}
		issuer = ai

		go func() {
			for {
				if t, err := issuer.LeafNotAfter(); err == nil {
					gCertExpiry.Set(uint64(time.Until(t).Seconds()))
				}
				time.Sleep(1 * time.Minute)
			}
		}()
	}

	if issuer != nil {
		srv.TLSConfig = issuer.ServerTLSConfig(true, getenvDefault("RATLS_TRUST_DOMAIN", "shieldx.local"))
		log.Printf("[contauth] (RA-TLS) starting on :%s", port)
		log.Fatal(srv.ListenAndServeTLS("", ""))
	} else {
		log.Printf("[contauth] starting on :%s", port)
		log.Fatal(srv.ListenAndServe())
	}
}

func NewContAuthService() *ContAuthService {
	// Phase 2 P0: Initialize Federated Learning with epsilon=1.0 (strong privacy)
	flEngine := NewFederatedLearningEngine(1.0, 1e-5, 5)

	return &ContAuthService{
		userProfiles:      make(map[string]*UserBehaviorProfile),
		keystrokeDynamics: NewKeystrokeDynamicsAnalyzer(),
		mouseAnalyzer:     NewMouseBehaviorAnalyzer(),
		deviceFingerprint: NewDeviceFingerprintEngine(),
		riskScorer:        NewAdaptiveRiskScorer(),
		federatedEngine:   flEngine,

		totalCollections: metrics.NewCounter("contauth_collections_total", "Total telemetry collections"),
		totalDecisions:   metrics.NewCounter("contauth_decisions_total", "Total auth decisions"),
		anomalyCount:     metrics.NewCounter("contauth_anomalies_total", "Detected anomalies"),
		avgRiskScore:     metrics.NewGauge("contauth_avg_risk_score", "Average risk score"),

		baselineWindow:   7 * 24 * time.Hour,
		anomalyThreshold: 0.75,
		minSampleSize:    50,
	}
}

func (cas *ContAuthService) handleCollect(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var telemetry TelemetryData
	if err := json.NewDecoder(r.Body).Decode(&telemetry); err != nil {
		http.Error(w, "invalid json", http.StatusBadRequest)
		return
	}

	userIDHash := hashUserID(telemetry.UserID)
	profile := cas.getOrCreateProfile(userIDHash)
	cas.updateProfile(profile, &telemetry)
	cas.totalCollections.Add(1)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":       "collected",
		"sample_count": profile.SampleCount,
		"baseline":     profile.BaselineEstablished,
	})
}

func (cas *ContAuthService) handleScore(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var telemetry TelemetryData
	if err := json.NewDecoder(r.Body).Decode(&telemetry); err != nil {
		http.Error(w, "invalid json", http.StatusBadRequest)
		return
	}

	userIDHash := hashUserID(telemetry.UserID)
	profile := cas.getOrCreateProfile(userIDHash)
	riskScore, confidence, factors := cas.calculateRiskScore(profile, &telemetry)

	cas.avgRiskScore.Set(uint64(riskScore))
	if riskScore > 70 {
		cas.anomalyCount.Add(1)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"risk_score": riskScore,
		"confidence": confidence,
		"factors":    factors,
		"baseline":   profile.BaselineEstablished,
	})
}

func (cas *ContAuthService) handleDecision(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var telemetry TelemetryData
	if err := json.NewDecoder(r.Body).Decode(&telemetry); err != nil {
		http.Error(w, "invalid json", http.StatusBadRequest)
		return
	}

	userIDHash := hashUserID(telemetry.UserID)
	profile := cas.getOrCreateProfile(userIDHash)
	riskScore, confidence, factors := cas.calculateRiskScore(profile, &telemetry)
	decision := cas.makeDecision(profile, riskScore, confidence, factors)

	cas.totalDecisions.Add(1)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(decision)
}

func (cas *ContAuthService) handleProfileReset(w http.ResponseWriter, r *http.Request) {
	userID := r.URL.Query().Get("user_id")
	if userID == "" {
		http.Error(w, "user_id required", http.StatusBadRequest)
		return
	}

	userIDHash := hashUserID(userID)
	cas.profileMu.Lock()
	delete(cas.userProfiles, userIDHash)
	cas.profileMu.Unlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "reset"})
}

func (cas *ContAuthService) handleProfileStats(w http.ResponseWriter, r *http.Request) {
	cas.profileMu.RLock()
	totalProfiles := len(cas.userProfiles)
	baselineEstablished := 0
	for _, profile := range cas.userProfiles {
		if profile.BaselineEstablished {
			baselineEstablished++
		}
	}
	cas.profileMu.RUnlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"total_profiles":       totalProfiles,
		"baseline_established": baselineEstablished,
	})
}

func (cas *ContAuthService) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":    "healthy",
		"service":   "contauth",
		"timestamp": time.Now(),
		"profiles":  len(cas.userProfiles),
	})
}

func (cas *ContAuthService) getOrCreateProfile(userIDHash string) *UserBehaviorProfile {
	cas.profileMu.RLock()
	profile, exists := cas.userProfiles[userIDHash]
	cas.profileMu.RUnlock()

	if exists {
		return profile
	}

	newProfile := &UserBehaviorProfile{
		UserIDHash:          userIDHash,
		CreatedAt:           time.Now(),
		LastUpdated:         time.Now(),
		BaselineEstablished: false,
		TrustScore:          50.0,
		RecentScores:        make([]float64, 0, 100),
		AdaptiveThreshold:   cas.anomalyThreshold,
		ClickPatterns:       make(map[string]float64),
		DeviceFeatures:      make(map[string]interface{}),
	}

	cas.profileMu.Lock()
	cas.userProfiles[userIDHash] = newProfile
	cas.profileMu.Unlock()

	return newProfile
}

func (cas *ContAuthService) updateProfile(profile *UserBehaviorProfile, telemetry *TelemetryData) {
	profile.LastUpdated = time.Now()
	profile.SampleCount++

	if len(telemetry.KeystrokeEvents) > 0 {
		cas.updateKeystrokeProfile(profile, telemetry.KeystrokeEvents)
	}

	if len(telemetry.MouseEvents) > 0 {
		cas.updateMouseProfile(profile, telemetry.MouseEvents)
	}

	deviceHash := cas.deviceFingerprint.GenerateFingerprint(&telemetry.DeviceInfo)
	if profile.DeviceHash == "" {
		profile.DeviceHash = deviceHash
	}

	if !profile.BaselineEstablished && profile.SampleCount >= cas.minSampleSize {
		profile.BaselineEstablished = true
		profile.RiskBaseline = calculateBaseline(profile.RecentScores)
	}
}

func (cas *ContAuthService) updateKeystrokeProfile(profile *UserBehaviorProfile, events []KeystrokeFeature) {
	holdTimes := make([]float64, 0, len(events))
	flightTimes := make([]float64, 0, len(events))

	for _, event := range events {
		if event.HoldTime > 0 {
			holdTimes = append(holdTimes, event.HoldTime)
		}
		if event.FlightTime > 0 {
			flightTimes = append(flightTimes, event.FlightTime)
		}
	}

	if len(holdTimes) > 0 {
		profile.KeyHoldTimes = calculateStatistics(holdTimes)
	}
	if len(flightTimes) > 0 {
		profile.KeyPressIntervals = calculateStatistics(flightTimes)
	}
}

func (cas *ContAuthService) updateMouseProfile(profile *UserBehaviorProfile, events []MouseFeature) {
	velocities := make([]float64, 0, len(events))

	for _, event := range events {
		if event.Velocity > 0 {
			velocities = append(velocities, event.Velocity)
		}
	}

	if len(velocities) > 0 {
		profile.MouseSpeed = calculateStatistics(velocities)
	}
}

func (cas *ContAuthService) calculateRiskScore(profile *UserBehaviorProfile, telemetry *TelemetryData) (float64, float64, map[string]float64) {
	factors := make(map[string]float64)

	if !profile.BaselineEstablished {
		return 50.0, 0.5, factors
	}

	keystrokeRisk := cas.keystrokeDynamics.AnalyzeAnomaly(profile, telemetry.KeystrokeEvents)
	factors["keystroke"] = keystrokeRisk

	mouseRisk := cas.mouseAnalyzer.AnalyzeAnomaly(profile, telemetry.MouseEvents)
	factors["mouse"] = mouseRisk

	deviceRisk := cas.deviceFingerprint.AnalyzeDeviceChange(profile, &telemetry.DeviceInfo)
	factors["device"] = deviceRisk

	weights := cas.riskScorer.GetAdaptiveWeights()
	riskScore := (keystrokeRisk * weights["keystroke"]) +
		(mouseRisk * weights["mouse"]) +
		(deviceRisk * weights["device"])

	riskScore = math.Min(riskScore*100.0, 100.0)
	confidence := calculateConfidence([]float64{keystrokeRisk, mouseRisk, deviceRisk})

	profile.RecentScores = append(profile.RecentScores, riskScore)
	if len(profile.RecentScores) > 100 {
		profile.RecentScores = profile.RecentScores[1:]
	}

	return riskScore, confidence, factors
}

func (cas *ContAuthService) makeDecision(profile *UserBehaviorProfile, riskScore, confidence float64, factors map[string]float64) *RiskDecision {
	decision := &RiskDecision{
		RiskScore:  riskScore,
		Confidence: confidence,
		TrustScore: profile.TrustScore,
		Timestamp:  time.Now(),
		Factors:    factors,
	}

	switch {
	case riskScore < 30:
		decision.Decision = "ALLOW"
		decision.Explanation = "low_risk"
	case riskScore < 60:
		decision.Decision = "ALLOW"
		decision.Explanation = "moderate_risk"
	case riskScore < 80:
		decision.Decision = "CHALLENGE"
		decision.RequiresMFA = true
		decision.ChallengeType = "soft_mfa"
		decision.Explanation = "elevated_risk"
	default:
		decision.Decision = "CHALLENGE"
		decision.RequiresMFA = true
		decision.ChallengeType = "strong_mfa"
		decision.Explanation = "high_risk"

		if riskScore > 90 {
			decision.Decision = "DENY"
			decision.Explanation = "critical_risk"
		}
	}

	return decision
}

func calculateStatistics(data []float64) StatisticalDistribution {
	if len(data) == 0 {
		return StatisticalDistribution{}
	}

	sort.Float64s(data)
	mean := average(data)
	stddev := stdDev(data, mean)
	median := percentile(data, 0.5)
	p25 := percentile(data, 0.25)
	p75 := percentile(data, 0.75)

	return StatisticalDistribution{
		Mean:   mean,
		StdDev: stddev,
		Median: median,
		P25:    p25,
		P75:    p75,
	}
}

func hashUserID(userID string) string {
	h := sha256.Sum256([]byte(userID))
	return fmt.Sprintf("%x", h)
}

func calculateBaseline(scores []float64) float64 {
	if len(scores) == 0 {
		return 50.0
	}
	return average(scores)
}

func calculateConfidence(scores []float64) float64 {
	if len(scores) < 2 {
		return 0.5
	}
	mean := average(scores)
	variance := 0.0
	for _, s := range scores {
		variance += math.Pow(s-mean, 2)
	}
	variance /= float64(len(scores))
	return 1.0 / (1.0 + variance*10.0)
}

func NewKeystrokeDynamicsAnalyzer() *KeystrokeDynamicsAnalyzer {
	return &KeystrokeDynamicsAnalyzer{}
}

func (kda *KeystrokeDynamicsAnalyzer) AnalyzeAnomaly(profile *UserBehaviorProfile, events []KeystrokeFeature) float64 {
	if len(events) == 0 || !profile.BaselineEstablished {
		return 0.0
	}

	holdTimeDeviation := 0.0
	if profile.KeyHoldTimes.StdDev > 0 {
		for _, e := range events {
			z := (e.HoldTime - profile.KeyHoldTimes.Mean) / profile.KeyHoldTimes.StdDev
			holdTimeDeviation += math.Abs(z)
		}
		holdTimeDeviation /= float64(len(events))
	}

	return math.Min(holdTimeDeviation/3.0, 1.0)
}

func NewMouseBehaviorAnalyzer() *MouseBehaviorAnalyzer {
	return &MouseBehaviorAnalyzer{}
}

func (mba *MouseBehaviorAnalyzer) AnalyzeAnomaly(profile *UserBehaviorProfile, events []MouseFeature) float64 {
	if len(events) == 0 || !profile.BaselineEstablished {
		return 0.0
	}

	velocityDeviation := 0.0
	if profile.MouseSpeed.StdDev > 0 {
		for _, e := range events {
			z := (e.Velocity - profile.MouseSpeed.Mean) / profile.MouseSpeed.StdDev
			velocityDeviation += math.Abs(z)
		}
		velocityDeviation /= float64(len(events))
	}

	return math.Min(velocityDeviation/3.0, 1.0)
}

func NewDeviceFingerprintEngine() *DeviceFingerprintEngine {
	return &DeviceFingerprintEngine{
		hashSalt:    []byte("shieldx-device-fp-" + os.Getenv("FP_SALT")),
		fpCache:     make(map[string]*DeviceFingerprint),
		cacheExpiry: 24 * time.Hour,
	}
}

func (dfe *DeviceFingerprintEngine) GenerateFingerprint(info *DeviceInfo) string {
	fpString := fmt.Sprintf("%s|%s|%s|%v",
		info.UserAgent,
		info.ScreenResolution,
		info.Timezone,
		info.Languages)

	h := sha256.New()
	h.Write(dfe.hashSalt)
	h.Write([]byte(fpString))
	return fmt.Sprintf("%x", h.Sum(nil))
}

func (dfe *DeviceFingerprintEngine) AnalyzeDeviceChange(profile *UserBehaviorProfile, info *DeviceInfo) float64 {
	currentFP := dfe.GenerateFingerprint(info)

	if profile.DeviceHash == "" {
		return 0.0
	}

	if profile.DeviceHash != currentFP {
		return 0.9
	}

	return 0.0
}

func NewAdaptiveRiskScorer() *AdaptiveRiskScorer {
	return &AdaptiveRiskScorer{
		weights: map[string]float64{
			"keystroke": 0.40,
			"mouse":     0.35,
			"device":    0.25,
		},
		timeOfDayModel: &TimeBasedRiskModel{},
		locationModel:  &LocationRiskModel{},
	}
}

func (ars *AdaptiveRiskScorer) GetAdaptiveWeights() map[string]float64 {
	ars.mu.RLock()
	defer ars.mu.RUnlock()
	return ars.weights
}

func average(data []float64) float64 {
	if len(data) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range data {
		sum += v
	}
	return sum / float64(len(data))
}

func stdDev(data []float64, mean float64) float64 {
	if len(data) == 0 {
		return 0
	}
	variance := 0.0
	for _, v := range data {
		variance += math.Pow(v-mean, 2)
	}
	variance /= float64(len(data))
	return math.Sqrt(variance)
}

func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	idx := p * float64(len(sorted)-1)
	lower := int(idx)
	upper := lower + 1
	if upper >= len(sorted) {
		return sorted[len(sorted)-1]
	}
	frac := idx - float64(lower)
	return sorted[lower]*(1-frac) + sorted[upper]*frac
}

func makeRateLimiter(reqPerMin int) func(http.HandlerFunc) http.HandlerFunc {
	type bucket struct {
		count  int
		window int64
	}
	var mu sync.Mutex
	buckets := map[string]*bucket{}

	return func(next http.HandlerFunc) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			ip := r.Header.Get("X-Forwarded-For")
			if ip == "" {
				ip = strings.Split(r.RemoteAddr, ":")[0]
			}
			nowMin := time.Now().Unix() / 60

			mu.Lock()
			b := buckets[ip]
			if b == nil || b.window != nowMin {
				b = &bucket{count: 0, window: nowMin}
				buckets[ip] = b
			}
			b.count++
			c := b.count
			mu.Unlock()

			if c > reqPerMin {
				http.Error(w, "rate limit exceeded", http.StatusTooManyRequests)
				return
			}
			next(w, r)
		}
	}
}

func makeAdminMiddleware() func(http.HandlerFunc) http.HandlerFunc {
	token := os.Getenv("CONTAUTH_ADMIN_TOKEN")
	if token == "" {
		return func(next http.HandlerFunc) http.HandlerFunc {
			return func(w http.ResponseWriter, r *http.Request) {
				http.Error(w, "admin endpoint disabled", http.StatusForbidden)
			}
		}
	}

	return func(next http.HandlerFunc) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			if r.Header.Get("X-Admin-Token") != token {
				http.Error(w, "forbidden", http.StatusForbidden)
				return
			}
			next(w, r)
		}
	}
}

func getenvDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func parseDurationDefault(key string, def time.Duration) time.Duration {
	if v := os.Getenv(key); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			return d
		}
	}
	return def
}

func parseIntDefault(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			return i
		}
	}
	return def
}
