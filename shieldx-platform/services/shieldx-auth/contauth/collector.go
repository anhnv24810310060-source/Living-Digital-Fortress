package main

import (
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"math"

	"shieldx/pkg/security/cryptoatrest"

	_ "github.com/lib/pq"
)

type SessionTelemetry struct {
	SessionID       string                 `json:"session_id"`
	UserID          string                 `json:"user_id"`
	DeviceID        string                 `json:"device_id"`
	IPAddress       string                 `json:"ip_address"`
	UserAgent       string                 `json:"user_agent"`
	KeystrokeData   []KeystrokeEvent       `json:"keystroke_data"`
	MouseData       []MouseEvent           `json:"mouse_data"`
	AccessPatterns  []AccessEvent          `json:"access_patterns"`
	GeolocationData GeolocationInfo        `json:"geolocation"`
	DeviceMetrics   DeviceFingerprint      `json:"device_metrics"`
	Timestamp       time.Time              `json:"timestamp"`
	Metadata        map[string]interface{} `json:"metadata"`
}

type KeystrokeEvent struct {
	Key       string  `json:"key"`
	Timestamp int64   `json:"timestamp"`
	Duration  float64 `json:"duration"`
	Pressure  float64 `json:"pressure"`
}

type MouseEvent struct {
	X         int     `json:"x"`
	Y         int     `json:"y"`
	Timestamp int64   `json:"timestamp"`
	EventType string  `json:"event_type"`
	Velocity  float64 `json:"velocity"`
}

type AccessEvent struct {
	Resource  string    `json:"resource"`
	Action    string    `json:"action"`
	Timestamp time.Time `json:"timestamp"`
	Success   bool      `json:"success"`
}

type GeolocationInfo struct {
	Latitude  float64 `json:"latitude"`
	Longitude float64 `json:"longitude"`
	Country   string  `json:"country"`
	City      string  `json:"city"`
	ISP       string  `json:"isp"`
}

type DeviceFingerprint struct {
	ScreenResolution string  `json:"screen_resolution"`
	Timezone         string  `json:"timezone"`
	Language         string  `json:"language"`
	Platform         string  `json:"platform"`
	CPUCores         int     `json:"cpu_cores"`
	Memory           int64   `json:"memory"`
	BatteryLevel     float64 `json:"battery_level"`
	NetworkType      string  `json:"network_type"`
}

type RiskScore struct {
	SessionID       string    `json:"session_id"`
	OverallScore    float64   `json:"overall_score"`
	KeystrokeScore  float64   `json:"keystroke_score"`
	MouseScore      float64   `json:"mouse_score"`
	LocationScore   float64   `json:"location_score"`
	DeviceScore     float64   `json:"device_score"`
	BehaviorScore   float64   `json:"behavior_score"`
	ReputationScore float64   `json:"reputation_score"`
	RiskFactors     []string  `json:"risk_factors"`
	Recommendation  string    `json:"recommendation"`
	CalculatedAt    time.Time `json:"calculated_at"`
}

type AuthDecision struct {
	SessionID  string    `json:"session_id"`
	Action     string    `json:"action"`
	Confidence float64   `json:"confidence"`
	Reason     string    `json:"reason"`
	Challenge  string    `json:"challenge,omitempty"`
	ExpiresAt  time.Time `json:"expires_at"`
}

type ContAuthCollector struct {
	db  *sql.DB
	enc *cryptoatrest.Encryptor
	// in-memory cache to avoid repeated DB hits for hot sessions
	cache *riskCache
}

func NewContAuthCollector(dbURL string) (*ContAuthCollector, error) {
	db, err := sql.Open("postgres", dbURL)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}

	if err := db.Ping(); err != nil {
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	db.SetMaxOpenConns(25)
	db.SetMaxIdleConns(5)
	db.SetConnMaxLifetime(5 * time.Minute)

	enc, err := cryptoatrest.NewFromEnv("CONTAUTH_ENC_KEY")
	if err != nil {
		// To respect constraint "encrypt telemetry at rest", fail fast if key missing
		return nil, fmt.Errorf("encryption key missing: %w", err)
	}
	collector := &ContAuthCollector{db: db, enc: enc, cache: newRiskCache(parseTTL(getenv("CONTAUTH_RISK_TTL"), 2*time.Minute), 8192)}
	if err := collector.migrate(); err != nil {
		return nil, fmt.Errorf("migration failed: %w", err)
	}

	return collector, nil
}

func (c *ContAuthCollector) migrate() error {
	query := `
	CREATE TABLE IF NOT EXISTS session_telemetry (
		id SERIAL PRIMARY KEY,
		session_id VARCHAR(255) NOT NULL,
		user_id VARCHAR(255) NOT NULL,
		device_id VARCHAR(255) NOT NULL,
		ip_address INET NOT NULL,
		user_agent TEXT,
		keystroke_data JSONB,
		mouse_data JSONB,
		access_patterns JSONB,
		geolocation_data JSONB,
		device_metrics JSONB,
		metadata JSONB,
		timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
		created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
	);

	CREATE TABLE IF NOT EXISTS risk_scores (
		id SERIAL PRIMARY KEY,
		session_id VARCHAR(255) NOT NULL,
		overall_score FLOAT NOT NULL,
		keystroke_score FLOAT NOT NULL,
		mouse_score FLOAT NOT NULL,
		location_score FLOAT NOT NULL,
		device_score FLOAT NOT NULL,
		behavior_score FLOAT NOT NULL,
		reputation_score FLOAT NOT NULL,
		risk_factors JSONB,
		recommendation VARCHAR(100),
		calculated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
	);

	CREATE TABLE IF NOT EXISTS auth_decisions (
		id SERIAL PRIMARY KEY,
		session_id VARCHAR(255) NOT NULL,
		action VARCHAR(50) NOT NULL,
		confidence FLOAT NOT NULL,
		reason TEXT,
		challenge VARCHAR(100),
		expires_at TIMESTAMP WITH TIME ZONE,
		created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
	);

	CREATE TABLE IF NOT EXISTS user_baselines (
		user_id VARCHAR(255) PRIMARY KEY,
		avg_keystroke_interval FLOAT,
		avg_keystroke_duration FLOAT,
		typical_mouse_velocity FLOAT,
		common_locations JSONB,
		device_fingerprints JSONB,
		access_patterns JSONB,
		last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW()
	);

	CREATE INDEX IF NOT EXISTS idx_session_telemetry_session_id ON session_telemetry(session_id);
	CREATE INDEX IF NOT EXISTS idx_session_telemetry_user_id ON session_telemetry(user_id);
	CREATE INDEX IF NOT EXISTS idx_session_telemetry_timestamp ON session_telemetry(timestamp);
	CREATE INDEX IF NOT EXISTS idx_risk_scores_session_id ON risk_scores(session_id);
	CREATE INDEX IF NOT EXISTS idx_auth_decisions_session_id ON auth_decisions(session_id);`

	_, err := c.db.Exec(query)
	return err
}

func (c *ContAuthCollector) CollectTelemetry(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var telemetry SessionTelemetry
	if err := json.NewDecoder(r.Body).Decode(&telemetry); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	if telemetry.SessionID == "" || telemetry.UserID == "" {
		http.Error(w, "Missing required fields", http.StatusBadRequest)
		return
	}

	telemetry.Timestamp = time.Now()
	// Constraint: do not store raw biometric data; hash or extract features only
	// Compute feature summaries
	avgInt, avgDur := summarizeKeystrokes(telemetry.KeystrokeData)
	avgMouseVel := summarizeMouse(telemetry.MouseData)
	kHash := hashKeystrokes(telemetry.KeystrokeData)
	mHash := hashMouse(telemetry.MouseData)
	// Keep only minimal, non-identifying summaries
	telemetry.KeystrokeData = nil
	telemetry.MouseData = nil
	if telemetry.Metadata == nil {
		telemetry.Metadata = map[string]interface{}{}
	}
	telemetry.Metadata["ks_sig"] = kHash
	telemetry.Metadata["mouse_sig"] = mHash
	telemetry.Metadata["ks_avg_interval"] = avgInt
	telemetry.Metadata["ks_avg_duration"] = avgDur
	telemetry.Metadata["mouse_avg_velocity"] = avgMouseVel

	if err := c.storeTelemetry(telemetry); err != nil {
		log.Printf("Failed to store telemetry: %v", err)
		http.Error(w, "Failed to store telemetry", http.StatusInternalServerError)
		return
	}
	// Update behavioral baselines asynchronously (no raw biometrics)
	if c.db != nil {
		go func(user string) {
			if err := c.updateBaselineFromMetadata(user); err != nil {
				log.Printf("[contauth] update baseline: %v", err)
			}
		}(telemetry.UserID)
	}

	response := map[string]interface{}{
		"success":    true,
		"session_id": telemetry.SessionID,
		"message":    "Telemetry collected successfully",
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(response); err != nil {
		log.Printf("Failed to encode response: %v", err)
	}
}

func (c *ContAuthCollector) CalculateRiskScore(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var request struct {
		SessionID string `json:"session_id"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	if request.SessionID == "" {
		http.Error(w, "Missing session_id", http.StatusBadRequest)
		return
	}

	// Fast path: if cached risk exists and still valid, return it directly
	if c.cache != nil {
		if rs, ok := c.cache.Get(request.SessionID); ok {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(rs)
			return
		}
	}

	telemetry, err := c.getRecentTelemetry(request.SessionID)
	if err != nil {
		log.Printf("Failed to get telemetry: %v", err)
		http.Error(w, "Failed to get session data", http.StatusInternalServerError)
		return
	}

	if telemetry == nil {
		http.Error(w, "No telemetry data found for session", http.StatusNotFound)
		return
	}

	riskScore := c.calculateRisk(*telemetry)

	// Optional: consult ML Orchestrator for ensemble anomaly score
	if mlURL := getenv("MLO_URL"); mlURL != "" {
		if ens := c.queryMLO(*telemetry, mlURL); ens != nil {
			// Combine conservatively: weighted average and max guard
			wgt := parseFloat(getenv("MLO_WEIGHT"), 0.35)
			combined := wgt*ens.Score + (1-wgt)*riskScore.OverallScore
			if ens.IsAnomaly {
				combined = maxFloat(combined, 0.7) // bump if strong anomaly flagged
			}
			// Clamp
			if combined < 0 {
				combined = 0
			} else if combined > 1 {
				combined = 1
			}
			riskScore.OverallScore = combined
			// Adjust recommendation accordingly
			if combined > 0.8 {
				riskScore.Recommendation = "block"
				riskScore.RiskFactors = appendUnique(riskScore.RiskFactors, "ml_anomaly_high")
			} else if combined > 0.6 {
				riskScore.Recommendation = "challenge"
				riskScore.RiskFactors = appendUnique(riskScore.RiskFactors, "ml_anomaly_elevated")
			}
		}
	}

	if err := c.storeRiskScore(riskScore); err != nil {
		log.Printf("Failed to store risk score: %v", err)
	}
	// Security event log (no PII): session, score, recommendation
	log.Printf("[contauth] risk session=%s overall=%.3f rec=%s", sanitizeID(request.SessionID), riskScore.OverallScore, riskScore.Recommendation)

	// populate cache
	if c.cache != nil {
		c.cache.Set(request.SessionID, riskScore)
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(riskScore); err != nil {
		log.Printf("Failed to encode response: %v", err)
	}
}

func (c *ContAuthCollector) GetAuthDecision(w http.ResponseWriter, r *http.Request) {
	sessionID := r.URL.Query().Get("session_id")
	if sessionID == "" {
		http.Error(w, "Missing session_id parameter", http.StatusBadRequest)
		return
	}

	// prefer cached risk to reduce DB load
	var riskScore *RiskScore
	if c.cache != nil {
		if rs, ok := c.cache.Get(sessionID); ok {
			tmp := rs
			riskScore = &tmp
		} else {
			var err error
			riskScore, err = c.getLatestRiskScore(sessionID)
			if err != nil {
				log.Printf("Failed to get risk score: %v", err)
				http.Error(w, "Failed to get risk assessment", http.StatusInternalServerError)
				return
			}
		}
	} else {
		var err error
		riskScore, err = c.getLatestRiskScore(sessionID)
		if err != nil {
			log.Printf("Failed to get risk score: %v", err)
			http.Error(w, "Failed to get risk assessment", http.StatusInternalServerError)
			return
		}
	}

	if riskScore == nil {
		http.Error(w, "No risk assessment found for session", http.StatusNotFound)
		return
	}

	decision := c.makeAuthDecision(*riskScore)

	if err := c.storeAuthDecision(decision); err != nil {
		log.Printf("Failed to store auth decision: %v", err)
	}
	// Security decision log
	log.Printf("[contauth] decision session=%s action=%s conf=%.2f", sanitizeID(sessionID), decision.Action, decision.Confidence)

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(decision); err != nil {
		log.Printf("Failed to encode response: %v", err)
	}
}

func (c *ContAuthCollector) storeTelemetry(telemetry SessionTelemetry) error {
	if c.db == nil {
		return nil
	}
	query := `
	INSERT INTO session_telemetry 
	(session_id, user_id, device_id, ip_address, user_agent, keystroke_data, 
	 mouse_data, access_patterns, geolocation_data, device_metrics, metadata, timestamp)
	VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)`

	// Encrypt blobs at rest
	keystrokeJSON, _ := json.Marshal(telemetry.KeystrokeData)
	mouseJSON, _ := json.Marshal(telemetry.MouseData)
	accessJSON, _ := json.Marshal(telemetry.AccessPatterns)
	geoJSON, _ := json.Marshal(telemetry.GeolocationData)
	deviceJSON, _ := json.Marshal(telemetry.DeviceMetrics)
	metadataJSON, _ := json.Marshal(telemetry.Metadata)

	encKS, encMS, encAP, encGeo, encDev, encMeta := string(keystrokeJSON), string(mouseJSON), string(accessJSON), string(geoJSON), string(deviceJSON), string(metadataJSON)
	if c.enc != nil {
		if v, err := c.enc.Encrypt(keystrokeJSON); err == nil {
			b, _ := json.Marshal(v)
			encKS = string(b)
		}
		if v, err := c.enc.Encrypt(mouseJSON); err == nil {
			b, _ := json.Marshal(v)
			encMS = string(b)
		}
		if v, err := c.enc.Encrypt(accessJSON); err == nil {
			b, _ := json.Marshal(v)
			encAP = string(b)
		}
		if v, err := c.enc.Encrypt(geoJSON); err == nil {
			b, _ := json.Marshal(v)
			encGeo = string(b)
		}
		if v, err := c.enc.Encrypt(deviceJSON); err == nil {
			b, _ := json.Marshal(v)
			encDev = string(b)
		}
		if v, err := c.enc.Encrypt(metadataJSON); err == nil {
			b, _ := json.Marshal(v)
			encMeta = string(b)
		}
	}

	_, err := c.db.Exec(query,
		telemetry.SessionID, telemetry.UserID, telemetry.DeviceID,
		telemetry.IPAddress, telemetry.UserAgent,
		encKS, encMS, encAP,
		encGeo, encDev, encMeta,
		telemetry.Timestamp)

	return err
}

func (c *ContAuthCollector) getRecentTelemetry(sessionID string) (*SessionTelemetry, error) {
	if c.db == nil {
		return nil, fmt.Errorf("database disabled")
	}
	query := `
	SELECT session_id, user_id, device_id, ip_address, user_agent,
		   keystroke_data, mouse_data, access_patterns, geolocation_data,
		   device_metrics, metadata, timestamp
	FROM session_telemetry 
	WHERE session_id = $1 
	ORDER BY timestamp DESC 
	LIMIT 1`

	var telemetry SessionTelemetry
	var keystrokeJSON, mouseJSON, accessJSON, geoJSON, deviceJSON, metadataJSON string

	err := c.db.QueryRow(query, sessionID).Scan(
		&telemetry.SessionID, &telemetry.UserID, &telemetry.DeviceID,
		&telemetry.IPAddress, &telemetry.UserAgent,
		&keystrokeJSON, &mouseJSON, &accessJSON,
		&geoJSON, &deviceJSON, &metadataJSON,
		&telemetry.Timestamp)

	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}

	// Decrypt blobs
	if c.enc != nil {
		// Unquote JSON string to raw base64 before decrypt
		if b64 := unquoteIfQuoted(keystrokeJSON); b64 != "" {
			if b, err := c.enc.Decrypt(b64); err == nil {
				_ = json.Unmarshal(b, &telemetry.KeystrokeData)
			}
		}
		if b64 := unquoteIfQuoted(mouseJSON); b64 != "" {
			if b, err := c.enc.Decrypt(b64); err == nil {
				_ = json.Unmarshal(b, &telemetry.MouseData)
			}
		}
		if b64 := unquoteIfQuoted(accessJSON); b64 != "" {
			if b, err := c.enc.Decrypt(b64); err == nil {
				_ = json.Unmarshal(b, &telemetry.AccessPatterns)
			}
		}
		if b64 := unquoteIfQuoted(geoJSON); b64 != "" {
			if b, err := c.enc.Decrypt(b64); err == nil {
				_ = json.Unmarshal(b, &telemetry.GeolocationData)
			}
		}
		if b64 := unquoteIfQuoted(deviceJSON); b64 != "" {
			if b, err := c.enc.Decrypt(b64); err == nil {
				_ = json.Unmarshal(b, &telemetry.DeviceMetrics)
			}
		}
		if b64 := unquoteIfQuoted(metadataJSON); b64 != "" {
			if b, err := c.enc.Decrypt(b64); err == nil {
				_ = json.Unmarshal(b, &telemetry.Metadata)
			}
		}
	} else {
		// fallback: assume plaintext JSON (tests)
		_ = json.Unmarshal([]byte(keystrokeJSON), &telemetry.KeystrokeData)
		_ = json.Unmarshal([]byte(mouseJSON), &telemetry.MouseData)
		_ = json.Unmarshal([]byte(accessJSON), &telemetry.AccessPatterns)
		_ = json.Unmarshal([]byte(geoJSON), &telemetry.GeolocationData)
		_ = json.Unmarshal([]byte(deviceJSON), &telemetry.DeviceMetrics)
		_ = json.Unmarshal([]byte(metadataJSON), &telemetry.Metadata)
	}

	return &telemetry, nil
}

func (c *ContAuthCollector) calculateRisk(telemetry SessionTelemetry) RiskScore {
	riskScore := RiskScore{
		SessionID:    telemetry.SessionID,
		CalculatedAt: time.Now(),
		RiskFactors:  []string{},
	}

	riskScore.KeystrokeScore = c.calculateKeystrokeRisk(telemetry.KeystrokeData, telemetry.UserID)
	riskScore.MouseScore = c.calculateMouseRisk(telemetry.MouseData, telemetry.UserID)
	riskScore.LocationScore = c.calculateLocationRisk(telemetry.GeolocationData, telemetry.UserID)
	riskScore.DeviceScore = c.calculateDeviceRisk(telemetry.DeviceMetrics, telemetry.UserID)
	riskScore.BehaviorScore = c.calculateBehaviorRisk(telemetry.AccessPatterns, telemetry.UserID)
	riskScore.ReputationScore = c.calculateReputationRisk(telemetry.IPAddress, telemetry.UserAgent)

	weights := map[string]float64{
		"keystroke":  0.25,
		"mouse":      0.15,
		"location":   0.20,
		"device":     0.15,
		"behavior":   0.15,
		"reputation": 0.10,
	}

	riskScore.OverallScore =
		riskScore.KeystrokeScore*weights["keystroke"] +
			riskScore.MouseScore*weights["mouse"] +
			riskScore.LocationScore*weights["location"] +
			riskScore.DeviceScore*weights["device"] +
			riskScore.BehaviorScore*weights["behavior"] +
			riskScore.ReputationScore*weights["reputation"]

	if riskScore.OverallScore > 0.8 {
		riskScore.Recommendation = "block"
		riskScore.RiskFactors = append(riskScore.RiskFactors, "high_risk_score")
	} else if riskScore.OverallScore > 0.6 {
		riskScore.Recommendation = "challenge"
		riskScore.RiskFactors = append(riskScore.RiskFactors, "elevated_risk")
	} else {
		riskScore.Recommendation = "allow"
	}

	return riskScore
}

func (c *ContAuthCollector) calculateKeystrokeRisk(keystrokes []KeystrokeEvent, userID string) float64 {
	if len(keystrokes) < 5 {
		// Fallback to aggregated metadata if available
		base := c.getUserBaseline(userID)
		if base != nil {
			// Without raw data, default to moderate-low risk
			return 0.35
		}
		return 0.5
	}

	var intervals []float64
	var durations []float64

	for i := 1; i < len(keystrokes); i++ {
		interval := float64(keystrokes[i].Timestamp - keystrokes[i-1].Timestamp)
		intervals = append(intervals, interval)
		durations = append(durations, keystrokes[i].Duration)
	}

	avgInterval := average(intervals)
	avgDuration := average(durations)

	intervalVariance := variance(intervals, avgInterval)
	durationVariance := variance(durations, avgDuration)

	baseline := c.getUserBaseline(userID)

	intervalDeviation := 0.0
	durationDeviation := 0.0

	if baseline != nil {
		if baseline.AvgKeystrokeInterval > 0 {
			intervalDeviation = abs(avgInterval-baseline.AvgKeystrokeInterval) / baseline.AvgKeystrokeInterval
		}
		if baseline.AvgKeystrokeDuration > 0 {
			durationDeviation = abs(avgDuration-baseline.AvgKeystrokeDuration) / baseline.AvgKeystrokeDuration
		}
	}

	riskScore := (intervalVariance/1000 + durationVariance/100 + intervalDeviation + durationDeviation) / 4

	return min(riskScore, 1.0)
}

// hashKeystrokes produces a stable feature hash; does not store raw keys
func hashKeystrokes(ks []KeystrokeEvent) string {
	h := sha256.New()
	for _, e := range ks {
		// Only timing features; no raw key values
		fmt.Fprintf(h, "%d:%f:%f|", e.Timestamp, e.Duration, e.Pressure)
	}
	return hex.EncodeToString(h.Sum(nil))
}

// hashMouse summarizes mouse behavior without raw coordinates
func hashMouse(ms []MouseEvent) string {
	h := sha256.New()
	for _, e := range ms {
		fmt.Fprintf(h, "%d:%f:%s|", e.Timestamp, e.Velocity, e.EventType)
	}
	return hex.EncodeToString(h.Sum(nil))
}

func (c *ContAuthCollector) calculateMouseRisk(mouseEvents []MouseEvent, userID string) float64 {
	if len(mouseEvents) < 10 {
		base := c.getUserBaseline(userID)
		if base != nil {
			return 0.25
		}
		return 0.3
	}

	var velocities []float64
	for _, event := range mouseEvents {
		if event.Velocity > 0 {
			velocities = append(velocities, event.Velocity)
		}
	}

	if len(velocities) == 0 {
		return 0.5
	}

	avgVelocity := average(velocities)
	velocityVariance := variance(velocities, avgVelocity)

	baseline := c.getUserBaseline(userID)
	velocityDeviation := 0.0

	if baseline != nil && baseline.TypicalMouseVelocity > 0 {
		velocityDeviation = abs(avgVelocity-baseline.TypicalMouseVelocity) / baseline.TypicalMouseVelocity
	}

	riskScore := (velocityVariance/10000 + velocityDeviation) / 2
	return min(riskScore, 1.0)
}

func (c *ContAuthCollector) calculateLocationRisk(geo GeolocationInfo, userID string) float64 {
	baseline := c.getUserBaseline(userID)
	if baseline == nil {
		return 0.2
	}

	if geo.Country == "" {
		return 0.8
	}

	commonLocations := []string{"US", "CA", "GB"}
	for _, location := range commonLocations {
		if geo.Country == location {
			return 0.1
		}
	}

	return 0.7
}

func (c *ContAuthCollector) calculateDeviceRisk(device DeviceFingerprint, userID string) float64 {
	baseline := c.getUserBaseline(userID)
	if baseline == nil {
		return 0.3
	}

	riskFactors := 0
	totalFactors := 5

	if device.ScreenResolution == "" {
		riskFactors++
	}
	if device.Timezone == "" {
		riskFactors++
	}
	if device.Language == "" {
		riskFactors++
	}
	if device.Platform == "" {
		riskFactors++
	}
	if device.CPUCores <= 0 {
		riskFactors++
	}

	return float64(riskFactors) / float64(totalFactors)
}

func (c *ContAuthCollector) calculateBehaviorRisk(accessPatterns []AccessEvent, userID string) float64 {
	if len(accessPatterns) == 0 {
		return 0.5
	}

	riskScore := 0.0

	failureCount := 0
	for _, event := range accessPatterns {
		if !event.Success {
			failureCount++
		}
	}

	failureRate := float64(failureCount) / float64(len(accessPatterns))
	if failureRate > 0.3 {
		riskScore += 0.5
	}

	return min(riskScore, 1.0)
}

func (c *ContAuthCollector) calculateReputationRisk(ipAddress, userAgent string) float64 {
	riskScore := 0.0

	suspiciousAgents := []string{"bot", "crawler", "scanner"}
	for _, suspicious := range suspiciousAgents {
		if strings.Contains(strings.ToLower(userAgent), suspicious) {
			riskScore += 0.3
		}
	}

	if ipAddress == "" {
		riskScore += 0.5
	}

	return min(riskScore, 1.0)
}

func (c *ContAuthCollector) makeAuthDecision(riskScore RiskScore) AuthDecision {
	decision := AuthDecision{
		SessionID:  riskScore.SessionID,
		Confidence: 1.0 - riskScore.OverallScore,
		ExpiresAt:  time.Now().Add(15 * time.Minute),
	}

	if riskScore.OverallScore > 0.8 {
		decision.Action = "block"
		decision.Reason = "High risk score detected"
	} else if riskScore.OverallScore > 0.6 {
		decision.Action = "challenge"
		decision.Reason = "Elevated risk requires additional verification"
		decision.Challenge = "mfa_required"
	} else if riskScore.OverallScore > 0.4 {
		decision.Action = "monitor"
		decision.Reason = "Moderate risk, continue monitoring"
	} else {
		decision.Action = "allow"
		decision.Reason = "Low risk, normal behavior"
	}

	return decision
}

func (c *ContAuthCollector) storeRiskScore(riskScore RiskScore) error {
	if c.db == nil {
		return nil
	}
	query := `
	INSERT INTO risk_scores 
	(session_id, overall_score, keystroke_score, mouse_score, location_score,
	 device_score, behavior_score, reputation_score, risk_factors, recommendation)
	VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)`

	riskFactorsJSON, _ := json.Marshal(riskScore.RiskFactors)

	_, err := c.db.Exec(query,
		riskScore.SessionID, riskScore.OverallScore,
		riskScore.KeystrokeScore, riskScore.MouseScore, riskScore.LocationScore,
		riskScore.DeviceScore, riskScore.BehaviorScore, riskScore.ReputationScore,
		string(riskFactorsJSON), riskScore.Recommendation)

	return err
}

func (c *ContAuthCollector) storeAuthDecision(decision AuthDecision) error {
	if c.db == nil {
		return nil
	}
	query := `
	INSERT INTO auth_decisions (session_id, action, confidence, reason, challenge, expires_at)
	VALUES ($1, $2, $3, $4, $5, $6)`

	_, err := c.db.Exec(query,
		decision.SessionID, decision.Action, decision.Confidence,
		decision.Reason, decision.Challenge, decision.ExpiresAt)

	return err
}

func (c *ContAuthCollector) getLatestRiskScore(sessionID string) (*RiskScore, error) {
	if c.db == nil {
		return nil, fmt.Errorf("database disabled")
	}
	query := `
	SELECT session_id, overall_score, keystroke_score, mouse_score, location_score,
		   device_score, behavior_score, reputation_score, risk_factors, 
		   recommendation, calculated_at
	FROM risk_scores 
	WHERE session_id = $1 
	ORDER BY calculated_at DESC 
	LIMIT 1`

	var riskScore RiskScore
	var riskFactorsJSON string

	err := c.db.QueryRow(query, sessionID).Scan(
		&riskScore.SessionID, &riskScore.OverallScore,
		&riskScore.KeystrokeScore, &riskScore.MouseScore, &riskScore.LocationScore,
		&riskScore.DeviceScore, &riskScore.BehaviorScore, &riskScore.ReputationScore,
		&riskFactorsJSON, &riskScore.Recommendation, &riskScore.CalculatedAt)

	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}

	json.Unmarshal([]byte(riskFactorsJSON), &riskScore.RiskFactors)
	return &riskScore, nil
}

type UserBaseline struct {
	UserID               string
	AvgKeystrokeInterval float64
	AvgKeystrokeDuration float64
	TypicalMouseVelocity float64
}

func (c *ContAuthCollector) getUserBaseline(userID string) *UserBaseline {
	return &UserBaseline{
		UserID:               userID,
		AvgKeystrokeInterval: 150.0,
		AvgKeystrokeDuration: 80.0,
		TypicalMouseVelocity: 500.0,
	}
}

func average(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

func variance(values []float64, mean float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range values {
		sum += (v - mean) * (v - mean)
	}
	return sum / float64(len(values))
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

func min(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

func (c *ContAuthCollector) Close() error {
	return c.db.Close()
}

// updateBaselineFromMetadata computes aggregated per-user baselines from metadata fields
// ks_avg_interval, ks_avg_duration, mouse_avg_velocity across recent N sessions and UPSERTs into user_baselines.
func (c *ContAuthCollector) updateBaselineFromMetadata(userID string) error {
	if c.db == nil || userID == "" {
		return nil
	}
	// Aggregate last 100 rows to limit cost
	q := `
		WITH recent AS (
			SELECT metadata
			FROM session_telemetry
			WHERE user_id = $1
			ORDER BY timestamp DESC
			LIMIT 100
		)
		SELECT 
			AVG( (metadata->>'ks_avg_interval')::double precision ) AS avg_int,
			AVG( (metadata->>'ks_avg_duration')::double precision ) AS avg_dur,
			AVG( (metadata->>'mouse_avg_velocity')::double precision ) AS avg_mouse
		FROM recent;`
	var avgInt, avgDur, avgMouse sql.NullFloat64
	if err := c.db.QueryRow(q, userID).Scan(&avgInt, &avgDur, &avgMouse); err != nil {
		return err
	}
	// Upsert into user_baselines
	uq := `
		INSERT INTO user_baselines (user_id, avg_keystroke_interval, avg_keystroke_duration, typical_mouse_velocity, last_updated)
		VALUES ($1, $2, $3, $4, NOW())
		ON CONFLICT (user_id) DO UPDATE SET
			avg_keystroke_interval = EXCLUDED.avg_keystroke_interval,
			avg_keystroke_duration = EXCLUDED.avg_keystroke_duration,
			typical_mouse_velocity = EXCLUDED.typical_mouse_velocity,
			last_updated = NOW();`
	_, err := c.db.Exec(uq,
		userID,
		nullOrZero(avgInt),
		nullOrZero(avgDur),
		nullOrZero(avgMouse),
	)
	return err
}

func nullOrZero(n sql.NullFloat64) float64 {
	if n.Valid {
		return n.Float64
	}
	return 0
}

// summarizeKeystrokes returns average inter-key interval and average key duration.
func summarizeKeystrokes(keystrokes []KeystrokeEvent) (avgInterval float64, avgDuration float64) {
	if len(keystrokes) < 2 {
		if len(keystrokes) == 1 {
			return 0, keystrokes[0].Duration
		}
		return 0, 0
	}
	intervals := make([]float64, 0, len(keystrokes)-1)
	durations := make([]float64, 0, len(keystrokes))
	for i := 1; i < len(keystrokes); i++ {
		intervals = append(intervals, float64(keystrokes[i].Timestamp-keystrokes[i-1].Timestamp))
	}
	for _, k := range keystrokes {
		durations = append(durations, k.Duration)
	}
	return average(intervals), average(durations)
}

// summarizeMouse returns average velocity.
func summarizeMouse(events []MouseEvent) float64 {
	if len(events) == 0 {
		return 0
	}
	v := make([]float64, 0, len(events))
	for _, e := range events {
		if e.Velocity > 0 {
			v = append(v, e.Velocity)
		}
	}
	return average(v)
}

// unquoteIfQuoted removes surrounding quotes from a JSON string value if present.
func unquoteIfQuoted(s string) string {
	if len(s) >= 2 && ((s[0] == '"' && s[len(s)-1] == '"') || (s[0] == '\'' && s[len(s)-1] == '\'')) {
		var out string
		if err := json.Unmarshal([]byte(s), &out); err == nil {
			return out
		}
	}
	return s
}

// -------------- High-performance helpers and ML integration --------------

// Minimal TTL cache for RiskScore keyed by sessionID
type riskCache struct {
	mu    sync.RWMutex
	data  map[string]cachedRisk
	ttl   time.Duration
	maxSz int
}
type cachedRisk struct {
	v   RiskScore
	exp time.Time
}

func newRiskCache(ttl time.Duration, max int) *riskCache {
	if ttl <= 0 {
		ttl = 2 * time.Minute
	}
	if max < 1024 {
		max = 1024
	}
	return &riskCache{data: make(map[string]cachedRisk, max), ttl: ttl, maxSz: max}
}
func (rc *riskCache) Get(k string) (RiskScore, bool) {
	rc.mu.RLock()
	cr, ok := rc.data[k]
	rc.mu.RUnlock()
	if !ok || time.Now().After(cr.exp) {
		if ok {
			rc.mu.Lock()
			delete(rc.data, k)
			rc.mu.Unlock()
		}
		return RiskScore{}, false
	}
	return cr.v, true
}
func (rc *riskCache) Set(k string, v RiskScore) {
	rc.mu.Lock()
	defer rc.mu.Unlock()
	if len(rc.data) >= rc.maxSz {
		// simple random eviction to keep O(1)
		for kk := range rc.data {
			delete(rc.data, kk)
			break
		}
	}
	rc.data[k] = cachedRisk{v: v, exp: time.Now().Add(rc.ttl)}
}

// ML orchestrator client (best-effort, no internals exposed)
type mlAnalyzeResult struct {
	IsAnomaly  bool    `json:"is_anomaly"`
	Score      float64 `json:"score"`
	Confidence float64 `json:"confidence"`
}

func (c *ContAuthCollector) queryMLO(t SessionTelemetry, baseURL string) *mlAnalyzeResult {
	// Build features from privacy-safe summaries only
	ksInt, ksDur := summarizeKeystrokes(t.KeystrokeData)
	mVel := summarizeMouse(t.MouseData)
	failRate := 0.0
	if n := len(t.AccessPatterns); n > 0 {
		fails := 0
		for _, a := range t.AccessPatterns {
			if !a.Success {
				fails++
			}
		}
		failRate = float64(fails) / float64(n)
	}
	deviceCompleteness := 0.0
	total := 5.0
	miss := 0.0
	if t.DeviceMetrics.ScreenResolution == "" {
		miss++
	}
	if t.DeviceMetrics.Timezone == "" {
		miss++
	}
	if t.DeviceMetrics.Language == "" {
		miss++
	}
	if t.DeviceMetrics.Platform == "" {
		miss++
	}
	if t.DeviceMetrics.CPUCores <= 0 {
		miss++
	}
	deviceCompleteness = 1.0 - miss/total
	repRisk := c.calculateReputationRisk(t.IPAddress, t.UserAgent)

	feat := []float64{ksInt, ksDur, mVel, failRate, deviceCompleteness, repRisk}
	payload := map[string]any{
		"timestamp":    time.Now(),
		"source":       "contauth",
		"event_type":   "contauth_session",
		"tenant_id":    t.UserID,
		"features":     feat,
		"threat_score": 0,
	}
	b, _ := json.Marshal(payload)
	// timeout and small client to avoid blocking
	to := time.Duration(parseInt(getenv("MLO_TIMEOUT_MS"), 500)) * time.Millisecond
	req, _ := http.NewRequest(http.MethodPost, strings.TrimRight(baseURL, "/")+"/analyze", strings.NewReader(string(b)))
	req.Header.Set("Content-Type", "application/json")
	cli := &http.Client{Timeout: to}
	resp, err := cli.Do(req)
	if err != nil {
		return nil
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return nil
	}
	var out mlAnalyzeResult
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil
	}
	// sanitize
	if out.Score < 0 || out.Score > 1 || math.IsNaN(out.Score) || math.IsInf(out.Score, 0) {
		return nil
	}
	return &out
}

// env helpers (localized)
func getenv(key string) string { return strings.TrimSpace(os.Getenv(key)) }
func parseFloat(s string, def float64) float64 {
	if s == "" {
		return def
	}
	if v, err := strconv.ParseFloat(s, 64); err == nil {
		return v
	}
	return def
}
func parseInt(s string, def int) int {
	if s == "" {
		return def
	}
	if v, err := strconv.Atoi(s); err == nil {
		return v
	}
	return def
}
func parseTTL(s string, def time.Duration) time.Duration {
	if s == "" {
		return def
	}
	if d, err := time.ParseDuration(s); err == nil {
		return d
	}
	return def
}
func maxFloat(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}
func appendUnique(arr []string, v string) []string {
	for _, x := range arr {
		if x == v {
			return arr
		}
	}
	return append(arr, v)
}

// sanitize identifiers before logging (truncate)
func sanitizeID(id string) string {
	if len(id) > 12 {
		return id[:12] + "*"
	}
	return id
}
