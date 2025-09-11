package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

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
	SessionID        string    `json:"session_id"`
	OverallScore     float64   `json:"overall_score"`
	KeystrokeScore   float64   `json:"keystroke_score"`
	MouseScore       float64   `json:"mouse_score"`
	LocationScore    float64   `json:"location_score"`
	DeviceScore      float64   `json:"device_score"`
	BehaviorScore    float64   `json:"behavior_score"`
	ReputationScore  float64   `json:"reputation_score"`
	RiskFactors      []string  `json:"risk_factors"`
	Recommendation   string    `json:"recommendation"`
	CalculatedAt     time.Time `json:"calculated_at"`
}

type AuthDecision struct {
	SessionID   string    `json:"session_id"`
	Action      string    `json:"action"`
	Confidence  float64   `json:"confidence"`
	Reason      string    `json:"reason"`
	Challenge   string    `json:"challenge,omitempty"`
	ExpiresAt   time.Time `json:"expires_at"`
}

type ContAuthCollector struct {
	db *sql.DB
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

	collector := &ContAuthCollector{db: db}
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

	if err := c.storeTelemetry(telemetry); err != nil {
		log.Printf("Failed to store telemetry: %v", err)
		http.Error(w, "Failed to store telemetry", http.StatusInternalServerError)
		return
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

	if err := c.storeRiskScore(riskScore); err != nil {
		log.Printf("Failed to store risk score: %v", err)
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

	riskScore, err := c.getLatestRiskScore(sessionID)
	if err != nil {
		log.Printf("Failed to get risk score: %v", err)
		http.Error(w, "Failed to get risk assessment", http.StatusInternalServerError)
		return
	}

	if riskScore == nil {
		http.Error(w, "No risk assessment found for session", http.StatusNotFound)
		return
	}

	decision := c.makeAuthDecision(*riskScore)

	if err := c.storeAuthDecision(decision); err != nil {
		log.Printf("Failed to store auth decision: %v", err)
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(decision); err != nil {
		log.Printf("Failed to encode response: %v", err)
	}
}

func (c *ContAuthCollector) storeTelemetry(telemetry SessionTelemetry) error {
	query := `
	INSERT INTO session_telemetry 
	(session_id, user_id, device_id, ip_address, user_agent, keystroke_data, 
	 mouse_data, access_patterns, geolocation_data, device_metrics, metadata, timestamp)
	VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)`

	keystrokeJSON, _ := json.Marshal(telemetry.KeystrokeData)
	mouseJSON, _ := json.Marshal(telemetry.MouseData)
	accessJSON, _ := json.Marshal(telemetry.AccessPatterns)
	geoJSON, _ := json.Marshal(telemetry.GeolocationData)
	deviceJSON, _ := json.Marshal(telemetry.DeviceMetrics)
	metadataJSON, _ := json.Marshal(telemetry.Metadata)

	_, err := c.db.Exec(query,
		telemetry.SessionID, telemetry.UserID, telemetry.DeviceID,
		telemetry.IPAddress, telemetry.UserAgent,
		string(keystrokeJSON), string(mouseJSON), string(accessJSON),
		string(geoJSON), string(deviceJSON), string(metadataJSON),
		telemetry.Timestamp)

	return err
}

func (c *ContAuthCollector) getRecentTelemetry(sessionID string) (*SessionTelemetry, error) {
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

	json.Unmarshal([]byte(keystrokeJSON), &telemetry.KeystrokeData)
	json.Unmarshal([]byte(mouseJSON), &telemetry.MouseData)
	json.Unmarshal([]byte(accessJSON), &telemetry.AccessPatterns)
	json.Unmarshal([]byte(geoJSON), &telemetry.GeolocationData)
	json.Unmarshal([]byte(deviceJSON), &telemetry.DeviceMetrics)
	json.Unmarshal([]byte(metadataJSON), &telemetry.Metadata)

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
			intervalDeviation = abs(avgInterval - baseline.AvgKeystrokeInterval) / baseline.AvgKeystrokeInterval
		}
		if baseline.AvgKeystrokeDuration > 0 {
			durationDeviation = abs(avgDuration - baseline.AvgKeystrokeDuration) / baseline.AvgKeystrokeDuration
		}
	}

	riskScore := (intervalVariance/1000 + durationVariance/100 + intervalDeviation + durationDeviation) / 4

	return min(riskScore, 1.0)
}

func (c *ContAuthCollector) calculateMouseRisk(mouseEvents []MouseEvent, userID string) float64 {
	if len(mouseEvents) < 10 {
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
		velocityDeviation = abs(avgVelocity - baseline.TypicalMouseVelocity) / baseline.TypicalMouseVelocity
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
	query := `
	INSERT INTO auth_decisions (session_id, action, confidence, reason, challenge, expires_at)
	VALUES ($1, $2, $3, $4, $5, $6)`

	_, err := c.db.Exec(query,
		decision.SessionID, decision.Action, decision.Confidence,
		decision.Reason, decision.Challenge, decision.ExpiresAt)

	return err
}

func (c *ContAuthCollector) getLatestRiskScore(sessionID string) (*RiskScore, error) {
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
	UserID                  string
	AvgKeystrokeInterval    float64
	AvgKeystrokeDuration    float64
	TypicalMouseVelocity    float64
}

func (c *ContAuthCollector) getUserBaseline(userID string) *UserBaseline {
	return &UserBaseline{
		UserID:                  userID,
		AvgKeystrokeInterval:    150.0,
		AvgKeystrokeDuration:    80.0,
		TypicalMouseVelocity:    500.0,
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