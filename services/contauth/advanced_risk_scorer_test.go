package main

import (
	"testing"
	"time"
)

func TestAdvancedRiskScorer_CleanSession(t *testing.T) {
	scorer := NewAdvancedRiskScorer()

	telemetry := &SessionTelemetry{
		SessionID: "test-session-1",
		UserID:    "user123",
		Timestamp: time.Now(),
		Metadata: map[string]interface{}{
			"ks_avg_interval":    150.0,
			"ks_avg_duration":    80.0,
			"mouse_avg_velocity": 120.0,
		},
		GeolocationData: GeolocationInfo{
			Country: "US",
		},
		DeviceMetrics: DeviceFingerprint{
			ScreenResolution: "1920x1080",
			Platform:         "Windows",
			Timezone:         "America/New_York",
		},
		AccessPatterns: []AccessEvent{
			{Resource: "/api/data", Action: "read", Success: true},
		},
	}

	baseline := &UserBaseline{
		UserID:               "user123",
		AvgKeystrokeInterval: 155.0,
		AvgKeystrokeDuration: 78.0,
		TypicalMouseVelocity: 118.0,
	}

	score := scorer.CalculateRisk(telemetry, baseline)

	if score.OverallScore > 30 {
		t.Errorf("Expected low risk score for clean session, got %.2f", score.OverallScore)
	}

	if score.Recommendation == "DENY" {
		t.Errorf("Expected ALLOW/MONITOR, got %s", score.Recommendation)
	}
}

func TestAdvancedRiskScorer_LocationAnomaly(t *testing.T) {
	scorer := NewAdvancedRiskScorer()

	telemetry := &SessionTelemetry{
		SessionID: "test-session-2",
		UserID:    "user123",
		Timestamp: time.Now(),
		Metadata: map[string]interface{}{
			"ks_avg_interval": 150.0,
			"ks_avg_duration": 80.0,
		},
		GeolocationData: GeolocationInfo{
			Country: "RU", // Changed from US
		},
		DeviceMetrics: DeviceFingerprint{
			Platform: "Windows",
		},
	}

	baseline := &UserBaseline{
		UserID:               "user123",
		AvgKeystrokeInterval: 155.0,
		AvgKeystrokeDuration: 78.0,
		// TypicalCountry (extended):        "US",
		// TypicalPlatform (extended):       "Windows",
	}

	score := scorer.CalculateRisk(telemetry, baseline)

	if score.LocationScore < 60 {
		t.Errorf("Expected high location risk for country change, got %.2f", score.LocationScore)
	}

	if len(score.RiskFactors) == 0 {
		t.Error("Expected risk factors to be populated")
	}
}

func TestAdvancedRiskScorer_DeviceMismatch(t *testing.T) {
	scorer := NewAdvancedRiskScorer()

	telemetry := &SessionTelemetry{
		SessionID: "test-session-3",
		UserID:    "user123",
		Timestamp: time.Now(),
		Metadata: map[string]interface{}{
			"ks_avg_interval": 150.0,
			"ks_avg_duration": 80.0,
		},
		DeviceMetrics: DeviceFingerprint{
			ScreenResolution: "2560x1440",
			Platform:         "Linux", // Changed from Windows
		},
	}

	baseline := &UserBaseline{
		UserID:               "user123",
		AvgKeystrokeInterval: 155.0,
		AvgKeystrokeDuration: 78.0,
		// TypicalScreenRes (extended):      "1920x1080",
		// TypicalPlatform (extended):       "Windows",
	}

	score := scorer.CalculateRisk(telemetry, baseline)

	if score.DeviceScore < 40 {
		t.Errorf("Expected elevated device risk for platform change, got %.2f", score.DeviceScore)
	}
}

func TestAdvancedRiskScorer_HighFailureRate(t *testing.T) {
	scorer := NewAdvancedRiskScorer()

	telemetry := &SessionTelemetry{
		SessionID: "test-session-4",
		UserID:    "user123",
		Timestamp: time.Now(),
		Metadata: map[string]interface{}{
			"ks_avg_interval": 150.0,
		},
		AccessPatterns: []AccessEvent{
			{Resource: "/api/data", Action: "read", Success: false},
			{Resource: "/api/data", Action: "read", Success: false},
			{Resource: "/api/data", Action: "read", Success: false},
			{Resource: "/api/data", Action: "read", Success: true},
		},
	}

	baseline := &UserBaseline{
		UserID:               "user123",
		AvgKeystrokeInterval: 155.0,
	}

	score := scorer.CalculateRisk(telemetry, baseline)

	if score.BehaviorScore < 50 {
		t.Errorf("Expected high behavior risk for failure rate, got %.2f", score.BehaviorScore)
	}
}

func TestAdvancedRiskScorer_NightAccess(t *testing.T) {
	scorer := NewAdvancedRiskScorer()

	// 3 AM access
	nightTime := time.Date(2024, 1, 15, 3, 0, 0, 0, time.UTC)

	telemetry := &SessionTelemetry{
		SessionID: "test-session-5",
		UserID:    "user123",
		Timestamp: nightTime,
		Metadata: map[string]interface{}{
			"ks_avg_interval": 150.0,
		},
	}

	baseline := &UserBaseline{
		UserID:               "user123",
		AvgKeystrokeInterval: 155.0,
	}

	score := scorer.CalculateRisk(telemetry, baseline)

	// Temporal anomaly should be detected
	if score.OverallScore < 20 {
		t.Errorf("Expected elevated risk for night access, got %.2f", score.OverallScore)
	}
}

func TestAdvancedRiskScorer_NoBaseline(t *testing.T) {
	scorer := NewAdvancedRiskScorer()

	telemetry := &SessionTelemetry{
		SessionID: "test-session-6",
		UserID:    "new-user",
		Timestamp: time.Now(),
		Metadata: map[string]interface{}{
			"ks_avg_interval": 150.0,
		},
	}

	score := scorer.CalculateRisk(telemetry, nil)

	// Should have moderate risk for new user (no baseline)
	if score.OverallScore > 50 {
		t.Errorf("Expected moderate risk for no baseline, got %.2f", score.OverallScore)
	}

	if score.Recommendation == "DENY" {
		t.Errorf("Should not deny new users immediately, got %s", score.Recommendation)
	}
}

func TestAdvancedRiskScorer_MultipleFactors(t *testing.T) {
	scorer := NewAdvancedRiskScorer()

	// High-risk session: wrong country, wrong device, night time, failures
	nightTime := time.Date(2024, 1, 15, 3, 0, 0, 0, time.UTC)

	telemetry := &SessionTelemetry{
		SessionID: "test-session-7",
		UserID:    "user123",
		Timestamp: nightTime,
		Metadata: map[string]interface{}{
			"ks_avg_interval": 350.0, // Very different from baseline
			"ks_avg_duration": 150.0,
		},
		GeolocationData: GeolocationInfo{
			Country: "CN",
		},
		DeviceMetrics: DeviceFingerprint{
			Platform: "Linux",
		},
		AccessPatterns: []AccessEvent{
			{Resource: "/api/admin", Action: "write", Success: false},
			{Resource: "/api/admin", Action: "write", Success: false},
		},
	}

	baseline := &UserBaseline{
		UserID:               "user123",
		AvgKeystrokeInterval: 155.0,
		AvgKeystrokeDuration: 78.0,
		// TypicalCountry (extended):        "US",
		// TypicalPlatform (extended):       "Windows",
	}

	score := scorer.CalculateRisk(telemetry, baseline)

	if score.OverallScore < 60 {
		t.Errorf("Expected high risk score for multiple anomalies, got %.2f", score.OverallScore)
	}

	if score.Recommendation != "DENY" && score.Recommendation != "MFA_REQUIRED" {
		t.Errorf("Expected strict action for high risk, got %s", score.Recommendation)
	}

	if len(score.RiskFactors) < 3 {
		t.Errorf("Expected multiple risk factors, got %d", len(score.RiskFactors))
	}
}

func TestSecureHash(t *testing.T) {
	data := []byte("sensitive_biometric_data")

	hash1 := SecureHash(data)
	hash2 := SecureHash(data)

	// Hashes should be different due to salt
	if hash1 == hash2 {
		t.Error("Expected different hashes with salt")
	}

	// Both should be valid
	if !VerifyHash(data, hash1) {
		t.Error("Hash1 verification failed")
	}
	if !VerifyHash(data, hash2) {
		t.Error("Hash2 verification failed")
	}

	// Wrong data should not verify
	wrongData := []byte("wrong_data")
	if VerifyHash(wrongData, hash1) {
		t.Error("Should not verify wrong data")
	}
}

func TestFeatureExtractor_Keystroke(t *testing.T) {
	extractor := NewFeatureExtractor()

	events := []KeystrokeEvent{
		{Key: "a", Timestamp: 1000, Duration: 80, Pressure: 0.5},
		{Key: "b", Timestamp: 1150, Duration: 75, Pressure: 0.6},
		{Key: "c", Timestamp: 1300, Duration: 82, Pressure: 0.55},
	}

	features := extractor.ExtractKeystrokeFeatures(events)

	if features["event_count"] != 3 {
		t.Errorf("Expected 3 events, got %.0f", features["event_count"])
	}

	if features["avg_interval"] <= 0 {
		t.Error("Expected positive average interval")
	}

	if features["avg_duration"] <= 0 {
		t.Error("Expected positive average duration")
	}

	if features["std_interval"] < 0 {
		t.Error("Expected non-negative std deviation")
	}
}

func TestFeatureExtractor_Mouse(t *testing.T) {
	extractor := NewFeatureExtractor()

	events := []MouseEvent{
		{X: 100, Y: 100, Timestamp: 1000, EventType: "move", Velocity: 120},
		{X: 150, Y: 120, Timestamp: 1050, EventType: "move", Velocity: 130},
		{X: 200, Y: 140, Timestamp: 1100, EventType: "click", Velocity: 0},
	}

	features := extractor.ExtractMouseFeatures(events)

	if features["total_events"] != 3 {
		t.Errorf("Expected 3 events, got %.0f", features["total_events"])
	}

	if features["click_count"] != 1 {
		t.Errorf("Expected 1 click, got %.0f", features["click_count"])
	}

	if features["move_count"] != 2 {
		t.Errorf("Expected 2 moves, got %.0f", features["move_count"])
	}
}
