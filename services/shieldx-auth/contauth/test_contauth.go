package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestCollectTelemetry(t *testing.T) {
	collector := &ContAuthCollector{}

	telemetry := SessionTelemetry{
		SessionID: "test_session_123",
		UserID:    "test_user_456",
		DeviceID:  "test_device_789",
		IPAddress: "192.168.1.100",
		UserAgent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
		KeystrokeData: []KeystrokeEvent{
			{Key: "a", Timestamp: 1000, Duration: 80.0, Pressure: 0.5},
			{Key: "b", Timestamp: 1150, Duration: 75.0, Pressure: 0.6},
			{Key: "c", Timestamp: 1300, Duration: 85.0, Pressure: 0.4},
		},
		MouseData: []MouseEvent{
			{X: 100, Y: 200, Timestamp: 1000, EventType: "move", Velocity: 500.0},
			{X: 150, Y: 250, Timestamp: 1100, EventType: "move", Velocity: 450.0},
		},
		AccessPatterns: []AccessEvent{
			{Resource: "/dashboard", Action: "GET", Timestamp: time.Now(), Success: true},
			{Resource: "/profile", Action: "GET", Timestamp: time.Now(), Success: true},
		},
		GeolocationData: GeolocationInfo{
			Latitude:  37.7749,
			Longitude: -122.4194,
			Country:   "US",
			City:      "San Francisco",
			ISP:       "Comcast",
		},
		DeviceMetrics: DeviceFingerprint{
			ScreenResolution: "1920x1080",
			Timezone:         "America/Los_Angeles",
			Language:         "en-US",
			Platform:         "Win32",
			CPUCores:         8,
			Memory:           16384,
			BatteryLevel:     0.85,
			NetworkType:      "wifi",
		},
	}

	jsonData, err := json.Marshal(telemetry)
	if err != nil {
		t.Fatalf("Failed to marshal telemetry: %v", err)
	}

	req := httptest.NewRequest("POST", "/contauth/telemetry", bytes.NewBuffer(jsonData))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	collector.CollectTelemetry(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	var response map[string]interface{}
	if err := json.NewDecoder(w.Body).Decode(&response); err != nil {
		t.Fatalf("Failed to decode response: %v", err)
	}

	if !response["success"].(bool) {
		t.Error("Expected success to be true")
	}

	if response["session_id"] != telemetry.SessionID {
		t.Errorf("Expected session_id %s, got %s", telemetry.SessionID, response["session_id"])
	}
}

func TestCalculateKeystrokeRisk(t *testing.T) {
	collector := &ContAuthCollector{}

	keystrokes := []KeystrokeEvent{
		{Key: "h", Timestamp: 1000, Duration: 80.0},
		{Key: "e", Timestamp: 1150, Duration: 75.0},
		{Key: "l", Timestamp: 1300, Duration: 85.0},
		{Key: "l", Timestamp: 1450, Duration: 78.0},
		{Key: "o", Timestamp: 1600, Duration: 82.0},
	}

	riskScore := collector.calculateKeystrokeRisk(keystrokes, "test_user")

	if riskScore < 0.0 || riskScore > 1.0 {
		t.Errorf("Risk score should be between 0 and 1, got %f", riskScore)
	}
}

func TestCalculateMouseRisk(t *testing.T) {
	collector := &ContAuthCollector{}

	mouseEvents := []MouseEvent{
		{X: 100, Y: 100, Timestamp: 1000, Velocity: 500.0},
		{X: 150, Y: 120, Timestamp: 1100, Velocity: 450.0},
		{X: 200, Y: 140, Timestamp: 1200, Velocity: 480.0},
		{X: 250, Y: 160, Timestamp: 1300, Velocity: 520.0},
		{X: 300, Y: 180, Timestamp: 1400, Velocity: 490.0},
		{X: 350, Y: 200, Timestamp: 1500, Velocity: 510.0},
		{X: 400, Y: 220, Timestamp: 1600, Velocity: 470.0},
		{X: 450, Y: 240, Timestamp: 1700, Velocity: 530.0},
		{X: 500, Y: 260, Timestamp: 1800, Velocity: 495.0},
		{X: 550, Y: 280, Timestamp: 1900, Velocity: 505.0},
	}

	riskScore := collector.calculateMouseRisk(mouseEvents, "test_user")

	if riskScore < 0.0 || riskScore > 1.0 {
		t.Errorf("Risk score should be between 0 and 1, got %f", riskScore)
	}
}

func TestCalculateLocationRisk(t *testing.T) {
	collector := &ContAuthCollector{}

	// Test known location (US)
	geoUS := GeolocationInfo{
		Country: "US",
		City:    "San Francisco",
	}

	riskScore := collector.calculateLocationRisk(geoUS, "test_user")
	if riskScore != 0.1 {
		t.Errorf("Expected low risk (0.1) for US location, got %f", riskScore)
	}

	// Test unknown location
	geoUnknown := GeolocationInfo{
		Country: "XX",
		City:    "Unknown",
	}

	riskScore = collector.calculateLocationRisk(geoUnknown, "test_user")
	if riskScore != 0.7 {
		t.Errorf("Expected high risk (0.7) for unknown location, got %f", riskScore)
	}

	// Test empty location
	geoEmpty := GeolocationInfo{}

	riskScore = collector.calculateLocationRisk(geoEmpty, "test_user")
	if riskScore != 0.8 {
		t.Errorf("Expected very high risk (0.8) for empty location, got %f", riskScore)
	}
}

func TestCalculateDeviceRisk(t *testing.T) {
	collector := &ContAuthCollector{}

	// Test complete device fingerprint
	deviceComplete := DeviceFingerprint{
		ScreenResolution: "1920x1080",
		Timezone:         "America/Los_Angeles",
		Language:         "en-US",
		Platform:         "Win32",
		CPUCores:         8,
	}

	riskScore := collector.calculateDeviceRisk(deviceComplete, "test_user")
	if riskScore != 0.0 {
		t.Errorf("Expected no risk (0.0) for complete device fingerprint, got %f", riskScore)
	}

	// Test incomplete device fingerprint
	deviceIncomplete := DeviceFingerprint{
		ScreenResolution: "",
		Timezone:         "",
		Language:         "en-US",
		Platform:         "Win32",
		CPUCores:         0,
	}

	riskScore = collector.calculateDeviceRisk(deviceIncomplete, "test_user")
	if riskScore != 0.6 {
		t.Errorf("Expected high risk (0.6) for incomplete device fingerprint, got %f", riskScore)
	}
}

func TestCalculateBehaviorRisk(t *testing.T) {
	collector := &ContAuthCollector{}

	// Test normal access patterns
	normalAccess := []AccessEvent{
		{Resource: "/dashboard", Success: true},
		{Resource: "/profile", Success: true},
		{Resource: "/settings", Success: true},
	}

	riskScore := collector.calculateBehaviorRisk(normalAccess, "test_user")
	if riskScore != 0.0 {
		t.Errorf("Expected no risk (0.0) for normal access patterns, got %f", riskScore)
	}

	// Test high failure rate
	failureAccess := []AccessEvent{
		{Resource: "/admin", Success: false},
		{Resource: "/admin", Success: false},
		{Resource: "/admin", Success: false},
		{Resource: "/dashboard", Success: true},
	}

	riskScore = collector.calculateBehaviorRisk(failureAccess, "test_user")
	if riskScore != 0.5 {
		t.Errorf("Expected high risk (0.5) for high failure rate, got %f", riskScore)
	}
}

func TestCalculateReputationRisk(t *testing.T) {
	collector := &ContAuthCollector{}

	// Test normal user agent
	normalUA := "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
	riskScore := collector.calculateReputationRisk("192.168.1.100", normalUA)
	if riskScore != 0.0 {
		t.Errorf("Expected no risk (0.0) for normal user agent, got %f", riskScore)
	}

	// Test suspicious user agent
	suspiciousUA := "bot/1.0 crawler scanner"
	riskScore = collector.calculateReputationRisk("192.168.1.100", suspiciousUA)
	if riskScore != 0.9 {
		t.Errorf("Expected high risk (0.9) for suspicious user agent, got %f", riskScore)
	}

	// Test empty IP
	riskScore = collector.calculateReputationRisk("", normalUA)
	if riskScore != 0.5 {
		t.Errorf("Expected moderate risk (0.5) for empty IP, got %f", riskScore)
	}
}

func TestMakeAuthDecision(t *testing.T) {
	collector := &ContAuthCollector{}

	// Test low risk - should allow
	lowRisk := RiskScore{
		SessionID:    "test_session",
		OverallScore: 0.2,
	}

	decision := collector.makeAuthDecision(lowRisk)
	if decision.Action != "allow" {
		t.Errorf("Expected 'allow' action for low risk, got %s", decision.Action)
	}

	// Test moderate risk - should monitor
	moderateRisk := RiskScore{
		SessionID:    "test_session",
		OverallScore: 0.5,
	}

	decision = collector.makeAuthDecision(moderateRisk)
	if decision.Action != "monitor" {
		t.Errorf("Expected 'monitor' action for moderate risk, got %s", decision.Action)
	}

	// Test elevated risk - should challenge
	elevatedRisk := RiskScore{
		SessionID:    "test_session",
		OverallScore: 0.7,
	}

	decision = collector.makeAuthDecision(elevatedRisk)
	if decision.Action != "challenge" {
		t.Errorf("Expected 'challenge' action for elevated risk, got %s", decision.Action)
	}
	if decision.Challenge != "mfa_required" {
		t.Errorf("Expected 'mfa_required' challenge, got %s", decision.Challenge)
	}

	// Test high risk - should block
	highRisk := RiskScore{
		SessionID:    "test_session",
		OverallScore: 0.9,
	}

	decision = collector.makeAuthDecision(highRisk)
	if decision.Action != "block" {
		t.Errorf("Expected 'block' action for high risk, got %s", decision.Action)
	}
}

func TestRiskScoreCalculation(t *testing.T) {
	collector := &ContAuthCollector{}

	telemetry := SessionTelemetry{
		SessionID: "test_session",
		UserID:    "test_user",
		KeystrokeData: []KeystrokeEvent{
			{Key: "a", Timestamp: 1000, Duration: 80.0},
			{Key: "b", Timestamp: 1150, Duration: 75.0},
			{Key: "c", Timestamp: 1300, Duration: 85.0},
			{Key: "d", Timestamp: 1450, Duration: 78.0},
			{Key: "e", Timestamp: 1600, Duration: 82.0},
		},
		MouseData: []MouseEvent{
			{X: 100, Y: 100, Timestamp: 1000, Velocity: 500.0},
			{X: 150, Y: 120, Timestamp: 1100, Velocity: 450.0},
			{X: 200, Y: 140, Timestamp: 1200, Velocity: 480.0},
			{X: 250, Y: 160, Timestamp: 1300, Velocity: 520.0},
			{X: 300, Y: 180, Timestamp: 1400, Velocity: 490.0},
			{X: 350, Y: 200, Timestamp: 1500, Velocity: 510.0},
			{X: 400, Y: 220, Timestamp: 1600, Velocity: 470.0},
			{X: 450, Y: 240, Timestamp: 1700, Velocity: 530.0},
			{X: 500, Y: 260, Timestamp: 1800, Velocity: 495.0},
			{X: 550, Y: 280, Timestamp: 1900, Velocity: 505.0},
		},
		GeolocationData: GeolocationInfo{Country: "US"},
		DeviceMetrics: DeviceFingerprint{
			ScreenResolution: "1920x1080",
			Timezone:         "America/Los_Angeles",
			Language:         "en-US",
			Platform:         "Win32",
			CPUCores:         8,
		},
		AccessPatterns: []AccessEvent{
			{Resource: "/dashboard", Success: true},
			{Resource: "/profile", Success: true},
		},
		IPAddress: "192.168.1.100",
		UserAgent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
	}

	riskScore := collector.calculateRisk(telemetry)

	// Verify overall score is calculated
	if riskScore.OverallScore < 0.0 || riskScore.OverallScore > 1.0 {
		t.Errorf("Overall risk score should be between 0 and 1, got %f", riskScore.OverallScore)
	}

	// Verify individual scores are calculated
	if riskScore.KeystrokeScore < 0.0 || riskScore.KeystrokeScore > 1.0 {
		t.Errorf("Keystroke risk score should be between 0 and 1, got %f", riskScore.KeystrokeScore)
	}

	if riskScore.MouseScore < 0.0 || riskScore.MouseScore > 1.0 {
		t.Errorf("Mouse risk score should be between 0 and 1, got %f", riskScore.MouseScore)
	}

	// Verify recommendation is set
	if riskScore.Recommendation == "" {
		t.Error("Recommendation should not be empty")
	}

	validRecommendations := []string{"allow", "challenge", "block"}
	validRecommendation := false
	for _, valid := range validRecommendations {
		if riskScore.Recommendation == valid {
			validRecommendation = true
			break
		}
	}

	if !validRecommendation {
		t.Errorf("Invalid recommendation: %s", riskScore.Recommendation)
	}
}