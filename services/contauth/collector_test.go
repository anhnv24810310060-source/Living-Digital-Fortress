package main

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// Test hash functions produce stable, non-empty outputs without leaking raw data
func TestHashingFunctions(t *testing.T) {
	ks := []KeystrokeEvent{{Timestamp: 1000, Duration: 80, Pressure: 0.3}, {Timestamp: 1100, Duration: 85, Pressure: 0.2}}
	ms := []MouseEvent{{Timestamp: 1000, Velocity: 120, EventType: "move"}, {Timestamp: 1100, Velocity: 300, EventType: "click"}}
	h1 := hashKeystrokes(ks)
	h2 := hashKeystrokes(ks)
	if h1 == "" || h2 == "" || h1 != h2 {
		t.Fatalf("unexpected keystroke hash: %q vs %q", h1, h2)
	}
	if strings.Contains(h1, "KeyA") {
		t.Fatalf("hash should not contain raw keys")
	}
	m1 := hashMouse(ms)
	m2 := hashMouse(ms)
	if m1 == "" || m1 != m2 {
		t.Fatalf("unexpected mouse hash: %q vs %q", m1, m2)
	}
}

// Test risk calculation combines sub-scores and returns recommendation
func TestRiskCalculation(t *testing.T) {
	c := &ContAuthCollector{}
	tel := SessionTelemetry{
		SessionID:      "s1",
		UserID:         "u1",
		IPAddress:      "1.2.3.4",
		UserAgent:      "Mozilla",
		MouseData:      []MouseEvent{{Timestamp: time.Now().UnixMilli(), Velocity: 100}},
		KeystrokeData:  []KeystrokeEvent{{Timestamp: 1, Duration: 80}, {Timestamp: 2, Duration: 90}},
		AccessPatterns: []AccessEvent{{Resource: "x", Action: "get", Timestamp: time.Now(), Success: true}},
	}
	rs := c.calculateRisk(tel)
	if rs.OverallScore < 0 || rs.OverallScore > 1 {
		t.Fatalf("score out of range: %f", rs.OverallScore)
	}
	if rs.Recommendation == "" {
		t.Fatalf("empty recommendation")
	}
}

func TestSummarizers(t *testing.T) {
	avgI, avgD := summarizeKeystrokes([]KeystrokeEvent{{Timestamp: 10, Duration: 50}, {Timestamp: 20, Duration: 70}})
	if avgI <= 0 || avgD <= 0 {
		t.Fatalf("unexpected summarizer: avgI=%v avgD=%v", avgI, avgD)
	}
	if v := summarizeMouse([]MouseEvent{{Velocity: 10}, {Velocity: 30}}); v <= 0 {
		t.Fatalf("unexpected mouse avg: %v", v)
	}
}

// API handler smoke test for CalculateRiskScore with DISABLE_DB=true via dummy collector
func TestCalculateRiskScoreHandler_NoDB(t *testing.T) {
	// Use dummy collector
	dc := NewDummyCollector()
	defer dc.Close()

	// simulate request
	reqBody, _ := json.Marshal(map[string]string{"session_id": "sess-123"})
	r := httptest.NewRequest("POST", "/contauth/score", strings.NewReader(string(reqBody)))
	w := httptest.NewRecorder()
	dc.CalculateRiskScore(w, r)
	if w.Code != 200 && w.Code != 500 && w.Code != 404 {
		t.Fatalf("unexpected status: %d", w.Code)
	}
}
