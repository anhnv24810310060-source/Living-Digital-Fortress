package main

import (
    "encoding/json"
    "net/http"
    "time"
)

// DummyCollector is a no-op collector used when the database is disabled.
type DummyCollector struct{}

func NewDummyCollector() *DummyCollector { return &DummyCollector{} }

func (d *DummyCollector) Close() error { return nil }

func (d *DummyCollector) CollectTelemetry(w http.ResponseWriter, r *http.Request) {
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(map[string]any{"success": true, "message": "telemetry accepted (noop)"})
}

func (d *DummyCollector) CalculateRiskScore(w http.ResponseWriter, r *http.Request) {
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(map[string]any{
        "overall_score": 0.3,
        "recommendation": "allow",
        "calculated_at": time.Now(),
    })
}

func (d *DummyCollector) GetAuthDecision(w http.ResponseWriter, r *http.Request) {
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(map[string]any{
        "action": "allow",
        "confidence": 0.8,
        "expires_at": time.Now().Add(5 * time.Minute),
    })
}
