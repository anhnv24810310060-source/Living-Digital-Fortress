package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"time"

	"shieldx/pkg/metrics"
	otelobs "shieldx/pkg/observability/otel"
)

type MLOrchestrator struct {
	anomalyDetector  *AnomalyDetector
	featureExtractor *FeatureExtractor
}

type AnomalyDetector struct {
	threshold float64
	trained   bool
}

type FeatureExtractor struct {
	windowSize time.Duration
}

type TelemetryEvent struct {
	Timestamp   time.Time `json:"timestamp"`
	Source      string    `json:"source"`
	EventType   string    `json:"event_type"`
	TenantID    string    `json:"tenant_id"`
	Features    []float64 `json:"features"`
	ThreatScore float64   `json:"threat_score"`
}

type AnomalyResult struct {
	IsAnomaly   bool    `json:"is_anomaly"`
	Score       float64 `json:"score"`
	Confidence  float64 `json:"confidence"`
	Explanation string  `json:"explanation"`
}

func main() {
	orchestrator := &MLOrchestrator{
		anomalyDetector:  &AnomalyDetector{threshold: 0.5},
		featureExtractor: &FeatureExtractor{windowSize: 30 * time.Second},
	}

	mux := http.NewServeMux()
	reg := metrics.NewRegistry()
	httpMetrics := metrics.NewHTTPMetrics(reg, "ml_orchestrator")
	mux.HandleFunc("/analyze", orchestrator.handleAnalyze)
	mux.HandleFunc("/train", orchestrator.handleTrain)
	mux.Handle("/metrics", reg)

	port := os.Getenv("ML_ORCHESTRATOR_PORT")
	if port == "" {
		port = "8087"
	}

	// OpenTelemetry tracing (no-op unless built with otelotlp and endpoint set)
	shutdown := otelobs.InitTracer("ml_orchestrator")
	defer shutdown(context.Background())

	// Wrap with tracing + metrics middleware
	h := httpMetrics.Middleware(mux)
	h = otelobs.WrapHTTPHandler("ml_orchestrator", h)

	log.Printf("ML Orchestrator starting on port %s", port)
	log.Fatal(http.ListenAndServe(":"+port, h))
}

func (m *MLOrchestrator) handleAnalyze(w http.ResponseWriter, r *http.Request) {
	var event TelemetryEvent
	if err := json.NewDecoder(r.Body).Decode(&event); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	result := m.anomalyDetector.Predict(event)
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

func (m *MLOrchestrator) handleTrain(w http.ResponseWriter, r *http.Request) {
	var events []TelemetryEvent
	if err := json.NewDecoder(r.Body).Decode(&events); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	m.anomalyDetector.Train(events)
	
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{"status": "trained"})
}

// Prometheus metrics moved to /metrics via Registry

func (ad *AnomalyDetector) Predict(event TelemetryEvent) AnomalyResult {
	if !ad.trained {
		return AnomalyResult{
			IsAnomaly:   false,
			Score:       0.0,
			Confidence:  0.0,
			Explanation: "Model not trained",
		}
	}

	// Simple anomaly detection based on threat score
	score := event.ThreatScore
	isAnomaly := score > ad.threshold
	
	return AnomalyResult{
		IsAnomaly:   isAnomaly,
		Score:       score,
		Confidence:  0.8,
		Explanation: "Threat score analysis",
	}
}

func (ad *AnomalyDetector) Train(events []TelemetryEvent) {
	// Simple training - just mark as trained
	ad.trained = len(events) > 0
}