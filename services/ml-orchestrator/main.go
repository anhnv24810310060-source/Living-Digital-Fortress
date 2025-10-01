package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"

	"shieldx/pkg/metrics"
	otelobs "shieldx/pkg/observability/otel"
	logcorr "shieldx/pkg/observability/logcorr"
	"shieldx/pkg/ratls"
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
	mux.HandleFunc("/health", orchestrator.handleHealth)
	mux.HandleFunc("/whoami", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"service": "ml-orchestrator",
			"ratls_enabled": os.Getenv("RATLS_ENABLE") == "true",
		})
	})
	mux.Handle("/metrics", reg)

	port := os.Getenv("ML_ORCHESTRATOR_PORT")
	if port == "" {
		port = "8087"
	}

	// OpenTelemetry tracing (no-op unless built with otelotlp and endpoint set)
	shutdown := otelobs.InitTracer("ml_orchestrator")
	defer shutdown(context.Background())

	// Wrap with metrics, log-correlation, then tracing
	h := httpMetrics.Middleware(mux)
	h = logcorr.Middleware(h)
	h = otelobs.WrapHTTPHandler("ml_orchestrator", h)

	// RA-TLS optional enablement
	gCertExpiry := metrics.NewGauge("ratls_cert_expiry_seconds", "Seconds until current RA-TLS cert expiry")
	reg.RegisterGauge(gCertExpiry)
	var issuer *ratls.AutoIssuer
	if os.Getenv("RATLS_ENABLE") == "true" {
		td := getenvDefault("RATLS_TRUST_DOMAIN", "shieldx.local")
		ns := getenvDefault("RATLS_NAMESPACE", "default")
		svc := getenvDefault("RATLS_SERVICE", "ml-orchestrator")
		rotate := parseDurationDefault("RATLS_ROTATE_EVERY", 45*time.Minute)
		valid := parseDurationDefault("RATLS_VALIDITY", 60*time.Minute)
		ai, err := ratls.NewDevIssuer(ratls.Identity{TrustDomain: td, Namespace: ns, Service: svc}, rotate, valid)
		if err != nil { log.Fatalf("[ml-orchestrator] RA-TLS init: %v", err) }
		issuer = ai
		go func(){ for { if t, err := issuer.LeafNotAfter(); err==nil { gCertExpiry.Set(uint64(time.Until(t).Seconds())) }; time.Sleep(1*time.Minute) } }()
	}
	addr := fmt.Sprintf(":%s", port)
	srv := &http.Server{ Addr: addr, Handler: h }
	if issuer != nil {
		srv.TLSConfig = issuer.ServerTLSConfig(true, getenvDefault("RATLS_TRUST_DOMAIN", "shieldx.local"))
		log.Printf("ML Orchestrator (RA-TLS) starting on %s", addr)
		log.Fatal(srv.ListenAndServeTLS("", ""))
	} else {
		log.Printf("ML Orchestrator starting on %s", addr)
		log.Fatal(srv.ListenAndServe())
	}
}

// env helpers aligned with other services
func getenvDefault(key, def string) string { if v := os.Getenv(key); v != "" { return v }; return def }
func parseDurationDefault(key string, def time.Duration) time.Duration { if v := os.Getenv(key); v != "" { if d, err := time.ParseDuration(v); err==nil { return d } }; return def }

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

func (m *MLOrchestrator) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status": "healthy",
		"timestamp": time.Now(),
		"service": "ml-orchestrator",
	})
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