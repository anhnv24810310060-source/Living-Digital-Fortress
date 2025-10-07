package server

import (
    "context"
    "encoding/json"
    "io"
    "net/http"
    "time"
    "shieldx/services/ai-service/internal/ml"
    "shieldx/shared/eventbus"
    sharedlog "shieldx/shared/logging"
)

// Server holds ML components and exposes HTTP handlers.
type Server struct {
    anomaly *ml.AnomalyDetector
    bus     *eventbus.Bus
}

// New constructs a Server with default anomaly detector.
func New() *Server { return &Server{anomaly: ml.NewAnomalyDetector(-0.5, 512, 10 * time.Minute)} }

// WithBus attaches an event bus and registers subscriptions.
func (s *Server) WithBus(bus *eventbus.Bus) *Server {
    s.bus = bus
    if bus != nil {
        bus.Register(eventbusSubscriber{server: s})
    }
    return s
}

// eventbusSubscriber implements eventbus.Subscriber for honeypot.request events.
type eventbusSubscriber struct { server *Server }

func (es eventbusSubscriber) Topics() []string { return []string{"honeypot.request"} }

func (es eventbusSubscriber) Handle(ctx context.Context, evt eventbus.Event) {
    // Convert generic payload into a basic TelemetryEvent for anomaly detection.
    // For now we simulate feature extraction by using placeholder numeric features.
    if es.server == nil { return }
    features := []float64{1, 0, 0} // TODO: derive from payload realistically
    te := ml.TelemetryEvent{Timestamp: time.Now().UTC(), Source: evt.Source, EventType: evt.Type, SessionID: "sess", Features: features}
    result, err := es.server.anomaly.Predict(te)
    if err != nil {
        sharedlog.Errorf("predict failed: %v", err)
        return
    }
    if es.server.bus != nil {
        _ = es.server.bus.Publish(ctx, eventbus.Event{Type: "analysis.result", Source: "ai-service", Payload: map[string]any{
            "is_anomaly": result.IsAnomaly,
            "score": result.Score,
            "confidence": result.Confidence,
        }})
    }
}

func (s *Server) HealthHandler(w http.ResponseWriter, _ *http.Request) {
    w.WriteHeader(http.StatusOK)
    _, _ = w.Write([]byte("ok"))
}

// TrainRequest is the input payload for /train.
type TrainRequest struct {
    Events []ml.TelemetryEvent `json:"events"`
}

// AnalyzeRequest is the input payload for /analyze.
type AnalyzeRequest struct {
    Event ml.TelemetryEvent `json:"event"`
}

// AnalyzeResponse is the response from /analyze.
type AnalyzeResponse struct {
    *ml.AnomalyResult `json:"result"`
}

func writeJSON(w http.ResponseWriter, status int, v any) {
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(status)
    _ = json.NewEncoder(w).Encode(v)
}

func (s *Server) TrainHandler(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodPost {
        w.WriteHeader(http.StatusMethodNotAllowed)
        return
    }
    body, err := io.ReadAll(r.Body)
    if err != nil {
        writeJSON(w, http.StatusBadRequest, map[string]string{"error": "unable to read body"})
        return
    }
    var req TrainRequest
    if err := json.Unmarshal(body, &req); err != nil {
        writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid json"})
        return
    }
    if len(req.Events) == 0 {
        writeJSON(w, http.StatusBadRequest, map[string]string{"error": "no events provided"})
        return
    }
    ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
    defer cancel()
    if err := s.anomaly.Train(ctx, req.Events); err != nil {
        sharedlog.Errorf("train failed: %v", err)
        writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
        return
    }
    writeJSON(w, http.StatusOK, map[string]string{"status": "trained", "events":  string(rune(len(req.Events)))})
}

func (s *Server) AnalyzeHandler(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodPost {
        w.WriteHeader(http.StatusMethodNotAllowed)
        return
    }
    body, err := io.ReadAll(r.Body)
    if err != nil {
        writeJSON(w, http.StatusBadRequest, map[string]string{"error": "unable to read body"})
        return
    }
    var req AnalyzeRequest
    if err := json.Unmarshal(body, &req); err != nil {
        writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid json"})
        return
    }
    result, err := s.anomaly.Predict(req.Event)
    if err != nil {
        writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
        return
    }
    writeJSON(w, http.StatusOK, AnalyzeResponse{AnomalyResult: result})
}
