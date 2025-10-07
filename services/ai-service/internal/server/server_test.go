package server

import (
    "bytes"
    "encoding/json"
    "net/http"
    "net/http/httptest"
    "testing"
    "time"
    "shieldx/services/ai-service/internal/ml"
)

func newTestServer() *Server { return New() }

func TestHealth(t *testing.T) {
    s := newTestServer()
    req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
    w := httptest.NewRecorder()
    s.HealthHandler(w, req)
    if w.Code != http.StatusOK { t.Fatalf("expected 200 got %d", w.Code) }
}

func TestTrainAndAnalyze(t *testing.T) {
    s := newTestServer()
    events := []ml.TelemetryEvent{{
        Timestamp: time.Now(),
        Source: "test",
        EventType: "connection",
        SessionID: "abc",
        Features: []float64{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18},
    }}
    tr := TrainRequest{Events: events}
    b, _ := json.Marshal(tr)
    trainReq := httptest.NewRequest(http.MethodPost, "/train", bytes.NewReader(b))
    trainW := httptest.NewRecorder()
    s.TrainHandler(trainW, trainReq)
    if trainW.Code != http.StatusOK { t.Fatalf("train expected 200 got %d body=%s", trainW.Code, trainW.Body.String()) }

    ar := AnalyzeRequest{Event: events[0]}
    ba, _ := json.Marshal(ar)
    analyzeReq := httptest.NewRequest(http.MethodPost, "/analyze", bytes.NewReader(ba))
    analyzeW := httptest.NewRecorder()
    s.AnalyzeHandler(analyzeW, analyzeReq)
    if analyzeW.Code != http.StatusOK { t.Fatalf("analyze expected 200 got %d body=%s", analyzeW.Code, analyzeW.Body.String()) }
}
