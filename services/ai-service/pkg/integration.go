package aihook

import (
    "time"
    "context"
    "shieldx/shared/eventbus"
    sharedlog "shieldx/shared/logging"
    "shieldx/services/ai-service/internal/ml"
)

// RegisterInMemoryAnomalyConsumer sets up a minimal anomaly detector subscriber on the provided bus.
// It mirrors the internal server's event handling for honeypot.request and publishes analysis.result.
func RegisterInMemoryAnomalyConsumer(bus *eventbus.Bus) {
    if bus == nil { return }
    det := ml.NewAnomalyDetector(-0.5, 512, 10*time.Minute)
    bus.Register(anomalySub{det: det, bus: bus})
}

type anomalySub struct {
    det *ml.AnomalyDetector
    bus *eventbus.Bus
}

func (a anomalySub) Topics() []string { return []string{"honeypot.request"} }

func (a anomalySub) Handle(ctx context.Context, evt eventbus.Event) {
    features := []float64{1,0,0}
    te := ml.TelemetryEvent{Timestamp: time.Now().UTC(), Source: evt.Source, EventType: evt.Type, SessionID: "sess", Features: features}
    res, err := a.det.Predict(te)
    if err != nil { sharedlog.Errorf("predict failed: %v", err); return }
    _ = a.bus.Publish(ctx, eventbus.Event{Type: "analysis.result", Source: "ai-service", Payload: map[string]any{
        "is_anomaly": res.IsAnomaly,
        "score": res.Score,
        "confidence": res.Confidence,
    }})
}
