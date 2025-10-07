package main

import (
    "context"
    "fmt"
    "log"
    "net/http"
    "sync/atomic"
    "time"
    sharedlog "shieldx/shared/logging"
    "shieldx/shared/eventbus"
)

var analysisCount atomic.Int64

type analysisSubscriber struct{}
func (analysisSubscriber) Topics() []string { return []string{"analysis.result"} }
func (analysisSubscriber) Handle(_ context.Context, evt eventbus.Event) { analysisCount.Add(1) }

func metricsHandler(w http.ResponseWriter, _ *http.Request) {
    w.Header().Set("Content-Type", "text/plain")
    fmt.Fprintf(w, "analysis_results_total %d\n", analysisCount.Load())
}
func healthHandler(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte("ok")) }

func main() {
    sharedlog.Infof("starting monitoring-service bootstrap")
    mux := http.NewServeMux()
    mux.HandleFunc("/healthz", healthHandler)
    mux.HandleFunc("/metrics", metricsHandler)
    // If embedded bus mode, create a bus and register subscriber (future: external injection)
    bus := eventbus.NewBus(1024)
    bus.Register(analysisSubscriber{})
    srv := &http.Server{Addr: ":7030", Handler: mux, ReadHeaderTimeout: 5 * time.Second}
    sharedlog.Infof("monitoring-service listening on :7030")
    if err := srv.ListenAndServe(); err != nil { log.Fatalf("monitoring-service stopped: %v", err) }
}
