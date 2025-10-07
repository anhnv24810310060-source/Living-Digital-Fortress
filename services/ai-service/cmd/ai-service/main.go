package main

import (
    "log"
    "net/http"
    "os"
    "time"
    sharedlog "shieldx/shared/logging"
    "shieldx/shared/eventbus"
    "shieldx/services/ai-service/internal/server"
)

func main() {
    sharedlog.Infof("starting ai-service bootstrap")

    srvLogic := server.New()
    if os.Getenv("AI_SERVICE_EMBED_BUS") == "1" {
        bus := eventbus.NewBus(1024)
        srvLogic.WithBus(bus)
        sharedlog.Infof("ai-service embedded event bus enabled (standalone mode)")
    }
    mux := http.NewServeMux()
    mux.HandleFunc("/healthz", srvLogic.HealthHandler)
    mux.HandleFunc("/train", srvLogic.TrainHandler)
    mux.HandleFunc("/analyze", srvLogic.AnalyzeHandler)

    httpSrv := &http.Server{Addr: ":7010", Handler: mux, ReadHeaderTimeout: 5 * time.Second}
    sharedlog.Infof("ai-service listening on :7010")
    if err := httpSrv.ListenAndServe(); err != nil {
        log.Fatalf("ai-service stopped: %v", err)
    }
}
