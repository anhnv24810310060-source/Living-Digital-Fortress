package main

import (
    "log"
    "net/http"
    "time"
    sharedlog "shieldx/shared/logging"
    "shieldx/shared/eventbus"
    decoyhttp "shieldx/services/honeypot-service/internal/decoys/http"
    decoyssh "shieldx/services/honeypot-service/internal/decoys/ssh"
    decoyredis "shieldx/services/honeypot-service/internal/decoys/redis"
)

func healthHandler(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte("ok")) }

func main() {
    sharedlog.Infof("starting honeypot-service bootstrap")
    // Shared in-memory event bus (will be externalized later)
    bus := eventbus.NewBus(1024)
    // Start placeholder decoys (non-blocking)
    _ = decoyhttp.New(":7110", bus).Start()
    _ = decoyssh.New(":7120").Start()
    _ = decoyredis.New(":7130").Start()

    mux := http.NewServeMux()
    mux.HandleFunc("/healthz", healthHandler)
    srv := &http.Server{Addr: ":7020", Handler: mux, ReadHeaderTimeout: 5 * time.Second}
    sharedlog.Infof("honeypot-service listening on :7020")
    if err := srv.ListenAndServe(); err != nil { log.Fatalf("honeypot-service stopped: %v", err) }
}
