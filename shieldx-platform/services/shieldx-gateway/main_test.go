//go:build integration

package main

import (
    "net/http"
    "net/http/httptest"
    "testing"
    "time"
)

// This is a lightweight sanity test to ensure the health check logic handles HTTPS endpoints.
// It uses httptest and does not spin real RA-TLS, so it only validates basic flow.
// For full mTLS verification, run an integration environment (compose) and observe health logs.
func TestHealthLoop_Basic(t *testing.T) {
    // Start a dummy healthy server
    ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        if r.URL.Path == "/health" { w.WriteHeader(http.StatusOK); w.Write([]byte("ok")); return }
        w.WriteHeader(http.StatusNotFound)
    }))
    defer ts.Close()

    sg := &ShieldXGateway{
        services:       map[string]*ServiceEndpoint{},
        httpClients:    map[string]*http.Client{},
        shutdownChan:   make(chan struct{}),
        config:         &GatewayConfig{HealthCheckInterval: 50 * time.Millisecond},
    }
    ep := &ServiceEndpoint{URLs: []string{ts.URL}, HealthyURLs: []string{}}
    sg.services["test"] = ep
    sg.httpClients["test"] = &http.Client{Timeout: 2 * time.Second}

    go sg.healthCheckLoop("test", ep)
    time.Sleep(200 * time.Millisecond)
    close(sg.shutdownChan)

    ep.mutex.RLock()
    defer ep.mutex.RUnlock()
    if len(ep.HealthyURLs) == 0 {
        t.Fatalf("expected healthy URLs to be non-empty")
    }
}
