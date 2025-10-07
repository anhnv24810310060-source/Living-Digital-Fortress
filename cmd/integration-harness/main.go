package main

import (
    "context"
    "fmt"
    "io"
    "net/http"
    "sync/atomic"
    "time"
    sharedlog "shieldx/shared/logging"
    "shieldx/shared/eventbus"
    aihook "shieldx/services/ai-service/pkg"
    hphook "shieldx/services/honeypot-service/pkg"
)

// Minimal monitoring in-process component (reuse subscriber type logic)
type monitoring struct {
    bus *eventbus.Bus
    addr string
}

func startMonitoring(bus *eventbus.Bus, addr string) *monitoring {
    m := &monitoring{bus: bus, addr: addr}
    // subscriber increments a counter via closure
    bus.Register(analysisSub{})
    mux := http.NewServeMux()
    mux.HandleFunc("/metrics", func(w http.ResponseWriter, _ *http.Request) { fmt.Fprintf(w, "analysis_results_total %d\n", analysisCount.Load()) })
    mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte("ok")) })
    go http.ListenAndServe(addr, mux)
    return m
}

// replicate subscriber/counter (cannot import from monitoring-service cmd package)
var analysisCount atomicCounter

type atomicCounter struct{ v int64 }
func (c *atomicCounter) Add(delta int64) { atomic.AddInt64(&c.v, delta) }
func (c *atomicCounter) Load() int64 { return atomic.LoadInt64(&c.v) }

type analysisSub struct{}
func (analysisSub) Topics() []string { return []string{"analysis.result"} }
func (analysisSub) Handle(_ context.Context, evt eventbus.Event) { analysisCount.Add(1) }

func main() {
    sharedlog.Infof("integration harness starting (Option A shared in-memory bus)")
    bus := eventbus.NewBus(2048)

    // Register AI anomaly consumer subscriber
    aihook.RegisterInMemoryAnomalyConsumer(bus)

    // Start honeypot http decoy
    _ = hphook.StartHTTPDecoy(":7110", bus)

    // Start minimal monitoring component
    startMonitoring(bus, ":7031")

    // Give servers time to start
    time.Sleep(300 * time.Millisecond)

    // Fire a synthetic honeypot request
    resp, err := http.Get("http://localhost:7110/test-path")
    if err != nil { sharedlog.Errorf("honeypot request failed: %v", err) } else { io.Copy(io.Discard, resp.Body); resp.Body.Close() }

    // Allow event propagation
    time.Sleep(500 * time.Millisecond)

    // Fetch metrics
    mresp, err := http.Get("http://localhost:7031/metrics")
    if err != nil { sharedlog.Errorf("metrics fetch failed: %v", err) } else { b, _ := io.ReadAll(mresp.Body); mresp.Body.Close(); fmt.Printf("\n--- Metrics Snapshot ---\n%s\n", string(b)) }

    sharedlog.Infof("integration harness completed demo run")
}
