package metrics

import (
    "fmt"
    "net/http"
    "sync/atomic"
)

// Minimal Prometheus-like counters and gauges

type Counter struct { v uint64; name string; help string }

func NewCounter(name, help string) *Counter { return &Counter{name: name, help: help} }
func (c *Counter) Inc() { atomic.AddUint64(&c.v, 1) }
func (c *Counter) Add(n uint64) { atomic.AddUint64(&c.v, n) }
func (c *Counter) Value() uint64 { return atomic.LoadUint64(&c.v) }
func (c *Counter) Expose(w http.ResponseWriter) {
    if c.help != "" { fmt.Fprintf(w, "# HELP %s %s\n", c.name, c.help) }
    fmt.Fprintf(w, "# TYPE %s counter\n%s %d\n", c.name, c.name, c.Value())
}

type Gauge struct { v uint64; name string; help string }

func NewGauge(name, help string) *Gauge { return &Gauge{name: name, help: help} }
func (g *Gauge) Set(n uint64) { atomic.StoreUint64(&g.v, n) }
func (g *Gauge) Value() uint64 { return atomic.LoadUint64(&g.v) }
func (g *Gauge) Expose(w http.ResponseWriter) {
    if g.help != "" { fmt.Fprintf(w, "# HELP %s %s\n", g.name, g.help) }
    fmt.Fprintf(w, "# TYPE %s gauge\n%s %d\n", g.name, g.name, g.Value())
}

type Registry struct { counters []*Counter; gauges []*Gauge }

func NewRegistry() *Registry { return &Registry{} }
func (r *Registry) Register(c *Counter) { r.counters = append(r.counters, c) }
func (r *Registry) RegisterGauge(g *Gauge) { r.gauges = append(r.gauges, g) }
func (r *Registry) ServeHTTP(w http.ResponseWriter, _ *http.Request) {
    w.Header().Set("Content-Type", "text/plain; version=0.0.4")
    for _, c := range r.counters { c.Expose(w) }
    for _, g := range r.gauges { g.Expose(w) }
}


