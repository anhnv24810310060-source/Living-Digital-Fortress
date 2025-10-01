package metrics

import (
    "fmt"
    "math"
    "net/http"
    "os"
    "regexp"
    "sort"
    "strings"
    "sync"
    "sync/atomic"
    "time"
)

// Minimal Prometheus-like counters, gauges, and histograms

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

// Histogram is a simple, thread-safe, cumulative bucket histogram
type Histogram struct {
    name    string
    help    string
    buckets []float64 // sorted, finite bucket boundaries; +Inf implied
    counts  []uint64  // same len as buckets
    sum     float64
    cnt     uint64
    mu      sync.Mutex
}

func defaultBuckets() []float64 {
    // seconds, similar to Prometheus default HTTP buckets
    return []float64{0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5}
}

func NewHistogram(name, help string, buckets []float64) *Histogram {
    if len(buckets) == 0 { buckets = defaultBuckets() }
    cp := make([]float64, len(buckets))
    copy(cp, buckets)
    sort.Float64s(cp)
    return &Histogram{name: name, help: help, buckets: cp, counts: make([]uint64, len(cp))}
}

func (h *Histogram) Observe(v float64) {
    if math.IsNaN(v) || math.IsInf(v, 0) { return }
    h.mu.Lock()
    defer h.mu.Unlock()
    // find first bucket >= v
    i := sort.SearchFloat64s(h.buckets, v)
    if i < len(h.counts) { h.counts[i]++ } else { // falls into +Inf; Prometheus expects cumulative buckets, we'll emit +Inf as total count
        // do nothing here; +Inf is emitted via total count
    }
    h.cnt++
    h.sum += v
}

func (h *Histogram) Expose(w http.ResponseWriter) {
    if h.help != "" { fmt.Fprintf(w, "# HELP %s %s\n", h.name, h.help) }
    fmt.Fprintf(w, "# TYPE %s histogram\n", h.name)
    // cumulative
    var cum uint64
    for i, b := range h.buckets {
        cum += h.counts[i]
        fmt.Fprintf(w, "%s_bucket{le=\"%g\"} %d\n", h.name, b, cum)
    }
    // +Inf bucket equals total count
    fmt.Fprintf(w, "%s_bucket{le=\"+Inf\"} %d\n", h.name, h.cnt)
    fmt.Fprintf(w, "%s_sum %g\n", h.name, h.sum)
    fmt.Fprintf(w, "%s_count %d\n", h.name, h.cnt)
}

type Registry struct { counters []*Counter; gauges []*Gauge; histos []*Histogram }

func NewRegistry() *Registry { return &Registry{} }
func (r *Registry) Register(c *Counter) { r.counters = append(r.counters, c) }
func (r *Registry) RegisterGauge(g *Gauge) { r.gauges = append(r.gauges, g) }
func (r *Registry) RegisterHistogram(h *Histogram) { r.histos = append(r.histos, h) }
// Note: ServeHTTP is defined later to include labeled metrics as well.

// ---- Labeled metrics (basic) ----

// Label encoding helpers using a stable order and a key string.
// We encode only values joined by a unit separator; order is fixed per metric.
// This keeps map keys comparable and lightweight.
// Unsafe for values containing the unit separator; choose a rare rune.
const labelSep = "\x1f"

func labelsKey(order []string, labels map[string]string) string {
    if len(order) == 0 { return "" }
    vals := make([]string, len(order))
    for i, k := range order { vals[i] = labels[k] }
    return strings.Join(vals, labelSep)
}

func writeLabels(w http.ResponseWriter, order []string, key string) {
    if len(order) == 0 { return }
    vals := strings.Split(key, labelSep)
    fmt.Fprint(w, "{")
    for i := range order {
        if i > 0 { fmt.Fprint(w, ",") }
        v := ""
        if i < len(vals) { v = vals[i] }
        fmt.Fprintf(w, "%s=\"%s\"", order[i], v)
    }
    fmt.Fprint(w, "}")
}

type LabeledCounter struct{
    name string; help string
    labelOrder []string
    mu sync.Mutex
    m map[string]uint64
}

func NewLabeledCounter(name, help string, labelOrder []string) *LabeledCounter {
    return &LabeledCounter{name: name, help: help, labelOrder: labelOrder, m: map[string]uint64{}}
}
func (c *LabeledCounter) Inc(labels map[string]string) {
    c.mu.Lock(); defer c.mu.Unlock()
    k := labelsKey(c.labelOrder, labels)
    c.m[k] = c.m[k] + 1
}
func (c *LabeledCounter) Expose(w http.ResponseWriter) {
    if c.help != "" { fmt.Fprintf(w, "# HELP %s %s\n", c.name, c.help) }
    fmt.Fprintf(w, "# TYPE %s counter\n", c.name)
    c.mu.Lock(); defer c.mu.Unlock()
    for k, v := range c.m {
        fmt.Fprintf(w, "%s", c.name)
        writeLabels(w, c.labelOrder, k)
        fmt.Fprintf(w, " %d\n", v)
    }
}

type LabeledHistogram struct{
    name string; help string
    labelOrder []string
    buckets []float64
    mu sync.Mutex
    counts map[string][]uint64
    cnt map[string]uint64
    sum map[string]float64
}

func NewLabeledHistogram(name, help string, labelOrder []string, buckets []float64) *LabeledHistogram {
    if len(buckets) == 0 { buckets = defaultBuckets() }
    cp := make([]float64, len(buckets)); copy(cp, buckets); sort.Float64s(cp)
    return &LabeledHistogram{name: name, help: help, labelOrder: labelOrder, buckets: cp, counts: map[string][]uint64{}, cnt: map[string]uint64{}, sum: map[string]float64{}}
}
func (h *LabeledHistogram) Observe(labels map[string]string, v float64) {
    if math.IsNaN(v) || math.IsInf(v,0) { return }
    k := labelsKey(h.labelOrder, labels)
    h.mu.Lock(); defer h.mu.Unlock()
    arr, ok := h.counts[k]
    if !ok { arr = make([]uint64, len(h.buckets)); h.counts[k] = arr }
    i := sort.SearchFloat64s(h.buckets, v)
    if i < len(arr) { arr[i]++ }
    h.cnt[k] = h.cnt[k] + 1
    h.sum[k] = h.sum[k] + v
}
func (h *LabeledHistogram) Expose(w http.ResponseWriter) {
    if h.help != "" { fmt.Fprintf(w, "# HELP %s %s\n", h.name, h.help) }
    fmt.Fprintf(w, "# TYPE %s histogram\n", h.name)
    h.mu.Lock(); defer h.mu.Unlock()
    for k, arr := range h.counts {
        var cum uint64
        for i, b := range h.buckets {
            cum += arr[i]
            // Merge existing labels with le into a single label set
            fmt.Fprintf(w, "%s_bucket", h.name)
            // Print merged label set: existing labels + le
            vals := strings.Split(k, labelSep)
            fmt.Fprint(w, "{")
            for idx := range h.labelOrder {
                if idx > 0 { fmt.Fprint(w, ",") }
                v := ""
                if idx < len(vals) { v = vals[idx] }
                fmt.Fprintf(w, "%s=\"%s\"", h.labelOrder[idx], v)
            }
            // append le
            if len(h.labelOrder) > 0 { fmt.Fprint(w, ",") }
            fmt.Fprintf(w, "le=\"%g\"}", b)
            fmt.Fprintf(w, " %d\n", cum)
        }
        // +Inf
        fmt.Fprintf(w, "%s_bucket", h.name)
        // merged labels with +Inf
        vals := strings.Split(k, labelSep)
        fmt.Fprint(w, "{")
        for idx := range h.labelOrder {
            if idx > 0 { fmt.Fprint(w, ",") }
            v := ""
            if idx < len(vals) { v = vals[idx] }
            fmt.Fprintf(w, "%s=\"%s\"", h.labelOrder[idx], v)
        }
        if len(h.labelOrder) > 0 { fmt.Fprint(w, ",") }
        fmt.Fprint(w, "le=\"+Inf\"}")
        fmt.Fprintf(w, " %d\n", h.cnt[k])
        fmt.Fprintf(w, "%s_sum", h.name); writeLabels(w, h.labelOrder, k); fmt.Fprintf(w, " %g\n", h.sum[k])
        fmt.Fprintf(w, "%s_count", h.name); writeLabels(w, h.labelOrder, k); fmt.Fprintf(w, " %d\n", h.cnt[k])
    }
}

type RegistryLabeled struct{ counters []*LabeledCounter; histos []*LabeledHistogram }

func (r *Registry) RegisterLabeledCounter(c *LabeledCounter) { rLabeled.counters = append(rLabeled.counters, c) }
func (r *Registry) RegisterLabeledHistogram(h *LabeledHistogram) { rLabeled.histos = append(rLabeled.histos, h) }

var rLabeled = &RegistryLabeled{}

func (r *Registry) ServeHTTP(w http.ResponseWriter, _ *http.Request) {
    w.Header().Set("Content-Type", "text/plain; version=0.0.4")
    for _, c := range r.counters { c.Expose(w) }
    for _, g := range r.gauges { g.Expose(w) }
    for _, h := range r.histos { h.Expose(w) }
    for _, c := range rLabeled.counters { c.Expose(w) }
    for _, h := range rLabeled.histos { h.Expose(w) }
}

// HTTPMetrics exposes basic HTTP request metrics
type HTTPMetrics struct {
    RequestsTotal *Counter
    ErrorsTotal   *Counter
    Duration      *Histogram // seconds
    RequestsByPath *LabeledCounter
    DurationByPath *LabeledHistogram
    // path normalization / cardinality controls
    pathAllowlist []string       // exact prefixes to keep as-is
    pathRegexps   []*regexp.Regexp // regex allowlist to keep as-is
    pathMode      string         // "heuristic" (default) or "strict"
}

func NewHTTPMetrics(reg *Registry, service string) *HTTPMetrics {
    m := &HTTPMetrics{
        RequestsTotal: NewCounter(service+"_http_requests_total", "Total HTTP requests"),
        ErrorsTotal:   NewCounter(service+"_http_errors_total", "Total HTTP 5xx responses"),
        Duration:      NewHistogram(service+"_http_request_duration_seconds", "HTTP request duration seconds", nil),
        RequestsByPath: NewLabeledCounter(service+"_http_requests_by_path_total", "Total HTTP requests by method and path", []string{"method","path"}),
        DurationByPath: NewLabeledHistogram(service+"_http_request_duration_by_path_seconds", "HTTP request duration seconds by method and path", []string{"method","path"}, nil),
        // default allowlist excludes root "/" to avoid disabling normalization globally
        pathAllowlist: allowlistFromEnv(service, []string{"/health","/healthz","/metrics"}),
        pathRegexps:   regexAllowlistFromEnv(service),
        pathMode:      pathModeFromEnv(service),
    }
    if reg != nil {
        reg.Register(m.RequestsTotal)
        reg.Register(m.ErrorsTotal)
        reg.RegisterHistogram(m.Duration)
        reg.RegisterLabeledCounter(m.RequestsByPath)
        reg.RegisterLabeledHistogram(m.DurationByPath)
    }
    return m
}

// statusRecorder wraps ResponseWriter to capture the final status code.
type statusRecorder struct{ http.ResponseWriter; status int }

// WriteHeader captures the status and forwards the call.
func (sr *statusRecorder) WriteHeader(code int) {
    sr.status = code
    sr.ResponseWriter.WriteHeader(code)
}

func (m *HTTPMetrics) Middleware(next http.Handler) http.Handler {
    if next == nil { return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(404) }) }
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        sr := &statusRecorder{ResponseWriter: w, status: 200}
        start := time.Now()
        // Serve wrapped response writer to capture status
        next.ServeHTTP(sr, r)
        // The above indirection keeps original interfaces; however, if handlers don't call WriteHeader explicitly, status stays 200
        // Update metrics
        m.RequestsTotal.Inc()
        if sr.status >= 500 { m.ErrorsTotal.Inc() }
        m.Duration.Observe(time.Since(start).Seconds())
        // Per-path labels (note: potential cardinality risk)
        p := normalizePath(r.URL.Path, m.pathAllowlist, m.pathRegexps, m.pathMode)
        meth := r.Method
        m.RequestsByPath.Inc(map[string]string{"method": meth, "path": p})
        m.DurationByPath.Observe(map[string]string{"method": meth, "path": p}, time.Since(start).Seconds())
    })
}

// normalizePath reduces path cardinality by:
// - keeping known low-cardinality paths as-is (allowlist prefixes)
// - keeping paths that match configured regex allowlist as-is
// - replacing path segments that look like IDs (hex, UUID-like, digits) with :id (heuristic mode)
// - or collapsing to ":other" if not in allowlist/regex (strict mode)
func normalizePath(path string, allow []string, rxps []*regexp.Regexp, mode string) string {
    if path == "" { return "/" }
    for _, pref := range allow {
        if pref != "" && strings.HasPrefix(path, pref) { return path }
    }
    for _, rx := range rxps {
        if rx != nil && rx.MatchString(path) { return path }
    }
    // strict mode: collapse to a single low-cardinality bucket
    if strings.EqualFold(mode, "strict") {
        return ":other"
    }
    // heuristic mode (default): replace likely-id segments with :id
    segs := strings.Split(path, "/")
    for i, s := range segs {
        if s == "" { continue }
        if looksLikeID(s) { segs[i] = ":id" }
    }
    np := strings.Join(segs, "/")
    if !strings.HasPrefix(np, "/") { np = "/" + np }
    return np
}

func looksLikeID(s string) bool {
    if len(s) >= 8 { // long enough to be an id
        // hex-only?
        hex := true
        for i := 0; i < len(s); i++ {
            c := s[i]
            if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F') || c == '-') { hex = false; break }
        }
        if hex { return true }
    }
    // purely digits
    digits := true
    for i := 0; i < len(s); i++ { if s[i] < '0' || s[i] > '9' { digits = false; break } }
    if digits && len(s) > 3 { return true }
    return false
}

// allowlistFromEnv reads SERVICE_HTTP_PATH_ALLOWLIST (comma-separated) if set.
// Also supports global HTTP_PATH_ALLOWLIST as fallback.
func allowlistFromEnv(service string, def []string) []string {
    // SERVICE name envs use uppercase and dashes/space normalized to underscore
    key := strings.ToUpper(service) + "_HTTP_PATH_ALLOWLIST"
    if v := os.Getenv(key); v != "" {
        return splitCSV(v)
    }
    if v := os.Getenv("HTTP_PATH_ALLOWLIST"); v != "" {
        return splitCSV(v)
    }
    return def
}

func splitCSV(v string) []string {
    parts := strings.Split(v, ",")
    out := make([]string, 0, len(parts))
    for _, p := range parts {
        s := strings.TrimSpace(p)
        if s != "" { out = append(out, s) }
    }
    return out
}

// regexAllowlistFromEnv reads SERVICE_HTTP_PATH_REGEX (comma-separated) if set, else HTTP_PATH_REGEX.
func regexAllowlistFromEnv(service string) []*regexp.Regexp {
    var raw string
    if v := os.Getenv(strings.ToUpper(service) + "_HTTP_PATH_REGEX"); v != "" {
        raw = v
    } else if v := os.Getenv("HTTP_PATH_REGEX"); v != "" {
        raw = v
    }
    if raw == "" { return nil }
    parts := splitCSV(raw)
    out := make([]*regexp.Regexp, 0, len(parts))
    for _, expr := range parts {
        if expr == "" { continue }
        if rx, err := regexp.Compile(expr); err == nil {
            out = append(out, rx)
        }
    }
    return out
}

// pathModeFromEnv reads SERVICE_HTTP_PATH_MODE or HTTP_PATH_MODE; values: heuristic (default) | strict
func pathModeFromEnv(service string) string {
    if v := os.Getenv(strings.ToUpper(service) + "_HTTP_PATH_MODE"); v != "" {
        vv := strings.ToLower(strings.TrimSpace(v))
        if vv == "strict" || vv == "heuristic" { return vv }
    }
    if v := os.Getenv("HTTP_PATH_MODE"); v != "" {
        vv := strings.ToLower(strings.TrimSpace(v))
        if vv == "strict" || vv == "heuristic" { return vv }
    }
    return "heuristic"
}


