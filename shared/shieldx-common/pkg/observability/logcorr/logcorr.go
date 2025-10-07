package logcorr

import (
    "log"
    "net/http"
    "time"

    "go.opentelemetry.io/otel/trace"
)

// Middleware logs one access line per request with trace_id/span_id when available
// and adds X-Trace-Id/X-Span-Id headers to the response.
func Middleware(next http.Handler) http.Handler {
    if next == nil {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { http.Error(w, "not found", http.StatusNotFound) })
    }
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        sr := &recorder{ResponseWriter: w, status: 200}
        // call next
        next.ServeHTTP(sr, r)

        // extract trace/span ids
        sc := trace.SpanContextFromContext(r.Context())
        tid := ""
        sid := ""
        if sc.HasTraceID() { tid = sc.TraceID().String() }
        if sc.HasSpanID() { sid = sc.SpanID().String() }
        if tid != "" { sr.Header().Set("X-Trace-Id", tid) }
        if sid != "" { sr.Header().Set("X-Span-Id", sid) }

        dur := time.Since(start)
        // single-line access log; safe minimal fields for correlation
        if tid != "" || sid != "" {
            log.Printf("access method=%s path=%s status=%d dur_ms=%d trace_id=%s span_id=%s", r.Method, r.URL.Path, sr.status, dur.Milliseconds(), tid, sid)
        } else {
            log.Printf("access method=%s path=%s status=%d dur_ms=%d", r.Method, r.URL.Path, sr.status, dur.Milliseconds())
        }
    })
}

type recorder struct{ http.ResponseWriter; status int }
func (r *recorder) WriteHeader(code int) { r.status = code; r.ResponseWriter.WriteHeader(code) }
