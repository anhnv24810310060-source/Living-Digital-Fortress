package otelobs

import (
	"log"
	"net/http"
	"time"

	"go.opentelemetry.io/otel/trace"
)

// HTTPTraceLogMiddleware logs a compact access line with trace_id/span_id per request
// and sets response headers Trace-Id and Span-Id for easy correlation.
// Non-invasive: does not change existing loggers; just adds one line at end.
func HTTPTraceLogMiddleware(next http.Handler) http.Handler {
	if next == nil {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(404) })
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// capture start time and wrap writer to record status/bytes
		start := time.Now()
		sr := &statusRecorder{ResponseWriter: w, status: 200}
		next.ServeHTTP(sr, r)
		// extract trace ids
		sc := trace.SpanContextFromContext(r.Context())
		traceID := "-"
		spanID := "-"
		if sc.IsValid() {
			traceID = sc.TraceID().String()
			spanID = sc.SpanID().String()
			// add correlation headers
			sr.Header().Set("Trace-Id", traceID)
			sr.Header().Set("Span-Id", spanID)
		}
		dur := time.Since(start)
		// simple access log line
		log.Printf("access method=%s path=%s status=%d dur_ms=%d trace_id=%s span_id=%s", r.Method, r.URL.Path, sr.status, dur.Milliseconds(), traceID, spanID)
	})
}

type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (sr *statusRecorder) WriteHeader(code int) {
	sr.status = code
	sr.ResponseWriter.WriteHeader(code)
}
