//go:build otelotlp

package otelobs

import (
    "net/http"

    "go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
    "go.opentelemetry.io/otel/propagation"
    "go.opentelemetry.io/otel"
)

// WrapHTTPHandler decorates the handler with otelhttp to produce server spans.
func WrapHTTPHandler(serviceName string, h http.Handler) http.Handler {
    // Ensure we inject standard W3C tracecontext; Jaeger via Collector understands it.
    otel.SetTextMapPropagator(propagation.TraceContext{})
    return otelhttp.NewHandler(h, serviceName)
}

// WrapHTTPTransport decorates an http.RoundTripper so client requests create spans
// and automatically propagate context via W3C traceparent headers.
func WrapHTTPTransport(t http.RoundTripper) http.RoundTripper {
    if t == nil {
        return otelhttp.DefaultClient.Transport
    }
    // Use default propagators
    return otelhttp.NewTransport(t)
}
