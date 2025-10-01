//go:build !otelotlp

package otelobs

import "net/http"

// WrapHTTPHandler is a no-op by default. Build with -tags otelotlp to enable tracing.
func WrapHTTPHandler(serviceName string, h http.Handler) http.Handler { return h }

// WrapHTTPTransport is a no-op by default. Build with -tags otelotlp to enable trace propagation.
func WrapHTTPTransport(t http.RoundTripper) http.RoundTripper { return t }
