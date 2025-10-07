//go:build !otelotlp

package otelobs

import (
	"context"
)

// InitTracer is a no-op by default to keep builds lightweight and avoid heavy OTLP deps.
// To enable OTLP tracing, build with -tags otelotlp which provides a real implementation.
func InitTracer(serviceName string) func(context.Context) error {
	return func(context.Context) error { return nil }
}
