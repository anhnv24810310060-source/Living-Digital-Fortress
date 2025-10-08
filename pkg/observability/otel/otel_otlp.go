//go:build otelotlp

package otelobs

import (
	"context"
	"log"
	"os"

	"go.opentelemetry.io/otel"
	otlptracehttp "go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/sdk/resource"
	"go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.21.0"
)

// InitTracer sets up an OTLP HTTP exporter and returns a shutdown func.
func InitTracer(serviceName string) func(context.Context) error {
	endpoint := os.Getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
	if endpoint == "" {
		log.Printf("[otel] no OTEL_EXPORTER_OTLP_ENDPOINT; tracing disabled for %s", serviceName)
		return func(context.Context) error { return nil }
	}
	exp, err := otlptracehttp.New(context.Background(), otlptracehttp.WithEndpoint(endpoint), otlptracehttp.WithInsecure())
	if err != nil {
		log.Printf("[otel] exporter init error: %v", err)
		return func(context.Context) error { return nil }
	}
	res, err := resource.New(context.Background(), resource.WithAttributes(semconv.ServiceName(serviceName)))
	if err != nil {
		log.Printf("[otel] resource init error: %v", err)
	}
	tp := trace.NewTracerProvider(trace.WithBatcher(exp), trace.WithResource(res))
	otel.SetTracerProvider(tp)
	return tp.Shutdown
}
