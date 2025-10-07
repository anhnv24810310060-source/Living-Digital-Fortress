package otelobs

import (
	"context"
	"fmt"
	"os"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.4.0"
)

// TracerConfig holds OpenTelemetry tracer configuration
type TracerConfig struct {
	ServiceName    string
	ServiceVersion string
	Environment    string
	Endpoint       string  // OTLP endpoint (default: localhost:4318)
	SamplingRate   float64 // 0.0 to 1.0 (default: 0.1 for 10%)
}

// InitTracerWithConfig initializes OpenTelemetry tracer with detailed configuration
func InitTracerWithConfig(cfg TracerConfig) (func(context.Context) error, error) {
	ctx := context.Background()

	// Default values
	if cfg.Endpoint == "" {
		cfg.Endpoint = "localhost:4318"
	}
	if cfg.SamplingRate == 0 {
		cfg.SamplingRate = 0.1 // 10% default sampling
	}
	if cfg.Environment == "" {
		cfg.Environment = "production"
	}

	// Create OTLP HTTP exporter
	exporter, err := otlptracehttp.New(ctx,
		otlptracehttp.WithEndpoint(cfg.Endpoint),
		otlptracehttp.WithInsecure(), // Use WithTLSClientConfig for production
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create OTLP exporter: %w", err)
	}

	// Create resource with service information
	res, err := resource.New(ctx,
		resource.WithAttributes(
			semconv.ServiceNameKey.String(cfg.ServiceName),
			semconv.ServiceVersionKey.String(cfg.ServiceVersion),
			attribute.String("environment", cfg.Environment),
		),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create resource: %w", err)
	}

	// Create tracer provider with sampling
	tp := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exporter,
			sdktrace.WithBatchTimeout(5*time.Second),
			sdktrace.WithMaxExportBatchSize(512),
		),
		sdktrace.WithResource(res),
		sdktrace.WithSampler(sdktrace.TraceIDRatioBased(cfg.SamplingRate)),
	)

	// Set global tracer provider
	otel.SetTracerProvider(tp)

	// Return shutdown function
	return func(ctx context.Context) error {
		ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
		defer cancel()
		return tp.Shutdown(ctx)
	}, nil
}

// GetTracerConfig creates config from environment variables
func GetTracerConfigFromEnv(serviceName, serviceVersion string) TracerConfig {
	// Read from environment or use defaults
	endpoint := getEnv("OTEL_EXPORTER_OTLP_ENDPOINT", "localhost:4318")
	environment := getEnv("ENVIRONMENT", "production")
	samplingRate := 0.1 // Default 10%

	return TracerConfig{
		ServiceName:    serviceName,
		ServiceVersion: serviceVersion,
		Environment:    environment,
		Endpoint:       endpoint,
		SamplingRate:   samplingRate,
	}
}

func getEnv(key, defaultValue string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return defaultValue
}
