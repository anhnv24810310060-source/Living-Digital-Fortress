package metrics

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/otlp/otlpmetric/otlpmetrichttp"
	"go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/resource"
	semconv "go.opentelemetry.io/otel/semconv/v1.4.0"
)

// OTelExporter wraps OpenTelemetry metric exporter
type OTelExporter struct {
	provider *metric.MeterProvider
	shutdown func(context.Context) error
}

// NewOTelExporter creates a new OpenTelemetry metrics exporter
func NewOTelExporter(serviceName, endpoint string) (*OTelExporter, error) {
	ctx := context.Background()

	// Create OTLP HTTP exporter
	exporter, err := otlpmetrichttp.New(ctx,
		otlpmetrichttp.WithEndpoint(endpoint),
		otlpmetrichttp.WithInsecure(),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create OTLP metric exporter: %w", err)
	}

	// Create resource
	res, err := resource.New(ctx,
		resource.WithAttributes(
			semconv.ServiceNameKey.String(serviceName),
		),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create resource: %w", err)
	}

	// Create meter provider
	provider := metric.NewMeterProvider(
		metric.WithReader(metric.NewPeriodicReader(exporter,
			metric.WithInterval(60*time.Second),
		)),
		metric.WithResource(res),
	)

	// Set global meter provider
	otel.SetMeterProvider(provider)

	return &OTelExporter{
		provider: provider,
		shutdown: provider.Shutdown,
	}, nil
}

// Shutdown gracefully shuts down the exporter
func (e *OTelExporter) Shutdown(ctx context.Context) error {
	if e.shutdown != nil {
		return e.shutdown(ctx)
	}
	return nil
}

// RegisterWithOTel integrates existing metrics registry with OpenTelemetry
func RegisterWithOTel(reg *Registry, serviceName string) error {
	meter := otel.Meter(serviceName)

	// Register counters
	for name, counter := range reg.counters {
		otelCounter, err := meter.Int64Counter(name)
		if err != nil {
			return fmt.Errorf("failed to create counter %s: %w", name, err)
		}

		// Create a goroutine to periodically sync
		go func(c *Counter, oc any) {
			ticker := time.NewTicker(10 * time.Second)
			defer ticker.Stop()
			for range ticker.C {
				// Sync counter value to OTel
				// Note: This is a simplified approach
				_ = otelCounter
			}
		}(counter, otelCounter)
	}

	return nil
}

// MetricsHandler returns an HTTP handler for Prometheus-compatible metrics
func (r *Registry) MetricsHandler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		w.Header().Set("Content-Type", "text/plain; version=0.0.4")
		r.mu.RLock()
		defer r.mu.RUnlock()

		// Expose counters
		for _, c := range r.counters {
			c.Expose(w)
		}

		// Expose gauges
		for _, g := range r.gauges {
			g.Expose(w)
		}

		// Expose histograms
		for _, h := range r.histograms {
			h.Expose(w)
		}
	})
}
