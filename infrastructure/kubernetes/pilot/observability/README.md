# Observability Stack - Living Digital Fortress

## Overview

Comprehensive observability stack for monitoring, tracing, and SLO tracking across all services in the Living Digital Fortress platform.

## Components

### 1. OpenTelemetry Collector
- **Port**: 4318 (HTTP), 4317 (gRPC)
- **Purpose**: Unified telemetry collection and processing
- **Features**:
  - Trace collection and sampling (10%)
  - Metrics aggregation
  - Resource enrichment
  - Export to Prometheus and Tempo

### 2. Prometheus
- **Port**: 9090
- **Purpose**: Metrics storage and querying
- **Retention**: 30 days
- **Features**:
  - SLO recording rules
  - Alert rules
  - Service discovery

### 3. Grafana
- **Port**: 3000
- **Credentials**: admin / fortress123
- **Purpose**: Visualization and dashboards
- **Dashboards**:
  - SLO Overview
  - Service Health Matrix
  - Error Budget Tracking
  - Latency Distribution

### 4. Tempo
- **Port**: 3200
- **Purpose**: Distributed tracing backend
- **Retention**: 7 days
- **Features**:
  - Trace storage and querying
  - Service graph generation
  - Metrics from traces

### 5. Alertmanager
- **Port**: 9093
- **Purpose**: Alert routing and notification
- **Integrations**:
  - Slack (warnings and critical)
  - PagerDuty (critical only)

## Quick Start

### Start the Stack

```bash
make otel-up
```

This will start all observability services:
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000
- Tempo: http://localhost:3200
- OTLP Endpoint: http://localhost:4318

### Configure Services

Set environment variables for your services:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318"
export OTEL_SERVICE_NAME="your-service-name"
export OTEL_SERVICE_VERSION="1.0.0"
```

### Stop the Stack

```bash
make otel-down
```

## SLO Monitoring

### Target Services (October 2025)

1. **Ingress**
   - Availability: 99.9%
   - P95 Latency: 100ms
   - P99 Latency: 200ms

2. **ShieldX Gateway**
   - Availability: 99.9%
   - P95 Latency: 50ms
   - P99 Latency: 100ms

3. **ContAuth**
   - Availability: 99.95%
   - P95 Latency: 150ms
   - P99 Latency: 300ms

4. **Verifier Pool**
   - Availability: 99.9%
   - P95 Latency: 200ms
   - P99 Latency: 500ms

5. **ML Orchestrator**
   - Availability: 99.5%
   - P95 Latency: 500ms
   - P99 Latency: 1000ms

### Check SLO Status

```bash
make slo-check
```

## Instrumentation Guide

### Go Services

```go
import (
    "shieldx/pkg/observability/otel"
    "shieldx/pkg/observability/slo"
)

func main() {
    // Initialize OpenTelemetry
    cfg := otel.GetTracerConfigFromEnv("my-service", "1.0.0")
    shutdown, err := otel.InitTracerWithConfig(cfg)
    if err != nil {
        log.Fatal(err)
    }
    defer shutdown(context.Background())

    // Register SLO
    sloMgr := slo.NewSLOManager()
    serviceSLO := sloMgr.RegisterSLO(
        "my-service",
        0.999,                    // 99.9% availability
        100*time.Millisecond,     // P95 target
        200*time.Millisecond,     // P99 target
    )

    // Record requests
    start := time.Now()
    success := handleRequest()
    serviceSLO.RecordRequest(time.Since(start), success)
}
```

### Python Services

```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Initialize tracer
trace.set_tracer_provider(TracerProvider())
otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces")
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(otlp_exporter)
)

tracer = trace.get_tracer(__name__)

# Use in code
with tracer.start_as_current_span("operation"):
    # Your code here
    pass
```

## Metrics Reference

### Common Metrics

- `{service}_requests_total` - Total request count
- `{service}_requests_success` - Successful requests
- `{service}_requests_errors` - Failed requests
- `{service}_request_duration_seconds` - Request latency histogram

### eBPF Enhanced Metrics

- `ebpf_syscall_latency_seconds` - Syscall latency by service
- `ebpf_network_io_bytes` - Network I/O by container
- `ebpf_file_access_total` - File access operations

## Alert Rules

### Critical Alerts

- **SLOAvailabilityBreach**: Availability below target
- **ErrorBudgetExhausted**: Error budget will be depleted in < 1 hour
- **LatencyP99Breach**: P99 latency exceeds SLO

### Warning Alerts

- **ErrorBudgetLow**: Error budget below 20%
- **LatencyP95High**: P95 latency approaching target

## Dashboards

### SLO Overview Dashboard

View all service SLOs in one place:
- Current availability
- Error budget remaining
- Latency percentiles
- Traffic volume

### Service Health Matrix

Detailed view of each service:
- Request rate (RPS)
- Success rate
- Latency distribution
- Resource utilization

### Error Budget Tracking

Monitor error budget consumption:
- Budget burn rate (fast/slow)
- Historical consumption
- Projected exhaustion time

## Troubleshooting

### No Metrics Appearing

1. Check OTLP endpoint configuration
2. Verify network connectivity
3. Check OpenTelemetry Collector logs:
   ```bash
   docker logs otel-collector
   ```

### Traces Not Showing

1. Verify sampling rate (default 10%)
2. Check Tempo connectivity
3. Ensure trace context propagation

### High Cardinality Warnings

1. Review metric labels
2. Use label_replace in Prometheus
3. Adjust collector configuration

## Best Practices

1. **Sampling**: Use 10% sampling for traces in production
2. **Labels**: Keep cardinality < 1000 per metric
3. **Retention**: Balance storage vs. historical data needs
4. **Alerting**: Set up both fast and slow burn alerts
5. **Documentation**: Document custom metrics and dashboards

## Maintenance

### Weekly Tasks

- Review SLO compliance
- Check error budget consumption
- Analyze latency trends
- Update alert thresholds if needed

### Monthly Tasks

- Review storage usage
- Update retention policies
- Rotate credentials
- Update documentation

## Resources

- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/)
- [Grafana Dashboard Guide](https://grafana.com/docs/grafana/latest/dashboards/)
- [SLO Implementation Guide](../docs/slo-dashboard.md)

---

**Last Updated**: October 1, 2025  
**Maintained By**: Platform Team  
**Status**: Production Ready
