# Observability & SLO Dashboard Guide

This guide captures the minimum viable observability stack for v1.0-alpha. It focuses on the five critical services in the roadmap and explains how to wire their metrics, traces, error budgets, and dashboards.

## Scope

- **Services covered:** `ingress`, `shieldx-gateway`, `contauth`, `verifier-pool`, `ml-service`
- **Signals:** request rate, latency (P95/P99), error rate, saturation (selected gauges), trace correlation
- **Destinations:** Prometheus (metrics), OpenTelemetry Collector (traces), Grafana dashboards and SLO board

## Golden Signals & SLOs

| Service | SLO Type | Target | Measurement | Notes |
|---------|----------|--------|-------------|-------|
| ingress | Availability | 99.9% monthly | `1 - (errors_total / requests_total)` | 5xx + policy deny counted as "error" |
| shieldx-gateway | Latency | P99 < 250 ms | `histogram_quantile(0.99, rate(shieldx_gateway_http_request_duration_seconds_bucket[5m]))` | track by route group |
| contauth | Latency | P95 < 350 ms | `histogram_quantile(0.95, rate(contauth_http_request_duration_seconds_bucket[5m]))` | exclude `/healthz` |
| verifier-pool | Availability | 99.95% | same formula as ingress | treat verifier errors as 5xx |
| ml-service | Freshness | 95% ingest < 2 s | trace span attribute `ml.pipeline.duration_ms` + metric `mlservice_http_request_duration_seconds` | combine traces + histogram |

### Error Budget Policy Hooks

- Error budgets and burn-rate alert formats live in `pilot/docs/error-budget-policy.md`.
- For each service, instrument two alerts:
  - **Fast burn:** budget exhausted in < 1 hour → PagerDuty.
  - **Slow burn:** budget exhausted in < 6 hours → Slack + ticket.

## Metrics Sources

| Component | Endpoint | Key Metrics |
|-----------|----------|-------------|
| ingress | `:8080/metrics` | `ingress_http_requests_total`, `ingress_http_errors_total`, `ingress_http_request_duration_seconds_bucket`, WG gauges |
| shieldx-gateway | `:8443/metrics` | `shieldx_gateway_http_requests_total`, integration counters, path histograms |
| contauth | `:8082/metrics` | `contauth_http_requests_total`, authentication outcome counters |
| verifier-pool | `:8084/metrics` | `verifier_pool_http_requests_total`, cryptographic latencies |
| ml-service | `:5000/metrics` | `mlservice_http_requests_total`, `mlservice_http_request_duration_seconds` |

### OpenTelemetry Tracing

- Go services wrap handlers with `otelobs.WrapHTTPHandler` and call `InitTracer(serviceName)`. Set `OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318` to enable.
- `ml-service` now performs optional tracing: set the same environment variable (and optional `OTEL_EXPORTER_OTLP_HEADERS`) to stream Flask spans and outbound requests.
- Traces carry span attributes:
  - `http.method`, `http.target`, `http.status_code`
  - `shieldx.slo.route_group` (supplied by middleware labels when present)
  - `ml.pipeline.duration_ms` for feature ingestion

## Prometheus Recipes

```promql
# Request SLO error rate (ingress example)
sum(rate(ingress_http_errors_total[5m]))
  /
sum(rate(ingress_http_requests_total[5m]))

# P99 latency (shieldx-gateway)
histogram_quantile(0.99,
  sum by (le) (rate(shieldx_gateway_http_request_duration_seconds_bucket[5m]))
)

# Error budget burn (fast) for 99.9% SLO across 1h window
max(
  (1 - (1 - sum(rate(ingress_http_errors_total[1m])) / sum(rate(ingress_http_requests_total[1m])))) / (1 - 0.999)
)
```

### Recording Rules

Create recording rules (Prometheus or Collector) for:

- `service:slo_error_ratio:rate5m`
- `service:latency_p95:rate5m`
- `service:error_budget_burn_fast`
- `service:error_budget_burn_slow`

This keeps dashboard panels light and shares the same definitions with alert rules.

## Grafana Dashboard Layout

1. **Top bar:** live SLI (availability %, P95, P99), remaining error budget (hours).
2. **Golden signals per service:** stacked panels with rate, latency histogram, saturation gauge.
3. **Error budget burn chart:** overlay fast/slow burn and error annotations.
4. **Trace exemplar panel:** embed Jaeger/Tempo link using trace IDs emitted by `HTTPTraceLogMiddleware`.
5. **ML ingestion freshness:** combined metric/trace view to spot slow feature writes.

Use templates for `service`, `route_group`, and (future) `tenant` labels.

## Collector Configuration Snippet

```yaml
receivers:
  otlp:
    protocols:
      http:
        endpoint: 0.0.0.0:4318
exporters:
  prometheus:
    endpoint: 0.0.0.0:9464
  otlp/jaeger:
    endpoint: tempo:4318
service:
  pipelines:
    metrics:
      receivers: [otlp]
      exporters: [prometheus]
    traces:
      receivers: [otlp]
      exporters: [otlp/jaeger]
```

## Runbook Checklist

- [ ] Confirm `/metrics` endpoint responds and exposes HTTP histograms.
- [ ] Deploy with `OTEL_EXPORTER_OTLP_ENDPOINT` set whenever tracing is required.
- [ ] Spot-check Jaeger/Tempo for exemplar traces after deployments.
- [ ] Keep `Nhật Ký Cập Nhật.md` aligned with dashboard changes.
- [ ] Export Grafana JSON when panels change; version in infra repo.

## Revision History

| Date | Change |
|------|--------|
| 2025-10-01 | Refocused document on SLO-first dashboard and documented OTEL-enabled ml-service. |
| 2024-01-15 | Legacy business KPI sheet (superseded). |
