# ADR 0001: Observability SLOs and Alerting

Date: 2025-10-02

## Status
Accepted

## Context
We need production-grade observability with clear SLOs, Prometheus metrics, and alerting rules that signal user-impacting issues timely and with low noise.

## Decision
- Adopt Prometheus for metrics, Grafana for visualization, and OTEL for traces.
- Define service SLOs (availability, latency) and encode alerting in `pilot/observability/alert-rules.yml`.
- Each service exposes `/metrics` and basic health endpoints.
- Keep alerts simple: error ratio, high latency, certificate expiry.

## Consequences
- Consistent metrics taxonomy across services.
- SLO dashboards can be built from the provided counters and histograms.
- Alert rules evolve with usage; initial thresholds are conservative and can be tuned per environment.
