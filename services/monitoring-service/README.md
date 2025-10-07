# monitoring-service

Purpose: Central metrics & (future) alert pipeline endpoint. Will ingest internal service stats & export Prometheus metrics.

Phase 1: Skeleton with `/metrics` & `/healthz`.
Future migration: `pkg/metrics`, parts of `observability`, tracing exporters.
