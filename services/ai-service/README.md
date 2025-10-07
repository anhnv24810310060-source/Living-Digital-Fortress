# ai-service

Purpose: Hosts ML inference endpoints (anomaly detection, threat scoring) decoupled from orchestration & honeypot capture layers.

Phase 1: Skeleton only. Phase 2 will migrate code from `pkg/ml` and related orchestrator glue.

Entrypoint: `cmd/ai-service/main.go` providing `/healthz` readiness endpoint.

Depends on: `shieldx/shared` for logging/config (future: metrics, tracing).
