# honeypot-service

Purpose: Runs decoy protocol listeners & session/event normalization before forwarding to pipeline / AI.

Phase 1: Skeleton only. Future migration sources: `services/decoy-*`, `pkg/deception`, relevant parts of `pkg/guardian`.

Provides `/healthz` readiness endpoint (port 7020).
