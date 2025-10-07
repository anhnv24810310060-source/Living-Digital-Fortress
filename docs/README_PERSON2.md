# PERSON 2: Security & ML Services

This document summarizes Guardian, Continuous Authentication, and ML Orchestrator service endpoints, security constraints, and production toggles.

## Guardian (PORT 9090)
- Endpoints:
  - POST /guardian/execute
  - GET  /guardian/status/:id
  - GET  /guardian/report/:id
  - GET  /wch/pubkey, POST /wch/recv, POST /wch/recv-udp
  - GET  /health, GET /metrics
- Isolation & sandbox
  - Default: env runner via `pkg/sandbox` (noop unless SANDBOX_DOCKER=1)
  - Optional Firecracker backend: set `GUARDIAN_SANDBOX_BACKEND=firecracker`, `FC_KERNEL_PATH`, `FC_ROOTFS_PATH`.
    - Limits via env: `FC_VCPU` (default 1), `FC_MEM_MIB` (default 128), `FC_TIMEOUT_SEC` (default 30). Network is denied and filesystem is read-only by default.
- Threat scoring
  - Uses sandbox-provided score if available (Firecracker 0..100 normalized to [0,1]).
  - Falls back to heuristic on output length and patterns.

## Continuous Authentication (PORT 5002)
- Endpoints:
  - POST /contauth/telemetry (alias: /contauth/collect)
  - POST /contauth/score
  - GET  /contauth/decision?session_id=...
  - GET  /health, GET /metrics
- Security & privacy
  - No raw biometric sequences persisted; only hashed signatures and aggregate summaries are stored in metadata.
  - Encryption at rest via `pkg/security/cryptoatrest` with `CONTAUTH_ENC_KEY` (32B raw/hex/base64). Service fails init if missing (unless `DISABLE_DB=true`).
  - Per-IP rate limiting via `CONTAUTH_RL_REQS_PER_MIN` (default 240 req/min).
  - Body size guards: telemetry <= 1MB, score <= 256KB.
- Baseline learning
  - Baselines auto-updated from aggregated metadata across last 100 sessions, stored in `user_baselines`.

## ML Orchestrator (default PORT 8087)
- Endpoints:
  - POST /analyze (rate limited)
  - POST /train [admin]
  - POST /model/save [admin], POST /model/load [admin]
  - POST /model/version/save [admin], POST /model/version/rollback [admin], GET /model/version/list [admin]
  - GET  /model/mode [admin], POST /model/mode [admin]
  - GET  /health, GET /metrics
- Ensemble detector
  - Robust Mahalanobis + Isolation Forest with adjustable `ML_ENSEMBLE_WEIGHT`.
  - Admin endpoints require header `X-Admin-Token` matching `ML_API_ADMIN_TOKEN`.

## Production tips
- Ensure RA-TLS by setting `RATLS_ENABLE=true` and trust domain envs for ContAuth and ML Orchestrator.
- Configure `MLO_URL` in ContAuth to integrate anomaly scores.
- Export metrics to Prometheus via /metrics; add Grafana dashboards using provided metric names.
# PERSON 2: Security & ML Services

This document summarizes the Guardian, ContAuth, and ML Orchestrator services wiring and production constraints implemented.

## Guardian (PORT 9090)
- Endpoints:
  - POST /guardian/execute
  - GET /guardian/status/:id
  - GET /guardian/report/:id
  - GET /healthz, GET /metrics
- Isolation: uses pkg/sandbox.NewFromEnv. Default build uses a safe noop runner; enable Docker runner by setting SANDBOX_DOCKER=1 and Docker available. Timeout hard-capped at 30s.
- Threat scoring: heuristic on output; does not expose internals in report (hash + preview).

Run:
  GUARDIAN_PORT=9090 go run ./services/guardian

## Continuous Authentication (PORT 5002)
- Endpoints:
  - POST /contauth/collect (alias of /contauth/telemetry)
  - POST /contauth/score
  - GET  /contauth/decision?session_id=...
  - GET  /health, GET /metrics
- Data-at-rest encryption: AES-256-GCM via pkg/security/cryptoatrest; requires env CONTAUTH_ENC_KEY (32-byte raw/base64/hex). In NO-DB mode (DISABLE_DB=true) encryption is bypassed and handlers are no-op.
- Privacy: raw keystroke and mouse sequences are not stored. Only hashed signatures and aggregate features are kept in metadata.

Run (no DB):
  DISABLE_DB=true PORT=5002 go run ./services/contauth

Run (with DB and encryption):
  export DATABASE_URL=postgres://contauth_user:contauth_pass2024@localhost:5432/contauth
  export CONTAUTH_ENC_KEY=$(head -c 32 /dev/urandom | hexdump -e '32/1 "%02x"')
  PORT=5002 go run ./services/contauth

## ML Orchestrator
No changes required here; provides anomaly detection endpoints and model persistence. See services/ml-orchestrator.

## Constraints enforced
- No untrusted code runs outside sandbox.
- No raw biometric sequences persisted; only feature hashes/summaries.
- Telemetry encrypted at rest (production mode).
- Guardian jobs timeout at 30s.
- No model internals exposed via API.
