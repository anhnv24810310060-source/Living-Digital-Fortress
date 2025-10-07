# Modularization Roadmap

This document tracks the incremental extraction of independent service modules to reduce contributor conflicts and clarify boundaries.

## Goals
1. Independent build/test scopes for AI, Honeypot, Monitoring.
2. Clear contracts (events, metrics, APIs) instead of implicit package coupling.
3. Enable new contributors to work inside one service without touching orchestrator / core code.

## Phase Overview
| Phase | Description | Output |
|-------|-------------|--------|
| 1 | Introduce skeleton modules + go.work | ai-service, honeypot-service, monitoring-service, shared |
| 2 | Migrate pure library code (pkg/ml -> ai-service/internal/ml, pkg/metrics -> monitoring-service/internal/metrics) | Reduced root pkg surface |
| 3 | Extract HTTP / gRPC endpoints & config out of orchestrator into services | Independent runtime processes |
| 4 | Introduce message/event bus abstraction (if needed) | Decoupled async flows |
| 5 | Hardening: structured logging, tracing, load tests per service | Reliability, perf baselines |

## Current Mapping (Initial Pass)
| Current Path | Proposed Destination | Rationale |
|--------------|----------------------|-----------|
| `pkg/ml/*` | `services/ai-service/internal/ml` | ML inference logic isolated from orchestrator |
| `pkg/metrics/*` | `services/monitoring-service/internal/metrics` | Central metrics exporter |
| `pkg/guardian/*` | `services/honeypot-service/internal/guardian` | Runtime detection/honeypot protections |
| `services/decoy-*` | `services/honeypot-service/internal/decoys/*` | All decoy protocol implementations grouped |
| `pkg/deception/*` | `services/honeypot-service/internal/deception` | Unified deception primitives |
| `pkg/observability/*` | split: monitoring-service + shared/logging | Consolidate telemetry |
| `pkg/policy/*` | `shared/policy` (migrated) | Cross-cutting evaluation logic central |
| `pkg/auth/*` | `shared/auth` (migrated; future auth-service) | Separate auth concerns |
| `pkg/ledger/*` | `shared/ledger` (migrated) | Common append-only forensic log |

## Shared Module Scope
`shared` contains ONLY stable primitives: logging (temporary), config helper, core DTOs (TelemetryEvent), future error contracts. Avoid pulling high-churn logic here.

## Service Contracts (Draft)
| Service | Inbound API | Outbound | Notes |
|---------|-------------|----------|-------|
| ai-service | HTTP: `/analyze` (future), `/healthz` | Event bus / sync reply | Stateless inference |
| honeypot-service | Protocol listeners (future), HTTP mgmt `/healthz` | Events -> ai-service / monitoring | High I/O; isolation prevents cascading failures |
| monitoring-service | `/metrics`, `/healthz` | N/A | Gateway for Prometheus + future alert hooks |

## Migration Guidelines
1. Move files WITHOUT changing package names first; rename packages after tests pass.
2. Replace direct imports of `pkg/ml` with `shieldx/services/ai-service/internal/ml` gradually.
3. Guard each migration with: (a) build root, (b) build moved module, (c) integration smoke (docker-compose TBD).
4. Keep public APIs narrow—prefer internal packages inside each service.

## Developer Workflow (Phase 1)
```bash
go work sync                     # ensure workspace recognized
go build ./services/ai-service/... 
go build ./services/honeypot-service/...
go build ./services/monitoring-service/...
```

## Next Steps (After Phase 1 Merge)
- Implement internal/ml inside ai-service by moving `pkg/ml`.
- Create minimal `/analyze` endpoint.
- Migrate metrics exporter code.
- Add docker-compose targets for new services.

## Phase 2 Progress (ongoing)
Date: 2025-10-07

Completed:
- Moved `pkg/ml` -> `services/ai-service/internal/ml` (imports updated, tests target path updated)
- Moved `pkg/metrics` -> `shared/metrics` (original plan to put under monitoring-service/internal changed to avoid internal visibility violations across many services)
- Updated Makefile + deployment scripts referencing old paths
- Full workspace build green after refactor

Pending:
- (Done) Implement `/analyze` HTTP endpoint in ai-service using anomaly detector
- Expose real Prometheus registry via monitoring-service instead of placeholder
- Migrate any remaining observability helpers from `pkg/observability` once audited

Adjustments:
API Progress:
- ai-service now exposes: `/healthz`, `/train`, `/analyze` (JSON)

Honeypot Progress:
- Moved `pkg/deception` -> `services/honeypot-service/internal/deception`
- Added decoy placeholders: http(:7110), ssh(:7120), redis(:7130)
- Moved `pkg/guardian` -> `services/honeypot-service/internal/guardian`

Observability Progress:
- Moved `pkg/observability/otel` -> `shared/observability/otel`
- Moved `pkg/observability/logcorr` -> `shared/observability/logcorr`
- Moved `pkg/observability/slo` -> `services/monitoring-service/internal/slo`
- Updated all imports to new shared / service paths


- Metrics placed in `shared` as they are cross-cutting; monitoring-service will become the exporter/aggregator rather than sole owner of metrics primitives.

Wave Update (2025-10-07):
- Migrated auth → shared/auth
- Migrated policy → shared/policy
- Migrated ledger → shared/ledger
- Updated all imports from `shieldx/pkg/{auth,policy,ledger}` to `shieldx/shared/{auth,policy,ledger}`
- Added smoke tests: `shared/auth/auth_smoke_test.go`, `shared/ledger/ledger_test.go`
- `shared/go.mod` tidied to include new dependencies (OPA, Redis, JWT)

---
Maintained by: Modularization Working Group (update this section as owners change).
