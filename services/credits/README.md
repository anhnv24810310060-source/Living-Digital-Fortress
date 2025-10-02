# Credits Service (PORT 5004)

High-performance credits ledger with strict invariants and production hardening.

## Endpoints
- POST /credits/purchase
- POST /credits/topup
- POST /credits/consume
- POST /credits/reserve
- POST /credits/commit
- POST /credits/cancel
- GET  /credits/balance/:tenant
- GET  /credits/history?tenant_id=...&limit=50
- GET  /credits/report?tenant_id=...
- POST /credits/threshold
- GET  /health
- GET  /metrics

Notes
- Supports Idempotency-Key header for POST endpoints.
- Immutable audit logs with HMAC chain, set AUDIT_HMAC_KEY in production.
- Optional Redis cache for hot balances (set REDIS_ADDR, REDIS_PASSWORD).

## Environment
- PORT (default 5004)
- DATABASE_URL (postgres URL)
- MIGRATE_ON_START=true to run built-in migrations
- BACKUP_BEFORE_MIGRATE=true to attempt a pg_dump before migrating
- BACKUP_CMD optional shell command to run instead of pg_dump
- CREDITS_API_KEY optional Bearer token for auth (health/metrics public)
- AUDIT_HMAC_KEY secret key for audit chain HMAC
- REDIS_ADDR, REDIS_PASSWORD for caching balances
- CREDITS_HTTP_PATH_* to tune metrics path normalization (see pkg/metrics)

## Constraints
- No negative balances; enforced at DB level and with guarded UPDATEs
- All credit movements occur within DB transactions
- Idempotent operations stored for 24h
- Reservation expiry is enforced by background scheduler

## Run (dev)
```bash
# DB
docker compose up -d
# Service
go run .
```

## Health
- GET /health returns JSON status
- GET /metrics exposes Prometheus text format metrics