# Ingress Security Controls

- TLS 1.3 enforced.
- mTLS required when RATLS_REQUIRE_CLIENT_CERT=true (default) or in static TLS with INGRESS_TLS_CLIENT_CA_FILE.
- Optional SAN allowlist via INGRESS_ALLOWED_CLIENT_SAN_PREFIXES (comma-separated prefixes: spiffe://..., dns-prefix).
- Fast deny filters:
  - INGRESS_DENY_PATH_PREFIXES=/admin,/internal
  - INGRESS_DENY_QUERY_KEYS=debug,token
- Rate limiting:
  - IP limiter (env: CONNECT_MAX_BYTES, LIMIT_PER_MIN, etc.)
  - Optional Redis (REDIS_ADDR) for distributed limits.

Metrics exposed on /metrics and health on /healthz.