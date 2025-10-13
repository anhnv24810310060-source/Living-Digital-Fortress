# Ingress Redirect Setup Guide (Open Source)

This guide explains how to configure “old → new” host redirects at the ShieldX Ingress, including security checks and end-to-end deployment steps.

## 1. Concepts

- old: The hostname that customers point DNS to (lands at Ingress). Example: `old.com`, `a.tenant.io`.
- new: The target hostname customers should be redirected to after security checks. Example: `new.com`, `portal.customer.com`.
- Redirect type: HTTP 3xx (default 307). The client receives a Location header to `new` and makes a new request there. Path/query/fragment are preserved.

## 2. Prerequisites

- Go toolchain matching `go.mod` (>= 1.25). For production use Docker or Kubernetes.
- TLS/mTLS configured for Ingress (RA-TLS or static certs) as per project docs.
- Optional: Redis for rate-limiting and state replication.

## 3. Environment variables for Redirect

- INGRESS_REDIRECT: Comma-separated rules `from->to`.
  - `from`: exact host (e.g., `old.com`), wildcard (`*.tenant.io`), or `default`.
  - `to`: target hostname (no scheme/port). Scheme is controlled by `INGRESS_REDIRECT_SCHEME`.
- INGRESS_REDIRECT_SCHEME: `https` (default) or `http`.
- INGRESS_REDIRECT_CODE: HTTP code, default `307` (supports 301/302/307).
- INGRESS_REDIRECT_ALLOW: allowlist of target hosts. If omitted, it is derived from all `to` hosts and the `default` target.

Security-first:
- Always set `INGRESS_REDIRECT_ALLOW` in production to avoid open redirects.
- Redirect occurs after admission HMAC and denylist checks.

## 4. Minimal local run

```bash
cd services/shieldx-gateway/ingress
export INGRESS_PORT=8081
export INGRESS_REDIRECT="old.com->new.com, *.tenant.io->portal.customer.com, default->landing.customer.com"
export INGRESS_REDIRECT_SCHEME=https
export INGRESS_REDIRECT_CODE=307
export INGRESS_REDIRECT_ALLOW="new.com,portal.customer.com,landing.customer.com"
# Optional: disable admission for local test
unset ADMISSION_SECRET

# Run (requires Go >= 1.25)
go run .
```

Verify:
```bash
curl -sI -H 'Host: old.com' http://127.0.0.1:8081/path?a=1 | sed -n '1p;/^Location:/p'
# Expect: 307 ... Location: https://new.com/path?a=1

curl -sI -H 'Host: a.tenant.io' http://127.0.0.1:8081/ | sed -n '1p;/^Location:/p'
# Expect: 307 ... Location: https://portal.customer.com/

curl -sI -H 'Host: other.io' http://127.0.0.1:8081/ | sed -n '1p;/^Location:/p'
# Expect: 307 ... Location: https://landing.customer.com/
```

## 5. Docker Compose deployment

Add env to the `ingress` service and bring it up:

```yaml
services:
  ingress:
    image: yourrepo/shieldx-ingress:latest
    ports:
      - "8081:8081"
    environment:
      - INGRESS_PORT=8081
      - INGRESS_REDIRECT=old.com->new.com, *.tenant.io->portal.customer.com, default->landing.customer.com
      - INGRESS_REDIRECT_SCHEME=https
      - INGRESS_REDIRECT_CODE=307
      - INGRESS_REDIRECT_ALLOW=new.com,portal.customer.com,landing.customer.com
      # Security (recommended):
      - RATLS_ENABLE=true
      - RATLS_REQUIRE_CLIENT_CERT=true
      - INGRESS_ALLOWED_CLIENT_SAN_PREFIXES=shieldx-client:,workload:
      - ADMISSION_SECRET_FILE=/run/secrets/admission_hmac
    secrets:
      - admission_hmac
secrets:
  admission_hmac:
    file: ./secrets/admission_hmac.txt
```

Then:
```bash
docker compose up -d ingress
docker logs -f ingress
```

Test from your machine with Host header (simulating DNS):
```bash
curl -sI -H 'Host: old.com' http://127.0.0.1:8081/
```

## 6. Kubernetes (outline)

- Create a Deployment for the ingress container with the same env vars.
- Mount secrets (Admission HMAC, TLS if using static certs) via Secret objects.
- Expose via Service/LoadBalancer and set DNS A/CAA records accordingly.
- NetworkPolicies should restrict `/metrics` and inter-service traffic.

## 7. Hardening checklist (prod)

- [ ] Set `INGRESS_REDIRECT_ALLOW`
- [ ] Enable RA-TLS or static mTLS with client CA (no plaintext HTTP)
- [ ] Configure Admission HMAC with short replay window and nonce
- [ ] Configure OPA policies and rate limiting
- [ ] Trust only `X-Forwarded-For` from trusted proxies (if any)
- [ ] Pin container image versions; scan with Trivy

## 8. Troubleshooting

- No redirect happening:
  - Check `INGRESS_REDIRECT` syntax and logs; ensure env is visible to the process.
  - Ensure the host header in the request matches a rule or `default`.
- 403/401 instead of redirect:
  - Admission/denylist/rate-limit may be blocking. Disable locally or adjust configuration.
- Wrong Location scheme:
  - Set `INGRESS_REDIRECT_SCHEME=https`.
- Open redirect warning:
  - Add `INGRESS_REDIRECT_ALLOW` with explicit allowed targets.

## 9. Example rule sets

- Single host migration:
  - `INGRESS_REDIRECT="old.com->new.com"`
- Multi-tenant consolidation:
  - `INGRESS_REDIRECT="*.tenant.io->portal.customer.com, default->landing.customer.com"`
- Blue/green by host:
  - `INGRESS_REDIRECT="app.example.com->app-green.example.com, default->app-blue.example.com"`

