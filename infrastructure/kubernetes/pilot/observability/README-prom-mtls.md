Prometheus mTLS profile (prom-mtls)
===================================

This optional profile runs a second Prometheus instance that scrapes services over HTTPS with mutual TLS.

Included files
- docker-compose.prom-mtls.yml: launches prometheus-mtls on port 9091 using prometheus-scrape-mtls.yml
- prometheus-scrape-mtls.yml: HTTPS scrape jobs with TLS 1.3 and client certificates
- Expected files mounted inside the container:
  - /etc/prometheus/tls/client.crt
  - /etc/prometheus/tls/client.key
  - /etc/prometheus/tls/ca.crt

CA / issuer model in the demo
- The RA-TLS demo services use an in-memory CA per service instance by default (pkg/ratls.AutoIssuer).
- A generic Prometheus client certificate will not be trusted unless your services share a common CA (recommended for real deployments) or you relax client cert requirement (RATLS_REQUIRE_CLIENT_CERT=false) for testing.
- Treat prom-mtls as a template intended for environments with a shared RA-TLS CA across services.

How to use
1) Provision client certs trusted by your services (shared CA):
   - In production, issue a client cert for Prometheus from the same CA that signs your services.
   - Place client.crt, client.key, and ca.crt under pilot/observability/tls-prom/ (mounted read-only in the container).
2) Start the extra Prometheus instance alongside the default stack:
   - Include docker-compose.prom-mtls.yml in your compose invocation.
3) Ensure targets run with HTTPS and, if desired, RATLS_REQUIRE_CLIENT_CERT=true.

Demo-only alternative
- Without a shared CA, continue scraping via HTTP (default Prometheus at 9090), or temporarily set RATLS_REQUIRE_CLIENT_CERT=false on targets to confirm basic TLS connectivity (not for production).

Mint a Prometheus client cert (shared CA scenario)
If your services share an RA-TLS issuer, mint a client certificate for Prometheus from the same issuer. Outline:
- Create or reuse the shared issuer (trust domain: shieldx.local, namespace: default, service: prometheus)
- Serialize the leaf cert and private key to PEM as client.crt/client.key and export the CA to ca.crt
- Mount them to pilot/observability/tls-prom/

Notes
- Integrate Prometheus into the same issuing workflow as your workloads (SPIFFE/SPIRE or internal CA).
- Keep RATLS_REQUIRE_CLIENT_CERT=true in production.
- prometheus-scrape-mtls.yml is strict (TLS 1.3, insecure_skip_verify: false).

Troubleshooting
- Target down in mTLS Prometheus:
  - Check that Prometheus’s client cert chains to the target’s trusted CA
  - Verify server name/identity and trust domain policy (if enforced)
  - Temporarily relax RATLS_REQUIRE_CLIENT_CERT=false to isolate basic HTTPS issues
