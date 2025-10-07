# RA-TLS Rollout Guide

This guide summarizes how services enable mutual TLS with SPIFFE identities using the in-repo RA-TLS library.

Environment variables (defaults in parentheses):
- RATLS_ENABLE: "true" to enable RA-TLS
- RATLS_TRUST_DOMAIN (shieldx.local)
- RATLS_NAMESPACE (default)
- RATLS_SERVICE (<service-name>)
- RATLS_ROTATE_EVERY (45m)
- RATLS_VALIDITY (60m)

Wiring pattern:
- Create issuer: NewDevIssuer(Identity{TrustDomain, Namespace, Service}, rotate, validity)
- Server: server.TLSConfig = issuer.ServerTLSConfig(true, trustDomain)
- Client: transport.TLSClientConfig = issuer.ClientTLSConfig()
- Metric: emit gauge ratls_cert_expiry_seconds as seconds until LeafNotAfter()

Alerting:
- Prometheus rule RATLSCertExpiringSoon fires when ratls_cert_expiry_seconds < 600 for 5m

Notes:
- In production, replace Dev issuer with an attested issuer backed by secure enclave/TEE.
- Ensure all intra-mesh calls use https and mTLS once enabled.