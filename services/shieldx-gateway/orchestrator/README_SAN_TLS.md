# Orchestrator TLS 1.3 + mTLS with SAN allowlist

Environment variables:

- RATLS_ENABLE=true (recommended for dev/staging) or ORCH_TLS_CERT_FILE/ORCH_TLS_KEY_FILE with ORCH_TLS_CLIENT_CA_FILE
- RATLS_TRUST_DOMAIN=shieldx.local
- ORCH_ALLOWED_CLIENT_SAN_PREFIXES=spiffe://shieldx.local/ns/default/sa/ingress
- ORCH_LB_ALGO=ewma (round_robin|least_conn|ewma|p2c|rendezvous)
- ORCH_POOL_<SERVICE>=http://127.0.0.1:9090
- ORCH_POOL_ALGO_<SERVICE>=p2c
- ORCH_POOL_WEIGHTS_<SERVICE>={"http://127.0.0.1:9090":2.0}

The server enforces TLS 1.3 minimum. When SAN prefixes are set, only client certs whose SAN (URI/DNS/IP or CN) starts with one of the prefixes are accepted.