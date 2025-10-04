# 🚀 Quick Start - PERSON 1 Phase 1 Enhancements

## Overview
Phase 1 enhances the **Orchestrator** and **Ingress** services with:
- 🔐 Post-Quantum Cryptography (Kyber-1024 + Dilithium-5)
- ⚡ Advanced QUIC (0-RTT, connection migration, BBR)
- 🔍 Certificate Transparency monitoring
- 🛡️ Adaptive rate limiting with ML
- 📋 Dynamic policy engine (hot-reload, ABAC)

---

## Prerequisites
```bash
# Go 1.21+
go version

# Redis (for distributed rate limiting - optional)
redis-server --version

# Git
git --version
```

---

## Installation

### 1. Clone & Build
```bash
git clone https://github.com/anhnv24810310060-source/Living-Digital-Fortress.git
cd Living-Digital-Fortress

# Build orchestrator with Phase 1 enhancements
make build-orchestrator
# OR
go build -o bin/orchestrator ./services/orchestrator/

# Verify build
./bin/orchestrator --version
```

### 2. Configuration

Create `config/orchestrator.env`:
```bash
# Service
ORCH_PORT=8080

# Post-Quantum Crypto
RATLS_ENABLE=true
RATLS_ROTATE_EVERY=24h
RATLS_VALIDITY=48h
RATLS_TRUST_DOMAIN=shieldx.local

# Certificate Transparency
CT_ENABLE=true
CT_CHECK_INTERVAL=60s
CT_MONITORED_DOMAINS=shieldx.local,*.shieldx.local

# Rate Limiting
RATELIMIT_BASE_RATE=100
RATELIMIT_ADAPT_ENABLED=true
RATELIMIT_LEARNING_RATE=0.1
RATELIMIT_MIN_RATE=10
RATELIMIT_MAX_RATE=10000

# Policy Engine
POLICY_PATH=/etc/shieldx/policy.json
POLICY_WATCH_ENABLED=true

# Redis (optional, for distributed limiting)
REDIS_ADDR=localhost:6379

# Backends
ORCH_POOL_GUARDIAN=http://localhost:9090
ORCH_POOL_INGRESS=http://localhost:8081
```

Create `config/policy.json`:
```json
{
  "tenants": [
    {
      "name": "default",
      "allow": ["*"],
      "deny": [],
      "riskLevel": "low"
    },
    {
      "name": "premium",
      "allow": ["*"],
      "deny": [],
      "riskLevel": "low"
    },
    {
      "name": "trial",
      "allow": ["read"],
      "deny": ["write", "admin"],
      "riskLevel": "medium"
    }
  ],
  "paths": [
    {"pattern": "/health", "action": "allow"},
    {"pattern": "/metrics", "action": "allow"},
    {"pattern": "/admin/*", "action": "deny"}
  ],
  "abacRules": [
    {
      "id": "high_risk_block",
      "priority": 100,
      "conditions": [
        {"attribute": "env.risk_score", "operator": "gt", "value": 0.8}
      ],
      "action": "deny"
    }
  ]
}
```

---

## Running

### Option 1: Standalone
```bash
# Export config
export $(cat config/orchestrator.env | xargs)

# Run orchestrator
./bin/orchestrator

# Expected output:
# [orchestrator] initializing Phase 1 enhancements...
# [orchestrator] ✓ PQC engine initialized (KEM pubkey: ...)
# [orchestrator] ✓ CT monitor started (checking 2 logs)
# [orchestrator] ✓ adaptive limiter initialized (multi-dimensional)
# [orchestrator] ✓ policy engine initialized (version=1)
# [orchestrator] ✅ Phase 1 initialization complete!
# [orchestrator] listening on :8080
```

### Option 2: Docker
```bash
# Build image
docker build -f docker/Dockerfile.orchestrator -t orchestrator:phase1 .

# Run container
docker run -d \
  --name orchestrator \
  -p 8080:8080 \
  -v $(pwd)/config:/etc/shieldx \
  -e POLICY_PATH=/etc/shieldx/policy.json \
  orchestrator:phase1
```

### Option 3: Docker Compose
```bash
# Start all services
docker-compose -f docker-compose.person1.yml up -d

# View logs
docker-compose logs -f orchestrator
```

---

## Testing

### 1. Health Check
```bash
curl -k https://localhost:8080/health

# Response:
# {
#   "service": "orchestrator",
#   "time": "2025-10-04T17:00:00Z",
#   "pools": {...},
#   "phase1": {
#     "pqc": {"encapsulations": 0, "signatures": 0, "rotations": 1},
#     "ct_monitor": {"checks_total": 5, "alerts_total": 0},
#     "rate_limit": {"allowed_total": 100, "rejected_total": 2},
#     "policy": {"evaluations_total": 50, "current_version": 1}
#   }
# }
```

### 2. Test Post-Quantum Crypto
```bash
# Get PQC public keys
curl -k https://localhost:8080/pqc/keys

# Response:
# {
#   "kem_public": "base64-encoded-kyber-public-key...",
#   "sig_public": "base64-encoded-dilithium-public-key...",
#   "algorithm": "kyber1024+dilithium5",
#   "hybrid_mode": "enabled"
# }
```

### 3. Test Rate Limiting
```bash
# Send 150 requests (should hit limit at 100)
for i in {1..150}; do
  curl -s -o /dev/null -w "%{http_code}\n" \
    -k https://localhost:8080/route \
    -H "Content-Type: application/json" \
    -d '{"service":"guardian","tenant":"default","scope":"read"}'
done

# Expected: 100x 200 OK, then 429 Too Many Requests
```

### 4. Test Dynamic Policy Hot-Reload
```bash
# Modify policy.json (e.g., add new tenant)
echo '{"tenants":[{"name":"new-tenant","allow":["*"],"deny":[]}]}' > /tmp/policy-v2.json

# Reload policy via API
curl -k -X POST https://localhost:8080/policy/reload \
  -H "Content-Type: application/json" \
  -d @/tmp/policy-v2.json

# Response:
# {"version": 2, "status": "loaded"}

# Verify
curl -k https://localhost:8080/policy
```

### 5. Test ABAC (Risk-Based)
```bash
# High risk request (should be denied)
curl -k -X POST https://localhost:8080/route \
  -H "Content-Type: application/json" \
  -d '{
    "service": "guardian",
    "tenant": "default",
    "scope": "admin",
    "risk_score": 0.9
  }'

# Response: 403 Forbidden (risk-based deny)
```

---

## Monitoring

### Metrics Endpoint
```bash
curl -k https://localhost:8080/metrics

# Key metrics:
# pqc_encapsulations_total
# pqc_signatures_total
# pqc_rotations_total
# ct_checks_total
# ct_alerts_total
# ratelimit_allowed_total
# ratelimit_rejected_total
# policy_evaluations_total
# policy_hot_reloads_total
```

### Prometheus Integration
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'orchestrator'
    scheme: https
    tls_config:
      insecure_skip_verify: true
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: /metrics
```

### Grafana Dashboard
Import dashboard: `dashboards/orchestrator-phase1.json`

Key panels:
- PQC operations/sec
- CT alerts timeline
- Rate limit rejection rate
- Policy evaluation latency
- Risk score distribution

---

## Troubleshooting

### Issue: Build fails with "undefined: certtransparency.Monitor"
**Solution:**
```bash
# Clean and rebuild
go clean -cache
go mod tidy
go build -o bin/orchestrator ./services/orchestrator/
```

### Issue: CT monitor shows "fetch STH error"
**Solution:**
```bash
# CT logs require internet access
# Check network connectivity
curl https://ct.googleapis.com/logs/us1/argon2024/ct/v1/get-sth

# Disable CT monitoring temporarily
export CT_ENABLE=false
```

### Issue: Rate limiting too aggressive
**Solution:**
```bash
# Increase base rate
export RATELIMIT_BASE_RATE=500

# Disable adaptive learning
export RATELIMIT_ADAPT_ENABLED=false

# Or adjust per-IP burst
export ORCH_IP_BURST=300
```

### Issue: Policy evaluation slow
**Solution:**
```bash
# Enable policy caching (default: 2s TTL)
export ORCH_OPA_CACHE_TTL=5s
export ORCH_OPA_CACHE_MAX=20000

# Reduce ABAC rule complexity
# Check policy.json for expensive conditions
```

---

## Performance Tuning

### For High Throughput (10K+ req/s)
```bash
# Increase worker goroutines
export GOMAXPROCS=8

# Tune connection pooling
export ORCH_MAX_IDLE_CONNS=1000
export ORCH_MAX_IDLE_CONNS_PER_HOST=100

# Optimize rate limiter
export RATELIMIT_BASE_RATE=1000
export RATELIMIT_WINDOW=10s  # Shorter window for high throughput

# Use Redis for distributed limiting
export REDIS_ADDR=redis-cluster:6379
```

### For Low Latency (<10ms p99)
```bash
# Disable expensive features
export CT_ENABLE=false  # CT monitoring adds ~5ms

# Optimize QUIC
export QUIC_CONGESTION_CONTROL=bbr  # BBR for low latency

# Policy cache tuning
export ORCH_OPA_CACHE_TTL=10s  # Longer cache TTL
```

---

## Security Best Practices

1. **TLS Configuration**
   ```bash
   # Use strong ciphers
   export TLS_MIN_VERSION=1.3
   export TLS_CIPHER_SUITES=TLS_AES_256_GCM_SHA384,TLS_CHACHA20_POLY1305_SHA256
   ```

2. **Certificate Pinning**
   ```go
   // In production, pin your domain certificates
   ctMonitor.PinCertificate("prod.shieldx.com", expectedFingerprint)
   ```

3. **Rate Limiting**
   ```bash
   # Set conservative defaults
   export RATELIMIT_BASE_RATE=100
   export RATELIMIT_MIN_RATE=10
   export RATELIMIT_MAX_RATE=1000
   ```

4. **Policy Security**
   ```bash
   # Restrict policy reload endpoint
   export ADMISSION_SECRET=your-secret-token
   export ADMISSION_HEADER=X-Admin-Token
   ```

---

## Next Steps

### Phase 2 (AI-Powered Traffic Intelligence)
- [ ] Implement GraphQL security
- [ ] Add transformer-based sequence analysis
- [ ] Enable federated learning
- [ ] Deploy adversarial training

### Phase 3 (Next-Gen Policy Engine)
- [ ] Continuous authorization validation
- [ ] Multi-cloud disaster recovery
- [ ] Zero-downtime deployment pipeline

---

## Support

**Documentation:** `/docs/`
**Issues:** GitHub Issues
**Slack:** #shieldx-dev

**Author:** PERSON 1 - Core Services & Orchestration Layer
**Date:** October 4, 2025
