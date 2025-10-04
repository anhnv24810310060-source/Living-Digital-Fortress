# 🚀 Quick Start - PERSON 1 Advanced Enhancements

**5-Minute Setup Guide** for Phase 1-3 Production Features

---

## 📋 Prerequisites

```bash
# Check Go version
go version  # Requires Go 1.21+

# Check system
uname -a    # Linux/macOS recommended
```

---

## ⚡ Quick Deploy

### Step 1: Environment Configuration

Create `.env` file:

```bash
cat > .env <<EOF
# ===== Phase 1: Quantum-Safe Security =====
PHASE1_ENABLE_PQC=true
PHASE1_PQC_ALGORITHM=hybrid               # kyber1024, dilithium5, hybrid, sphincs
PHASE1_PQC_ROTATION=24h
PHASE1_ENABLE_MULTI_SIG=true

# QUIC Enhancements
INGRESS_QUIC_ADDR=:4433
QUIC_ENABLE_0RTT=true
QUIC_ENABLE_MIGRATION=true
QUIC_ENABLE_MULTIPATH=true
QUIC_MAX_PATHS=4
QUIC_SCHEDULER=minrtt                     # roundrobin, minrtt, weighted

# Certificate Transparency
CT_MONITOR_ENABLE=true
CT_LOG_URLS=https://ct.googleapis.com/logs/argon2023/

# ===== Phase 2: AI-Powered Intelligence =====
# Behavioral Analytics
ANALYTICS_ENABLE=true
ANALYTICS_BUFFER_SIZE=10000
ANALYTICS_WINDOW_SIZE=1440
ANALYTICS_ANOMALY_THRESHOLD=3.0

# Adaptive Rate Limiting
RATELIMIT_ENABLE_ML=true
RATELIMIT_ADJUSTMENT_CYCLE=5m
RATELIMIT_IP_LIMIT=200
RATELIMIT_USER_LIMIT=500

# GraphQL Security
GRAPHQL_MAX_DEPTH=10
GRAPHQL_MAX_COMPLEXITY=1000
GRAPHQL_DISABLE_INTROSPECTION=true

# ===== Phase 3: Next-Gen Policy =====
# ABAC
ABAC_ENABLE=true
ABAC_RISK_THRESHOLD=0.7

# Continuous Authorization
CONTINUOUS_AUTH_ENABLE=true
CONTINUOUS_AUTH_INTERVAL=5m

# A/B Testing
ABTEST_ENABLE=true
ABTEST_TEST_TRAFFIC_PCT=0.1
ABTEST_AUTO_ROLLBACK=true

# ===== Core Services =====
ORCH_PORT=8080
ORCH_LB_ALGO=p2c                          # round_robin, least_conn, ewma, p2c
RATLS_ENABLE=true
RATLS_TRUST_DOMAIN=shieldx.local
EOF
```

### Step 2: Build

```bash
# Build orchestrator with all enhancements
make build-orchestrator

# Or build all services
make build
```

### Step 3: Start Services

```bash
# Terminal 1: Start orchestrator
./bin/orchestrator

# Terminal 2: Start ingress (if needed)
./bin/ingress

# Terminal 3: Start guardian (if needed)
./bin/guardian
```

---

## 🧪 Test Enhancements

### Test 1: Post-Quantum Cryptography

```bash
# Get PQC public keys
curl -k https://localhost:8080/pqc/keys

# Response:
{
  "kem_public": "base64-kyber-public-key...",
  "sig_public": "base64-dilithium-public-key...",
  "sphincs_public": "base64-sphincs-public-key...",  # NEW
  "algorithm": "hybrid",
  "multi_sig_enabled": true                            # NEW
}

# Test Kyber encapsulation
curl -X POST -k https://localhost:8080/pqc/encapsulate

# Test multi-signature (NEW)
curl -X POST -k https://localhost:8080/pqc/multi-sign \
  -H "Content-Type: application/json" \
  -d '{"message":"test message"}'
```

### Test 2: Multipath QUIC (NEW)

```bash
# Check QUIC server status
curl -k https://localhost:8080/quic/status

# Response:
{
  "enabled": true,
  "multipath": true,
  "active_paths": 2,
  "scheduler": "minrtt",
  "0rtt_enabled": true
}
```

### Test 3: Behavioral Analytics (NEW)

```bash
# Publish test events
for i in {1..100}; do
  curl -X POST https://localhost:8080/analytics/event \
    -H "Content-Type: application/json" \
    -d "{\"type\":\"request\",\"source\":\"192.168.1.$i\",\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}"
done

# Check analytics metrics
curl https://localhost:8080/analytics/metrics

# Response:
{
  "events_processed": 100,
  "anomalies_detected": 2,
  "bots_detected": 5,
  "ddos_detected": 0,
  "exfiltration": 0
}
```

### Test 4: Adaptive Rate Limiting (NEW)

```bash
# Test rate limiting with different risk levels
# Low risk (trusted IP)
curl https://localhost:8080/route \
  -H "X-Client-IP: 1.2.3.4" \
  -H "X-Risk-Level: low" \
  -d '{"service":"guardian"}'

# High risk (untrusted IP)
curl https://localhost:8080/route \
  -H "X-Client-IP: 5.6.7.8" \
  -H "X-Risk-Level: high" \
  -d '{"service":"guardian"}'

# Check rate limit metrics
curl https://localhost:8080/metrics | grep ratelimit
```

### Test 5: ABAC Policies (NEW)

```bash
# List ABAC policies
curl https://localhost:8080/abac/policies

# Evaluate ABAC policy
curl -X POST https://localhost:8080/abac/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "alice",
    "user_role": "admin",
    "resource_type": "database",
    "data_sensitivity": "confidential",
    "action_type": "read"
  }'

# Response:
{
  "action": "allow",
  "policy_id": "admin-sensitive",
  "risk_score": 0.3
}
```

### Test 6: A/B Testing (NEW)

```bash
# Start a policy A/B test
curl -X POST https://localhost:8080/abtest/experiments \
  -H "Content-Type: application/json" \
  -d '{
    "name": "strict-rate-limit-test",
    "description": "Test 50% stricter rate limits",
    "traffic_pct": 0.1,
    "duration": "1h"
  }'

# Check experiment metrics
curl https://localhost:8080/abtest/experiments/latest

# Stop experiment
curl -X POST https://localhost:8080/abtest/experiments/latest/stop
```

---

## 📊 Monitoring

### Prometheus Metrics

```bash
# All enhanced metrics
curl https://localhost:8080/metrics | grep -E "(pqc|quic|analytics|ratelimit|abac|abtest)"

# PQC metrics
pqc_encapsulations_total 1234
pqc_decapsulations_total 1200
pqc_signatures_total 567
pqc_multi_signatures_total 89        # NEW
pqc_rotations_total 12

# QUIC metrics
quic_accepts_total 5678
quic_0rtt_accepts_total 3456
quic_multipath_connections 234       # NEW
quic_failovers_total 5               # NEW

# Analytics metrics
analytics_events_processed_total 100000
analytics_anomalies_detected_total 45
analytics_bots_detected_total 123
analytics_ddos_detected_total 2

# Rate limiting metrics
ratelimit_allowed_total 98765
ratelimit_throttled_total 1234
ratelimit_adaptations_total 144      # NEW

# ABAC metrics
abac_evaluations_total 5432
abac_allow_decisions_total 4321
abac_deny_decisions_total 1111

# A/B testing metrics
abtest_experiments_total 5
abtest_active_experiments 1
abtest_rollbacks_total 1
```

### Health Check

```bash
curl https://localhost:8080/health

# Response:
{
  "service": "orchestrator",
  "status": "healthy",
  "phase1": {
    "pqc_enabled": true,
    "quic_multipath": true,
    "ct_monitoring": true
  },
  "phase2": {
    "analytics_enabled": true,
    "adaptive_ratelimit": true,
    "bot_detection": ">99.5%"
  },
  "phase3": {
    "abac_enabled": true,
    "continuous_auth": true,
    "abtest_active": 1
  },
  "uptime": "2h15m30s",
  "version": "1.0.0"
}
```

---

## 🔧 Common Configurations

### Production Setup

```env
# Maximum security
PHASE1_PQC_ALGORITHM=hybrid
PHASE1_ENABLE_MULTI_SIG=true
QUIC_ENABLE_MULTIPATH=true
GRAPHQL_DISABLE_INTROSPECTION=true
ABTEST_AUTO_ROLLBACK=true

# High performance
ORCH_LB_ALGO=p2c
QUIC_SCHEDULER=minrtt
RATELIMIT_ENABLE_ML=true
ANALYTICS_BUFFER_SIZE=50000
```

### Development Setup

```env
# Relaxed security for testing
PHASE1_PQC_ALGORITHM=hybrid
GRAPHQL_DISABLE_INTROSPECTION=false
ABTEST_TEST_TRAFFIC_PCT=0.5

# Verbose logging
LOG_LEVEL=debug
METRICS_ENABLE=true
```

### High-Traffic Setup

```env
# Optimize for throughput
ORCH_LB_ALGO=p2c
ANALYTICS_BUFFER_SIZE=100000
RATELIMIT_IP_LIMIT=500
QUIC_MAX_PATHS=8

# ML tuning
RATELIMIT_ADJUSTMENT_CYCLE=3m
RATELIMIT_TARGET_CPU=0.8
```

---

## 🐛 Troubleshooting

### Issue: PQC Initialization Failed

```bash
# Check logs
tail -f data/orchestrator-access.log | grep pqc

# Common fixes:
# 1. Ensure crypto libraries are available
# 2. Check key rotation interval
# 3. Verify file permissions on data/
```

### Issue: QUIC Connection Fails

```bash
# Check QUIC status
curl -k https://localhost:8080/quic/status

# Common fixes:
# 1. Ensure UDP port 4433 is open
# 2. Check firewall rules
# 3. Verify TLS certificates
```

### Issue: Analytics Buffer Full

```bash
# Increase buffer size
export ANALYTICS_BUFFER_SIZE=50000

# Or reduce event rate
export ANALYTICS_AGGREGATION=5m
```

### Issue: Rate Limit Too Aggressive

```bash
# Check current adjustment factor
curl https://localhost:8080/ratelimit/metrics

# Adjust thresholds
export RATELIMIT_TARGET_CPU=0.8
export RATELIMIT_TARGET_LATENCY=150

# Or disable ML adjustment
export RATELIMIT_ENABLE_ML=false
```

---

## 📚 Next Steps

1. **Load Testing**: Run `make load-test` to verify performance
2. **Security Audit**: Review logs in `data/orchestrator-security.log`
3. **Monitoring Setup**: Configure Prometheus + Grafana dashboards
4. **Production Deployment**: Follow [DEPLOYMENT.md](./DEPLOYMENT.md)

---

## 🎯 Key Performance Indicators

After deployment, monitor these KPIs:

```bash
# Latency (target: <150ms P99)
curl https://localhost:8080/metrics | grep http_request_duration

# Throughput (target: >10k req/s)
curl https://localhost:8080/metrics | grep http_requests_total

# Error rate (target: <1%)
curl https://localhost:8080/metrics | grep http_request_errors

# Bot detection (target: >99.5%)
curl https://localhost:8080/analytics/metrics | jq '.bots_detected'

# Quantum resistance
curl https://localhost:8080/pqc/keys | jq '.algorithm'
# Should return: "hybrid"
```

---

## 🆘 Getting Help

- **Logs**: `tail -f data/*.log`
- **Metrics**: http://localhost:8080/metrics
- **Health**: http://localhost:8080/health
- **Docs**: See `PERSON1_ADVANCED_FINAL_DELIVERY.md`

---

**Ready for Production?** ✅

Run final checklist:
```bash
make check-deployment
```

---

Built with ❤️ by PERSON 1 | October 4, 2025
