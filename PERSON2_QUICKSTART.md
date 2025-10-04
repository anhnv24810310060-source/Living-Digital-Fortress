# PERSON 2: Security & ML Services - Quick Start Guide

**Last Updated:** 2025-10-04  
**Status:** Production Ready ✅

---

## 🚀 Quick Start (5 Minutes)

### Prerequisites
```bash
# Check Go version (>= 1.21 required)
go version

# Check Docker (optional, for full stack)
docker --version

# Install dependencies
go mod download
```

---

## 🎯 Option 1: Test Individual Services

### 1. Guardian Service (Port 9090)

#### Start Guardian:
```bash
cd /workspaces/Living-Digital-Fortress

# Set environment
export GUARDIAN_PORT=9090
export GUARDIAN_CREDITS_URL=http://localhost:5004

# Build and run
go build -o bin/guardian ./services/guardian
./bin/guardian
```

#### Test Endpoints:
```bash
# Health check
curl http://localhost:9090/health
# Response: ok

# Execute payload in sandbox (safe test)
curl -X POST http://localhost:9090/guardian/execute \
  -H "Content-Type: application/json" \
  -d '{
    "payload": "echo hello",
    "tenant_id": "test-tenant",
    "cost": 10
  }'
# Response: {"id":"j-1","status":"queued",...}

# Get execution status
curl http://localhost:9090/guardian/status/j-1
# Response: {"id":"j-1","status":"done","threat_score_100":15,...}

# Get full report
curl http://localhost:9090/guardian/report/j-1
# Response: Detailed threat analysis with eBPF features
```

#### Test Malicious Payload Detection:
```bash
# Test shell injection (should score high)
curl -X POST http://localhost:9090/guardian/execute \
  -H "Content-Type: application/json" \
  -d '{
    "payload": "/bin/sh -c \"rm -rf /\"",
    "tenant_id": "test-tenant"
  }'

# Check score (should be 70-90)
curl http://localhost:9090/guardian/report/j-2
```

#### View Metrics:
```bash
# Prometheus metrics
curl http://localhost:9090/metrics | grep guardian
```

---

### 2. ContAuth Service (Port 5002)

#### Start ContAuth:
```bash
cd /workspaces/Living-Digital-Fortress

# Set environment
export PORT=5002
export DATABASE_URL=postgres://contauth_user:contauth_pass2024@localhost:5432/contauth
export DISABLE_DB=true  # For quick testing without DB

# Build and run
go build -o bin/contauth ./services/contauth
./bin/contauth
```

#### Test Endpoints:
```bash
# Health check
curl http://localhost:5002/health
# Response: {"status":"healthy","service":"contauth"}

# Collect telemetry (privacy-preserving)
curl -X POST http://localhost:5002/contauth/collect \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "session_id": "sess-abc",
    "keystroke_dwell_times": [100, 120, 110, 105, 115],
    "keystroke_flight_times": [50, 55, 48, 52],
    "mouse_movements": [
      {"delta_x": 10, "delta_y": 5, "speed": 150},
      {"delta_x": -5, "delta_y": 8, "speed": 120}
    ],
    "device_fingerprint": "device-fp-123",
    "time_of_day": 14
  }'
# Response: {"status":"collected"}

# Calculate risk score
curl -X POST http://localhost:5002/contauth/score \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "session_id": "sess-abc",
    "keystroke_dwell_times": [100, 120, 110],
    "device_fingerprint": "device-fp-123"
  }'
# Response: {"score":25,"risk_level":"low","confidence":0.3,...}

# Get authentication decision
curl -X GET "http://localhost:5002/contauth/decision?user_id=user123"
# Response: {"decision":"allow","confidence":0.7,"risk_score":25,...}
```

#### Test Anomaly Detection:
```bash
# Normal behavior
curl -X POST http://localhost:5002/contauth/collect \
  -d '{"user_id":"user456","keystroke_dwell_times":[100,105,102,108]}'

# Collect 5+ samples to build baseline
for i in {1..5}; do
  curl -X POST http://localhost:5002/contauth/collect \
    -d '{"user_id":"user456","keystroke_dwell_times":[100,105,102,108]}'
  sleep 0.5
done

# Test anomalous behavior (very different typing speed)
curl -X POST http://localhost:5002/contauth/score \
  -d '{"user_id":"user456","keystroke_dwell_times":[300,320,310,305]}'
# Response: Higher risk score due to deviation
```

---

### 3. ML Orchestrator (Port 8083)

#### Start ML Orchestrator:
```bash
cd /workspaces/Living-Digital-Fortress

# Set environment
export ML_PORT=8083
export ML_ENSEMBLE_WEIGHT=0.6
export ML_AB_PERCENT=10
export ML_MODEL_STORAGE=/tmp/shieldx-models

# Build and run
go build -o bin/ml-orchestrator ./services/ml-orchestrator
./bin/ml-orchestrator
```

#### Test Endpoints:
```bash
# Health check
curl http://localhost:8083/health
# Response: ok

# Analyze telemetry
curl -X POST http://localhost:8083/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "source": "test",
    "tenant_id": "t1",
    "features": [0.5, 0.8, 0.3, 0.1, 0.9],
    "threat_score": 0.6
  }'
# Response: {"is_anomaly":false,"score":0.42,"confidence":0.85}

# Train anomaly detector
curl -X POST http://localhost:8083/train \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [
      {"features": [0.1, 0.2, 0.3, 0.4]},
      {"features": [0.15, 0.22, 0.28, 0.38]},
      {"features": [0.12, 0.19, 0.31, 0.41]}
    ]
  }'
# Response: {"status":"trained","samples":3}

# View metrics
curl http://localhost:8083/metrics | grep ml_
```

#### Test Model Management:
```bash
# Register new model
curl -X POST http://localhost:8083/model/register \
  -H "Content-Type: application/json" \
  -d '{
    "name": "anomaly_detector_v2",
    "algorithm": "isolation_forest",
    "accuracy": 0.95
  }'

# List models
curl http://localhost:8083/model/list

# Activate model
curl -X POST http://localhost:8083/model/activate \
  -d '{"version":"v1696435200","deployed_by":"engineer@shieldx.io"}'

# Start A/B test
curl -X POST http://localhost:8083/model/abtest/start \
  -d '{
    "model_a": "v1696435200",
    "model_b": "v1696435300",
    "traffic_split": 0.10,
    "min_samples": 100
  }'

# Evaluate A/B test
curl http://localhost:8083/model/abtest/evaluate
```

---

## 🎯 Option 2: Test Integration (Full Stack)

### Start All Services:
```bash
# Terminal 1: Guardian
export GUARDIAN_PORT=9090
go run ./services/guardian/main.go

# Terminal 2: ContAuth
export PORT=5002
export DISABLE_DB=true
go run ./services/contauth/main.go

# Terminal 3: ML Orchestrator
export ML_PORT=8083
go run ./services/ml-orchestrator/main.go
```

### End-to-End Test:
```bash
# 1. Collect user telemetry (ContAuth)
curl -X POST http://localhost:5002/contauth/collect \
  -d '{"user_id":"e2e-user","keystroke_dwell_times":[100,105,102]}'

# 2. Calculate risk score (ContAuth)
RISK=$(curl -s -X POST http://localhost:5002/contauth/score \
  -d '{"user_id":"e2e-user"}' | jq -r '.score')
echo "Risk Score: $RISK"

# 3. If risk is acceptable, execute in sandbox (Guardian)
if [ "$RISK" -lt 60 ]; then
  JOB_ID=$(curl -s -X POST http://localhost:9090/guardian/execute \
    -d '{"payload":"echo test","tenant_id":"e2e-user"}' | jq -r '.id')
  echo "Job ID: $JOB_ID"
  
  # 4. Get threat score (Guardian)
  sleep 2
  THREAT=$(curl -s http://localhost:9090/guardian/report/$JOB_ID | jq -r '.threat_score_100')
  echo "Threat Score: $THREAT"
  
  # 5. Analyze with ML (ML Orchestrator)
  curl -X POST http://localhost:8083/analyze \
    -d "{\"threat_score\":$THREAT,\"features\":[]}"
fi
```

---

## 🧪 Run Test Suite

### Unit Tests:
```bash
# All tests
go test ./pkg/... -v

# Specific packages
go test ./pkg/ebpf -v
go test ./pkg/guardian -v
go test ./pkg/contauth -v
go test ./pkg/ml -v

# With coverage
go test ./pkg/... -cover -coverprofile=coverage.out
go tool cover -html=coverage.out
```

### Benchmarks:
```bash
# eBPF performance
go test ./pkg/ebpf -bench=. -benchmem

# Threat scoring performance
go test ./pkg/guardian -bench=. -benchmem

# Auth scoring performance
go test ./pkg/contauth -bench=. -benchmem
```

### Integration Tests:
```bash
# Ensure services are running, then:
cd /workspaces/Living-Digital-Fortress

# Guardian integration
./scripts/test_guardian_integration.sh

# ContAuth integration  
./scripts/test_contauth_integration.sh

# ML integration
./scripts/test_ml_integration.sh
```

---

## 📊 Monitoring & Observability

### Prometheus Metrics:
```bash
# Guardian metrics
curl http://localhost:9090/metrics | grep -E "(guardian_|ebpf_)"

# ContAuth metrics
curl http://localhost:5002/metrics | grep "contauth_"

# ML metrics
curl http://localhost:8083/metrics | grep "ml_"
```

### Key Metrics to Watch:
```
# Guardian
guardian_jobs_created_total
guardian_jobs_timeout_total
guardian_threats_blocked_total
ebpf_dangerous_syscalls_total

# ContAuth
contauth_risk_score_high_total
contauth_anomaly_detected_total
contauth_auth_denied_total

# ML
ml_predictions_total
ml_inference_latency_seconds
ml_anomaly_detected_total
```

### Logs:
```bash
# Guardian logs
tail -f data/ledger-guardian.log

# ContAuth logs
tail -f data/ledger-contauth.log

# All security events
tail -f data/ledger-*.log | grep -E "(THREAT|ANOMALY|DENIED)"
```

---

## 🐛 Troubleshooting

### Guardian Issues:

**Issue**: Sandbox timeout
```bash
# Increase timeout
export GUARDIAN_TIMEOUT_SEC=60

# Check job status
curl http://localhost:9090/guardian/status/j-123
```

**Issue**: High memory usage
```bash
# Reduce job TTL
export GUARDIAN_JOB_TTL_SEC=300

# Check current jobs
curl http://localhost:9090/guardian/jobs
```

### ContAuth Issues:

**Issue**: Risk score always 50 (insufficient baseline)
```bash
# Collect more samples (need 5+ for baseline)
for i in {1..10}; do
  curl -X POST http://localhost:5002/contauth/collect \
    -d '{"user_id":"testuser","keystroke_dwell_times":[100,105,102]}'
done

# Then try scoring again
```

**Issue**: Database connection failed
```bash
# Use in-memory mode for testing
export DISABLE_DB=true

# Or check database connection
psql $DATABASE_URL -c "SELECT 1"
```

### ML Issues:

**Issue**: Model not found
```bash
# List available models
curl http://localhost:8083/model/list

# Register a new model
curl -X POST http://localhost:8083/model/register \
  -d '{"name":"detector","algorithm":"iforest"}'
```

**Issue**: High inference latency
```bash
# Check metrics
curl http://localhost:8083/metrics | grep ml_inference_latency

# Consider model optimization or caching
```

---

## 🎓 Best Practices

### Security:
1. ✅ **Always use TLS in production** (RATLS_ENABLE=true)
2. ✅ **Enable rate limiting** (default: 60 req/min for Guardian)
3. ✅ **Monitor threat scores** (alert on >80)
4. ✅ **Review audit logs daily**
5. ✅ **Rotate secrets regularly** (HMAC keys, DB passwords)

### Performance:
1. ✅ **Use caching** (5 min TTL for threat scores)
2. ✅ **Tune eBPF buffer size** (8K events = good balance)
3. ✅ **Monitor latency** (P95 < 100ms target)
4. ✅ **Limit concurrent jobs** (max 100 active sandboxes)
5. ✅ **Enable connection pooling** (database)

### Operations:
1. ✅ **Backup baselines** (export user profiles regularly)
2. ✅ **Test rollback** (practice model rollback procedure)
3. ✅ **Monitor A/B tests** (evaluate after 1000+ samples)
4. ✅ **Review patterns** (update threat signatures monthly)
5. ✅ **Capacity planning** (scale before 80% CPU/memory)

---

## 📚 Next Steps

### For Development:
1. Read architecture docs: `/docs/`
2. Review API specs: `/api/openapi.yaml`
3. Study code examples: `/examples/`
4. Join team channel: `#shieldx-security`

### For Production:
1. Complete security audit
2. Load testing (1000+ req/sec)
3. Disaster recovery plan
4. Monitoring dashboards (Grafana)
5. On-call runbook

---

## 🆘 Support

### Documentation:
- Architecture: `/docs/DATABASE_LAYER.md`
- API Reference: `/api/openapi.yaml`
- Troubleshooting: This guide

### Team Contacts:
- **PERSON 2**: Security & ML Services
- **Slack**: `#shieldx-security`
- **Email**: `security@shieldx.io`

### Emergency:
```bash
# Stop all services
pkill -f guardian
pkill -f contauth
pkill -f ml-orchestrator

# Check running processes
ps aux | grep -E "(guardian|contauth|ml-orchestrator)"

# View recent errors
tail -n 100 data/ledger-*.log | grep ERROR
```

---

**Last Updated:** 2025-10-04  
**Maintained By:** PERSON 2 - Security & ML Services Team  
**Status:** ✅ Production Ready
