# PERSON 1 - Quick Start Guide

🚀 **Orchestrator & Ingress Services - Production Deployment Guide**

---

## 📦 Prerequisites

```bash
# Install dependencies
go version  # Requires Go 1.21+
docker --version
kubectl version --client

# Clone repository
cd /workspaces/Living-Digital-Fortress
```

---

## 🔧 Configuration

### 1. Environment Variables

Create `.env.orchestrator`:

```bash
# Service
SERVICE_VERSION=1.0.0
ORCH_PORT=8080
DEV_MODE=false  # IMPORTANT: Set to false in production

# TLS/mTLS (P0 REQUIRED)
TLS_CERT_FILE=/etc/shieldx/certs/orchestrator.crt
TLS_KEY_FILE=/etc/shieldx/certs/orchestrator.key
TLS_CA_FILE=/etc/shieldx/certs/ca.crt
TLS_ALLOWED_SANS=spiffe://shieldx.local/ns/default/sa/,svc-

# Rate Limiting (P0)
ORCH_IP_BURST=200
REDIS_ADDR=redis:6379

# OPA Policy (P0)
ORCH_OPA_POLICY_PATH=/etc/shieldx/policies/routing.rego
ORCH_OPA_ENFORCE=1
ORCH_POLICY_PATH=/etc/shieldx/policies/base-policy.json

# Load Balancing (P1)
ORCH_LB_ALGO=p2c  # Options: round_robin, least_conn, ewma, p2c, rendezvous
ORCH_P2C_CONN_PENALTY=5.0

# Backend Pools
ORCH_BACKENDS_JSON={"guardian":["https://guardian:9090"],"ingress":["https://ingress:8081"]}

# Logging (P1)
ACCESS_LOG_PATH=data/access-orchestrator.log
MASK_PII=true

# Observability
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
```

### 2. Generate TLS Certificates

```bash
# Generate CA
openssl genrsa -out ca.key 4096
openssl req -new -x509 -days 3650 -key ca.key -out ca.crt \
  -subj "/CN=ShieldX-CA/O=ShieldX/C=US"

# Generate orchestrator certificate
openssl genrsa -out orchestrator.key 4096
openssl req -new -key orchestrator.key -out orchestrator.csr \
  -subj "/CN=orchestrator/O=ShieldX/C=US"

# Create SAN configuration
cat > orchestrator.ext <<EOF
subjectAltName = @alt_names
[alt_names]
URI.1 = spiffe://shieldx.local/ns/default/sa/orchestrator
DNS.1 = orchestrator
DNS.2 = localhost
IP.1 = 127.0.0.1
EOF

# Sign certificate
openssl x509 -req -in orchestrator.csr -CA ca.crt -CAkey ca.key \
  -CAcreateserial -out orchestrator.crt -days 365 \
  -extensions v3_req -extfile orchestrator.ext

# Verify SAN
openssl x509 -in orchestrator.crt -text -noout | grep -A1 "Subject Alternative Name"
```

### 3. Create OPA Policy

Create `policies/routing.rego`:

```rego
package shieldx.routing

import future.keywords.if

default allow := false

# Allow all health checks
allow if {
    input.path == "/health"
}

allow if {
    input.path == "/metrics"
}

# Allow trusted tenants
allow if {
    input.tenant == "trusted-tenant"
    input.scope == "api"
}

# Deny known bad actors
deny if {
    input.ip in ["192.0.2.1", "198.51.100.1"]
}

# Divert suspicious patterns
action := "divert" if {
    contains(input.path, "../")
}

action := "divert" if {
    contains(input.path, "etc/passwd")
}

# Tarpit slow scan attacks
action := "tarpit" if {
    input.tenant == "unknown"
    count(input.path) > 100
}
```

---

## 🏃 Running the Service

### Option 1: Standalone Binary

```bash
# Build
go build -o bin/orchestrator ./services/orchestrator

# Run
./bin/orchestrator
```

### Option 2: Docker

```bash
# Build image
docker build -t shieldx/orchestrator:latest -f docker/Dockerfile.orchestrator .

# Run container
docker run -d \
  --name orchestrator \
  -p 8080:8080 \
  -v $(pwd)/certs:/etc/shieldx/certs:ro \
  -v $(pwd)/policies:/etc/shieldx/policies:ro \
  -v $(pwd)/data:/data \
  --env-file .env.orchestrator \
  shieldx/orchestrator:latest
```

### Option 3: Kubernetes

```bash
# Create namespace
kubectl create namespace shieldx-system

# Create secrets
kubectl create secret generic orchestrator-tls \
  --from-file=tls.crt=orchestrator.crt \
  --from-file=tls.key=orchestrator.key \
  --from-file=ca.crt=ca.crt \
  -n shieldx-system

kubectl create configmap orchestrator-policy \
  --from-file=routing.rego=policies/routing.rego \
  -n shieldx-system

# Deploy
kubectl apply -f pilot/orchestrator-deployment.yml
kubectl apply -f pilot/orchestrator-service.yml

# Check status
kubectl get pods -n shieldx-system
kubectl logs -n shieldx-system -l app=orchestrator -f
```

---

## ✅ Verification

### 1. Health Check

```bash
# Basic health
curl http://localhost:8080/health

# Expected response:
{
  "service": "orchestrator",
  "status": "healthy",
  "timestamp": "2025-10-03T12:34:56Z",
  "version": "1.0.0",
  "uptime": "1m30s",
  "checks": {
    "backend_pools": {"status": "pass", "message": "Healthy backends: 2/2 across 2 pools"},
    "redis": {"status": "pass", "message": "Redis connected"},
    "opa_engine": {"status": "pass", "message": "OPA engine functional"},
    "system_resources": {"status": "pass"},
    "certificate": {"status": "pass", "message": "Certificate valid for 364d"}
  }
}
```

### 2. Metrics Check

```bash
# Get Prometheus metrics
curl http://localhost:8080/metrics

# Verify key metrics
curl -s http://localhost:8080/metrics | grep -E "orchestrator_route_total|orchestrator_health"
```

### 3. Rate Limiting Test

```bash
# Send burst of requests
for i in {1..250}; do
  curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8080/health
done

# Should see:
# 200 (first 200 requests)
# 429 (requests 201-250, rate limited)
```

### 4. Policy Enforcement Test

```bash
# Allowed request
curl -X POST http://localhost:8080/route \
  -H "Content-Type: application/json" \
  -d '{
    "service": "guardian",
    "tenant": "trusted-tenant",
    "scope": "api",
    "path": "/api/v1/test"
  }'

# Expected: 200 OK with target backend

# Denied request (bad tenant)
curl -X POST http://localhost:8080/route \
  -H "Content-Type: application/json" \
  -d '{
    "service": "guardian",
    "tenant": "untrusted",
    "scope": "api",
    "path": "/api/v1/test"
  }'

# Expected: 403 Forbidden
```

### 5. mTLS Test

```bash
# Without client cert (should fail)
curl --cacert ca.crt https://localhost:8080/health

# With client cert (should succeed)
curl --cacert ca.crt \
  --cert client.crt \
  --key client.key \
  https://localhost:8080/health
```

### 6. Load Balancing Test

```bash
# Test round-robin
for i in {1..10}; do
  curl -s -X POST http://localhost:8080/route \
    -H "Content-Type: application/json" \
    -d '{
      "service": "guardian",
      "tenant": "test",
      "algo": "round_robin"
    }' | jq -r '.target'
done

# Should distribute evenly across backends
```

---

## 📊 Monitoring

### Grafana Dashboard

Import dashboard from `dashboards/orchestrator-overview.json`:

**Key Panels:**
1. Request Rate (RPS)
2. Error Rate (%)
3. P50/P95/P99 Latency
4. Backend Health
5. Rate Limit Hits
6. OPA Cache Hit Rate
7. TLS Certificate Expiry

### Prometheus Alerts

```yaml
groups:
  - name: orchestrator
    rules:
      - alert: OrchestatorDown
        expr: up{job="orchestrator"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Orchestrator service is down"

      - alert: OrchestatorHighErrorRate
        expr: rate(orchestrator_route_error_total[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Orchestrator error rate above 5%"

      - alert: OrchestatorNoHealthyBackends
        expr: sum(orchestrator_backend_healthy) == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "No healthy backends available"
```

---

## 🔧 Troubleshooting

### Issue: Service won't start

**Check logs:**
```bash
tail -f data/ledger-orchestrator.log
```

**Common causes:**
- Missing TLS certificates → Generate certs following step 2
- Port already in use → Check `netstat -tulpn | grep 8080`
- Invalid OPA policy → Test with `opa eval --data policies/routing.rego`

### Issue: High latency

**Check backend health:**
```bash
curl http://localhost:8080/health?detailed=true | jq '.checks.backend_pools'
```

**Check metrics:**
```bash
curl http://localhost:8080/metrics | grep orchestrator_health_probe_seconds
```

**Solution:** Adjust health probe interval or circuit breaker thresholds

### Issue: Rate limiting too aggressive

**Check current limits:**
```bash
# View environment
env | grep ORCH_IP_BURST
```

**Adjust limits:**
```bash
export ORCH_IP_BURST=500
# Restart service
```

---

## 🚀 Load Testing

### Using hey

```bash
# Install hey
go install github.com/rakyll/hey@latest

# Test throughput
hey -n 100000 -c 100 -m POST \
  -H "Content-Type: application/json" \
  -d '{"service":"guardian","tenant":"test","path":"/api"}' \
  http://localhost:8080/route

# Expected results:
# - Requests/sec: >10,000
# - Average latency: <10ms
# - P99 latency: <50ms
```

### Using k6

```javascript
// load-test.js
import http from 'k6/http';
import { check } from 'k6';

export let options = {
  stages: [
    { duration: '30s', target: 100 },
    { duration: '1m', target: 100 },
    { duration: '30s', target: 0 },
  ],
};

export default function () {
  let payload = JSON.stringify({
    service: 'guardian',
    tenant: 'test',
    path: '/api/v1/test',
  });

  let res = http.post('http://localhost:8080/route', payload, {
    headers: { 'Content-Type': 'application/json' },
  });

  check(res, {
    'status is 200': (r) => r.status === 200,
    'response time < 50ms': (r) => r.timings.duration < 50,
  });
}
```

Run:
```bash
k6 run load-test.js
```

---

## 📝 Next Steps

1. ✅ Verify all P0 requirements working
2. ✅ Configure monitoring dashboards
3. ✅ Set up alerting rules
4. ⏳ Run load tests and tune performance
5. ⏳ Coordinate with PERSON 2 for Guardian integration
6. ⏳ Coordinate with PERSON 3 for Credits integration
7. ⏳ Production deployment approval

---

## 📞 Support

- **Documentation:** `docs/`
- **Code issues:** Create GitHub issue
- **Security concerns:** Email security@shieldx.io
- **Slack:** #shieldx-dev

---

**Last Updated:** October 3, 2025  
**Maintained by:** PERSON 1 (Orchestrator Team)
