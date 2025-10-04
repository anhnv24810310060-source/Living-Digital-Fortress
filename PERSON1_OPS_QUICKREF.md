# 🚀 PERSON 1: Production Operations Quick Reference

**Version:** 1.0.0  
**Last Updated:** 2025-10-04  
**Status:** PRODUCTION

---

## 📌 Quick Start (30 seconds)

```bash
# 1. Set environment
export RATLS_ENABLE=true
export ORCH_PORT=8080
export INGRESS_PORT=8081
export ORCH_LB_ALGO=lrt

# 2. Start services
./orchestrator &
./ingress &

# 3. Verify
curl -k https://localhost:8080/health
curl -k https://localhost:8081/health
```

---

## 🎛️ Essential Environment Variables

### **Must Configure (Required)**
```bash
# TLS
RATLS_ENABLE=true                    # Enable RA-TLS
RATLS_TRUST_DOMAIN=shieldx.local     # SPIFFE trust domain

# Backend Pools
ORCH_POOL_API=http://backend1:9090,http://backend2:9090

# Policy
ORCH_POLICY_PATH=configs/policy.json
```

### **Recommended (Performance)**
```bash
# Load Balancing
ORCH_LB_ALGO=lrt                     # lrt, p2c_enhanced, peak_ewma

# Connection Pool
QUIC_POOL_MAX_IDLE_PER_HOST=10
QUIC_POOL_MAX_IDLE_TIME=90s

# Circuit Breaker
ORCH_CB_MIN_THRESHOLD=3
ORCH_CB_MAX_THRESHOLD=20
```

### **Optional (Advanced)**
```bash
# Phase 1 Features
PHASE1_ENABLE_PQC=true
PHASE1_ENABLE_CT=true
PHASE1_ENABLE_GRAPHQL_SEC=true
PHASE1_ENABLE_BEHAVIOR=true
PHASE1_ENABLE_ADAPTIVE_RL=true

# QUIC
QUIC_ENABLE_0RTT=true
QUIC_ENABLE_MIGRATION=true
QUIC_ENABLE_MULTIPATH=true
QUIC_CONGESTION_CONTROL=bbr          # cubic, bbr, reno
```

---

## 📊 Key Metrics to Watch

### **SLIs (Service Level Indicators)**
```prometheus
# Request Success Rate (Target: >99.9%)
rate(orchestrator_route_total[5m]) - rate(orchestrator_route_error_total[5m])

# P99 Latency (Target: <10ms)
histogram_quantile(0.99, orchestrator_route_duration_seconds)

# Backend Health (Target: >95%)
orchestrator_health_ratio_x10000 / 10000

# Connection Pool Hit Rate (Target: >85%)
quic_pool_hits_total / quic_pool_gets_total
```

### **Critical Alerts**
```
1. Error Rate >1%        → Check backends health
2. P99 Latency >20ms     → Review load balancer algorithm
3. Circuit Breakers Open → Investigate failing backends
4. Policy Version Mismatch → Check hot reload status
```

---

## 🔧 Common Operations

### **1. Add Backend to Pool**
```bash
curl -k -X PUT https://localhost:8080/admin/pools/api \
  -H "Content-Type: application/json" \
  -d '{
    "urls": ["http://new-backend:9090"],
    "algo": "lrt",
    "weights": {"http://new-backend:9090": 1.5}
  }'
```

### **2. Remove Backend from Pool**
```bash
curl -k -X DELETE https://localhost:8080/admin/pools/api/backend-url
```

### **3. Reload Policy (Hot Reload)**
```bash
# Edit policy file
vim configs/policy.json

# Wait 3 seconds (automatic reload)
sleep 3

# Verify new version
curl -k https://localhost:8080/policy/v2/status
```

### **4. Rollback Policy**
```bash
curl -k -X POST https://localhost:8080/policy/v2/rollback
```

### **5. Check Circuit Breaker Status**
```bash
curl -k https://localhost:8080/circuit-breakers | jq
```

### **6. Reset Circuit Breaker**
```bash
# Remove and re-add backend
curl -k -X DELETE https://localhost:8080/admin/pools/api
curl -k -X PUT https://localhost:8080/admin/pools/api -d '...'
```

---

## 🚨 Troubleshooting Runbook

### **Problem: High Error Rate**

**Symptoms:**
- `orchestrator_route_error_total` increasing
- HTTP 5xx responses

**Investigation:**
```bash
# 1. Check backend health
curl -k https://localhost:8080/health | jq '.pools'

# 2. Check circuit breaker status
curl -k https://localhost:8080/circuit-breakers | jq

# 3. Review recent logs
tail -100 data/orchestrator-access.log | jq 'select(.event=="route.error")'
```

**Resolution:**
1. If backends unhealthy → Restart backends
2. If circuit breaker open → Wait for auto-recovery or reset
3. If policy denying → Review policy rules

---

### **Problem: High Latency**

**Symptoms:**
- P99 latency >20ms
- Slow responses

**Investigation:**
```bash
# 1. Check load balancer distribution
curl -k https://localhost:8080/metrics | grep orchestrator_lb_pick_total

# 2. Check backend EWMA
curl -k https://localhost:8080/health | jq '.pools[].backends[] | {url, ewma, conns}'

# 3. Check connection pool
curl -k https://localhost:8081/metrics | grep quic_pool
```

**Resolution:**
1. Try different LB algorithm: `lrt` → `p2c_enhanced`
2. Increase connection pool size
3. Add more backends
4. Check network latency to backends

---

### **Problem: Policy Not Reloading**

**Symptoms:**
- Policy version not incrementing
- Changes not taking effect

**Investigation:**
```bash
# 1. Check file permissions
ls -la configs/policy.json

# 2. Check policy validation
tail -50 data/orchestrator-access.log | grep policy

# 3. Check file modification time
stat configs/policy.json
```

**Resolution:**
1. Fix file permissions: `chmod 644 configs/policy.json`
2. Validate JSON syntax: `jq . configs/policy.json`
3. Restart service if stuck

---

### **Problem: Memory Leak**

**Symptoms:**
- Memory usage constantly increasing
- OOM errors

**Investigation:**
```bash
# 1. Check Go memory stats
curl -k https://localhost:8080/metrics | grep go_memstats

# 2. Profile heap
go tool pprof http://localhost:8080/debug/pprof/heap

# 3. Check goroutine count
curl -k https://localhost:8080/metrics | grep go_goroutines
```

**Resolution:**
1. Check connection pool for leaks
2. Review policy engine history size
3. Restart service as temporary fix
4. Report to development team

---

## 🔍 Debug Commands

### **Connection Pool Stats**
```bash
curl -k https://localhost:8081/metrics | grep quic_pool | grep -v "#"
```

### **Load Balancer Distribution**
```bash
curl -k https://localhost:8080/metrics | \
  grep orchestrator_lb_pick_total | \
  awk '{print $2}' | \
  sort -n
```

### **Policy Version History**
```bash
curl -k https://localhost:8080/policy/v2/history | jq '.history[]'
```

### **Circuit Breaker Summary**
```bash
curl -k https://localhost:8080/circuit-breakers | \
  jq '.summary'
```

### **Behavioral Anomalies**
```bash
tail -1000 data/orchestrator-access.log | \
  jq 'select(.event=="behavior.anomaly")' | \
  jq -s 'group_by(.client_id) | .[] | {client: .[0].client_id, count: length}'
```

---

## 📈 Performance Tuning

### **High Throughput (>50k rps)**
```bash
export ORCH_LB_ALGO=p2c_enhanced
export ORCH_P2C_SUBSET_SIZE=5
export QUIC_POOL_MAX_IDLE_PER_HOST=20
export ORCH_CB_MIN_THRESHOLD=5
```

### **Low Latency (<5ms P99)**
```bash
export ORCH_LB_ALGO=lrt
export QUIC_ENABLE_0RTT=true
export QUIC_POOL_MAX_IDLE_TIME=60s
export ORCH_HEALTH_EVERY=3s
```

### **High Availability**
```bash
export QUIC_ENABLE_MULTIPATH=true
export ORCH_CB_MIN_THRESHOLD=3
export ORCH_CB_RECOVERY_TIMEOUT=10s
export PHASE1_ENABLE_ADAPTIVE_RL=true
```

---

## 🔒 Security Checklist

### **Before Production**
- [ ] TLS 1.3 enforced (`RATLS_ENABLE=true`)
- [ ] Client certificates required
- [ ] SAN allowlist configured
- [ ] Policy validation enabled
- [ ] Audit logging enabled
- [ ] Rate limiting configured
- [ ] No hard-coded credentials
- [ ] Firewall rules applied
- [ ] Monitoring alerts configured
- [ ] Backup & recovery tested

### **Regular Audits**
- [ ] Review audit logs weekly
- [ ] Check for policy violations
- [ ] Monitor behavioral anomalies
- [ ] Review CT alerts
- [ ] Verify certificate expiry dates
- [ ] Test rollback procedures

---

## 📞 Escalation

### **Severity Levels**

**P0 (Critical - Immediate Response)**
- Total service outage
- Security breach detected
- Data loss

**P1 (High - 1 hour response)**
- Degraded performance (>50%)
- All backends unhealthy
- Certificate expiry <24h

**P2 (Medium - 4 hour response)**
- Single backend failure
- High error rate (>1%)
- Policy hot reload failure

**P3 (Low - Next business day)**
- Minor performance degradation
- Non-critical metrics anomalies
- Documentation updates

### **On-Call Contacts**
```
PERSON 1 (Core Services):   [Contact Info]
PERSON 2 (Security/ML):     [Contact Info]
PERSON 3 (Infrastructure):  [Contact Info]
```

---

## 🔗 Useful Links

- **Metrics:** `https://localhost:8080/metrics`
- **Health:** `https://localhost:8080/health`
- **Grafana:** `http://grafana:3000`
- **Prometheus:** `http://prometheus:9090`
- **Documentation:** `/workspaces/Living-Digital-Fortress/PERSON1_PRODUCTION_COMPLETE.md`

---

## 📝 Change Log Format

```bash
# When making changes, log in CHANGELOG:
Date: 2025-10-04
Operator: [Name]
Change: [Description]
Impact: [Expected impact]
Rollback: [Rollback procedure]
Verified: [Y/N]
```

---

**Last Updated:** 2025-10-04  
**Maintained By:** PERSON 1 - Core Services & Orchestration Layer  
**Version:** 1.0.0
