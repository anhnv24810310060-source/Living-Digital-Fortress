# PERSON 3 - Production Deployment Guide
# Business Logic & Infrastructure

## 🎯 Overview

This guide covers the complete production deployment of **PERSON 3's components**:
- ✅ Credits Service (PORT 5004)
- ✅ Shadow Evaluation Service (PORT 5005)
- ✅ Locator Service (Service Discovery)
- ✅ Database Infrastructure (PostgreSQL + Redis)
- ✅ K8s Deployment with Monitoring

---

## 📋 P0 Requirements Checklist

### Credits Service ✅
- [x] **ACID Transactions** - Implemented with optimistic locking + retry
- [x] **Never Negative Balance** - Database constraint + application validation
- [x] **Immutable Audit Logs** - Blockchain-style audit chain with HMAC
- [x] **Payment Data Masking** - PCI DSS compliant masking
- [x] **Idempotency** - SHA256 hashed keys with 24h TTL
- [x] **Database Backup** - Automated backup before migrations
- [x] **Transaction Timeouts** - 30s timeout with exponential backoff

### Shadow Evaluation ✅
- [x] **Bayesian A/B Testing** - Thompson Sampling implementation
- [x] **Safe Deployment** - No deploy without conclusive test results
- [x] **Minimum Sample Size** - Enforced before deployment
- [x] **Confidence Level** - 95% confidence threshold
- [x] **Offline Evaluation** - Results stored before production deploy
- [x] **Rollback Mechanism** - Can revert to previous rule

### Database & Infrastructure ✅
- [x] **Connection Pooling** - Advanced pool with health monitoring
- [x] **K8s Production Manifests** - Resource limits, health checks, autoscaling
- [x] **Network Policies** - Restrict traffic between services
- [x] **PodSecurity** - Non-root containers, read-only filesystem
- [x] **Monitoring & Alerts** - Prometheus + Grafana dashboards
- [x] **Backup Automation** - Daily backups with retention policy

---

## 🚀 Pre-Deployment Steps

### 1. Environment Setup

```bash
# Set environment variables
export CREDITS_DATABASE_URL="postgres://user:pass@postgres-credits:5432/credits"
export SHADOW_DATABASE_URL="postgres://user:pass@postgres-shadow:5432/shadow"
export REDIS_ADDR="redis-master:6379"
export REDIS_PASSWORD="<secure-password>"
export AUDIT_HMAC_KEY="<generate-secure-key>"
export CREDITS_API_KEY="<generate-api-key>"
export SHADOW_API_KEY="<generate-api-key>"

# Verify Kubernetes cluster access
kubectl cluster-info
kubectl get nodes
```

### 2. Create Kubernetes Namespace

```bash
kubectl create namespace shieldx-prod
kubectl label namespace shieldx-prod name=shieldx-prod

# Create monitoring namespace
kubectl create namespace monitoring
```

### 3. Create Secrets

```bash
# Credits database secret
kubectl create secret generic credits-db-secret \
  --from-literal=connection-string="$CREDITS_DATABASE_URL" \
  -n shieldx-prod

# Shadow database secret
kubectl create secret generic shadow-db-secret \
  --from-literal=connection-string="$SHADOW_DATABASE_URL" \
  -n shieldx-prod

# Redis secret
kubectl create secret generic redis-secret \
  --from-literal=password="$REDIS_PASSWORD" \
  -n shieldx-prod

# Credits service secrets
kubectl create secret generic credits-secrets \
  --from-literal=audit-hmac-key="$AUDIT_HMAC_KEY" \
  --from-literal=api-key="$CREDITS_API_KEY" \
  -n shieldx-prod

# Shadow service secrets
kubectl create secret generic shadow-secrets \
  --from-literal=api-key="$SHADOW_API_KEY" \
  -n shieldx-prod
```

### 4. Deploy Databases

```bash
# Deploy PostgreSQL for Credits
kubectl apply -f pilot/k8s/postgres-credits.yaml

# Deploy PostgreSQL for Shadow
kubectl apply -f pilot/k8s/postgres-shadow.yaml

# Deploy Redis
kubectl apply -f pilot/k8s/redis.yaml

# Wait for databases to be ready
kubectl wait --for=condition=ready pod -l app=postgres-credits -n shieldx-prod --timeout=300s
kubectl wait --for=condition=ready pod -l app=postgres-shadow -n shieldx-prod --timeout=300s
kubectl wait --for=condition=ready pod -l app=redis -n shieldx-prod --timeout=300s
```

### 5. Run Database Migrations

```bash
# Dry run first to verify
./scripts/migrate-databases.sh --dry-run

# Run actual migrations (with automatic backup)
export BACKUP_DIR=/backups
export BACKUP_RETENTION_DAYS=30
./scripts/migrate-databases.sh

# Verify migrations
kubectl exec -it postgres-credits-0 -n shieldx-prod -- \
  psql -U credits_user -d credits -c "SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 5;"
```

---

## 🎬 Deployment Sequence

### Step 1: Deploy Credits Service

```bash
# Build and push Docker image
cd services/credits
docker build -t shieldx/credits-service:v1.0.0 -f ../../docker/Dockerfile.credits .
docker push shieldx/credits-service:v1.0.0

# Deploy to Kubernetes
kubectl apply -f pilot/k8s/credits-service.yaml

# Verify deployment
kubectl get pods -l app=credits-service -n shieldx-prod
kubectl logs -f deployment/credits-service -n shieldx-prod

# Test health endpoint
kubectl port-forward svc/credits-service 5004:5004 -n shieldx-prod &
curl http://localhost:5004/health
```

### Step 2: Deploy Shadow Evaluation Service

```bash
# Build and push Docker image
cd services/shadow
docker build -t shieldx/shadow-eval:v1.0.0 -f ../../docker/Dockerfile.shadow .
docker push shieldx/shadow-eval:v1.0.0

# Deploy to Kubernetes
kubectl apply -f pilot/k8s/shadow-eval-service.yaml

# Verify deployment
kubectl get pods -l app=shadow-eval -n shieldx-prod
kubectl logs -f deployment/shadow-eval-service -n shieldx-prod

# Test health endpoint
kubectl port-forward svc/shadow-eval-service 5005:5005 -n shieldx-prod &
curl http://localhost:5005/health
```

### Step 3: Deploy Locator Service

```bash
# Build and push Docker image
cd services/locator
docker build -t shieldx/locator-service:v1.0.0 .
docker push shieldx/locator-service:v1.0.0

# Deploy to Kubernetes
kubectl apply -f pilot/k8s/locator-service.yaml

# Verify deployment
kubectl get pods -l app=locator -n shieldx-prod
```

### Step 4: Deploy Monitoring Stack

```bash
# Deploy Prometheus
kubectl apply -f pilot/monitoring/prometheus-config.yaml
kubectl apply -f pilot/monitoring/prometheus-deployment.yaml

# Deploy Grafana
kubectl apply -f pilot/monitoring/grafana-deployment.yaml

# Access Grafana
kubectl port-forward svc/grafana 3000:3000 -n monitoring &
# Open http://localhost:3000 (admin/admin)
```

---

## 🧪 Post-Deployment Testing

### 1. Credits Service Tests

```bash
# Test credit purchase
curl -X POST http://credits-service.shieldx-prod.svc.cluster.local:5004/credits/topup \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $CREDITS_API_KEY" \
  -d '{
    "tenant_id": "test-tenant-001",
    "amount": 1000,
    "payment_method": "test",
    "payment_token": "tok_test_12345"
  }'

# Test credit consumption
curl -X POST http://credits-service.shieldx-prod.svc.cluster.local:5004/credits/consume \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $CREDITS_API_KEY" \
  -d '{
    "tenant_id": "test-tenant-001",
    "amount": 100,
    "description": "Test consumption"
  }'

# Check balance
curl http://credits-service.shieldx-prod.svc.cluster.local:5004/credits/balance/test-tenant-001 \
  -H "Authorization: Bearer $CREDITS_API_KEY"

# View metrics
curl http://credits-service.shieldx-prod.svc.cluster.local:5004/metrics
```

### 2. Shadow Evaluation Tests

```bash
# Create A/B test
curl -X POST http://shadow-eval-service.shieldx-prod.svc.cluster.local:5005/shadow/evaluate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $SHADOW_API_KEY" \
  -d '{
    "test_name": "rate-limit-test",
    "control_variant": "current-rule",
    "test_variant": "new-rule",
    "confidence_level": 0.95,
    "min_sample_size": 1000
  }'

# Get test results
curl http://shadow-eval-service.shieldx-prod.svc.cluster.local:5005/shadow/results/<test-id> \
  -H "Authorization: Bearer $SHADOW_API_KEY"
```

### 3. Load Testing

```bash
# Run load test on Credits Service
k6 run --vus 100 --duration 5m tests/load/credits-service.js

# Monitor during load test
kubectl top pods -n shieldx-prod
kubectl get hpa -n shieldx-prod -w
```

---

## 📊 Monitoring & Alerts

### Key Metrics to Watch

#### Credits Service
- `credits_operations_total{result="success"}` - Successful transactions
- `credits_operations_total{result="error"}` - Failed transactions
- `http_request_duration_seconds` - Response time
- `pg_stat_activity_count` - Database connections
- `credits_negative_balance_count` - ⚠️ MUST be 0 (P0)
- `credits_audit_chain_integrity_errors` - ⚠️ MUST be 0 (P0)

#### Shadow Evaluation
- `shadow_test_probability_best` - Best variant probability
- `shadow_test_sample_count` - Number of samples
- `shadow_deployments_without_testing` - ⚠️ MUST be 0 (P0)

### Access Dashboards

```bash
# Prometheus
kubectl port-forward svc/prometheus 9090:9090 -n monitoring
# http://localhost:9090

# Grafana
kubectl port-forward svc/grafana 3000:3000 -n monitoring
# http://localhost:3000
```

---

## 🚨 Troubleshooting

### Credits Service Issues

#### High error rate
```bash
# Check logs
kubectl logs -f deployment/credits-service -n shieldx-prod

# Check database connectivity
kubectl exec -it credits-service-xxx -n shieldx-prod -- \
  psql $DATABASE_URL -c "SELECT 1"

# Check metrics
curl http://credits-service:5004/metrics | grep credits_operations_total
```

#### Database connection pool exhausted
```bash
# Check active connections
kubectl exec -it postgres-credits-0 -n shieldx-prod -- \
  psql -U credits_user -d credits -c \
  "SELECT count(*) FROM pg_stat_activity WHERE datname='credits';"

# Increase pool size (if needed)
kubectl edit deployment credits-service -n shieldx-prod
# Update: resources.limits.cpu and resources.limits.memory
```

#### Negative balance detected (P0 CRITICAL)
```bash
# IMMEDIATE ACTION REQUIRED
# 1. Check audit logs
kubectl exec -it postgres-credits-0 -n shieldx-prod -- \
  psql -U credits_user -d credits -c \
  "SELECT * FROM credit_accounts WHERE balance < 0;"

# 2. Check audit chain
kubectl exec -it postgres-credits-0 -n shieldx-prod -- \
  psql -U credits_user -d credits -c \
  "SELECT * FROM audit_log_chain ORDER BY id DESC LIMIT 50;"

# 3. Rollback to last known good backup
./scripts/migrate-databases.sh --rollback --service=credits --backup=<backup-file>
```

### Shadow Service Issues

#### Unsafe deployment attempt (P0 CRITICAL)
```bash
# Check deployment logs
kubectl logs -f deployment/shadow-eval-service -n shieldx-prod | grep "unsafe"

# List pending tests
kubectl exec -it postgres-shadow-0 -n shieldx-prod -- \
  psql -U shadow_user -d shadow -c \
  "SELECT * FROM ab_tests WHERE status != 'conclusive';"
```

---

## 🔄 Rollback Procedures

### Service Rollback
```bash
# Rollback Credits Service
kubectl rollout undo deployment/credits-service -n shieldx-prod
kubectl rollout status deployment/credits-service -n shieldx-prod

# Rollback Shadow Service
kubectl rollout undo deployment/shadow-eval-service -n shieldx-prod
```

### Database Rollback
```bash
# List available backups
ls -lh /backups/

# Restore from backup
./scripts/migrate-databases.sh \
  --rollback \
  --service=credits \
  --backup=/backups/credits_backup_20251004_120000.sql.gz
```

---

## 📈 Performance Tuning

### Credits Service Optimization

```sql
-- Analyze query performance
EXPLAIN ANALYZE SELECT balance FROM credit_accounts WHERE tenant_id = 'xxx';

-- Refresh materialized view
SELECT refresh_credit_balances_summary();

-- Optimize indexes
REINDEX TABLE credit_accounts;
VACUUM ANALYZE credit_accounts;
```

### Connection Pool Tuning

```go
// Adjust in code based on workload
db.SetMaxOpenConns(100)      // Increase for high load
db.SetMaxIdleConns(25)        // Keep 25% of max
db.SetConnMaxLifetime(30 * time.Minute)
```

---

## 🎓 Best Practices

### DO ✅
- Always backup before migrations
- Use idempotency keys for all transactions
- Monitor P0 metrics continuously
- Test in shadow environment first
- Scale horizontally with K8s HPA
- Use read replicas for reporting queries

### DON'T ❌
- Never disable balance checks
- Never skip audit logging
- Never deploy untested rules to production
- Never expose payment data in logs
- Never run migrations in production without backup
- Never allow negative balances (P0 violation)

---

## 📞 Support & Escalation

### P0 Issues (Critical)
- Negative balance detected
- Audit chain broken
- Unsafe deployment to production
- **Action**: Page on-call engineer immediately

### P1 Issues (High)
- High error rate (>5%)
- Slow transactions (>1s p95)
- Database pool exhaustion
- **Action**: Investigate within 15 minutes

### P2 Issues (Medium)
- Low balance alerts
- Inconclusive A/B tests after 1 week
- **Action**: Investigate within 1 hour

---

## 📚 Additional Resources

- Architecture Docs: `/docs/architecture/person3-services.md`
- API Documentation: `/docs/api/credits-service.md`
- Runbook: `/docs/runbooks/person3-operations.md`
- Migration Guide: `/migrations/README.md`

---

## ✅ Deployment Checklist

### Pre-Deployment
- [ ] Secrets created in K8s
- [ ] Database backups verified
- [ ] Migration scripts tested (dry-run)
- [ ] Load testing completed
- [ ] Monitoring dashboards configured

### Deployment
- [ ] Databases deployed and healthy
- [ ] Migrations applied successfully
- [ ] Credits service deployed (3 replicas)
- [ ] Shadow service deployed (2 replicas)
- [ ] Locator service deployed
- [ ] Health checks passing
- [ ] HPA configured and working

### Post-Deployment
- [ ] All services responding to health checks
- [ ] Metrics being scraped by Prometheus
- [ ] Grafana dashboards showing data
- [ ] Alerts configured in AlertManager
- [ ] Load test passed at production scale
- [ ] Negative balance check: 0 violations
- [ ] Audit chain integrity: verified
- [ ] Shadow tests: properly isolated

---

## 🎉 Success Criteria

Deployment is successful when:

1. ✅ All services healthy for 30+ minutes
2. ✅ P95 latency < 100ms under load
3. ✅ Zero P0 violations detected
4. ✅ Database connection pool < 70% utilization
5. ✅ HPA scales up/down correctly
6. ✅ All alerts firing correctly
7. ✅ Backup automation working
8. ✅ Monitoring dashboards accurate

---

**Deployed by PERSON 3 - Business Logic & Infrastructure Team**
**Date**: 2025-10-04
**Version**: 1.0.0 (Production Ready)
