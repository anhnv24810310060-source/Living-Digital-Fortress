# 🚀 PERSON 3 - Quick Start Guide
## Get Production Services Running in 5 Minutes

---

## Prerequisites

```bash
# Required tools
- Docker & Docker Compose
- kubectl (Kubernetes CLI)
- PostgreSQL client
- Go 1.21+
- Redis CLI

# Verify installations
docker --version
kubectl version --client
psql --version
go version
redis-cli --version
```

---

## 🏃 Quick Start (Local Development)

### Step 1: Start Databases

```bash
# Navigate to project root
cd /workspaces/Living-Digital-Fortress

# Start PostgreSQL and Redis
docker-compose -f infra/docker-compose.data.yml up -d

# Wait for databases to be ready (30 seconds)
sleep 30

# Verify databases are running
docker ps | grep postgres
docker ps | grep redis
```

### Step 2: Run Database Migrations

```bash
# Set database URLs
export CREDITS_DATABASE_URL="postgres://credits_user:credits_pass2024@localhost:5432/credits"
export SHADOW_DATABASE_URL="postgres://shadow_user:shadow_pass2024@localhost:5432/shadow"

# Run migrations (with automatic backup)
chmod +x scripts/migrate-databases.sh
./scripts/migrate-databases.sh

# Verify migrations
psql $CREDITS_DATABASE_URL -c "\dt"
```

### Step 3: Start Credits Service

```bash
# Terminal 1: Credits Service
cd services/credits

# Set environment variables
export PORT=5004
export DATABASE_URL=$CREDITS_DATABASE_URL
export REDIS_ADDR="localhost:6379"
export AUDIT_HMAC_KEY="dev-hmac-key-change-in-production"
export CREDITS_API_KEY="dev-api-key-change-in-production"

# Build and run
go build -o credits-service .
./credits-service

# Expected output:
# [credits] Connection pool initialized: max_open=100, max_idle=25
# [credits] Redis enabled at localhost:6379 for balance cache
# [credits] service starting on :5004
```

### Step 4: Start Shadow Evaluation Service

```bash
# Terminal 2: Shadow Service
cd services/shadow

# Set environment variables
export PORT=5005
export DATABASE_URL=$SHADOW_DATABASE_URL
export SHADOW_API_KEY="dev-shadow-key-change-in-production"

# Build and run
go build -o shadow-service .
./shadow-service

# Expected output:
# Shadow evaluation service starting on port 5005
```

### Step 5: Test Services

```bash
# Terminal 3: Testing

# Test Credits Service health
curl http://localhost:5004/health
# Expected: {"status":"healthy","service":"credits"}

# Test Shadow Service health
curl http://localhost:5005/health
# Expected: {"status":"healthy","service":"shadow"}

# Create test account with credits
curl -X POST http://localhost:5004/credits/topup \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dev-api-key-change-in-production" \
  -d '{
    "tenant_id": "test-user-001",
    "amount": 1000,
    "payment_method": "test",
    "payment_token": "tok_test",
    "idempotency_key": "topup-001"
  }'

# Check balance
curl http://localhost:5004/credits/balance/test-user-001 \
  -H "Authorization: Bearer dev-api-key-change-in-production"
# Expected: {"balance":1000,"reserved":0,"available":1000}

# Consume credits
curl -X POST http://localhost:5004/credits/consume \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dev-api-key-change-in-production" \
  -d '{
    "tenant_id": "test-user-001",
    "amount": 100,
    "description": "Test consumption",
    "idempotency_key": "consume-001"
  }'

# Verify new balance
curl http://localhost:5004/credits/balance/test-user-001 \
  -H "Authorization: Bearer dev-api-key-change-in-production"
# Expected: {"balance":900,"reserved":0,"available":900}
```

---

## 🐳 Quick Start (Docker Compose)

### All-in-One Setup

```bash
# Create docker-compose.yml for PERSON 3 services
cd /workspaces/Living-Digital-Fortress

# Start all services
docker-compose -f docker-compose.person3.yml up -d

# Check logs
docker-compose -f docker-compose.person3.yml logs -f

# Services will be available at:
# - Credits: http://localhost:5004
# - Shadow: http://localhost:5005
# - PostgreSQL: localhost:5432
# - Redis: localhost:6379

# Stop all services
docker-compose -f docker-compose.person3.yml down
```

Create `docker-compose.person3.yml`:

```yaml
version: '3.8'

services:
  postgres-credits:
    image: postgres:16
    environment:
      POSTGRES_USER: credits_user
      POSTGRES_PASSWORD: credits_pass2024
      POSTGRES_DB: credits
    ports:
      - "5432:5432"
    volumes:
      - credits-data:/var/lib/postgresql/data

  postgres-shadow:
    image: postgres:16
    environment:
      POSTGRES_USER: shadow_user
      POSTGRES_PASSWORD: shadow_pass2024
      POSTGRES_DB: shadow
    ports:
      - "5433:5432"
    volumes:
      - shadow-data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis-data:/data

  credits-service:
    build:
      context: .
      dockerfile: docker/Dockerfile.credits
    ports:
      - "5004:5004"
    environment:
      PORT: 5004
      DATABASE_URL: postgres://credits_user:credits_pass2024@postgres-credits:5432/credits
      REDIS_ADDR: redis:6379
      AUDIT_HMAC_KEY: dev-hmac-key
      CREDITS_API_KEY: dev-api-key
    depends_on:
      - postgres-credits
      - redis

  shadow-service:
    build:
      context: .
      dockerfile: docker/Dockerfile.shadow
    ports:
      - "5005:5005"
    environment:
      PORT: 5005
      DATABASE_URL: postgres://shadow_user:shadow_pass2024@postgres-shadow:5432/shadow
      SHADOW_API_KEY: dev-shadow-key
    depends_on:
      - postgres-shadow

volumes:
  credits-data:
  shadow-data:
  redis-data:
```

---

## ☸️ Quick Start (Kubernetes)

### Deploy to Local K8s (Minikube/Kind)

```bash
# Start local Kubernetes cluster
minikube start
# OR
kind create cluster

# Create namespace
kubectl create namespace shieldx-dev

# Create secrets
kubectl create secret generic credits-db-secret \
  --from-literal=connection-string="postgres://user:pass@postgres:5432/credits" \
  -n shieldx-dev

kubectl create secret generic shadow-db-secret \
  --from-literal=connection-string="postgres://user:pass@postgres:5432/shadow" \
  -n shieldx-dev

kubectl create secret generic redis-secret \
  --from-literal=password="redis-pass" \
  -n shieldx-dev

# Deploy services
kubectl apply -f pilot/k8s/credits-service.yaml -n shieldx-dev
kubectl apply -f pilot/k8s/shadow-eval-service.yaml -n shieldx-dev

# Wait for pods to be ready
kubectl wait --for=condition=ready pod -l app=credits-service -n shieldx-dev --timeout=300s

# Check status
kubectl get pods -n shieldx-dev
kubectl get svc -n shieldx-dev

# Port forward to access services
kubectl port-forward svc/credits-service 5004:5004 -n shieldx-dev &
kubectl port-forward svc/shadow-eval-service 5005:5005 -n shieldx-dev &

# Test services
curl http://localhost:5004/health
curl http://localhost:5005/health
```

---

## 🧪 Run Tests

### Unit Tests

```bash
# Test Credits Service
cd services/credits
go test -v -cover ./...

# Expected coverage: >80%

# Test Transaction Engine specifically
go test -v -run TestTransactionEngine
```

### Integration Tests

```bash
# Setup test database
export DATABASE_URL="postgres://credits_user:credits_pass2024@localhost:5432/credits_test"

# Run integration tests
go test -v -tags=integration ./...
```

### Load Tests

```bash
# Install k6 (load testing tool)
brew install k6  # macOS
# OR
sudo apt-get install k6  # Linux

# Run load test
k6 run --vus 100 --duration 5m tests/load/credits-service.js

# Expected results:
# - RPS: >1,000
# - P95 latency: <100ms
# - Success rate: >99%
```

---

## 📊 View Metrics & Monitoring

### Prometheus (Local)

```bash
# Run Prometheus
docker run -d -p 9090:9090 \
  -v $(pwd)/pilot/monitoring/prometheus-config.yaml:/etc/prometheus/prometheus.yml \
  prom/prometheus

# Access: http://localhost:9090
# Query examples:
# - rate(credits_operations_total[5m])
# - histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
```

### Grafana (Local)

```bash
# Run Grafana
docker run -d -p 3000:3000 \
  -e "GF_SECURITY_ADMIN_PASSWORD=admin" \
  grafana/grafana

# Access: http://localhost:3000
# Login: admin/admin
# Add Prometheus datasource: http://host.docker.internal:9090
```

---

## 🐛 Troubleshooting

### Services won't start

```bash
# Check if ports are already in use
lsof -i :5004
lsof -i :5005

# Kill existing processes
kill -9 <PID>

# Check database connectivity
psql $CREDITS_DATABASE_URL -c "SELECT 1"

# Check Redis connectivity
redis-cli ping
```

### Database migration fails

```bash
# Check database connection
psql $CREDITS_DATABASE_URL -c "\conninfo"

# Check if migrations already ran
psql $CREDITS_DATABASE_URL -c "SELECT * FROM schema_migrations"

# Rollback and retry
./scripts/migrate-databases.sh --rollback --service=credits
./scripts/migrate-databases.sh
```

### High latency

```bash
# Check database connection pool
curl http://localhost:5004/metrics | grep db_pool

# Check Redis connectivity
redis-cli --latency

# Check slow queries
psql $CREDITS_DATABASE_URL -c "SELECT query, mean_exec_time FROM pg_stat_statements ORDER BY mean_exec_time DESC LIMIT 10"
```

### Out of memory

```bash
# Check memory usage
docker stats

# Increase memory limits in K8s manifests
resources:
  limits:
    memory: 4Gi  # Increase from 2Gi
```

---

## 📚 Common Operations

### Add credits to account

```bash
curl -X POST http://localhost:5004/credits/topup \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $CREDITS_API_KEY" \
  -d '{
    "tenant_id": "user-123",
    "amount": 5000,
    "payment_method": "stripe",
    "payment_token": "tok_visa"
  }'
```

### Check transaction history

```bash
curl "http://localhost:5004/credits/history?tenant_id=user-123&limit=10" \
  -H "Authorization: Bearer $CREDITS_API_KEY"
```

### Create A/B test

```bash
curl -X POST http://localhost:5005/shadow/evaluate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $SHADOW_API_KEY" \
  -d '{
    "test_name": "new-rate-limit",
    "control_variant": "current",
    "test_variant": "aggressive",
    "confidence_level": 0.95,
    "min_sample_size": 1000
  }'
```

### Check A/B test results

```bash
curl http://localhost:5005/shadow/results/<test-id> \
  -H "Authorization: Bearer $SHADOW_API_KEY"
```

---

## 🔐 Security Best Practices

### Production Environment Variables

```bash
# NEVER use these in production - generate secure values!
export AUDIT_HMAC_KEY=$(openssl rand -hex 32)
export CREDITS_API_KEY=$(openssl rand -base64 32)
export SHADOW_API_KEY=$(openssl rand -base64 32)

# Use secrets management (e.g., AWS Secrets Manager, HashiCorp Vault)
```

### Database Permissions

```sql
-- Create read-only user for reporting
CREATE USER credits_readonly WITH PASSWORD 'secure_password';
GRANT CONNECT ON DATABASE credits TO credits_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO credits_readonly;
```

---

## 🎓 Next Steps

1. **Read Full Documentation**: `/docs/PERSON3_PRODUCTION_DEPLOYMENT.md`
2. **Review API Docs**: `/api/openapi.yaml`
3. **Setup Monitoring**: Configure Prometheus + Grafana
4. **Load Test**: Verify performance under production load
5. **Security Audit**: Run security scans (gosec, trivy)
6. **Deploy to Staging**: Test full deployment process
7. **Production Rollout**: Follow deployment checklist

---

## 📞 Support

- **Documentation**: `/docs/` directory
- **Issues**: GitHub Issues
- **Monitoring**: Grafana dashboards
- **Alerts**: AlertManager

---

**Services are now running!** 🎉

Check health:
- Credits: http://localhost:5004/health
- Shadow: http://localhost:5005/health
- Metrics: http://localhost:5004/metrics

Happy coding! 🚀
