#!/bin/bash

# 🚀 PERSON 3 Quick Start - Production Deployment Guide
# This script helps you quickly deploy Credits, Shadow, and Camouflage services

set -euo pipefail

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║     🚀 PERSON 3 Services - Production Deployment 🚀      ║
║                                                           ║
║   Credits Service | Shadow Evaluation | Camouflage API   ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Step 1: Prerequisites
echo -e "${YELLOW}📋 Step 1: Checking Prerequisites...${NC}"
echo ""

check_command() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN}✓${NC} $1 is installed"
    else
        echo -e "${RED}✗${NC} $1 is not installed. Please install it first."
        exit 1
    fi
}

check_command kubectl
check_command docker
check_command pg_dump
echo ""

# Step 2: Environment Setup
echo -e "${YELLOW}🔧 Step 2: Environment Setup${NC}"
echo ""

read -p "Enter Kubernetes namespace (default: shieldx-prod): " NAMESPACE
NAMESPACE=${NAMESPACE:-shieldx-prod}

read -p "Enter database password: " -s DB_PASSWORD
echo ""

read -p "Enter Redis password: " -s REDIS_PASSWORD
echo ""

read -p "Enter API key for services: " -s API_KEY
echo ""

echo -e "${GREEN}✓${NC} Environment configured"
echo ""

# Step 3: Create Namespace and Secrets
echo -e "${YELLOW}📦 Step 3: Creating Namespace and Secrets...${NC}"
echo ""

kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
echo -e "${GREEN}✓${NC} Namespace created: $NAMESPACE"

# Create secrets
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Secret
metadata:
  name: credits-secrets
  namespace: $NAMESPACE
type: Opaque
stringData:
  DATABASE_URL: "postgres://credits_user:${DB_PASSWORD}@postgres-service:5432/credits"
  REDIS_PASSWORD: "${REDIS_PASSWORD}"
  CREDITS_API_KEY: "${API_KEY}"
  AUDIT_HMAC_KEY: "$(openssl rand -hex 32)"
  ENCRYPTION_KEY: "$(openssl rand -hex 32)"
---
apiVersion: v1
kind: Secret
metadata:
  name: shadow-secrets
  namespace: $NAMESPACE
type: Opaque
stringData:
  DATABASE_URL: "postgres://shadow_user:${DB_PASSWORD}@postgres-service:5432/shadow"
  SHADOW_API_KEY: "${API_KEY}"
EOF

echo -e "${GREEN}✓${NC} Secrets created"
echo ""

# Step 4: Deploy PostgreSQL
echo -e "${YELLOW}🗄️  Step 4: Deploying PostgreSQL...${NC}"
echo ""

cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: $NAMESPACE
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15-alpine
        env:
        - name: POSTGRES_PASSWORD
          value: "${DB_PASSWORD}"
        - name: POSTGRES_MULTIPLE_DATABASES
          value: "credits,shadow"
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
      volumes:
      - name: postgres-storage
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: $NAMESPACE
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
EOF

echo -e "${GREEN}✓${NC} PostgreSQL deployed"
kubectl wait --for=condition=ready pod -l app=postgres -n "$NAMESPACE" --timeout=120s
echo ""

# Step 5: Deploy Redis
echo -e "${YELLOW}💾 Step 5: Deploying Redis...${NC}"
echo ""

cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: $NAMESPACE
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        command: ["redis-server", "--requirepass", "${REDIS_PASSWORD}"]
        ports:
        - containerPort: 6379
---
apiVersion: v1
kind: Service
metadata:
  name: redis-service
  namespace: $NAMESPACE
spec:
  selector:
    app: redis
  ports:
  - port: 6379
EOF

echo -e "${GREEN}✓${NC} Redis deployed"
kubectl wait --for=condition=ready pod -l app=redis -n "$NAMESPACE" --timeout=120s
echo ""

# Step 6: Initialize Databases
echo -e "${YELLOW}🗃️  Step 6: Initializing Databases...${NC}"
echo ""

# Wait for postgres to be ready
sleep 10

# Create databases
kubectl exec -n "$NAMESPACE" deploy/postgres -- psql -U postgres -c "CREATE DATABASE credits;" || true
kubectl exec -n "$NAMESPACE" deploy/postgres -- psql -U postgres -c "CREATE DATABASE shadow;" || true
kubectl exec -n "$NAMESPACE" deploy/postgres -- psql -U postgres -c "CREATE USER credits_user WITH PASSWORD '${DB_PASSWORD}';" || true
kubectl exec -n "$NAMESPACE" deploy/postgres -- psql -U postgres -c "CREATE USER shadow_user WITH PASSWORD '${DB_PASSWORD}';" || true
kubectl exec -n "$NAMESPACE" deploy/postgres -- psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE credits TO credits_user;" || true
kubectl exec -n "$NAMESPACE" deploy/postgres -- psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE shadow TO shadow_user;" || true

echo -e "${GREEN}✓${NC} Databases initialized"
echo ""

# Step 7: Deploy Credits Service
echo -e "${YELLOW}💳 Step 7: Deploying Credits Service...${NC}"
echo ""

kubectl apply -f pilot/credits/credits-production.yaml
echo -e "${GREEN}✓${NC} Credits service deployed"
kubectl wait --for=condition=ready pod -l app=credits-service -n "$NAMESPACE" --timeout=300s || {
    echo -e "${RED}✗${NC} Credits service failed to start. Check logs:"
    echo "  kubectl logs -l app=credits-service -n $NAMESPACE"
}
echo ""

# Step 8: Deploy Shadow Service
echo -e "${YELLOW}🕵️  Step 8: Deploying Shadow Service...${NC}"
echo ""

kubectl apply -f pilot/shadow/shadow-production.yaml
echo -e "${GREEN}✓${NC} Shadow service deployed"
kubectl wait --for=condition=ready pod -l app=shadow-service -n "$NAMESPACE" --timeout=300s || {
    echo -e "${RED}✗${NC} Shadow service failed to start. Check logs:"
    echo "  kubectl logs -l app=shadow-service -n $NAMESPACE"
}
echo ""

# Step 9: Deploy Monitoring
echo -e "${YELLOW}📊 Step 9: Deploying Monitoring...${NC}"
echo ""

kubectl apply -f pilot/observability/monitoring-person3.yaml
echo -e "${GREEN}✓${NC} Monitoring configured"
echo ""

# Step 10: Health Checks
echo -e "${YELLOW}🏥 Step 10: Running Health Checks...${NC}"
echo ""

check_health() {
    local service=$1
    local port=$2
    echo -n "Checking $service... "
    
    if kubectl exec -n "$NAMESPACE" deploy/"$service" -- wget -q -O- "http://localhost:$port/health" &>/dev/null; then
        echo -e "${GREEN}✓ Healthy${NC}"
        return 0
    else
        echo -e "${RED}✗ Unhealthy${NC}"
        return 1
    fi
}

check_health "credits-service" "5004"
check_health "shadow-service" "5005"
echo ""

# Step 11: Display Service Info
echo -e "${YELLOW}📝 Step 11: Service Information${NC}"
echo ""

echo -e "${BLUE}Services deployed in namespace: ${GREEN}$NAMESPACE${NC}"
echo ""
echo "Credits Service:"
echo "  - Internal: http://credits-service:5004"
echo "  - Health: http://credits-service:5004/health"
echo "  - Metrics: http://credits-service:5004/metrics"
echo ""
echo "Shadow Service:"
echo "  - Internal: http://shadow-service:5005"
echo "  - Health: http://shadow-service:5005/health"
echo "  - Metrics: http://shadow-service:5005/metrics"
echo ""

# Step 12: Port Forwarding (Optional)
echo -e "${YELLOW}🔌 Step 12: Port Forwarding (Optional)${NC}"
echo ""
read -p "Setup port forwarding for local access? (y/n): " SETUP_PORTS

if [[ "$SETUP_PORTS" == "y" ]]; then
    echo ""
    echo "Starting port forwarding in background..."
    kubectl port-forward -n "$NAMESPACE" svc/credits-service 5004:5004 &>/dev/null &
    kubectl port-forward -n "$NAMESPACE" svc/shadow-service 5005:5005 &>/dev/null &
    
    echo -e "${GREEN}✓${NC} Port forwarding started"
    echo ""
    echo "You can now access services locally:"
    echo "  - Credits: http://localhost:5004"
    echo "  - Shadow: http://localhost:5005"
    echo ""
    echo "To stop port forwarding:"
    echo "  pkill -f 'kubectl port-forward'"
fi

# Step 13: Quick Test
echo ""
echo -e "${YELLOW}🧪 Step 13: Quick Smoke Test${NC}"
echo ""

if [[ "$SETUP_PORTS" == "y" ]]; then
    sleep 5
    echo "Testing Credits health endpoint..."
    curl -s http://localhost:5004/health | jq . || echo "Credits service not responding"
    echo ""
    echo "Testing Shadow health endpoint..."
    curl -s http://localhost:5005/health | jq . || echo "Shadow service not responding"
fi

# Summary
echo ""
echo -e "${GREEN}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║          ✅ DEPLOYMENT COMPLETED SUCCESSFULLY! ✅         ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo "Next Steps:"
echo "1. Monitor logs: kubectl logs -f -l app=credits-service -n $NAMESPACE"
echo "2. Check metrics: kubectl port-forward -n $NAMESPACE svc/credits-service 5004:5004"
echo "3. View dashboard: kubectl port-forward -n monitoring svc/grafana 3000:3000"
echo "4. Run tests: ./scripts/test-person3-services.sh"
echo ""
echo "Useful Commands:"
echo "  # View all pods"
echo "  kubectl get pods -n $NAMESPACE"
echo ""
echo "  # Get service logs"
echo "  kubectl logs -l app=credits-service -n $NAMESPACE"
echo ""
echo "  # Restart service"
echo "  kubectl rollout restart deployment/credits-service -n $NAMESPACE"
echo ""
echo "  # Scale service"
echo "  kubectl scale deployment/credits-service --replicas=5 -n $NAMESPACE"
echo ""
echo "  # Delete everything"
echo "  kubectl delete namespace $NAMESPACE"
echo ""
echo "Documentation: PERSON3_PRODUCTION_IMPLEMENTATION.md"
echo "Support: #shieldx-prod or person3-team@shieldx.io"
echo ""
echo -e "${GREEN}Happy deploying! 🚀${NC}"
