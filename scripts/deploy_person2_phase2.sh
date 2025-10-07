#!/bin/bash
# PERSON 2: Phase 2 Production Deployment Script
# Purpose: Deploy Advanced Security & ML Services
# Requirements: Docker, kubectl, go 1.21+

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="shieldx"
VERSION="2.0.0"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-localhost:5000}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}PERSON 2: Phase 2 Deployment${NC}"
echo -e "${GREEN}========================================${NC}"

# Pre-flight checks
echo -e "${YELLOW}[1/10] Pre-flight checks...${NC}"

command -v docker >/dev/null 2>&1 || { echo -e "${RED}Error: docker not found${NC}"; exit 1; }
command -v kubectl >/dev/null 2>&1 || { echo -e "${RED}Error: kubectl not found${NC}"; exit 1; }
command -v go >/dev/null 2>&1 || { echo -e "${RED}Error: go not found${NC}"; exit 1; }

echo -e "${GREEN}✓ All prerequisites installed${NC}"

# Build services
echo -e "${YELLOW}[2/10] Building Guardian service...${NC}"
cd services/guardian
CGO_ENABLED=0 GOOS=linux go build -o guardian main.go
echo -e "${GREEN}✓ Guardian built${NC}"

echo -e "${YELLOW}[3/10] Building ContAuth service...${NC}"
cd ../contauth-service
CGO_ENABLED=0 GOOS=linux go build -o contauth main.go federated_learning.go
echo -e "${GREEN}✓ ContAuth built${NC}"

echo -e "${YELLOW}[4/10] Building ML-Orchestrator...${NC}"
cd ../ml-orchestrator
CGO_ENABLED=0 GOOS=linux go build -o ml-orchestrator main.go
echo -e "${GREEN}✓ ML-Orchestrator built${NC}"

cd ../..

# Run unit tests
echo -e "${YELLOW}[5/10] Running unit tests...${NC}"
go test ./services/honeypot-service/internal/guardian/... -v -cover || { echo -e "${RED}Guardian tests failed${NC}"; exit 1; }
go test ./services/ai-service/internal/ml/... -v -cover || { echo -e "${RED}ML tests failed${NC}"; exit 1; }
go test ./services/contauth-service/... -v -cover || { echo -e "${RED}ContAuth tests failed${NC}"; exit 1; }
echo -e "${GREEN}✓ All tests passed${NC}"

# Build Docker images
echo -e "${YELLOW}[6/10] Building Docker images...${NC}"

# Guardian image
cat > Dockerfile.guardian <<EOF
FROM alpine:3.18
RUN apk add --no-cache ca-certificates
COPY services/guardian/guardian /usr/local/bin/guardian
RUN chmod +x /usr/local/bin/guardian
USER 1000:1000
EXPOSE 9090
CMD ["/usr/local/bin/guardian"]
EOF

docker build -f Dockerfile.guardian -t ${DOCKER_REGISTRY}/guardian:${VERSION} .
echo -e "${GREEN}✓ Guardian image built${NC}"

# ContAuth image
cat > Dockerfile.contauth <<EOF
FROM alpine:3.18
RUN apk add --no-cache ca-certificates
COPY services/contauth-service/contauth /usr/local/bin/contauth
RUN chmod +x /usr/local/bin/contauth
USER 1000:1000
EXPOSE 5002
CMD ["/usr/local/bin/contauth"]
EOF

docker build -f Dockerfile.contauth -t ${DOCKER_REGISTRY}/contauth:${VERSION} .
echo -e "${GREEN}✓ ContAuth image built${NC}"

# ML-Orchestrator image
cat > Dockerfile.ml-orchestrator <<EOF
FROM alpine:3.18
RUN apk add --no-cache ca-certificates
COPY services/ml-orchestrator/ml-orchestrator /usr/local/bin/ml-orchestrator
RUN chmod +x /usr/local/bin/ml-orchestrator
USER 1000:1000
EXPOSE 8087
CMD ["/usr/local/bin/ml-orchestrator"]
EOF

docker build -f Dockerfile.ml-orchestrator -t ${DOCKER_REGISTRY}/ml-orchestrator:${VERSION} .
echo -e "${GREEN}✓ ML-Orchestrator image built${NC}"

# Push images
echo -e "${YELLOW}[7/10] Pushing Docker images...${NC}"
docker push ${DOCKER_REGISTRY}/guardian:${VERSION}
docker push ${DOCKER_REGISTRY}/contauth:${VERSION}
docker push ${DOCKER_REGISTRY}/ml-orchestrator:${VERSION}
echo -e "${GREEN}✓ Images pushed${NC}"

# Create namespace
echo -e "${YELLOW}[8/10] Setting up Kubernetes namespace...${NC}"
kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
echo -e "${GREEN}✓ Namespace ready${NC}"

# Deploy ConfigMap
echo -e "${YELLOW}[9/10] Deploying ConfigMaps...${NC}"
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: person2-config
  namespace: ${NAMESPACE}
data:
  GUARDIAN_PORT: "9090"
  GUARDIAN_MAX_CONCURRENT: "32"
  GUARDIAN_SANDBOX_BACKEND: "firecracker"
  DETECTOR_USE_TRANSFORMER: "true"
  DETECTOR_USE_EBPF: "true"
  DETECTOR_USE_MEMORY_FORENSICS: "true"
  DETECTOR_MAX_LATENCY_MS: "100"
  CONTAUTH_PORT: "5002"
  CONTAUTH_FL_EPSILON: "1.0"
  CONTAUTH_FL_DELTA: "1e-5"
  CONTAUTH_FL_MIN_CLIENTS: "5"
  ML_ORCHESTRATOR_PORT: "8087"
  ML_ENSEMBLE_WEIGHT: "0.6"
EOF

# Create Secret for admin tokens
kubectl create secret generic person2-secrets \
  --from-literal=ML_API_ADMIN_TOKEN=$(openssl rand -base64 32) \
  --from-literal=CONTAUTH_ADMIN_TOKEN=$(openssl rand -base64 32) \
  --namespace=${NAMESPACE} \
  --dry-run=client -o yaml | kubectl apply -f -

echo -e "${GREEN}✓ ConfigMap deployed${NC}"

# Deploy services
echo -e "${YELLOW}[10/10] Deploying services...${NC}"

# Guardian Deployment
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: guardian
  namespace: ${NAMESPACE}
  labels:
    app: guardian
    version: ${VERSION}
spec:
  replicas: 3
  selector:
    matchLabels:
      app: guardian
  template:
    metadata:
      labels:
        app: guardian
        version: ${VERSION}
    spec:
      containers:
      - name: guardian
        image: ${DOCKER_REGISTRY}/guardian:${VERSION}
        ports:
        - containerPort: 9090
          name: http
        - containerPort: 9091
          name: metrics
        envFrom:
        - configMapRef:
            name: person2-config
        resources:
          requests:
            memory: "512Mi"
            cpu: "1000m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 9090
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 9090
          initialDelaySeconds: 5
          periodSeconds: 10
        securityContext:
          runAsNonRoot: true
          runAsUser: 1000
---
apiVersion: v1
kind: Service
metadata:
  name: guardian
  namespace: ${NAMESPACE}
spec:
  selector:
    app: guardian
  ports:
  - name: http
    port: 9090
    targetPort: 9090
  - name: metrics
    port: 9091
    targetPort: 9091
  type: ClusterIP
EOF

# ContAuth Deployment
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: contauth
  namespace: ${NAMESPACE}
  labels:
    app: contauth
    version: ${VERSION}
spec:
  replicas: 2
  selector:
    matchLabels:
      app: contauth
  template:
    metadata:
      labels:
        app: contauth
        version: ${VERSION}
    spec:
      containers:
      - name: contauth
        image: ${DOCKER_REGISTRY}/contauth:${VERSION}
        ports:
        - containerPort: 5002
          name: http
        envFrom:
        - configMapRef:
            name: person2-config
        - secretRef:
            name: person2-secrets
        resources:
          requests:
            memory: "256Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 5002
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 5002
          initialDelaySeconds: 5
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: contauth
  namespace: ${NAMESPACE}
spec:
  selector:
    app: contauth
  ports:
  - name: http
    port: 5002
    targetPort: 5002
  type: ClusterIP
EOF

# ML-Orchestrator Deployment
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-orchestrator
  namespace: ${NAMESPACE}
  labels:
    app: ml-orchestrator
    version: ${VERSION}
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ml-orchestrator
  template:
    metadata:
      labels:
        app: ml-orchestrator
        version: ${VERSION}
    spec:
      containers:
      - name: ml-orchestrator
        image: ${DOCKER_REGISTRY}/ml-orchestrator:${VERSION}
        ports:
        - containerPort: 8087
          name: http
        envFrom:
        - configMapRef:
            name: person2-config
        - secretRef:
            name: person2-secrets
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8087
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8087
          initialDelaySeconds: 5
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: ml-orchestrator
  namespace: ${NAMESPACE}
spec:
  selector:
    app: ml-orchestrator
  ports:
  - name: http
    port: 8087
    targetPort: 8087
  type: ClusterIP
EOF

echo -e "${GREEN}✓ All services deployed${NC}"

# Wait for rollout
echo -e "${YELLOW}Waiting for deployments to complete...${NC}"
kubectl rollout status deployment/guardian -n ${NAMESPACE} --timeout=5m
kubectl rollout status deployment/contauth -n ${NAMESPACE} --timeout=5m
kubectl rollout status deployment/ml-orchestrator -n ${NAMESPACE} --timeout=5m

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Deployment Summary${NC}"
echo -e "${GREEN}========================================${NC}"

# Show pod status
kubectl get pods -n ${NAMESPACE} -l 'app in (guardian,contauth,ml-orchestrator)'

# Show service endpoints
echo -e "\n${GREEN}Service Endpoints:${NC}"
kubectl get svc -n ${NAMESPACE}

echo -e "\n${GREEN}✓ Phase 2 Deployment Complete!${NC}"
echo -e "\n${YELLOW}Next Steps:${NC}"
echo -e "1. Run smoke tests: ./scripts/smoke_test_person2.sh"
echo -e "2. Check metrics: kubectl port-forward -n ${NAMESPACE} svc/guardian 9091:9091"
echo -e "3. View logs: kubectl logs -n ${NAMESPACE} -l app=guardian -f"
echo -e "4. Access Grafana dashboards for monitoring"

# Cleanup temp files
rm -f Dockerfile.guardian Dockerfile.contauth Dockerfile.ml-orchestrator
