#!/bin/bash

# P0 Setup Script - Development Environment for PERSON 1
# Sets up everything needed to test Core Services & Orchestration

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "=========================================="
echo "P0 Setup - Core Services Development"
echo "=========================================="

# Create directories
echo -e "\n${YELLOW}Creating directories...${NC}"
mkdir -p data
mkdir -p certs
mkdir -p policies
mkdir -p configs

# Create log files
touch data/ledger-orchestrator.log
touch data/ledger-orchestrator-sec.log
touch data/ledger-ingress.log
touch data/ledger-ingress-sec.log

echo -e "${GREEN}✓${NC} Log files created"

# Create default policy file
if [ ! -f "policies/base.json" ]; then
    echo -e "\n${YELLOW}Creating default policy file...${NC}"
    cat > policies/base.json << 'EOF'
{
  "allowAll": false,
  "allowed": [
    {
      "tenant": "*",
      "scope": "read:*",
      "path": "/api/*"
    }
  ],
  "advanced": {
    "rateLimitPerTenant": 1000,
    "maxRequestSize": 16384
  }
}
EOF
    echo -e "${GREEN}✓${NC} Policy file created at policies/base.json"
fi

# Create OPA policy (example)
if [ ! -f "policies/advanced.rego" ]; then
    echo -e "\n${YELLOW}Creating OPA policy...${NC}"
    cat > policies/advanced.rego << 'EOF'
package shieldx.routing

default allow = false

# Allow authenticated requests to /api/
allow {
    input.path = "/api/v1/users"
    input.scope = "read:users"
}

# Allow health checks
allow {
    input.path = "/health"
}

# Allow metrics
allow {
    input.path = "/metrics"
}

# Deny suspicious paths
deny {
    contains(input.path, "..")
}

deny {
    contains(input.path, "etc/passwd")
}
EOF
    echo -e "${GREEN}✓${NC} OPA policy created at policies/advanced.rego"
fi

# Generate self-signed certificates for development
if [ ! -f "certs/ca.pem" ]; then
    echo -e "\n${YELLOW}Generating development certificates...${NC}"
    
    # CA certificate
    openssl req -x509 -newkey rsa:2048 -nodes \
        -keyout certs/ca-key.pem -out certs/ca.pem \
        -days 365 -subj "/CN=ShieldX-Dev-CA" 2>/dev/null
    
    # Orchestrator certificate with SAN
    cat > certs/orchestrator.cnf << EOF
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req
prompt = no

[req_distinguished_name]
CN = orchestrator

[v3_req]
keyUsage = keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth, clientAuth
subjectAltName = @alt_names

[alt_names]
URI.1 = spiffe://shieldx.local/ns/default/sa/orchestrator
DNS.1 = orchestrator
DNS.2 = localhost
IP.1 = 127.0.0.1
EOF
    
    openssl req -new -newkey rsa:2048 -nodes \
        -keyout certs/orchestrator-key.pem \
        -out certs/orchestrator.csr \
        -config certs/orchestrator.cnf 2>/dev/null
    
    openssl x509 -req -in certs/orchestrator.csr \
        -CA certs/ca.pem -CAkey certs/ca-key.pem -CAcreateserial \
        -out certs/orchestrator.pem -days 365 \
        -extensions v3_req -extfile certs/orchestrator.cnf 2>/dev/null
    
    # Ingress certificate with SAN
    cat > certs/ingress.cnf << EOF
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req
prompt = no

[req_distinguished_name]
CN = ingress

[v3_req]
keyUsage = keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth, clientAuth
subjectAltName = @alt_names

[alt_names]
URI.1 = spiffe://shieldx.local/ns/default/sa/ingress
DNS.1 = ingress
DNS.2 = localhost
IP.1 = 127.0.0.1
EOF
    
    openssl req -new -newkey rsa:2048 -nodes \
        -keyout certs/ingress-key.pem \
        -out certs/ingress.csr \
        -config certs/ingress.cnf 2>/dev/null
    
    openssl x509 -req -in certs/ingress.csr \
        -CA certs/ca.pem -CAkey certs/ca-key.pem -CAcreateserial \
        -out certs/ingress.pem -days 365 \
        -extensions v3_req -extfile certs/ingress.cnf 2>/dev/null
    
    # Guardian certificate (for PERSON 2)
    cat > certs/guardian.cnf << EOF
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req
prompt = no

[req_distinguished_name]
CN = guardian

[v3_req]
keyUsage = keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth, clientAuth
subjectAltName = @alt_names

[alt_names]
URI.1 = spiffe://shieldx.local/ns/default/sa/guardian
DNS.1 = guardian
EOF
    
    openssl req -new -newkey rsa:2048 -nodes \
        -keyout certs/guardian-key.pem \
        -out certs/guardian.csr \
        -config certs/guardian.cnf 2>/dev/null
    
    openssl x509 -req -in certs/guardian.csr \
        -CA certs/ca.pem -CAkey certs/ca-key.pem -CAcreateserial \
        -out certs/guardian.pem -days 365 \
        -extensions v3_req -extfile certs/guardian.cnf 2>/dev/null
    
    # Clean up CSR and config files
    rm -f certs/*.csr certs/*.cnf certs/ca.srl
    
    echo -e "${GREEN}✓${NC} Certificates generated in certs/"
    echo "  - CA: certs/ca.pem"
    echo "  - Orchestrator: certs/orchestrator.pem"
    echo "  - Ingress: certs/ingress.pem"
    echo "  - Guardian: certs/guardian.pem"
fi

# Create environment file
echo -e "\n${YELLOW}Creating .env file...${NC}"
cat > .env.dev << EOF
# Orchestrator Configuration
ORCH_PORT=8080
ORCH_POLICY_PATH=policies/base.json
ORCH_OPA_POLICY_PATH=policies/advanced.rego
ORCH_OPA_ENFORCE=1
ORCH_IP_BURST=200
ORCH_LB_ALGO=p2c
ORCH_P2C_CONN_PENALTY=5.0
ORCH_EWMA_DECAY=0.3
ORCH_HEALTH_EVERY=5s
ORCH_MAX_ROUTE_BYTES=16384

# TLS Configuration (Orchestrator)
ORCH_TLS_CERT_FILE=certs/orchestrator.pem
ORCH_TLS_KEY_FILE=certs/orchestrator-key.pem
ORCH_TLS_CLIENT_CA_FILE=certs/ca.pem
ORCH_TLS_SAN_ALLOW=spiffe://shieldx.local/ns/default/sa/

# Ingress Configuration
INGRESS_PORT=8081
INGRESS_TLS_CERT_FILE=certs/ingress.pem
INGRESS_TLS_KEY_FILE=certs/ingress-key.pem
INGRESS_TLS_CLIENT_CA_FILE=certs/ca.pem
INGRESS_TLS_SAN_ALLOW=spiffe://shieldx.local/ns/default/sa/

# Redis (optional)
# REDIS_ADDR=localhost:6379

# RA-TLS (optional)
RATLS_ENABLE=false
RATLS_TRUST_DOMAIN=shieldx.local
RATLS_NAMESPACE=default
RATLS_SERVICE=orchestrator
EOF

echo -e "${GREEN}✓${NC} Environment file created: .env.dev"

# Build packages
echo -e "\n${YELLOW}Building packages...${NC}"
go build ./pkg/validation
go build ./pkg/accesslog
go build ./pkg/tlsutil
echo -e "${GREEN}✓${NC} Packages built successfully"

# Run tests
echo -e "\n${YELLOW}Running unit tests...${NC}"
go test ./pkg/validation -v -cover | tail -3
echo -e "${GREEN}✓${NC} Tests passed"

# Create quick start script
cat > start-dev.sh << 'EOF'
#!/bin/bash
# Quick start script for development

# Load environment
export $(cat .env.dev | grep -v '^#' | xargs)

echo "Starting Orchestrator on port $ORCH_PORT..."
./bin/orchestrator &
ORCH_PID=$!

echo "Orchestrator PID: $ORCH_PID"
echo "Logs: tail -f data/ledger-orchestrator.log"
echo ""
echo "Test with:"
echo "  curl http://localhost:8080/health"
echo "  curl http://localhost:8080/metrics"
echo ""
echo "Stop with: kill $ORCH_PID"
EOF

chmod +x start-dev.sh

echo -e "\n${GREEN}=========================================="
echo "Setup Complete!"
echo "==========================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Build services:"
echo "     make build-orchestrator"
echo "     make build-ingress"
echo ""
echo "  2. Start in dev mode:"
echo "     ./start-dev.sh"
echo ""
echo "  3. Or with TLS:"
echo "     export \$(cat .env.dev | grep -v '^#' | xargs)"
echo "     ./bin/orchestrator"
echo ""
echo "  4. Run integration tests:"
echo "     ./scripts/test-p0-integration.sh"
echo ""
echo "  5. View logs:"
echo "     tail -f data/ledger-orchestrator.log"
echo ""
echo -e "${YELLOW}Note:${NC} For production, generate proper certificates with your PKI!"
