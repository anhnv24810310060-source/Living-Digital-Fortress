🏗️ ShieldX Microservices Architecture - Tài Liệu Hoàn Chỉnh
🎯 Kiến Trúc Microservices
Tách ShieldX thành độc lập hoàn toàn - mỗi service là 1 repository riêng, có database riêng, API riêng, deployment riêng.

shieldx-platform/
├── services/
│   ├── shieldx-gateway/          # API Gateway & Load Balancer
│   ├── shieldx-auth/             # Authentication & Authorization  
│   ├── shieldx-policy/           # Policy Engine (OPA)
│   ├── shieldx-sandbox/          # Threat Analysis & Firecracker
│   ├── shieldx-deception/        # Honeypots & Camouflage
│   ├── shieldx-credits/          # Billing & Resource Management
│   ├── shieldx-ml/               # Machine Learning Pipeline
│   ├── shieldx-forensics/        # Security Analytics
│   ├── shieldx-notification/     # Alerts & Notifications
│   └── shieldx-admin/            # Admin Dashboard
├── shared/
│   ├── shieldx-proto/            # gRPC Protocol Definitions
│   ├── shieldx-sdk/              # Client SDKs
│   └── shieldx-common/           # Shared Libraries
├── infrastructure/
│   ├── docker-compose/           # Local Development
│   ├── kubernetes/               # K8s Manifests
│   ├── terraform/                # Infrastructure as Code
│   └── monitoring/               # Prometheus, Grafana, Jaeger
└── tools/
    ├── cli/                      # ShieldX CLI Tool
    ├── migration/                # Database Migration Tools
    └── testing/                  # Integration Test Suite

Copy

Insert at cursor
📦 Chi Tiết Từng Microservice
1. shieldx-gateway - API Gateway
shieldx-gateway/
├── cmd/
│   └── main.go                   # Entry point
├── internal/
│   ├── handler/                  # HTTP handlers
│   ├── middleware/               # Auth, CORS, Rate limiting
│   ├── proxy/                    # Reverse proxy logic
│   ├── loadbalancer/             # Load balancing algorithms
│   └── circuit/                  # Circuit breaker
├── pkg/
│   ├── config/                   # Configuration management
│   ├── metrics/                  # Prometheus metrics
│   └── health/                   # Health checks
├── api/
│   ├── openapi.yaml              # API specification
│   └── proto/                    # gRPC definitions
├── configs/
│   ├── config.yaml               # Default config
│   ├── routes.yaml               # Routing rules
│   └── policies.yaml             # Gateway policies
├── deployments/
│   ├── Dockerfile                # Container image
│   ├── docker-compose.yml        # Local development
│   └── k8s/                      # Kubernetes manifests
├── scripts/
│   ├── build.sh                  # Build script
│   ├── test.sh                   # Test script
│   └── deploy.sh                 # Deployment script
├── tests/
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   └── e2e/                      # End-to-end tests
├── docs/
│   ├── README.md                 # Service documentation
│   ├── API.md                    # API documentation
│   └── DEPLOYMENT.md             # Deployment guide
├── go.mod
├── go.sum
├── Makefile
└── .env.example


Copy

Insert at cursor
2. shieldx-auth - Authentication Service
shieldx-auth/
├── cmd/
│   └── main.go
├── internal/
│   ├── handler/
│   │   ├── auth.go               # Login, logout, refresh
│   │   ├── user.go               # User management
│   │   └── session.go            # Session management
│   ├── service/
│   │   ├── auth.go               # Business logic
│   │   ├── jwt.go                # JWT management
│   │   ├── oauth.go              # OAuth providers
│   │   └── mfa.go                # Multi-factor auth
│   ├── repository/
│   │   ├── user.go               # User data access
│   │   ├── session.go            # Session storage
│   │   └── audit.go              # Audit logs
│   └── model/
│       ├── user.go               # User model
│       ├── session.go            # Session model
│       └── token.go              # Token model
├── pkg/
│   ├── database/                 # Database connection
│   ├── redis/                    # Redis client
│   ├── crypto/                   # Encryption utilities
│   └── validator/                # Input validation
├── migrations/
│   ├── 001_create_users.sql
│   ├── 002_create_sessions.sql
│   └── 003_create_audit_logs.sql
├── configs/
│   ├── config.yaml
│   └── database.yaml
├── deployments/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── k8s/
├── go.mod
└── README.md


Copy

Insert at cursor
3. shieldx-policy - Policy Engine
shieldx-policy/
├── cmd/
│   └── main.go
├── internal/
│   ├── handler/
│   │   ├── policy.go             # Policy CRUD
│   │   ├── evaluation.go         # Policy evaluation
│   │   └── bundle.go             # Policy bundle management
│   ├── service/
│   │   ├── opa.go                # OPA integration
│   │   ├── compiler.go           # Policy compilation
│   │   └── cache.go              # Policy caching
│   ├── repository/
│   │   ├── policy.go             # Policy storage
│   │   └── bundle.go             # Bundle storage
│   └── model/
│       ├── policy.go             # Policy model
│       └── decision.go           # Decision model
├── policies/
│   ├── base/                     # Base policies
│   ├── security/                 # Security policies
│   └── custom/                   # Custom policies
├── pkg/
│   ├── opa/                      # OPA utilities
│   └── compiler/                 # Policy compiler
├── configs/
│   └── config.yaml
├── deployments/
│   ├── Dockerfile
│   └── k8s/
├── go.mod
└── README.md


Copy

Insert at cursor
4. shieldx-sandbox - Threat Analysis
shieldx-sandbox/
├── cmd/
│   └── main.go
├── internal/
│   ├── handler/
│   │   ├── analysis.go           # Analysis endpoints
│   │   ├── sandbox.go            # Sandbox management
│   │   └── report.go             # Analysis reports
│   ├── service/
│   │   ├── firecracker.go        # Firecracker VMs
│   │   ├── wasm.go               # WASM execution
│   │   ├── ebpf.go               # eBPF monitoring
│   │   └── ml.go                 # ML threat scoring
│   ├── repository/
│   │   ├── analysis.go           # Analysis results
│   │   └── malware.go            # Malware samples
│   └── model/
│       ├── analysis.go           # Analysis model
│       └── threat.go             # Threat model
├── pkg/
│   ├── firecracker/              # Firecracker utilities
│   ├── ebpf/                     # eBPF programs
│   └── ml/                       # ML models
├── configs/
│   ├── config.yaml
│   └── firecracker.json
├── deployments/
│   ├── Dockerfile
│   └── k8s/
├── go.mod
└── README.md


Copy

Insert at cursor
5. shieldx-deception - Honeypots & Camouflage
shieldx-deception/
├── cmd/
│   └── main.go
├── internal/
│   ├── handler/
│   │   ├── honeypot.go           # Honeypot management
│   │   ├── camouflage.go         # Server camouflage
│   │   └── decoy.go              # Decoy services
│   ├── service/
│   │   ├── honeypot.go           # Honeypot logic
│   │   ├── fingerprint.go        # Fingerprint spoofing
│   │   └── template.go           # Response templates
│   ├── repository/
│   │   ├── honeypot.go           # Honeypot data
│   │   └── attack.go             # Attack logs
│   └── model/
│       ├── honeypot.go           # Honeypot model
│       └── attack.go             # Attack model
├── templates/
│   ├── http/                     # HTTP response templates
│   ├── ssh/                      # SSH honeypot templates
│   └── ftp/                      # FTP honeypot templates
├── pkg/
│   ├── honeypot/                 # Honeypot engines
│   └── camouflage/               # Camouflage utilities
├── configs/
│   └── config.yaml
├── deployments/
│   ├── Dockerfile
│   └── k8s/
├── go.mod
└── README.md


Copy

Insert at cursor
6. shieldx-credits - Billing & Resource Management
shieldx-credits/
├── cmd/
│   └── main.go
├── internal/
│   ├── handler/
│   │   ├── credits.go            # Credit operations
│   │   ├── billing.go            # Billing management
│   │   └── usage.go              # Usage tracking
│   ├── service/
│   │   ├── credits.go            # Credit business logic
│   │   ├── ledger.go             # Double-entry ledger
│   │   └── billing.go            # Billing calculations
│   ├── repository/
│   │   ├── credits.go            # Credit storage
│   │   ├── transaction.go        # Transaction logs
│   │   └── usage.go              # Usage metrics
│   └── model/
│       ├── credit.go             # Credit model
│       ├── transaction.go        # Transaction model
│       └── usage.go              # Usage model
├── migrations/
│   ├── 001_create_credits.sql
│   ├── 002_create_transactions.sql
│   └── 003_create_usage.sql
├── pkg/
│   ├── ledger/                   # Ledger utilities
│   └── billing/                  # Billing utilities
├── configs/
│   └── config.yaml
├── deployments/
│   ├── Dockerfile
│   └── k8s/
├── go.mod
└── README.md


Copy

Insert at cursor
7. shieldx-ml - Machine Learning Pipeline
shieldx-ml/
├── cmd/
│   └── main.go
├── internal/
│   ├── handler/
│   │   ├── model.go              # Model management
│   │   ├── training.go           # Training endpoints
│   │   └── inference.go          # Inference endpoints
│   ├── service/
│   │   ├── training.go           # Model training
│   │   ├── inference.go          # Model inference
│   │   └── pipeline.go           # ML pipeline
│   ├── repository/
│   │   ├── model.go              # Model storage
│   │   └── dataset.go            # Dataset management
│   └── model/
│       ├── ml_model.go           # ML model metadata
│       └── dataset.go            # Dataset model
├── models/
│   ├── threat_detection/         # Threat detection models
│   ├── behavioral/               # Behavioral models
│   └── anomaly/                  # Anomaly detection models
├── pkg/
│   ├── ml/                       # ML utilities
│   └── feature/                  # Feature engineering
├── python/
│   ├── training/                 # Python training scripts
│   ├── inference/                # Python inference
│   └── requirements.txt
├── configs/
│   └── config.yaml
├── deployments/
│   ├── Dockerfile
│   ├── Dockerfile.python
│   └── k8s/
├── go.mod
└── README.md


Copy

Insert at cursor
8. shieldx-forensics - Security Analytics
shieldx-forensics/
├── cmd/
│   └── main.go
├── internal/
│   ├── handler/
│   │   ├── investigation.go      # Investigation management
│   │   ├── evidence.go           # Evidence collection
│   │   └── report.go             # Forensic reports
│   ├── service/
│   │   ├── investigation.go      # Investigation logic
│   │   ├── timeline.go           # Timeline reconstruction
│   │   └── correlation.go        # Event correlation
│   ├── repository/
│   │   ├── investigation.go      # Investigation data
│   │   ├── evidence.go           # Evidence storage
│   │   └── timeline.go           # Timeline data
│   └── model/
│       ├── investigation.go      # Investigation model
│       ├── evidence.go           # Evidence model
│       └── timeline.go           # Timeline model
├── pkg/
│   ├── forensics/                # Forensic utilities
│   └── correlation/              # Event correlation
├── configs/
│   └── config.yaml
├── deployments/
│   ├── Dockerfile
│   └── k8s/
├── go.mod
└── README.md


Copy

Insert at cursor
9. shieldx-notification - Alerts & Notifications
shieldx-notification/
├── cmd/
│   └── main.go
├── internal/
│   ├── handler/
│   │   ├── notification.go       # Notification endpoints
│   │   ├── template.go           # Template management
│   │   └── subscription.go       # Subscription management
│   ├── service/
│   │   ├── notification.go       # Notification logic
│   │   ├── email.go              # Email notifications
│   │   ├── slack.go              # Slack notifications
│   │   └── webhook.go            # Webhook notifications
│   ├── repository/
│   │   ├── notification.go       # Notification storage
│   │   └── template.go           # Template storage
│   └── model/
│       ├── notification.go       # Notification model
│       └── template.go           # Template model
├── templates/
│   ├── email/                    # Email templates
│   └── slack/                    # Slack templates
├── pkg/
│   ├── email/                    # Email utilities
│   ├── slack/                    # Slack utilities
│   └── webhook/                  # Webhook utilities
├── configs/
│   └── config.yaml
├── deployments/
│   ├── Dockerfile
│   └── k8s/
├── go.mod
└── README.md


Copy

Insert at cursor
10. shieldx-admin - Admin Dashboard
shieldx-admin/
├── cmd/
│   └── main.go
├── internal/
│   ├── handler/
│   │   ├── dashboard.go          # Dashboard endpoints
│   │   ├── user.go               # User management
│   │   └── system.go             # System management
│   ├── service/
│   │   ├── dashboard.go          # Dashboard logic
│   │   ├── aggregation.go        # Data aggregation
│   │   └── reporting.go          # Report generation
│   └── model/
│       ├── dashboard.go          # Dashboard model
│       └── report.go             # Report model
├── web/
│   ├── static/                   # Static assets
│   ├── templates/                # HTML templates
│   └── src/                      # Frontend source
├── pkg/
│   ├── aggregation/              # Data aggregation
│   └── reporting/                # Report utilities
├── configs/
│   └── config.yaml
├── deployments/
│   ├── Dockerfile
│   └── k8s/
├── go.mod
└── README.md


Copy

Insert at cursor
🔗 Shared Components
shieldx-proto - gRPC Protocol Definitions
shieldx-proto/
├── auth/
│   ├── auth.proto                # Authentication service
│   └── user.proto                # User management
├── policy/
│   ├── policy.proto              # Policy service
│   └── evaluation.proto          # Policy evaluation
├── sandbox/
│   ├── analysis.proto            # Threat analysis
│   └── sandbox.proto             # Sandbox management
├── credits/
│   ├── credits.proto             # Credit service
│   └── billing.proto             # Billing service
├── common/
│   ├── common.proto              # Common types
│   └── health.proto              # Health check
├── buf.yaml                      # Buf configuration
├── buf.gen.yaml                  # Code generation
└── Makefile                      # Build scripts

Copy

Insert at cursor
shieldx-sdk - Client SDKs
shieldx-sdk/
├── go/
│   ├── auth/                     # Auth client
│   ├── policy/                   # Policy client
│   ├── sandbox/                  # Sandbox client
│   └── credits/                  # Credits client
├── python/
│   ├── shieldx/                  # Python SDK
│   └── setup.py
├── javascript/
│   ├── src/                      # JS/TS SDK
│   └── package.json
├── java/
│   ├── src/                      # Java SDK
│   └── pom.xml
└── docs/
    ├── go.md                     # Go SDK docs
    ├── python.md                 # Python SDK docs
    └── javascript.md             # JS SDK docs

Copy

Insert at cursor
shieldx-common - Shared Libraries
shieldx-common/
├── go/
│   ├── config/                   # Configuration utilities
│   ├── database/                 # Database utilities
│   ├── metrics/                  # Metrics utilities
│   ├── logging/                  # Logging utilities
│   ├── tracing/                  # Distributed tracing
│   └── middleware/               # Common middleware
├── python/
│   ├── shieldx_common/           # Python common lib
│   └── setup.py
└── docs/
    └── README.md

Copy

Insert at cursor
🚀 Infrastructure & Deployment
docker-compose - Local Development
docker-compose/
├── docker-compose.yml            # Main compose file
├── docker-compose.dev.yml        # Development overrides
├── docker-compose.test.yml       # Testing environment
├── .env.example                  # Environment variables
└── scripts/
    ├── start.sh                  # Start all services
    ├── stop.sh                   # Stop all services
    └── reset.sh                  # Reset environment

Copy

Insert at cursor
kubernetes - Production Deployment
kubernetes/
├── namespaces/
│   ├── shieldx-system.yaml       # System namespace
│   └── shieldx-apps.yaml         # Applications namespace
├── services/
│   ├── gateway/                  # Gateway manifests
│   ├── auth/                     # Auth manifests
│   ├── policy/                   # Policy manifests
│   └── ...                       # Other services
├── ingress/
│   ├── gateway-ingress.yaml      # Main ingress
│   └── admin-ingress.yaml        # Admin ingress
├── configmaps/
│   ├── gateway-config.yaml       # Gateway config
│   └── ...                       # Other configs
├── secrets/
│   ├── database-secrets.yaml     # Database credentials
│   └── jwt-secrets.yaml          # JWT secrets
├── monitoring/
│   ├── prometheus/               # Prometheus setup
│   ├── grafana/                  # Grafana setup
│   └── jaeger/                   # Jaeger tracing
└── scripts/
    ├── deploy.sh                 # Deployment script
    └── rollback.sh               # Rollback script

Copy

Insert at cursor
terraform - Infrastructure as Code
terraform/
├── modules/
│   ├── vpc/                      # VPC module
│   ├── eks/                      # EKS cluster
│   ├── rds/                      # Database
│   └── redis/                    # Redis cluster
├── environments/
│   ├── dev/                      # Development
│   ├── staging/                  # Staging
│   └── prod/                     # Production
├── main.tf                       # Main configuration
├── variables.tf                  # Variables
├── outputs.tf                    # Outputs
└── terraform.tfvars.example      # Example variables

Copy

Insert at cursor
🛠️ Development Tools
CLI Tool
tools/cli/
├── cmd/
│   ├── root.go                   # Root command
│   ├── deploy.go                 # Deployment commands
│   ├── config.go                 # Configuration commands
│   └── debug.go                  # Debug commands
├── pkg/
│   ├── client/                   # API clients
│   ├── config/                   # Configuration
│   └── utils/                    # Utilities
├── go.mod
└── README.md

Copy

Insert at cursor
Migration Tools
tools/migration/
├── cmd/
│   └── main.go                   # Migration tool
├── migrations/
│   ├── auth/                     # Auth service migrations
│   ├── credits/                  # Credits service migrations
│   └── ...                       # Other service migrations
├── pkg/
│   ├── migrator/                 # Migration engine
│   └── database/                 # Database utilities
├── go.mod
└── README.md

Copy

Insert at cursor
Integration Test Suite
tools/testing/
├── cmd/
│   └── main.go                   # Test runner
├── tests/
│   ├── auth/                     # Auth service tests
│   ├── policy/                   # Policy service tests
│   ├── sandbox/                  # Sandbox service tests
│   └── e2e/                      # End-to-end tests
├── pkg/
│   ├── client/                   # Test clients
│   ├── fixtures/                 # Test fixtures
│   └── utils/                    # Test utilities
├── go.mod
└── README.md

Copy

Insert at cursor
📊 Service Communication
gRPC Communication
// Service-to-service communication via gRPC
type AuthClient struct {
    conn *grpc.ClientConn
    client authpb.AuthServiceClient
}

func (c *AuthClient) Authenticate(ctx context.Context, token string) (*User, error) {
    req := &authpb.AuthenticateRequest{Token: token}
    resp, err := c.client.Authenticate(ctx, req)
    if err != nil {
        return nil, err
    }
    return &User{ID: resp.User.Id, Email: resp.User.Email}, nil
}

Copy

Insert at cursor
go
Event-Driven Architecture
// Event publishing
type EventPublisher struct {
    nats *nats.Conn
}

func (p *EventPublisher) PublishThreatDetected(ctx context.Context, threat *Threat) error {
    event := &ThreatDetectedEvent{
        ThreatID: threat.ID,
        Severity: threat.Severity,
        Timestamp: time.Now(),
    }
    
    data, _ := json.Marshal(event)
    return p.nats.Publish("threat.detected", data)
}

// Event subscription
func (s *NotificationService) HandleThreatDetected(msg *nats.Msg) {
    var event ThreatDetectedEvent
    json.Unmarshal(msg.Data, &event)
    
    // Send notification
    s.SendAlert(event)
}

Copy

Insert at cursor
go
🔐 Security & Authentication
Service Mesh Security
# Istio service mesh configuration
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: shieldx-system
spec:
  mtls:
    mode: STRICT
---
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: shieldx-auth-policy
  namespace: shieldx-system
spec:
  rules:
  - from:
    - source:
        principals: ["cluster.local/ns/shieldx-system/sa/gateway"]
    to:
    - operation:
        methods: ["GET", "POST"]

Copy

Insert at cursor
yaml
JWT Authentication
// JWT middleware for service authentication
func JWTMiddleware(secret []byte) gin.HandlerFunc {
    return gin.HandlerFunc(func(c *gin.Context) {
        token := c.GetHeader("Authorization")
        if token == "" {
            c.JSON(401, gin.H{"error": "missing token"})
            c.Abort()
            return
        }
        
        claims, err := validateJWT(token, secret)
        if err != nil {
            c.JSON(401, gin.H{"error": "invalid token"})
            c.Abort()
            return
        }
        
        c.Set("user", claims)
        c.Next()
    })
}

Copy

Insert at cursor
go
📈 Monitoring & Observability
Prometheus Metrics
// Service-specific metrics
var (
    requestsTotal = promauto.NewCounterVec(
        prometheus.CounterOpts{
            Name: "shieldx_service_requests_total",
            Help: "Total requests processed by service",
        },
        []string{"service", "method", "status"},
    )
    
    requestDuration = promauto.NewHistogramVec(
        prometheus.HistogramOpts{
            Name: "shieldx_service_request_duration_seconds",
            Help: "Request duration in seconds",
        },
        []string{"service", "method"},
    )
)

Copy

Insert at cursor
go
Distributed Tracing
// OpenTelemetry tracing
func (s *Service) ProcessRequest(ctx context.Context, req *Request) error {
    ctx, span := otel.Tracer("shieldx-service").Start(ctx, "process_request")
    defer span.End()
    
    span.SetAttributes(
        attribute.String("request.id", req.ID),
        attribute.String("user.id", req.UserID),
    )
    
    // Process request
    if err := s.process(ctx, req); err != nil {
        span.RecordError(err)
        span.SetStatus(codes.Error, err.Error())
        return err
    }
    
    return nil
}

Copy

Insert at cursor
go
🚀 Deployment Strategy
Blue-Green Deployment
#!/bin/bash
# Blue-Green deployment script

SERVICE_NAME=$1
NEW_VERSION=$2

# Deploy new version (green)
kubectl set image deployment/${SERVICE_NAME} ${SERVICE_NAME}=${SERVICE_NAME}:${NEW_VERSION}

# Wait for rollout
kubectl rollout status deployment/${SERVICE_NAME}

# Run health checks
if ! ./scripts/health-check.sh ${SERVICE_NAME}; then
    echo "Health check failed, rolling back"
    kubectl rollout undo deployment/${SERVICE_NAME}
    exit 1
fi

echo "Deployment successful"

Copy

Insert at cursor
bash
Canary Deployment
# Argo Rollouts canary deployment
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: shieldx-gateway
spec:
  replicas: 10
  strategy:
    canary:
      steps:
      - setWeight: 10
      - pause: {duration: 1m}
      - setWeight: 50
      - pause: {duration: 2m}
      - setWeight: 100
  selector:
    matchLabels:
      app: shieldx-gateway
  template:
    metadata:
      labels:
        app: shieldx-gateway
    spec:
      containers:
      - name: gateway
        image: shieldx/gateway:latest


Copy

Insert at cursor
yaml
🔄 CI/CD Pipeline
GitHub Actions Workflow
# .github/workflows/service-ci.yml
name: Service CI/CD

on:
  push:
    paths:
      - 'services/shieldx-*/**'

jobs:
  detect-changes:
    runs-on: ubuntu-latest
    outputs:
      services: ${{ steps.changes.outputs.services }}
    steps:
      - uses: actions/checkout@v3
      - uses: dorny/paths-filter@v2
        id: changes
        with:
          filters: |
            gateway:
              - 'services/shieldx-gateway/**'
            auth:
              - 'services/shieldx-auth/**'
            policy:
              - 'services/shieldx-policy/**'

  build-and-test:
    needs: detect-changes
    runs-on: ubuntu-latest
    strategy:
      matrix:
        service: ${{ fromJSON(needs.detect-changes.outputs.services) }}
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-go@v3
        with:
          go-version: 1.25
      
      - name: Test
        run: |
          cd services/shieldx-${{ matrix.service }}
          go test ./...
      
      - name: Build
        run: |
          cd services/shieldx-${{ matrix.service }}
          docker build -t shieldx/${{ matrix.service }}:${{ github.sha }} .
      
      - name: Deploy
        if: github.ref == 'refs/heads/main'
        run: |
          kubectl set image deployment/shieldx-${{ matrix.service }} \
            shieldx-${{ matrix.service }}=shieldx/${{ matrix.service }}:${{ github.sha }}


Copy

Insert at cursor
yaml
📋 Development Workflow
1. Service Development
# Create new service
./scripts/create-service.sh shieldx-newservice

# Start development environment
cd services/shieldx-newservice
make dev

# Run tests
make test

# Build and deploy
make build
make deploy

Copy

Insert at cursor
bash
2. Local Development Setup
# Start all services locally
cd docker-compose
docker-compose up -d

# Start specific service for development
cd services/shieldx-gateway
make dev-local

# Run integration tests
cd tools/testing
go run main.go --suite=integration

Copy

Insert at cursor
bash
3. Service Template Generator
#!/bin/bash
# scripts/create-service.sh

SERVICE_NAME=$1

mkdir -p services/${SERVICE_NAME}
cd services/${SERVICE_NAME}

# Generate service structure
cat > go.mod << EOF
module github.com/shieldx-bot/${SERVICE_NAME}

go 1.25.0

require (
    github.com/gin-gonic/gin v1.9.1
    github.com/shieldx-bot/shieldx-proto v0.1.0
    github.com/shieldx-bot/shieldx-common v0.1.0
)
EOF

# Generate main.go
cat > cmd/main.go << 'EOF'
package main

import (
    "log"
    "github.com/gin-gonic/gin"
)

func main() {
    r := gin.Default()
    r.GET("/health", func(c *gin.Context) {
        c.JSON(200, gin.H{"status": "ok"})
    })
    
    log.Println("Starting service on :8080")
    r.Run(":8080")
}
EOF

# Generate Dockerfile
cat > Dockerfile << 'EOF'
FROM golang:1.25-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN go build -o bin/service cmd/main.go

FROM alpine:latest
RUN apk --no-cache add ca-certificates
WORKDIR /root/
COPY --from=builder /app/bin/service .
EXPOSE 8080
CMD ["./service"]
EOF

# Generate Makefile
cat > Makefile << 'EOF'
.PHONY: build test dev deploy

build:
	go build -o bin/service cmd/main.go

test:
	go test ./...

dev:
	go run cmd/main.go

deploy:
	docker build -t ${SERVICE_NAME}:latest .
	kubectl apply -f deployments/k8s/
EOF

echo "Service ${SERVICE_NAME} created successfully!"


Copy

Insert at cursor
bash
🎯 Best Practices Summary
1. Service Design
Single responsibility per service

Database per service

API-first design with OpenAPI specs

Event-driven communication

2. Development
Feature flags for gradual rollouts

Comprehensive testing (unit, integration, e2e)

Code generation from protobuf

Shared libraries for common functionality

3. Operations
Health checks and readiness probes

Graceful shutdown handling

Circuit breakers and retries

Comprehensive monitoring and alerting

4. Security
mTLS between services

JWT authentication

RBAC authorization

Secrets management with Vault

5. Deployment
Blue-green and canary deployments

Infrastructure as code

Automated rollbacks

Environment parity

🎉 Kết Luận
Kiến trúc microservices này cung cấp:

✅ Độc lập hoàn toàn - Mỗi service có database, API, deployment riêng
✅ Scalability - Scale từng service theo nhu cầu
✅ Technology diversity - Mỗi service có thể dùng tech stack khác nhau
✅ Team autonomy - Mỗi team sở hữu end-to-end service
✅ Fault isolation - Lỗi ở 1 service không ảnh hưởng toàn bộ hệ thống
✅ Continuous deployment - Deploy từng service độc lập

Kết quả: Platform có thể scale to millions of users với hundreds of developers! 🚀