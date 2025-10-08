 

-----

# 🛡️ ShieldX Policy Service

[](https://golang.org)
[](https://www.openpolicyagent.org/)
[](https://opensource.org/licenses/Apache-2.0)
[](https://www.docker.com/)

**ShieldX Policy Service** is a central service for securely and flexibly managing, enforcing, and deploying security policies, using Open Policy Agent (OPA) as its core engine.

## 📋 Table of Contents

  - [🎯 Overview](https://www.google.com/search?q=%23-overview)
      - [Key Features](https://www.google.com/search?q=%23key-features)
      - [Technology Stack](https://www.google.com/search?q=%23technology-stack)
  - [🏗️ Architecture](https://www.google.com/search?q=%23%EF%B8%8F-architecture)
      - [System Architecture](https://www.google.com/search?q=%23system-architecture)
      - [Policy Deployment Flow](https://www.google.com/search?q=%23policy-deployment-flow)
  - [🚀 Quick Start](https://www.google.com/search?q=%23-quick-start)
      - [Prerequisites](https://www.google.com/search?q=%23prerequisites)
      - [Installation Guide](https://www.google.com/search?q=%23installation-guide)
  - [📡 API Reference](https://www.google.com/search?q=%23-api-reference)
      - [Policy Management](https://www.google.com/search?q=%23policy-management)
      - [Policy Evaluation](https://www.google.com/search?q=%23policy-evaluation)
      - [Policy Bundles](https://www.google.com/search?q=%23policy-bundles)
      - [Policy Rollout](https://www.google.com/search?q=%23policy-rollout)
      - [Policy Testing](https://www.google.com/search?q=%23policy-testing)
      - [Decision Logging](https://www.google.com/search?q=%23decision-logging)
  - [📝 Rego Policy Examples](https://www.google.com/search?q=%23-rego-policy-examples)
  - [🔄 Rollout Process](https://www.google.com/search?q=%23-rollout-process)
  - [🧪 Policy Testing](https://www.google.com/search?q=%23-policy-testing-1)
  - [💻 Development Guide](https://www.google.com/search?q=%23-development-guide)
  - [📊 Monitoring](https://www.google.com/search?q=%23-monitoring)
  - [🔧 Troubleshooting](https://www.google.com/search?q=%23-troubleshooting)
  - [🤝 Contributing](https://www.google.com/search?q=%23-contributing)
  - [📄 License](https://www.google.com/search?q=%23-license)

-----

## 🎯 Overview

### Key Features

  - **Centralized Policy Management**: Provides an API to create, read, update, and delete (CRUD) policies with version control.
  - **Real-Time Evaluation**: Executes Rego policies with extremely low latency (typically under 10ms).
  - **Policy Bundles**: Packages multiple policies into a single bundle, digitally signed with `Cosign` to ensure integrity.
  - **Gradual Rollout**: Safely deploys policy changes in stages (Canary, Beta, Production).
  - **Automatic Rollback**: Automatically reverts to the previous stable version if the new version's error rate exceeds a predefined threshold.
  - **Integrated Testing**: Provides an API to run test cases for policies directly within the system.
  - **Comprehensive Audit Trail**: Records every decision made by the policy engine for investigation and compliance purposes.
  - **Performance Optimization**: Uses Redis to cache decisions, minimizing latency for repeated requests.

### Technology Stack

| Component | Technology | Version |
| :--- | :--- | :--- |
| Language | Go | 1.25+ |
| Policy Engine | Open Policy Agent (OPA) | 0.60+ |
| Policy Language | Rego | - |
| Database | PostgreSQL | 15+ |
| Bundle Storage | MinIO / AWS S3 | - |
| Cache | Redis | 7+ |
| Digital Signing | Cosign | 2.0+ |

-----

## 🏗️ Architecture

### System Architecture

```plaintext
┌─────────────────────────────────────────────────────┐
│ ShieldX Policy Service (Port 5007)                  │
├─────────────────────────────────────────────────────┤
│ API Layer (REST & gRPC)                             │
│ - Policy, Bundle, & Rollout Management              │
│ - Policy Evaluation API                             │
│ - Prometheus Metrics Endpoint                       │
├─────────────────────────────────────────────────────┤
│ Policy Management Layer                             │
│ - CRUD & Version Control                            │
│ - Bundle Management & Signing                       │
├─────────────────────────────────────────────────────┤
│ OPA Evaluation Engine                               │
│ - Rego Compilation & Optimization                   │
│ - Decision Caching                                  │
│ - Policy Enforcement                                │
├─────────────────────────────────────────────────────┤
│ Rollout Manager                                     │
│ - Canary Deployment & Traffic Splitting             │
│ - Health Monitoring & Auto-Rollback                 │
│ - A/B Testing & Auto-Promotion                      │
├─────────────────────────────────────────────────────┤
│ Data Layer                                          │
│ - PostgreSQL (Metadata) - S3 (Bundles) - Redis (Cache)│
└─────────────────────────────────────────────────────┘
```

### Policy Deployment Flow

```plaintext
Policy Upload → Validate → Compile → Sign → Create Bundle
      │
      └─> Canary (1%) ── Monitor (10min) ─┬─> Success? ──> Beta (10%)
                                          │
                                          └─> Fail? ─────> Auto Rollback
                                                          (Notify Team)
```

-----

## 🚀 Quick Start

### Prerequisites

  - Go `1.25+`
  - OPA `0.60+`
  - PostgreSQL `15+`
  - MinIO / AWS S3
  - Redis `7+`
  - Cosign `2.0+`
  - Docker & Docker Compose

### Installation Guide

Follow these steps to install and run the service in a local environment.

```bash
# 1. Clone the repository and install dependencies
git clone https://github.com/shieldx-bot/shieldx.git
cd shieldx/services/shieldx-policy
go mod download

# 2. Install OPA (example for Linux)
curl -L -o opa https://openpolicyagent.org/downloads/latest/opa_linux_amd64
chmod +x opa && sudo mv opa /usr/local/bin/
opa version

# 3. Install Cosign (example for Linux)
wget "https://github.com/sigstore/cosign/releases/download/v2.2.3/cosign-linux-amd64" -O cosign
chmod +x cosign && sudo mv cosign /usr/local/bin/
cosign version

# 4. Start background services using Docker
docker run -d --name shieldx-postgres -e POSTGRES_USER=policy_user -e POSTGRES_PASSWORD=policy_pass -e POSTGRES_DB=shieldx_policy -p 5432:5432 postgres:15-alpine
docker run -d --name shieldx-minio -p 9000:9000 -p 9001:9001 -e MINIO_ROOT_USER=minioadmin -e MINIO_ROOT_PASSWORD=minioadmin minio/minio server /data --console-address ":9001"
docker run -d --name shieldx-redis -p 6379:6379 redis:7-alpine

# 5. Configure MinIO client and create a bucket
mc alias set local http://localhost:9000 minioadmin minioadmin
mc mb local/shieldx-policies

# 6. Configure environment variables
export POLICY_PORT=5007
export POLICY_DB_HOST=localhost
# ... (Add other environment variables)

# 7. Run database migrations
migrate -path ./migrations -database "postgresql://policy_user:policy_pass@localhost:5432/shieldx_policy?sslmode=disable" up

# 8. Generate a key pair for signing bundles
cosign generate-key-pair
# Move and protect the private key
mkdir -p /keys
mv cosign.key /keys/policy-signing.key
mv cosign.pub /keys/policy-signing.pub
chmod 600 /keys/policy-signing.key

# 9. Start the OPA server and the Policy Service
opa run --server --addr :8181 &
go build -o bin/shieldx-policy cmd/server/main.go
./bin/shieldx-policy

# 10. Verify
curl http://localhost:5007/health
# Expected result: {"status":"healthy"}
```

-----

## 📡 API Reference

**Base URL**: `http://localhost:5007/api/v1`
**Authentication**: `Authorization: Bearer <token>`

### Policy Management

#### 1\. Create New Policy

`POST /api/v1/policies`

\<details\>\<summary\>View Request Example\</summary\>

```json
{
  "tenant_id": "tenant-123",
  "name": "block_sql_injection_v2",
  "description": "Blocks common SQL injection patterns.",
  "type": "waf",
  "rego_code": "package shieldx.waf\n\ndefault allow = true\n\nallow = false {\n  regex.match(`(?i)(union|select|insert)`, input.request.body)\n}",
  "enabled": true,
  "priority": 100
}
```

\</details\>

#### 2\. List Policies

`GET /api/v1/policies?type=waf&enabled=true`

### Policy Evaluation

#### 1\. Evaluate a Policy

`POST /api/v1/evaluate`

\<details\>\<summary\>View Request Example\</summary\>

```json
{
  "policy_name": "block_sql_injection_v2",
  "input": {
    "request": {
      "method": "POST",
      "body": "SELECT * FROM users"
    }
  }
}
```

\</details\>
\<details\>\<summary\>View Response Example\</summary\>

```json
{
  "decision_id": "dec-xyz-123",
  "allowed": false,
  "violations": ["SQL injection detected"],
  "evaluation_time_ms": 3,
  "cached": false
}
```

\</details\>

### Policy Bundles

#### 1\. Create a Policy Bundle

`POST /api/v1/bundles`

\<details\>\<summary\>View Request Example\</summary\>

```json
{
  "name": "waf_policies_prod_20251008",
  "policy_names": ["block_sql_injection_v2", "block_xss_v1"],
  "sign": true
}
```

\</details\>

### Policy Rollout

#### 1\. Create a Rollout Process

`POST /api/v1/rollouts`

\<details\>\<summary\>View Request Example\</summary\>

```json
{
  "policy_name": "block_sql_injection_v2",
  "target_version": "v1.1.0",
  "strategy": "gradual",
  "stages": [
    {"name": "canary", "traffic_percentage": 1, "duration_minutes": 10},
    {"name": "beta", "traffic_percentage": 10, "duration_minutes": 30},
    {"name": "production", "traffic_percentage": 100}
  ],
  "auto_promote": true,
  "auto_rollback_threshold": 0.01
}
```

\</details\>

### Policy Testing

#### 1\. Run Test Cases for a Policy

`POST /api/v1/policies/{policy_id}/test`

\<details\>\<summary\>View Request Example\</summary\>

```json
{
  "test_cases": [
    {
      "name": "Block simple SQL injection",
      "input": {"request": {"body": "SELECT * FROM users"}},
      "expected_decision": {"allowed": false}
    },
    {
      "name": "Allow normal request",
      "input": {"request": {"body": "This is a normal request"}},
      "expected_decision": {"allowed": true}
    }
  ]
}
```

\</details\>

### Decision Logging

#### 1\. Query Decision History

`GET /api/v1/decisions?policy_id=pol-456&decision=deny`

-----

## 📝 Rego Policy Examples

#### 1\. WAF (Web Application Firewall) Policy

```rego
package shieldx.waf

import future.keywords.if

default allow = true

# Decision is "deny" if any violation is found
allow = false if {
    count(violations) > 0
}

violations[msg] {
    regex.match(`(?i)(union|select|insert|update|delete|drop)`, input.request.body)
    msg := "SQL injection detected in request body"
}

violations[msg] {
    regex.match(`(?i)(<script|javascript:|onerror=)`, input.request.path)
    msg := "XSS attack detected in request path"
}
```

#### 2\. RBAC (Role-Based Access Control) Policy

```rego
package shieldx.rbac

import future.keywords.if

default allow = false

# Admins can do anything
allow if {
    input.user.role == "admin"
}

# Users can only read their own data
allow if {
    input.user.role == "user"
    input.request.method == "GET"
    input.request.path == concat("/", ["/api/users", input.user.id])
}
```

-----

## 🔄 Rollout Process

We use a staged rollout process to minimize risk when updating policies.

| Stage | Traffic | Monitoring Time | Error Threshold | Success Criteria |
| :--- | :--- | :--- | :--- | :--- |
| **Canary** | 1% | 10 minutes | 1% | Success rate \> 99% |
| **Beta** | 10% | 30 minutes | 0.5% | Success rate \> 99.5% |
| **Stable** | 50% | 1 hour | 0.1% | Success rate \> 99.9% |
| **Production**| 100% | - | 0.05%| Success rate \> 99.95% |

-----

## 🧪 Policy Testing

Write test cases for policies using the Rego language itself.

```rego
# tests/waf_test.rego
package shieldx.waf

test_sql_injection_is_blocked {
    allow == false with input as {
        "request": {"body": "SELECT * FROM users WHERE id = 1 OR 1=1"}
    }
}

test_normal_request_is_allowed {
    allow == true with input as {
        "request": {"body": "This is a normal text"}
    }
}
```

Run tests using the OPA CLI:

```bash
# Run all tests in the directory
opa test ./policies/ ./tests/

# Run tests with a coverage report
opa test --coverage ./policies/ ./tests/
```

-----

## 💻 Development Guide

### Project Structure

```
shieldx-policy/
├── cmd/server/main.go
├── internal/
│   ├── api/          # Handlers, routes, middleware
│   ├── opa/          # OPA client and evaluator
│   ├── rollout/      # Staged rollout logic
│   ├── bundle/       # Bundle creation and signing logic
│   ├── repository/   # Interaction with DB, Redis, S3
│   └── models/       # Data structures
├── policies/         # Directory containing .rego files
├── tests/            # Contains .rego test files and Go tests
├── migrations/       # SQL migrations
└── go.mod
```

-----

## 📊 Monitoring

The service provides metrics in the Prometheus format.

```
# Policy evaluations
shieldx_policy_evaluation_total{policy_id,decision}
shieldx_policy_evaluation_duration_seconds{policy_id}

# Rollouts
shieldx_policy_rollout_traffic_percentage{policy_id,stage}
shieldx_policy_rollback_total{policy_id,reason}

# Cache
shieldx_policy_cache_hits_total
shieldx_policy_cache_misses_total
```

-----

## 🔧 Troubleshooting

#### Policy Compilation Error

```bash
# Check the syntax of a rego file
opa check ./policies/waf/sql_injection.rego

# Test a decision locally
opa eval -d ./policies/ -i input.json "data.shieldx.waf.allow"
```

#### Slow Evaluation

```bash
# Enable profiling to see which rules are time-consuming
opa eval --profile -d ./policies/ "data.shieldx.waf.allow"

# Check Redis cache status
redis-cli INFO stats | grep "keyspace_hits\|keyspace_misses"
```

#### Bundle Signing Error

```bash
# Re-verify the signature of a bundle
cosign verify --key /keys/policy-signing.pub bundle.tar.gz

# Check the file permissions of the key
ls -l /keys/policy-signing.key
```

-----

## 🤝 Contributing

Please see the `CONTRIBUTING.md` file for details.

## 📄 License

This project is licensed under the [Apache License 2.0](https://github.com/shieldx-bot/shieldx/blob/main/LICENSE).