
-----

# 🛡️ ShieldX Admin Service

[](https://golang.org)
[](https://opensource.org/licenses/Apache-2.0)
[](https://www.docker.com/)

**ShieldX Admin Service** is the central administrative service for the ShieldX platform. It is responsible for multi-tenant management, user administration, security policy configuration, and system-wide settings.

## 📋 Table of Contents

  - [🎯 Overview](https://www.google.com/search?q=%23-overview)
      - [Core Responsibilities](https://www.google.com/search?q=%23core-responsibilities)
      - [Key Features](https://www.google.com/search?q=%23key-features)
      - [Technology Stack](https://www.google.com/search?q=%23technology-stack)
  - [🏗️ Architecture](https://www.google.com/search?q=%23%EF%B8%8F-architecture)
      - [System Architecture](https://www.google.com/search?q=%23system-architecture)
      - [Request Flow](https://www.google.com/search?q=%23request-flow)
  - [🚀 Getting Started](https://www.google.com/search?q=%23-getting-started)
      - [Prerequisites](https://www.google.com/search?q=%23prerequisites)
      - [Installation Steps](https://www.google.com/search?q=%23installation-steps)
  - [📡 API Reference](https://www.google.com/search?q=%23-api-reference)
      - [Tenant Management](https://www.google.com/search?q=%23tenant-management)
      - [User Management](https://www.google.com/search?q=%23user-management)
      - [Policy Management](https://www.google.com/search?q=%23policy-management)
      - [Audit Logs](https://www.google.com/search?q=%23audit-logs)
      - [System Configuration](https://www.google.com/search?q=%23system-configuration)
  - [🗄️ Database Schema](https://www.google.com/search?q=%23%EF%B8%8F-database-schema)
  - [💻 Development Guide](https://www.google.com/search?q=%23-development-guide)
      - [Project Structure](https://www.google.com/search?q=%23project-structure)
      - [Coding Standards](https://www.google.com/search?q=%23coding-standards)
  - [🧪 Testing](https://www.google.com/search?q=%23-testing)
  - [🚢 Deployment](https://www.google.com/search?q=%23-deployment)
      - [Docker](https://www.google.com/search?q=%23docker)
      - [Docker Compose](https://www.google.com/search?q=%23docker-compose)
      - [Kubernetes](https://www.google.com/search?q=%23kubernetes)
  - [📊 Monitoring](https://www.google.com/search?q=%23-monitoring)
  - [🔧 Troubleshooting](https://www.google.com/search?q=%23-troubleshooting)
  - [📚 Resources](https://www.google.com/search?q=%23-resources)
  - [🤝 Contributing](https://www.google.com/search?q=%23-contributing)
  - [📄 License](https://www.google.com/search?q=%23-license)

-----

## 🎯 Overview

### Core Responsibilities

  - **Tenant Management**: Manages multi-tenant organizations/customers with complete data isolation.
  - **User Management**: Handles user administration and Role-Based Access Control (RBAC).
  - **Policy Configuration**: Manages the configuration of security policies.
  - **System Configuration**: Controls global system settings and preferences.
  - **Audit Logging**: Provides a comprehensive audit trail for compliance (SOC 2, ISO 27001).
  - **Service Health Monitoring**: Monitors and reports the health status of all ShieldX services.

### Key Features

✅ Multi-tenant architecture with complete data isolation
✅ Role-Based Access Control (RBAC) with granular permissions
✅ RESTful API with OpenAPI/Swagger documentation
✅ Comprehensive audit trail for compliance requirements
✅ Real-time metrics with Prometheus integration
✅ Graceful shutdown and health checks
✅ JWT-based authentication

### Technology Stack

| Component | Technology | Version |
| :--- | :--- | :--- |
| **Language** | Go | 1.25+ |
| **Framework** | Gin Web Framework | Latest |
| **Database** | PostgreSQL | 15+ |
| **Cache** | Redis | 7+ |
| **Metrics** | Prometheus | Latest |
| **Authentication** | JWT tokens | - |

-----

## 🏗️ Architecture

### System Architecture

```plaintext
┌──────────────────────────────────────────────────────────┐
│ ShieldX Admin Service                                    │
│ (Port 8082)                                              │
├──────────────────────────────────────────────────────────┤
│           ▲                      ▲                       ▲           │
│    ┌──────┴──────┐        ┌──────┴──────┐         ┌──────┴─────┐     │
│    │  HTTP API   │        │   gRPC API  │         │  Metrics   │     │
│    │   (REST)    │        │ (Internal)  │         │ /metrics   │     │
│    └──────┬──────┘        └──────┬──────┘         └──────┬─────┘     │
│           │                      │                       │           │
│    ┌──────▼──────────────────────▼───────────────────────▼─────────┐ │
│    │ Business Logic Layer                                         │ │
│    │ - Tenant Service      - User Service                         │ │
│    │ - Policy Service      - Config Service                       │ │
│    └──────────────────────────┬────────────────────────────────────┘ │
│                               │                                      │
│    ┌──────────────────────────▼────────────────────────────────────┐ │
│    │ Data Access Layer (Repository)                               │ │
│    │ - PostgreSQL Repository                                      │ │
│    │ - Redis Cache Layer                                          │ │
│    └──────────────────────────┬────────────────────────────────────┘ │
│                               │                                      │
└───────────────────────────────┼──────────────────────────────────────┘
                                │
                  ┌─────────────┴─────────────┐
            ┌─────▼─────┐               ┌─────▼─────┐
            │PostgreSQL │               │   Redis   │
            │ (Primary) │               │  (Cache)  │
            └───────────┘               └───────────┘
```

### Request Flow

```
Client Request
    ↓
Authentication Middleware (JWT Validation)
    ↓
Rate Limiting & Logging Middleware
    ↓
Request Handler (Controller)
    ↓
Business Logic (Service Layer)
    ↓
Data Access (Repository Layer)
    ↓
Database / Cache (PostgreSQL / Redis)
    ↓
Response (with Audit Log generation)
```

-----

## 🚀 Getting Started

### Prerequisites

  - Go 1.25+
  - PostgreSQL 15+
  - Redis 7+
  - Docker & Docker Compose
  - `migrate` CLI for database migrations

### Installation Steps

```bash
# 1. Clone the repository
git clone https://github.com/shieldx-bot/shieldx.git
cd shieldx/services/shieldx-admin

# 2. Install dependencies
go mod download

# 3. Setup PostgreSQL and Redis with Docker
docker run -d --name shieldx-postgres -e POSTGRES_USER=admin_user -e POSTGRES_PASSWORD=admin_pass -e POSTGRES_DB=shieldx_admin -p 5432:5432 postgres:15-alpine
docker run -d --name shieldx-redis -p 6379:6379 redis:7-alpine

# 4. Configure environment (create a .env file)
# You can copy from .env.example and modify it
export ADMIN_PORT=8082
export ADMIN_DB_HOST=localhost
export ADMIN_DB_USER=admin_user
export ADMIN_DB_PASSWORD=admin_pass
export ADMIN_DB_NAME=shieldx_admin
export ADMIN_REDIS_HOST=localhost
export ADMIN_JWT_SECRET="your-super-secret-key-of-at-least-32-characters"

# 5. Run database migrations
# Ensure you have the migrate CLI installed
migrate -path ./migrations -database "postgresql://admin_user:admin_pass@localhost:5432/shieldx_admin?sslmode=disable" up

# 6. Build & Run
go build -o bin/shieldx-admin cmd/server/main.go
./bin/shieldx-admin

# 7. Verify installation
curl http://localhost:8082/health
# Expected response should indicate a "healthy" status for all dependencies.
```

-----

## 📡 API Reference

**Base URL**: `http://localhost:8082/api/v1`
**Authentication**: All endpoints require a JWT token in the `Authorization: Bearer <token>` header.

### Tenant Management

#### 1\. Create Tenant

`POST /api/v1/tenants`

\<details\>\<summary\>View Request Example\</summary\>

```json
{
  "name": "Acme Corporation",
  "slug": "acme-corp",
  "email": "admin@acme.com",
  "plan": "enterprise",
  "settings": {
    "max_users": 100,
    "features": ["waf", "sandbox"]
  }
}
```

\</details\>
\<details\>\<summary\>View Response Example (201 Created)\</summary\>

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "Acme Corporation",
  "slug": "acme-corp",
  "plan": "enterprise",
  "status": "active",
  "created_at": "2025-10-08T17:09:32Z"
}
```

\</details\>

#### 2\. List Tenants

`GET /api/v1/tenants?page=1&limit=20`

#### 3\. Get Tenant Details

`GET /api/v1/tenants/{tenant_id}`

#### 4\. Update Tenant

`PUT /api/v1/tenants/{tenant_id}`

#### 5\. Delete Tenant (Soft Delete)

`DELETE /api/v1/tenants/{tenant_id}`

### User Management

#### 1\. Create User

`POST /api/v1/users`

\<details\>\<summary\>View Request Example\</summary\>

```json
{
  "tenant_id": "550e8400-e29b-41d4-a716-446655440000",
  "email": "john.doe@acme.com",
  "name": "John Doe",
  "role": "admin",
  "permissions": ["users:create", "users:read"]
}
```

\</details\>

#### 2\. List Users

`GET /api/v1/users?tenant_id={tenant_id}`

#### 3\. Deactivate User

`POST /api/v1/users/{user_id}/deactivate`

### Policy Management

#### 1\. Create Policy

`POST /api/v1/policies`

\<details\>\<summary\>View Request Example\</summary\>

```json
{
  "tenant_id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "Block SQL Injection",
  "type": "waf",
  "rules": {
    "pattern": "(?i)(union|select|insert)",
    "action": "block"
  },
  "enabled": true
}
```

\</details\>

#### 2\. List Policies

`GET /api/v1/policies?tenant_id={tenant_id}`

### Audit Logs

#### Query Audit Logs

`GET /api/v1/audit?tenant_id={tenant_id}&action=user.created`

\<details\>\<summary\>View Response Example\</summary\>

```json
{
  "data": [
    {
      "id": "770e8400-e29b-41d4-a716-446655440002",
      "tenant_id": "550e8400-e29b-41d4-a716-446655440000",
      "user_id": "660e8400-e29b-41d4-a716-446655440001",
      "action": "user.created",
      "resource": "users/660e8400-e29b-41d4-a716-446655440001",
      "timestamp": "2025-10-08T17:09:32Z",
      "ip_address": "192.168.1.100",
      "status": "success"
    }
  ],
  "pagination": { "total": 1500, "limit": 100, "offset": 0 }
}
```

\</details\>

### System Configuration

#### 1\. Get Configuration

`GET /api/v1/config`

#### 2\. Update Configuration

`PUT /api/v1/config`

-----

## 🗄️ Database Schema

A simplified view of the primary tables.

#### `tenants` Table

```sql
CREATE TABLE tenants (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) UNIQUE NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    settings JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE
);
```

#### `users` Table

```sql
CREATE TABLE users (
    id UUID PRIMARY KEY,
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    email VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    password_hash VARCHAR(255),
    role VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    UNIQUE(tenant_id, email)
);
```

#### `audit_logs` Table

```sql
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY,
    tenant_id UUID,
    user_id UUID,
    action VARCHAR(100) NOT NULL,
    resource VARCHAR(255),
    ip_address INET,
    details JSONB,
    status VARCHAR(20),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

-----

## 💻 Development Guide

### Project Structure

```plaintext
shieldx-admin/
├── cmd/server/main.go         # Application entry point
├── internal/
│   ├── api/                   # Handlers, routes, middleware
│   ├── service/               # Business logic layer
│   ├── repository/            # Data access layer
│   ├── models/                # Domain models
│   └── config/                # Configuration
├── pkg/                       # Shared packages (logger, jwt, etc.)
├── migrations/                # Database migrations
└── tests/                     # Unit and integration tests
```

### Coding Standards

  - **Error Handling**: Wrap errors with context for better traceability.
  - **Context Propagation**: Always pass `context.Context` as the first parameter in service and repository methods.
  - **Structured Logging**: Use structured logging (e.g., zerolog) for all log messages to make them machine-readable.

-----

## 🧪 Testing

```bash
# Run all tests
make test

# Run unit tests only
go test ./internal/... -v

# Run integration tests (requires a running test database)
go test ./tests/integration/... -v

# Run tests with code coverage
go test -coverprofile=coverage.out ./... && go tool cover -html=coverage.out
```

-----

## 🚢 Deployment

### Docker

A multi-stage `Dockerfile` is provided for building a lightweight, production-ready container image.

```bash
# Build the Docker image
docker build -t shieldx-admin:latest .

# Run the container
docker run -d -p 8082:8082 --name shieldx-admin -e ADMIN_DB_HOST=... shieldx-admin:latest
```

### Docker Compose

A `docker-compose.yml` file is included for easy local development and testing setups, orchestrating the admin service, PostgreSQL, and Redis.

```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down
```

### Kubernetes

Example Kubernetes manifests (`deployment.yaml`, `service.yaml`, `hpa.yaml`) are provided for deploying the service to a Kubernetes cluster. The deployment includes readiness/liveness probes and resource requests/limits for robust operation.

-----

## 📊 Monitoring

The service exposes Prometheus metrics at the `/metrics` endpoint. Key metrics include:

  - `shieldx_admin_http_requests_total`: Total number of HTTP requests.
  - `shieldx_admin_http_request_duration_seconds`: HTTP request latency.
  - `shieldx_admin_tenants_total`: Total number of tenants.
  - `shieldx_admin_users_total`: Total number of users.

-----

## 🔧 Troubleshooting

#### Database Connection Failed

  - **Check**: Ensure the PostgreSQL container is running (`docker ps`) and accessible.
  - **Verify**: Test the connection manually using `psql`. Check credentials and host in the `.env` file.

#### High Memory Usage

  - **Check**: Use `docker stats shieldx-admin` to monitor container resources.
  - **Profile**: Enable the pprof endpoint (`/debug/pprof`) to analyze memory usage and identify potential goroutine leaks.

#### Slow API Queries

  - **Enable**: Turn on slow query logging in PostgreSQL.
  - **Analyze**: Use `EXPLAIN ANALYZE` on suspected slow queries to check for missing indexes or inefficient query plans.

-----

## 📚 Resources

  - [ShieldX Architecture Documentation](https://www.google.com/search?q=https://docs.shieldx.dev/architecture)
  - [API Documentation (Swagger)](https://www.google.com/search?q=http://localhost:8082/swagger/index.html)
  - [Go Best Practices](https://go.dev/doc/effective_go)

-----

## 🤝 Contributing

We welcome contributions\! Please see `CONTRIBUTING.md` for details on our code of conduct, development setup, and pull request process.

-----

## 📄 License

This project is licensed under the Apache License 2.0. See the `LICENSE` file for details.