 

-----

# 🛡️ ShieldX API Gateway

[](https://golang.org)
[](https://opensource.org/licenses/Apache-2.0)
[](https://www.docker.com/)

**ShieldX API Gateway** is the single entry point for the entire ShieldX system. This service is responsible for routing, authenticating, rate-limiting, and monitoring all HTTP requests to internal microservices.

## 📋 Table of Contents

  - [🎯 Overview](https://www.google.com/search?q=%23-overview)
      - [Key Features](https://www.google.com/search?q=%23key-features)
      - [Technology Stack](https://www.google.com/search?q=%23technology-stack)
  - [🏗️ System Architecture](https://www.google.com/search?q=%23%EF%B8%8F-system-architecture)
  - [🚀 Quick Start](https://www.google.com/search?q=%23-quick-start)
      - [Prerequisites](https://www.google.com/search?q=%23prerequisites)
      - [Installation & Startup](https://www.google.com/search?q=%23installation--startup)
  - [📡 API Reference](https://www.google.com/search?q=%23-api-reference)
      - [Gateway Endpoints](https://www.google.com/search?q=%23gateway-endpoints)
      - [Proxied Routes](https://www.google.com/search?q=%23proxied-routes)
  - [🚦 Traffic Management](https://www.google.com/search?q=%23-traffic-management)
      - [Rate Limiting](https://www.google.com/search?q=%23rate-limiting)
      - [Circuit Breaker](https://www.google.com/search?q=%23circuit-breaker)
      - [Load Balancing](https://www.google.com/search?q=%23load-balancing)
  - [🔐 Security Features](https://www.google.com/search?q=%23-security-features)
      - [Centralized Authentication](https://www.google.com/search?q=%23centralized-authentication)
      - [CORS Configuration](https://www.google.com/search?q=%23cors-configuration)
      - [Request Validation](https://www.google.com/search?q=%23request-validation)
  - [💻 Development Guide](https://www.google.com/search?q=%23-development-guide)
      - [Project Structure](https://www.google.com/search?q=%23project-structure)
      - [Adding a New Route](https://www.google.com/search?q=%23adding-a-new-route)
      - [Creating Custom Middleware](https://www.google.com/search?q=%23creating-custom-middleware)
  - [🧪 Testing](https://www.google.com/search?q=%23-testing)
  - [📊 Monitoring & Logging](https://www.google.com/search?q=%23-monitoring--logging)
  - [🔧 Troubleshooting](https://www.google.com/search?q=%23-troubleshooting)
  - [📚 References](https://www.google.com/search?q=%23-references)
  - [📄 License](https://www.google.com/search?q=%23-license)

-----

## 🎯 Overview

### Key Features

  - **Intelligent Routing**: Routes requests to the appropriate microservices based on path, headers, or method.
  - **Rate Limiting**: Protects backend services from being overwhelmed by limiting the number of requests per IP or tenant.
  - **Authentication Gateway**: Centrally validates JWT tokens before forwarding requests.
  - **Data Transformation**: Flexibly modifies the headers and body of requests/responses.
  - **Load Balancing**: Distributes traffic evenly among multiple instances of a service.
  - **Circuit Breaking**: Automatically disconnects from services that are experiencing issues to prevent cascade failures.
  - **API Versioning**: Supports routing to different API versions (e.g., `/api/v1`, `/api/v2`).

### Technology Stack

  - **Language**: Go 1.25+
  - **Framework**: Gin Web Framework
  - **Proxy**: `net/http/httputil` Reverse Proxy
  - **Cache & Rate Limiting**: Redis 7+
  - **Service Discovery**: Consul or Kubernetes DNS
  - **Protocols**: HTTP/2, QUIC support

-----

## 🏗️ System Architecture

```plaintext
┌─────────────────────────────────────────────────────┐
│ ShieldX API Gateway (Port 8081)                     │
├─────────────────────────────────────────────────────┤
│ Ingress Layer                                       │
│ - TLS Termination                                   │
│ - Basic DDoS Protection                             │
│ - Initial Request Validation                        │
├─────────────────────────────────────────────────────┤
│ Middleware Pipeline                                 │
│ - Authentication                                    │
│ - Rate Limiting                                     │
│ - CORS                                              │
│ - Logging                                           │
│ - Circuit Breaker                                   │
├─────────────────────────────────────────────────────┤
│ Routing Engine                                      │
│ - Path-based routing                                │
│ - Header-based routing                              │
│ - Load balancing                                    │
├─────────────────────────────────────────────────────┤
│ Service Proxy                                       │
│ - HTTP Reverse Proxy                                │
│ - Request/Response transformation                   │
└─────────────────────────────────────────────────────┘
       │                      │                      │
┌──────▼──────┐        ┌──────▼──────┐        ┌──────▼──────┐
│ Admin Service│        │ Auth Service│        │Credit Service│
└─────────────┘        └─────────────┘        └─────────────┘
```

-----

## 🚀 Quick Start

### Prerequisites

  - Go `1.25` or newer
  - Redis `7` or newer
  - Docker & Docker Compose
  - (Optional) Consul for service discovery

### Installation & Startup

```bash
# 1. Clone the repository
git clone https://github.com/shieldx-bot/shieldx.git
cd shieldx/services/shieldx-gateway

# 2. Install dependencies
go mod download

# 3. Start Redis using Docker
docker run -d \
  --name shieldx-redis \
  -p 6379:6379 \
  redis:7-alpine

# 4. Configure environment variables (example for local services)
export GATEWAY_PORT=8081
export GATEWAY_REDIS_HOST=localhost
export GATEWAY_AUTH_SERVICE_URL=http://localhost:5001
export GATEWAY_ADMIN_SERVICE_URL=http://localhost:8082
export GATEWAY_CREDITS_SERVICE_URL=http://localhost:5004
# ...Add other services here

# 5. Build and run the application
go build -o bin/shieldx-gateway cmd/server/main.go
./bin/shieldx-gateway

# 6. Check the service status
# You should receive {"status": "ok"} if successful
curl http://localhost:8081/health
```

-----

## 📡 API Reference

**Base URL**: `http://localhost:8081`

### Gateway Endpoints

These are the endpoints provided by the Gateway itself for administration and monitoring.

  - `GET /health`: Checks the status of the Gateway and the services it connects to.
  - `GET /metrics`: Provides Prometheus metrics on the Gateway's performance.

### Proxied Routes

The Gateway automatically forwards requests with matching prefixes to the corresponding services:

| Request Path | Destination Service |
| :--- | :--- |
| `/api/v1/auth/*` | Auth Service (`:5001`) |
| `/api/v1/admin/*` | Admin Service (`:8082`) |
| `/api/v1/credits/*` | Credits Service (`:5004`) |
| `/api/v1/deception/*` | Deception Service (`:5005`) |
| `/api/v1/forensics/*` | Forensics Service (`:5006`) |

#### Example: Login via Gateway

When a client sends a `POST /api/v1/auth/login` request, the Gateway performs the following steps:

1.  Applies middleware: CORS, Rate Limiting, Logging.
2.  Validates the JWT token in the `Authorization` header.
3.  If valid, proxies the request to `http://<auth-service-url>/login`.
4.  Receives the response from the Auth Service.
5.  Adds custom headers (e.g., `X-Request-ID`) to the response.
6.  Returns the response to the client.

-----

## 🚦 Traffic Management

### Rate Limiting

Configuration is defined in the `configs/gateway.yaml` file.

```yaml
rate_limiting:
  enabled: true
  strategy: "redis" # Use Redis to share state between instances
  default_limit: 1000 # requests / minute per IP
  burst: 200 # Number of requests allowed to exceed the limit in a short burst

  # Override limits for specific tenants
  tenant_limits:
    "tenant-premium": 10000
    "tenant-free": 500

  # Override limits for specific endpoints
  endpoint_limits:
    "/api/v1/auth/login": 10 # Only 10 requests/minute to prevent brute-force attacks
```

### Circuit Breaker

Automatically disconnects from unresponsive backend services to protect the system.

```go
// Configuration for the Circuit Breaker
type CircuitBreakerConfig struct {
    MaxFailures  uint32        // Number of consecutive failures before opening the circuit (e.g., 5)
    Timeout      time.Duration // Timeout for requests to the service (e.g., 30s)
    ResetTimeout time.Duration // Time the circuit remains open before transitioning to Half-Open (e.g., 60s)
}
```

  - **`Closed` State**: Normal operation.
  - **`Open` State**: The backend service has failed; the Gateway immediately returns a `503 Service Unavailable` error without sending a request.
  - **`Half-Open` State**: After a period, the Gateway sends a few test requests to check if the service has recovered.

### Load Balancing

Supports multiple load balancing strategies when a service has multiple instances.

```yaml
load_balancing:
  strategy: "round_robin" # Other strategies: "least_connections", "ip_hash"
  health_check:
    enabled: true
    interval: "10s"
    timeout: "5s"
    path: "/health"
```

-----

## 🔐 Security Features

### Centralized Authentication

This middleware validates tokens and adds `user_id` and `tenant_id` information to the request context.

```go
// Middleware to validate JWT tokens
func AuthMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        authHeader := c.GetHeader("Authorization")
        // ... (Logic to extract and validate the token)

        valid, claims, err := verifyToken(token) // Send gRPC/HTTP request to Auth Service
        if !valid {
            c.JSON(http.StatusUnauthorized, gin.H{"error": "Invalid or expired token"})
            c.Abort()
            return
        }

        // Attach user info to the context for downstream services to use
        c.Set("x-user-id", claims.UserID)
        c.Set("x-tenant-id", claims.TenantID)
        c.Next()
    }
}
```

### CORS Configuration

```yaml
cors:
  enabled: true
  allowed_origins:
    - "https://app.shieldx.com"
    - "https://dashboard.shieldx.com"
  allowed_methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
  allowed_headers: ["Authorization", "Content-Type", "X-Request-ID"]
  max_age: 3600
```

### Request Validation

This middleware performs basic checks, such as request size.

```go
// Middleware to check basic rules
func RequestValidationMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        // Limit request body size to 10MB
        if c.Request.ContentLength > 10 * 1024 * 1024 {
            c.JSON(http.StatusRequestEntityTooLarge, gin.H{"error": "Request body too large"})
            c.Abort()
            return
        }
        c.Next()
    }
}
```

-----

## 💻 Development Guide

### Project Structure

```
shieldx-gateway/
├── cmd/server/main.go
├── internal/
│   ├── api/
│   │   ├── middleware/   # Gateway middleware
│   │   └── routes.go     # Route and proxy definitions
│   ├── proxy/
│   │   ├── reverse_proxy.go
│   │   └── load_balancer.go
│   ├── ratelimit/
│   │   └── redis_limiter.go
│   └── config/
├── configs/
│   └── gateway.yaml      # Main configuration file
└── tests/
```

### Adding a New Route

To add a new service, you need to update the `internal/api/routes.go` file.

```go
// internal/api/routes.go
func SetupRoutes(r *gin.Engine, proxy *proxy.ReverseProxy, cfg *config.Config) {
    api := r.Group("/api/v1")
    api.Use(middleware.RateLimit(cfg))
    api.Use(middleware.Auth(cfg))
    {
        api.Any("/auth/*path", proxy.ProxyTo("auth-service"))
        api.Any("/admin/*path", proxy.ProxyTo("admin-service"))
        // Add new route here
        api.Any("/new-service/*path", proxy.ProxyTo("new-service"))
    }
}
```

### Creating Custom Middleware

```go
// internal/api/middleware/request_id.go
func RequestIDMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        requestID := uuid.New().String()
        c.Set("x-request-id", requestID)
        c.Header("X-Request-ID", requestID)
        c.Next()
    }
}
```

-----

## 🧪 Testing

```bash
# Run all unit tests
go test ./... -v

# Run tests for a specific package
go test ./internal/api/middleware -v

# Run load testing with hey (10,000 requests from 100 concurrent clients)
hey -n 10000 -c 100 http://localhost:8081/health
```

-----

## 📊 Monitoring & Logging

### Prometheus Metrics

```
shieldx_gateway_requests_total{method,path,status}      # Total number of requests
shieldx_gateway_request_duration_seconds{path}          # Request latency
shieldx_gateway_rate_limit_exceeded_total               # Number of requests denied by rate limiting
shieldx_gateway_circuit_breaker_state{service}          # State of the circuit breaker (0=Closed, 1=Open)
```

### Structured Logging

```go
// Example of structured logging with Zerolog
log.Info().
    Str("method", c.Request.Method).
    Str("path", c.Request.URL.Path).
    Int("status", c.Writer.Status()).
    Dur("latency", latency).
    Str("request_id", c.GetString("x-request-id")).
    Msg("Incoming request")
```

-----

## 🔧 Troubleshooting

#### High Latency

```bash
# 1. Check the status of backend services
curl http://localhost:8081/health

# 2. View latency metrics
curl http://localhost:8081/metrics | grep shieldx_gateway_request_duration_seconds

# 3. Check Redis connection
redis-cli -h localhost PING
```

#### Circuit Breaker is Open

```bash
# 1. Check the backend service status directly
curl http://<service_url>/health

# 2. Manually reset the circuit breaker (if an admin endpoint exists)
curl -X POST http://localhost:8081/admin/circuits/auth-service/reset
```

-----

## 📚 References

  - [API Gateway Pattern](https://learn.microsoft.com/en-us/azure/architecture/patterns/gateway-routing)
  - [Rate Limiting Strategies and Techniques](https://www.nginx.com/blog/rate-limiting-nginx/)
  - [Circuit Breaker Pattern](https://martinfowler.com/bliki/CircuitBreaker.html)

-----

## 📄 License

This project is licensed under the [Apache License 2.0](https://github.com/shieldx-bot/shieldx/blob/main/LICENSE).