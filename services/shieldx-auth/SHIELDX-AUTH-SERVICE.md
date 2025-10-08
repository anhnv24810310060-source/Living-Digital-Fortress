 
-----

# 🛡️ ShieldX Auth Service

[](https://golang.org)
[](https://opensource.org/licenses/Apache-2.0)
[](https://www.docker.com/)

**ShieldX Auth Service** is the central authentication and authorization service for the ShieldX platform. It handles user identity, session management, and secure access control.

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
      - [Authentication Endpoints](https://www.google.com/search?q=%23authentication-endpoints)
      - [OAuth2 Endpoints](https://www.google.com/search?q=%23oauth2-endpoints)
      - [Password Management](https://www.google.com/search?q=%23password-management)
      - [Multi-Factor Authentication (MFA)](https://www.google.com/search?q=%23multi-factor-authentication-mfa)
      - [Session Management](https://www.google.com/search?q=%23session-management)
  - [🔐 Authentication Flow](https://www.google.com/search?q=%23-authentication-flow)
  - [💻 Development Guide](https://www.google.com/search?q=%23-development-guide)
      - [Project Structure](https://www.google.com/search?q=%23project-structure)
      - [JWT Token Structure](https://www.google.com/search?q=%23jwt-token-structure)
  - [🧪 Testing](https://www.google.com/search?q=%23-testing)
  - [🛡️ Security](https://www.google.com/search?q=%23%EF%B8%8F-security)
      - [Best Practices](https://www.google.com/search?q=%23best-practices)
      - [Security Checklist](https://www.google.com/search?q=%23security-checklist)
  - [📊 Monitoring](https://www.google.com/search?q=%23-monitoring)
  - [🔧 Troubleshooting](https://www.google.com/search?q=%23-troubleshooting)
  - [📚 Resources](https://www.google.com/search?q=%23-resources)
  - [🤝 Contributing](https://www.google.com/search?q=%23-contributing)
  - [📄 License](https://www.google.com/search?q=%23-license)

-----

## 🎯 Overview

### Core Responsibilities

  - **JWT Token Management**: Generate, verify, and refresh JSON Web Tokens.
  - **OAuth2 Integration**: Handle authentication via third-party providers like Google and GitHub.
  - **Session Management**: Manage user sessions and support token revocation.
  - **Multi-Factor Authentication (MFA)**: Provide support for TOTP-based 2FA.
  - **Device Fingerprinting**: Identify and track devices for enhanced security.
  - **Password Management**: Ensure secure password hashing, verification, and reset functionality.
  - **Rate Limiting**: Protect against brute-force attacks on authentication endpoints.

### Key Features

✅ JWT-based authentication with refresh tokens
✅ OAuth2 integration (Google, GitHub, etc.)
✅ Session management with Redis for scalability
✅ Multi-factor authentication (TOTP)
✅ Device fingerprinting and tracking
✅ Rate limiting and brute-force protection

### Technology Stack

| Component | Technology | Version |
| :--- | :--- | :--- |
| **Language** | Go | 1.25+ |
| **Framework** | Gin Web Framework | Latest |
| **Cache/Session Store** | Redis | 7+ |
| **Database** | PostgreSQL | 15+ |
| **Password Hashing** | bcrypt | - |
| **Metrics** | Prometheus | Latest |

-----

## 🏗️ Architecture

### System Architecture

```plaintext
┌─────────────────────────────────────────────────────────┐
│ ShieldX Auth Service (Port 5001)                        │
├─────────────────────────────────────────────────────────┤
│          ▲                    ▲                     ▲          │
│   ┌──────┴──────┐      ┌──────┴───────┐      ┌──────┴─────┐   │
│   │  HTTP API   │      │    OAuth2    │      │  Metrics   │   │
│   │   (REST)    │      │   Handlers   │      │ /metrics   │   │
│   └──────┬──────┘      └──────┬───────┘      └──────┬─────┘   │
│          │                    │                     │          │
│   ┌──────▼────────────────────▼─────────────────────▼────────┐ │
│   │ Authentication Logic Layer                             │ │
│   │ - JWT Service      - OAuth Service                     │ │
│   │ - Session Service  - MFA Service                       │ │
│   │ - Password Service - Rate Limiter                      │ │
│   └──────────────────────────┬─────────────────────────────┘ │
│                              │                               │
│   ┌──────────────────────────▼─────────────────────────────┐ │
│   │ Data Access Layer                                      │ │
│   │ - Redis (Sessions, Rate Limits, Blacklists)            │ │
│   │ - PostgreSQL (User Credentials, Audit Logs)            │ │
│   └──────────────────────────┬─────────────────────────────┘ │
│                              │                               │
└──────────────────────────────┼───────────────────────────────┘
                               │
                 ┌─────────────┴─────────────┐
           ┌─────▼─────┐               ┌─────▼─────┐
           │   Redis   │               │PostgreSQL │
           │(Sessions) │               │  (Users)  │
           └───────────┘               └───────────┘
```

### Request Flow

```plaintext
Client Request
    ↓
Rate Limiting Middleware
    ↓
Authentication Middleware (if required)
    ↓
Request Handler (Controller)
    ↓
Service Layer (Business Logic)
    ↓
Repository Layer (Data Access)
    ↓
Database / Cache (PostgreSQL / Redis)
    ↓
Response
```

-----

## 🚀 Getting Started

### Prerequisites

  - Go 1.25+
  - Redis 7+
  - PostgreSQL 15+
  - Docker & Docker Compose

### Installation Steps

```bash
# 1. Clone the repository
git clone https://github.com/shieldx-bot/shieldx.git
cd shieldx/services/shieldx-auth

# 2. Install dependencies
go mod download

# 3. Setup Redis and PostgreSQL using Docker
docker run -d --name shieldx-redis -p 6379:6379 redis:7-alpine
docker run -d --name shieldx-postgres -e POSTGRES_USER=auth_user -e POSTGRES_PASSWORD=auth_pass -e POSTGRES_DB=shieldx_auth -p 5432:5432 postgres:15-alpine

# 4. Configure environment variables (create a .env file)
# Copy from .env.example and modify as needed
export AUTH_PORT=5001
export AUTH_REDIS_HOST=localhost
export AUTH_DB_HOST=localhost
export AUTH_DB_USER=auth_user
export AUTH_DB_PASSWORD=auth_pass
export AUTH_DB_NAME=shieldx_auth
export AUTH_JWT_SECRET="your-super-secret-key-of-at-least-32-characters"

# 5. Run database migrations (if using PostgreSQL)
# Ensure you have the migrate CLI installed
migrate -path ./migrations -database "postgresql://auth_user:auth_pass@localhost:5432/shieldx_auth?sslmode=disable" up

# 6. Build and run the service
go build -o bin/shieldx-auth cmd/server/main.go
./bin/shieldx-auth

# 7. Verify the installation
curl http://localhost:5001/health
# Expected response should show "healthy" status for dependencies
```

-----

## 📡 API Reference

**Base URL**: `http://localhost:5001/api/v1/auth`

### Authentication Endpoints

#### 1\. User Login

`POST /login`

\<details\>\<summary\>View Request Example\</summary\>

```json
{
  "email": "user@example.com",
  "password": "SecurePassword123!",
  "device_fingerprint": "abc123xyz"
}
```

\</details\>
\<details\>\<summary\>View Response Example (200 OK)\</summary\>

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "user": {
    "id": "user-123",
    "email": "user@example.com",
    "role": "user"
  },
  "mfa_required": false
}
```

\</details\>

#### 2\. Refresh Token

`POST /refresh`

\<details\>\<summary\>View Request Example\</summary\>

```json
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

\</details\>

#### 3\. Verify Token

`POST /verify` (Requires `Authorization: Bearer <access_token>`)

#### 4\. Logout

`POST /logout` (Requires `Authorization: Bearer <access_token>`)

### OAuth2 Endpoints

#### 1\. Google OAuth Login

`GET /oauth/google` (Redirects to Google's consent screen)

#### 2\. GitHub OAuth Login

`GET /oauth/github` (Redirects to GitHub's authorization page)

#### 3\. OAuth Callback

`GET /oauth/callback?code=...&state=...` (Handles the callback and returns JWTs)

### Password Management

#### 1\. Request Password Reset

`POST /password/reset/request`

\<details\>\<summary\>View Request Example\</summary\>

```json
{
  "email": "user@example.com"
}
```

\</details\>

#### 2\. Reset Password

`POST /password/reset`

\<details\>\<summary\>View Request Example\</summary\>

```json
{
  "token": "reset-token-from-email-123",
  "new_password": "NewSecurePassword456!"
}
```

\</details\>

### Multi-Factor Authentication (MFA)

#### 1\. Enable MFA

`POST /mfa/enable` (Requires `Authorization: Bearer <access_token>`)

\<details\>\<summary\>View Response Example\</summary\>

```json
{
  "secret": "JBSWY3DPEHPK3PXP",
  "qr_code": "data:image/png;base64,iVBORw0KG...",
  "backup_codes": ["12345678", "87654321"]
}
```

\</details\>

#### 2\. Verify MFA

`POST /mfa/verify`

\<details\>\<summary\>View Request Example\</summary\>

```json
{
  "code": "123456"
}
```

\</details\>

### Session Management

#### 1\. List Active Sessions

`GET /sessions` (Requires `Authorization: Bearer <access_token>`)

#### 2\. Revoke Session

`DELETE /sessions/{session_id}` (Requires `Authorization: Bearer <access_token>`)

-----

## 🔐 Authentication Flow

#### Standard Login Flow

```plaintext
1. Client sends POST /login with email and password.
2. Service validates credentials against the database.
3. Service checks for rate limits.
4. Service generates a short-lived JWT access token (e.g., 1 hour).
5. Service generates a long-lived refresh token (e.g., 7 days).
6. Service stores the session in Redis.
7. Service returns both tokens to the client.
```

#### OAuth2 Flow

```plaintext
1. Client navigates to GET /oauth/google.
2. Service redirects the user to Google's OAuth consent screen.
3. User authenticates and grants permission.
4. Google redirects back to /oauth/callback with an authorization code.
5. Service exchanges the code for a Google access token.
6. Service retrieves user info from Google.
7. Service creates or finds the user in its database.
8. Service generates its own JWTs (access and refresh tokens).
9. Service returns its JWTs to the client.
```

-----

## 💻 Development Guide

### Project Structure

```plaintext
shieldx-auth/
├── cmd/server/main.go            # Application entry point
├── internal/
│   ├── api/                      # Handlers, routes, middleware
│   ├── service/                  # Business logic layer
│   ├── repository/               # Data access layer (Redis, Postgres)
│   ├── models/                   # Domain models
│   └── config/                   # Configuration loading
├── pkg/                          # Shared packages (jwt, crypto, etc.)
└── tests/                        # Unit and integration tests
```

### JWT Token Structure

```go
type Claims struct {
    UserID      string   `json:"user_id"`
    TenantID    string   `json:"tenant_id"`
    Role        string   `json:"role"`
    Permissions []string `json:"permissions"`
    SessionID   string   `json:"session_id"`
    jwt.RegisteredClaims
}
```

-----

## 🧪 Testing

```bash
# Run all tests
make test

# Run unit tests only
go test ./internal/... -v

# Run integration tests
go test ./tests/integration/... -v

# Run tests with code coverage
go test -coverprofile=coverage.out ./... && go tool cover -html=coverage.out
```

-----

## 🛡️ Security

### Best Practices

1.  **JWT Secret Management**: Use a strong, high-entropy secret (\>= 32 characters) loaded from environment variables or a secret manager. Implement a secret rotation strategy.
2.  **Password Hashing**: Use `bcrypt` with a cost factor of at least 12 to hash all user passwords.
3.  **Rate Limiting**: Apply strict rate limits on login and password reset endpoints to prevent brute-force attacks.
4.  **Token Revocation**: Implement a token blacklist in Redis to immediately revoke sessions upon logout or suspicious activity.
5.  **HTTPS Only**: Enforce HTTPS in production environments to protect data in transit.

### Security Checklist

  - [x] JWT secret is strong and stored securely.
  - [x] Passwords are hashed with bcrypt (cost ≥ 12).
  - [x] Rate limiting is enabled on sensitive endpoints.
  - [x] Token revocation is implemented.
  - [x] HTTPS is enforced in production.
  - [x] Input validation is performed on all endpoints.

-----

## 📊 Monitoring

### Prometheus Metrics

```go
var (
    loginAttempts = prometheus.NewCounterVec(
        prometheus.CounterOpts{
            Name: "shieldx_auth_login_attempts_total",
            Help: "Total number of login attempts by status.",
        },
        []string{"status"}, // e.g., "success", "failure"
    )
    
    activeSessions = prometheus.NewGauge(
        prometheus.GaugeOpts{
            Name: "shieldx_auth_active_sessions",
            Help: "Current number of active user sessions.",
        },
    )
)
```

-----

## 🔧 Troubleshooting

#### JWT Token Invalid

  - **Check**: Ensure the `AUTH_JWT_SECRET` environment variable is set correctly and matches across services.
  - **Verify**: Check the token's expiration by decoding it. Ensure the system clocks are synchronized.

#### OAuth Callback Failed

  - **Check**: Verify that the `Redirect URL` in your `.env` file exactly matches the one configured in the OAuth provider's dashboard (e.g., Google Cloud Console).
  - **Verify**: Ensure client ID and secret are correct.

-----

## 📚 Resources

  - [JWT Best Practices](https://www.google.com/search?q=https://jwt.io/best-practices/)
  - [OAuth 2.0 RFC 6749](https://tools.ietf.org/html/rfc6749)
  - [OWASP Authentication Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html)

-----

## 🤝 Contributing

Please see `CONTRIBUTING.md` for details.

## 📄 License

This project is licensed under the Apache License 2.0. See the `LICENSE` file for details.