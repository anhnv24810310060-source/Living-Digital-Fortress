# Authentication & Authorization Module

Production-ready authentication and authorization system for ShieldX.

## Features

### ✅ JWT Token Management
- RSA-256 signing (production-grade)
- Access & Refresh token pairs
- Token revocation with Redis
- Automatic token rotation
- PKCE support for OAuth2

### ✅ Session Management
- Redis-backed distributed sessions
- Session metadata support
- Automatic expiration
- Multi-device session tracking
- Session revocation

### ✅ RBAC Engine
- Role-Based Access Control
- Policy-based authorization with OPA
- Permission inheritance
- Fine-grained permissions (resource:action)
- Default roles: admin, user, service, auditor, operator

### ✅ OAuth2/OIDC Provider
- Authorization Code Flow
- PKCE (Proof Key for Code Exchange)
- Refresh Token Flow
- Client registration
- State parameter support

## Quick Start

### 1. Generate RSA Keys

```bash
# Generate private key
openssl genrsa -out private.pem 2048

# Extract public key
openssl rsa -in private.pem -pubout -out public.pem
```

### 2. Initialize Auth System

```go
package main

import (
    "context"
    "io/ioutil"
    "log"
    "time"
    
    "github.com/shieldx-bot/shieldx/shared/shieldx-common/pkg/auth"
)

func main() {
    // Load keys
    privateKey, _ := ioutil.ReadFile("private.pem")
    publicKey, _ := ioutil.ReadFile("public.pem")
    
    // Initialize JWT Manager
    jwtManager, err := auth.NewJWTManager(auth.JWTConfig{
        PrivateKeyPEM:   string(privateKey),
        PublicKeyPEM:    string(publicKey),
        AccessTokenTTL:  15 * time.Minute,
        RefreshTokenTTL: 7 * 24 * time.Hour,
        Issuer:          "shieldx-auth",
        RevokedTokenStore: auth.NewRedisRevokedStore(auth.RedisConfig{
            Addr: "localhost:6379",
        }),
    })
    if err != nil {
        log.Fatal(err)
    }
    
    // Initialize Session Manager
    sessionManager, err := auth.NewSessionManager(auth.SessionConfig{
        RedisAddr:  "localhost:6379",
        SessionTTL: 24 * time.Hour,
    })
    if err != nil {
        log.Fatal(err)
    }
    
    // Initialize RBAC Engine
    rbacEngine, err := auth.NewRBACEngine(auth.RBACConfig{
        Roles: auth.GetDefaultRoles(),
    })
    if err != nil {
        log.Fatal(err)
    }
    
    // Create middleware
    authMiddleware := auth.NewAuthMiddleware(auth.MiddlewareConfig{
        JWTManager:     jwtManager,
        SessionManager: sessionManager,
        RBACEngine:     rbacEngine,
        BypassPaths:    []string{"/health", "/metrics"},
    })
    
    // Use in HTTP server
    // http.Handle("/api/", authMiddleware.Authenticate(yourHandler))
}
```

### 3. Generate Tokens

```go
ctx := context.Background()

// Generate token pair for user
tokenPair, err := jwtManager.GenerateTokenPair(
    ctx,
    "user-123",           // User ID
    "tenant-456",         // Tenant ID
    "user@example.com",   // Email
    []string{"user"},     // Roles
    []string{"api:read", "dashboard:read"}, // Permissions
)

fmt.Printf("Access Token: %s\n", tokenPair.AccessToken)
fmt.Printf("Refresh Token: %s\n", tokenPair.RefreshToken)
```

### 4. Validate Tokens

```go
claims, err := jwtManager.ValidateToken(ctx, accessToken)
if err != nil {
    log.Fatal(err)
}

fmt.Printf("User ID: %s\n", claims.UserID)
fmt.Printf("Roles: %v\n", claims.Roles)
```

### 5. Refresh Tokens

```go
newTokenPair, err := jwtManager.RefreshToken(ctx, refreshToken)
if err != nil {
    log.Fatal(err)
}
```

### 6. Check Permissions

```go
// Check role-based permission
allowed, err := rbacEngine.CheckPermission(
    ctx,
    []string{"user"},  // User roles
    "api",             // Resource
    "read",            // Action
)

// Check with OPA policy
policyInput := map[string]interface{}{
    "user_id": "user-123",
    "roles":   []string{"user"},
    "resource": "api",
    "action":   "read",
}

allowed, err = rbacEngine.CheckPermissionWithPolicy(
    ctx,
    "api_access",
    policyInput,
)
```

## HTTP Middleware Usage

```go
router := http.NewServeMux()

// Public routes
router.HandleFunc("/health", healthHandler)

// Protected routes
protectedRouter := authMiddleware.Authenticate(router)

// Role-based protection
adminRouter := authMiddleware.RequireRole("admin")(adminHandler)

// Permission-based protection
apiRouter := authMiddleware.RequirePermission("api", "write")(apiHandler)

// Policy-based protection
policyRouter := authMiddleware.RequirePolicy(
    "data_access",
    func(r *http.Request, claims *auth.CustomClaims) map[string]interface{} {
        return map[string]interface{}{
            "user_id":    claims.UserID,
            "roles":      claims.Roles,
            "resource":   "sensitive_data",
            "action":     "read",
            "data_sensitivity": "confidential",
        }
    },
)(sensitiveHandler)
```

## OAuth2 Setup

```go
oauth2Provider, err := auth.NewOAuth2Provider(auth.OAuth2Config{
    Issuer:         "https://auth.shieldx.cloud",
    RedisClient:    redisClient,
    JWTManager:     jwtManager,
    SessionManager: sessionManager,
    Clients: []*auth.OAuth2Client{
        {
            ClientID:     "web-app",
            ClientSecret: "secret-key",
            RedirectURIs: []string{"https://app.example.com/callback"},
            GrantTypes:   []string{"authorization_code", "refresh_token"},
            Scopes:       []string{"openid", "profile", "email"},
        },
    },
})

// Register OAuth2 endpoints
http.HandleFunc("/oauth2/authorize", oauth2Provider.HandleAuthorize)
http.HandleFunc("/oauth2/token", oauth2Provider.HandleToken)
```

## Default Roles

| Role      | Description              | Permissions                          |
|-----------|--------------------------|--------------------------------------|
| `admin`   | Full system access       | `*:*`                                |
| `user`    | Standard user            | `profile:read,update`, `api:read`    |
| `service` | Service account          | `api:read,write`, `metrics:write`    |
| `auditor` | Read-only audit access   | `logs:read`, `audit:read`            |
| `operator`| Operational access       | `services:read,restart`, inherits user |

## Custom Roles

```go
customRole := &auth.Role{
    Name:        "developer",
    Description: "Developer access",
    Permissions: []auth.Permission{
        {Resource: "api", Actions: []string{"read", "write"}},
        {Resource: "logs", Actions: []string{"read"}},
        {Resource: "deployments", Actions: []string{"read", "create"}},
    },
    Inherits: []string{"user"},
}

rbacEngine.AddRole(customRole)
```

## Environment Variables

```bash
# JWT Keys
JWT_PRIVATE_KEY=path/to/private.pem
JWT_PUBLIC_KEY=path/to/public.pem

# Redis
REDIS_ADDR=localhost:6379
REDIS_PASSWORD=your-password
REDIS_DB=0

# Token TTL
ACCESS_TOKEN_TTL=15m
REFRESH_TOKEN_TTL=168h

# Session TTL
SESSION_TTL=24h
```

## Security Best Practices

1. **Use RSA-256** for production (never use HS256 with shared secrets)
2. **Short-lived access tokens** (15 minutes recommended)
3. **Long-lived refresh tokens** (7 days, with rotation)
4. **Enable token revocation** with Redis
5. **Use HTTPS only** in production
6. **Implement rate limiting** on auth endpoints
7. **Store secrets securely** (use Vault or similar)
8. **Rotate keys regularly** (quarterly recommended)
9. **Enable PKCE** for public clients
10. **Monitor failed auth attempts**

## Performance Notes

- JWT validation: ~0.1ms per request
- Session lookup: ~1ms with Redis
- RBAC check: ~0.05ms in-memory
- OPA policy eval: ~1-5ms depending on complexity

## Testing

```bash
go test ./pkg/auth/... -v
```

## License

MIT
