# ShieldX Authentication Service

Production-ready authentication and authorization service.

## Features

- ✅ JWT Authentication (RS256)
- ✅ Session Management (Redis)
- ✅ RBAC with OPA Policies
- ✅ OAuth2/OIDC Provider
- ✅ Token Refresh
- ✅ Multi-tenant Support

## Endpoints

### Public Endpoints

- `POST /auth/login` - Login and get tokens
- `POST /auth/register` - Register new user
- `POST /auth/refresh` - Refresh access token
- `GET /oauth2/authorize` - OAuth2 authorization
- `POST /oauth2/token` - OAuth2 token exchange

### Protected Endpoints

- `GET /api/profile` - Get user profile (authenticated)
- `GET /api/users` - List users (admin only)
- `GET /api/data` - Access data (requires api:read permission)
- `GET /admin/roles` - List roles (admin only)

## Quick Start

```bash
# Set environment variables
export REDIS_ADDR=localhost:6379
export PORT=8080

# Generate RSA keys (first time only)
openssl genrsa -out private.pem 2048
openssl rsa -in private.pem -pubout -out public.pem
export JWT_PRIVATE_KEY=$(cat private.pem)
export JWT_PUBLIC_KEY=$(cat public.pem)

# Run service
go run main.go
```

## Docker

```bash
docker build -t shieldx-auth-service .
docker run -p 8080:8080 \
  -e REDIS_ADDR=redis:6379 \
  -e JWT_PRIVATE_KEY="$(cat private.pem)" \
  -e JWT_PUBLIC_KEY="$(cat public.pem)" \
  shieldx-auth-service
```

## API Examples

### Login

```bash
curl -X POST http://localhost:8080/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}'
```

Response:
```json
{
  "access_token": "eyJhbGc...",
  "refresh_token": "eyJhbGc...",
  "token_type": "Bearer",
  "expires_in": 900,
  "session_id": "abc123..."
}
```

### Access Protected Resource

```bash
curl http://localhost:8080/api/profile \
  -H "Authorization: Bearer eyJhbGc..."
```

### Refresh Token

```bash
curl -X POST http://localhost:8080/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{"refresh_token":"eyJhbGc..."}'
```

### OAuth2 Authorization Flow

1. Redirect user to authorization endpoint:
```
http://localhost:8080/oauth2/authorize?
  client_id=shieldx-web-app&
  redirect_uri=http://localhost:3000/callback&
  response_type=code&
  scope=openid profile email api&
  state=random-state-value&
  code_challenge=BASE64URL(SHA256(code_verifier))&
  code_challenge_method=S256
```

2. Exchange code for tokens:
```bash
curl -X POST http://localhost:8080/oauth2/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=authorization_code" \
  -d "code=AUTHORIZATION_CODE" \
  -d "redirect_uri=http://localhost:3000/callback" \
  -d "client_id=shieldx-web-app" \
  -d "client_secret=demo-secret-change-in-production" \
  -d "code_verifier=CODE_VERIFIER"
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | `8080` |
| `REDIS_ADDR` | Redis address | `localhost:6379` |
| `REDIS_PASSWORD` | Redis password | `` |
| `JWT_PRIVATE_KEY` | RSA private key (PEM) | Generated |
| `JWT_PUBLIC_KEY` | RSA public key (PEM) | Generated |

## Default Users (Demo)

| Username | Password | Role |
|----------|----------|------|
| `admin` | any | `admin` |
| any other | any | `user` |

⚠️ **Replace with real user authentication in production!**

## Monitoring

```bash
# Health check
curl http://localhost:8080/health
```

## Security Notes

1. Change all default secrets in production
2. Use HTTPS in production
3. Implement rate limiting
4. Enable audit logging
5. Use secure Redis connection
6. Rotate JWT keys regularly
7. Implement account lockout
8. Add CAPTCHA for login
9. Enable 2FA for sensitive operations
10. Monitor for suspicious activity
