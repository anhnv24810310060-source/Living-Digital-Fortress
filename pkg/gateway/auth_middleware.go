package gateway


import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"net/http"
	"strings"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

// AuthConfig holds authentication configuration
type AuthConfig struct {
	JWTSecret     []byte
	APIKeyHeader  string
	BypassPaths   []string
	TokenDuration time.Duration
}

// Claims represents JWT claims
type Claims struct {
	UserID   string   `json:"user_id"`
	TenantID string   `json:"tenant_id"`
	Roles    []string `json:"roles"`
	jwt.RegisteredClaims
}

// AuthMiddleware provides authentication for HTTP requests
type AuthMiddleware struct {
	config AuthConfig
}

// ctxKey defines a private type to avoid key collisions in context
type ctxKey string

// claimsCtxKey is the context key used to store JWT claims
var claimsCtxKey ctxKey = "claims"

// ClaimsFromContext extracts claims from a request context
func ClaimsFromContext(ctx context.Context) (*Claims, bool) {
	c, ok := ctx.Value(claimsCtxKey).(*Claims)
	return c, ok
}

// NewAuthMiddleware creates a new authentication middleware
func NewAuthMiddleware(config AuthConfig) *AuthMiddleware {
	if config.APIKeyHeader == "" {
		config.APIKeyHeader = "X-API-Key"
	}
	if config.TokenDuration == 0 {
		config.TokenDuration = 24 * time.Hour
	}
	return &AuthMiddleware{config: config}
}

// Authenticate validates JWT or API key
func (am *AuthMiddleware) Authenticate(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Check if path should bypass authentication
		for _, path := range am.config.BypassPaths {
			if strings.HasPrefix(r.URL.Path, path) {
				next.ServeHTTP(w, r)
				return
			}
		}

		// Try JWT authentication first
		if authHeader := r.Header.Get("Authorization"); authHeader != "" {
			if strings.HasPrefix(authHeader, "Bearer ") {
				token := strings.TrimPrefix(authHeader, "Bearer ")
				if claims, err := am.validateJWT(token); err == nil {
					ctx := context.WithValue(r.Context(), claimsCtxKey, claims)
					next.ServeHTTP(w, r.WithContext(ctx))
					return
				}
			}
		}

		// Try API key authentication
		if apiKey := r.Header.Get(am.config.APIKeyHeader); apiKey != "" {
			if am.validateAPIKey(apiKey) {
				next.ServeHTTP(w, r)
				return
			}
		}

		// Authentication failed
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		json.NewEncoder(w).Encode(map[string]string{
			"error":   "unauthorized",
			"message": "valid JWT token or API key required",
		})
	})
}

// validateJWT validates a JWT token and returns claims
func (am *AuthMiddleware) validateJWT(tokenString string) (*Claims, error) {
	token, err := jwt.ParseWithClaims(tokenString, &Claims{}, func(token *jwt.Token) (interface{}, error) {
		return am.config.JWTSecret, nil
	})
	if err != nil {
		return nil, err
	}

	if claims, ok := token.Claims.(*Claims); ok && token.Valid {
		return claims, nil
	}
	return nil, jwt.ErrSignatureInvalid
}

// validateAPIKey validates an API key (placeholder - implement your logic)
func (am *AuthMiddleware) validateAPIKey(apiKey string) bool {
	// TODO: Implement real API key validation against database/cache
	// For now, constant-time comparison with demo key
	demoKey := "demo-api-key-shieldx-2025"
	return subtle.ConstantTimeCompare([]byte(apiKey), []byte(demoKey)) == 1
}

// GenerateJWT generates a new JWT token
func (am *AuthMiddleware) GenerateJWT(userID, tenantID string, roles []string) (string, error) {
	claims := Claims{
		UserID:   userID,
		TenantID: tenantID,
		Roles:    roles,
		RegisteredClaims: jwt.RegisteredClaims{
			ExpiresAt: jwt.NewNumericDate(time.Now().Add(am.config.TokenDuration)),
			IssuedAt:  jwt.NewNumericDate(time.Now()),
			NotBefore: jwt.NewNumericDate(time.Now()),
			Issuer:    "shieldx-gateway",
		},
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	return token.SignedString(am.config.JWTSecret)
}

// RequireRole checks if user has required role
func RequireRole(requiredRole string) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			claims, ok := ClaimsFromContext(r.Context())
			if !ok {
				http.Error(w, "unauthorized", http.StatusUnauthorized)
				return
			}

			hasRole := false
			for _, role := range claims.Roles {
				if role == requiredRole || role == "admin" {
					hasRole = true
					break
				}
			}

			if !hasRole {
				http.Error(w, "forbidden", http.StatusForbidden)
				return
			}

			next.ServeHTTP(w, r)
		})
	}
}
