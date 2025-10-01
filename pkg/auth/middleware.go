package auth
package auth

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"time"
)

// AuthMiddleware provides comprehensive authentication middleware
type AuthMiddleware struct {
	jwtManager     *JWTManager
	sessionManager *SessionManager
	rbacEngine     *RBACEngine
	bypassPaths    map[string]bool
}

// MiddlewareConfig configuration for auth middleware
type MiddlewareConfig struct {
	JWTManager     *JWTManager
	SessionManager *SessionManager
	RBACEngine     *RBACEngine
	BypassPaths    []string
}

// NewAuthMiddleware creates a new authentication middleware
func NewAuthMiddleware(config MiddlewareConfig) *AuthMiddleware {
	bypass := make(map[string]bool)
	for _, path := range config.BypassPaths {
		bypass[path] = true
	}

	return &AuthMiddleware{
		jwtManager:     config.JWTManager,
		sessionManager: config.SessionManager,
		rbacEngine:     config.RBACEngine,
		bypassPaths:    bypass,
	}
}

// Authenticate validates JWT token and loads user context
func (am *AuthMiddleware) Authenticate(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Check bypass paths
		if am.bypassPaths[r.URL.Path] {
			next.ServeHTTP(w, r)
			return
		}

		// Extract token from Authorization header
		authHeader := r.Header.Get("Authorization")
		if authHeader == "" {
			am.unauthorized(w, "missing authorization header")
			return
		}

		parts := strings.SplitN(authHeader, " ", 2)
		if len(parts) != 2 || parts[0] != "Bearer" {
			am.unauthorized(w, "invalid authorization header format")
			return
		}

		token := parts[1]

		// Validate token
		claims, err := am.jwtManager.ValidateToken(r.Context(), token)
		if err != nil {
			am.unauthorized(w, "invalid or expired token: "+err.Error())
			return
		}

		// Validate session if session manager is available
		if am.sessionManager != nil && claims.SessionID != "" {
			session, err := am.sessionManager.GetSession(r.Context(), claims.SessionID)
			if err != nil {
				am.unauthorized(w, "invalid session: "+err.Error())
				return
			}

			// Update session activity
			am.sessionManager.UpdateSessionActivity(r.Context(), session.SessionID)
		}

		// Add claims to context
		ctx := context.WithValue(r.Context(), ClaimsContextKey, claims)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

// RequireRole middleware checks if user has required role
func (am *AuthMiddleware) RequireRole(roles ...string) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			claims := GetClaimsFromContext(r.Context())
			if claims == nil {
				am.unauthorized(w, "no authentication context")
				return
			}

			hasRole := false
			for _, role := range roles {
				for _, userRole := range claims.Roles {
					if userRole == role || userRole == "admin" {
						hasRole = true
						break
					}
				}
				if hasRole {
					break
				}
			}

			if !hasRole {
				am.forbidden(w, "insufficient permissions")
				return
			}

			next.ServeHTTP(w, r)
		})
	}
}

// RequirePermission middleware checks if user has specific permission
func (am *AuthMiddleware) RequirePermission(resource, action string) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			claims := GetClaimsFromContext(r.Context())
			if claims == nil {
				am.unauthorized(w, "no authentication context")
				return
			}

			if am.rbacEngine == nil {
				// Fallback to checking permissions in claims
				allowed := am.checkClaimsPermission(claims, resource, action)
				if !allowed {
					am.forbidden(w, "insufficient permissions")
					return
				}
			} else {
				// Use RBAC engine
				allowed, err := am.rbacEngine.CheckPermission(r.Context(), claims.Roles, resource, action)
				if err != nil || !allowed {
					am.forbidden(w, "insufficient permissions")
					return
				}
			}

			next.ServeHTTP(w, r)
		})
	}
}

// RequirePolicy middleware checks custom OPA policy
func (am *AuthMiddleware) RequirePolicy(policyName string, inputBuilder func(*http.Request, *CustomClaims) map[string]interface{}) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			claims := GetClaimsFromContext(r.Context())
			if claims == nil {
				am.unauthorized(w, "no authentication context")
				return
			}

			if am.rbacEngine == nil {
				am.serverError(w, "RBAC engine not configured")
				return
			}

			// Build policy input
			input := inputBuilder(r, claims)

			// Evaluate policy
			allowed, err := am.rbacEngine.CheckPermissionWithPolicy(r.Context(), policyName, input)
			if err != nil {
				am.serverError(w, "policy evaluation failed: "+err.Error())
				return
			}

			if !allowed {
				am.forbidden(w, "access denied by policy")
				return
			}

			next.ServeHTTP(w, r)
		})
	}
}

// RateLimit middleware (basic implementation)
func (am *AuthMiddleware) RateLimit(requestsPerMinute int) func(http.Handler) http.Handler {
	// TODO: Implement Redis-based distributed rate limiting
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Placeholder - implement with Redis
			next.ServeHTTP(w, r)
		})
	}
}

// checkClaimsPermission checks permission from claims
func (am *AuthMiddleware) checkClaimsPermission(claims *CustomClaims, resource, action string) bool {
	for _, perm := range claims.Permissions {
		// Format: "resource:action"
		parts := strings.Split(perm, ":")
		if len(parts) != 2 {
			continue
		}
		if (parts[0] == resource || parts[0] == "*") && (parts[1] == action || parts[1] == "*") {
			return true
		}
	}
	return false
}

// Helper functions
func (am *AuthMiddleware) unauthorized(w http.ResponseWriter, message string) {
	am.jsonError(w, http.StatusUnauthorized, "unauthorized", message)
}

func (am *AuthMiddleware) forbidden(w http.ResponseWriter, message string) {
	am.jsonError(w, http.StatusForbidden, "forbidden", message)
}

func (am *AuthMiddleware) serverError(w http.ResponseWriter, message string) {
	am.jsonError(w, http.StatusInternalServerError, "internal_error", message)
}

func (am *AuthMiddleware) jsonError(w http.ResponseWriter, status int, errorType, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"error":     errorType,
		"message":   message,
		"timestamp": time.Now().Unix(),
	})
}

// Context helpers
type contextKey string

const ClaimsContextKey contextKey = "claims"

// GetClaimsFromContext extracts claims from request context
func GetClaimsFromContext(ctx context.Context) *CustomClaims {
	claims, ok := ctx.Value(ClaimsContextKey).(*CustomClaims)
	if !ok {
		return nil
	}
	return claims
}
