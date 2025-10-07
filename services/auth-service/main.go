package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"time"

	"shieldx/shared/auth"

	"github.com/redis/go-redis/v9"
)

func main() {
	// Load configuration from environment
	redisAddr := getEnv("REDIS_ADDR", "localhost:6379")
	redisPassword := getEnv("REDIS_PASSWORD", "")
	port := getEnv("PORT", "8080")

	// Generate keys for demo (in production, load from secure storage)
	privateKey, publicKey, err := auth.LoadKeysFromEnv()
	if err != nil {
		log.Fatal("Failed to load keys:", err)
	}

	// Initialize Redis client
	redisClient := redis.NewClient(&redis.Options{
		Addr:     redisAddr,
		Password: redisPassword,
		DB:       0,
	})

	// Test Redis connection
	ctx := context.Background()
	if err := redisClient.Ping(ctx).Err(); err != nil {
		log.Printf("Warning: Redis not available, using in-memory store: %v", err)
	}

	// Initialize JWT Manager
	jwtManager, err := auth.NewJWTManager(auth.JWTConfig{
		PrivateKeyPEM:   privateKey,
		PublicKeyPEM:    publicKey,
		AccessTokenTTL:  15 * time.Minute,
		RefreshTokenTTL: 7 * 24 * time.Hour,
		Issuer:          "shieldx-auth-service",
		RevokedTokenStore: auth.NewRedisRevokedStore(auth.RedisConfig{
			Addr:     redisAddr,
			Password: redisPassword,
			DB:       0,
		}),
	})
	if err != nil {
		log.Fatal("Failed to create JWT manager:", err)
	}

	// Initialize Session Manager
	sessionManager, err := auth.NewSessionManager(auth.SessionConfig{
		RedisAddr:     redisAddr,
		RedisPassword: redisPassword,
		RedisDB:       0,
		SessionTTL:    24 * time.Hour,
	})
	if err != nil {
		log.Printf("Warning: Session manager not available: %v", err)
	}

	// Initialize RBAC Engine
	rbacEngine, err := auth.NewRBACEngine(auth.RBACConfig{
		Roles: auth.GetDefaultRoles(),
	})
	if err != nil {
		log.Fatal("Failed to create RBAC engine:", err)
	}

	// Initialize OAuth2 Provider
	oauth2Provider, err := auth.NewOAuth2Provider(auth.OAuth2Config{
		Issuer:         "https://auth.shieldx.cloud",
		RedisClient:    redisClient,
		JWTManager:     jwtManager,
		SessionManager: sessionManager,
		Clients: []*auth.OAuth2Client{
			{
				ClientID:     "shieldx-web-app",
				ClientSecret: "demo-secret-change-in-production",
				RedirectURIs: []string{
					"http://localhost:3000/callback",
					"https://console.shieldx.cloud/callback",
				},
				GrantTypes: []string{"authorization_code", "refresh_token"},
				Scopes:     []string{"openid", "profile", "email", "api"},
				Name:       "ShieldX Web Console",
			},
			{
				ClientID:     "shieldx-mobile-app",
				ClientSecret: "mobile-secret-change-in-production",
				RedirectURIs: []string{
					"shieldx://callback",
				},
				GrantTypes: []string{"authorization_code", "refresh_token"},
				Scopes:     []string{"openid", "profile", "email", "api"},
				Name:       "ShieldX Mobile App",
			},
		},
	})
	if err != nil {
		log.Fatal("Failed to create OAuth2 provider:", err)
	}

	// Create auth middleware
	authMiddleware := auth.NewAuthMiddleware(auth.MiddlewareConfig{
		JWTManager:     jwtManager,
		SessionManager: sessionManager,
		RBACEngine:     rbacEngine,
		BypassPaths: []string{
			"/health",
			"/metrics",
			"/auth/login",
			"/auth/register",
			"/oauth2/authorize",
			"/oauth2/token",
		},
	})

	// Setup routes
	mux := http.NewServeMux()

	// Public endpoints
	mux.HandleFunc("/health", healthHandler)
	mux.HandleFunc("/auth/login", loginHandler(jwtManager, sessionManager))
	mux.HandleFunc("/auth/register", registerHandler)
	mux.HandleFunc("/auth/refresh", refreshHandler(jwtManager))

	// OAuth2 endpoints
	mux.HandleFunc("/oauth2/authorize", oauth2Provider.HandleAuthorize)
	mux.HandleFunc("/oauth2/token", oauth2Provider.HandleToken)

	// Protected endpoints
	mux.Handle("/api/profile", authMiddleware.Authenticate(http.HandlerFunc(profileHandler)))
	mux.Handle("/api/users", authMiddleware.Authenticate(
		authMiddleware.RequireRole("admin")(http.HandlerFunc(usersHandler)),
	))
	mux.Handle("/api/data", authMiddleware.Authenticate(
		authMiddleware.RequirePermission("api", "read")(http.HandlerFunc(dataHandler)),
	))

	// Admin endpoints
	mux.Handle("/admin/roles", authMiddleware.Authenticate(
		authMiddleware.RequireRole("admin")(http.HandlerFunc(rolesHandler(rbacEngine))),
	))

	log.Printf("🔐 ShieldX Auth Service starting on port %s", port)
	log.Printf("📝 JWT Manager initialized (RS256)")
	log.Printf("💾 Session Manager: %s", redisAddr)
	log.Printf("🔒 RBAC Engine loaded with %d default roles", len(auth.GetDefaultRoles()))
	log.Printf("🌐 OAuth2 Provider initialized with %d clients", 2)

	if err := http.ListenAndServe(":"+port, mux); err != nil {
		log.Fatal("Server failed:", err)
	}
}

// Handler functions
func healthHandler(w http.ResponseWriter, r *http.Request) {
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":    "healthy",
		"service":   "shieldx-auth",
		"timestamp": time.Now().Unix(),
	})
}

func loginHandler(jwtManager *auth.JWTManager, sessionManager *auth.SessionManager) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req struct {
			Username string `json:"username"`
			Password string `json:"password"`
		}

		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid request", http.StatusBadRequest)
			return
		}

		// TODO: Validate credentials against database
		// For demo, accept any non-empty username/password
		if req.Username == "" || req.Password == "" {
			http.Error(w, "Invalid credentials", http.StatusUnauthorized)
			return
		}

		// Create session
		var session *auth.SessionData
		var err error
		if sessionManager != nil {
			session, err = sessionManager.CreateSession(
				r.Context(),
				"user-"+req.Username,
				"tenant-default",
				req.Username+"@example.com",
				r.RemoteAddr,
				r.UserAgent(),
			)
			if err != nil {
				log.Printf("Failed to create session: %v", err)
			}
		}

		// Determine roles (demo logic)
		roles := []string{"user"}
		if req.Username == "admin" {
			roles = []string{"admin"}
		}

		// Generate token pair
		tokenPair, err := jwtManager.GenerateTokenPair(
			r.Context(),
			"user-"+req.Username,
			"tenant-default",
			req.Username+"@example.com",
			roles,
			[]string{"api:read", "api:write", "dashboard:read"},
		)
		if err != nil {
			http.Error(w, "Failed to generate token", http.StatusInternalServerError)
			return
		}

		response := map[string]interface{}{
			"access_token":  tokenPair.AccessToken,
			"refresh_token": tokenPair.RefreshToken,
			"token_type":    "Bearer",
			"expires_in":    900, // 15 minutes
		}

		if session != nil {
			response["session_id"] = session.SessionID
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}
}

func registerHandler(w http.ResponseWriter, r *http.Request) {
	// TODO: Implement user registration
	http.Error(w, "Not implemented", http.StatusNotImplemented)
}

func refreshHandler(jwtManager *auth.JWTManager) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req struct {
			RefreshToken string `json:"refresh_token"`
		}

		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid request", http.StatusBadRequest)
			return
		}

		tokenPair, err := jwtManager.RefreshToken(r.Context(), req.RefreshToken)
		if err != nil {
			http.Error(w, "Invalid refresh token", http.StatusUnauthorized)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"access_token":  tokenPair.AccessToken,
			"refresh_token": tokenPair.RefreshToken,
			"token_type":    "Bearer",
			"expires_in":    900,
		})
	}
}

func profileHandler(w http.ResponseWriter, r *http.Request) {
	claims := auth.GetClaimsFromContext(r.Context())
	if claims == nil {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"user_id":     claims.UserID,
		"tenant_id":   claims.TenantID,
		"email":       claims.Email,
		"roles":       claims.Roles,
		"permissions": claims.Permissions,
		"session_id":  claims.SessionID,
	})
}

func usersHandler(w http.ResponseWriter, r *http.Request) {
	// Admin only endpoint
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"users": []map[string]string{
			{"id": "user-1", "username": "admin", "role": "admin"},
			{"id": "user-2", "username": "user1", "role": "user"},
			{"id": "user-3", "username": "operator1", "role": "operator"},
		},
	})
}

func dataHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"data": []string{"item1", "item2", "item3"},
	})
}

func rolesHandler(rbacEngine *auth.RBACEngine) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		roles := auth.GetDefaultRoles()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"roles": roles,
		})
	}
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}
