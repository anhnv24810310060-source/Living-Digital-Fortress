package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"shieldx/core/maze_engine"
	"shieldx/pkg/ledger"
	"shieldx/pkg/metrics"
)

type CamouflageAPI struct {
	engine  *maze_engine.CamouflageEngine
	metrics *APIMetrics
}

type APIMetrics struct {
	TemplateRequests *metrics.Counter
	SessionCreated   *metrics.Counter
	LogRequests      *metrics.Counter
	Errors           *metrics.Counter
}

type SessionRequest struct {
	TemplateType string `json:"template_type"`
	ClientIP     string `json:"client_ip"`
	UserAgent    string `json:"user_agent"`
	ReconType    string `json:"recon_type"`
}

type SessionResponse struct {
	SessionID string                      `json:"session_id"`
	Template  *maze_engine.Template       `json:"template"`
	Success   bool                        `json:"success"`
	Message   string                      `json:"message"`
}

type LogRequest struct {
	Timestamp string `json:"timestamp"`
	ClientIP  string `json:"client_ip"`
	UserAgent string `json:"user_agent"`
	Pathname  string `json:"pathname"`
	ReconType string `json:"recon_type"`
	CFRay     string `json:"cf_ray"`
	Country   string `json:"country"`
}

func main() {
	port := getEnv("CAMOUFLAGE_API_PORT", "8091")
	templatesPath := getEnv("TEMPLATES_PATH", "./core/maze_engine/templates")

	// Initialize camouflage engine
	engine, err := maze_engine.NewCamouflageEngine(templatesPath)
	if err != nil {
		log.Fatalf("Failed to initialize camouflage engine: %v", err)
	}

	// Initialize API
	api := &CamouflageAPI{
		engine:  engine,
		metrics: initAPIMetrics(),
	}

	// Setup routes
	mux := http.NewServeMux()
	
	// Template endpoints
	mux.HandleFunc("/v1/camouflage/template/", api.getTemplateHandler)
	mux.HandleFunc("/v1/camouflage/templates", api.listTemplatesHandler)
	
	// Session endpoints
	mux.HandleFunc("/v1/camouflage/session", api.createSessionHandler)
	mux.HandleFunc("/v1/camouflage/session/", api.getSessionHandler)
	
	// Logging endpoint
	mux.HandleFunc("/v1/camouflage/log", api.logHandler)
	
	// Health and metrics
	mux.HandleFunc("/health", api.healthHandler)
	mux.HandleFunc("/metrics", api.metricsHandler)

	// Add middleware
	handler := api.authMiddleware(api.corsMiddleware(mux))

	server := &http.Server{
		Addr:         ":" + port,
		Handler:      handler,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 15 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	log.Printf("[camouflage-api] Starting server on port %s", port)
	log.Fatal(server.ListenAndServe())
}

func initAPIMetrics() *APIMetrics {
	return &APIMetrics{
		TemplateRequests: metrics.NewCounter("camouflage_api_template_requests_total", "Total template requests"),
		SessionCreated:   metrics.NewCounter("camouflage_api_sessions_created_total", "Total sessions created"),
		LogRequests:      metrics.NewCounter("camouflage_api_log_requests_total", "Total log requests"),
		Errors:           metrics.NewCounter("camouflage_api_errors_total", "Total API errors"),
	}
}

func (api *CamouflageAPI) getTemplateHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		api.metrics.Errors.Inc()
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Extract template name from path
	path := strings.TrimPrefix(r.URL.Path, "/v1/camouflage/template/")
	templateName := strings.Split(path, "/")[0]

	if templateName == "" {
		api.metrics.Errors.Inc()
		http.Error(w, "Template name required", http.StatusBadRequest)
		return
	}

	template, err := api.engine.GetTemplate(templateName)
	if err != nil {
		api.metrics.Errors.Inc()
		http.Error(w, fmt.Sprintf("Template not found: %v", err), http.StatusNotFound)
		return
	}

	api.metrics.TemplateRequests.Inc()

	// Log template request
	clientIP := r.Header.Get("X-Client-IP")
	reconType := r.Header.Get("X-Recon-Type")
	
	_ = ledger.AppendJSONLine("data/ledger-camouflage-api.log", "camouflage_api", "template_requested", map[string]any{
		"template_name": templateName,
		"client_ip":     clientIP,
		"recon_type":    reconType,
		"user_agent":    r.UserAgent(),
	})

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(template)
}

func (api *CamouflageAPI) listTemplatesHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		api.metrics.Errors.Inc()
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	templates := api.engine.ListTemplates()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"templates": templates,
		"count":     len(templates),
	})
}

func (api *CamouflageAPI) createSessionHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		api.metrics.Errors.Inc()
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req SessionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		api.metrics.Errors.Inc()
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	// Validate request
	if req.TemplateType == "" || req.ClientIP == "" {
		api.metrics.Errors.Inc()
		http.Error(w, "Missing required fields", http.StatusBadRequest)
		return
	}

	// Create session
	session, err := api.engine.CreateSession(req.TemplateType, req.ClientIP, req.UserAgent)
	if err != nil {
		api.metrics.Errors.Inc()
		http.Error(w, fmt.Sprintf("Failed to create session: %v", err), http.StatusInternalServerError)
		return
	}

	api.metrics.SessionCreated.Inc()

	response := SessionResponse{
		SessionID: session.ID,
		Template:  session.Template,
		Success:   true,
		Message:   "Session created successfully",
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (api *CamouflageAPI) logHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		api.metrics.Errors.Inc()
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req LogRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		api.metrics.Errors.Inc()
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	api.metrics.LogRequests.Inc()

	// Log reconnaissance attempt
	_ = ledger.AppendJSONLine("data/ledger-reconnaissance.log", "reconnaissance", "attempt_detected", map[string]any{
		"timestamp":  req.Timestamp,
		"client_ip":  req.ClientIP,
		"user_agent": req.UserAgent,
		"pathname":   req.Pathname,
		"recon_type": req.ReconType,
		"cf_ray":     req.CFRay,
		"country":    req.Country,
	})

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status":  "logged",
		"message": "Reconnaissance attempt logged successfully",
	})
}

func (api *CamouflageAPI) healthHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(200)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":    "healthy",
		"service":   "camouflage-api",
		"timestamp": time.Now().Format(time.RFC3339),
		"templates": len(api.engine.ListTemplates()),
	})
}

func (api *CamouflageAPI) authMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Skip auth for health check
		if r.URL.Path == "/health" || r.URL.Path == "/metrics" {
			next.ServeHTTP(w, r)
			return
		}

		auth := r.Header.Get("Authorization")
		if !strings.HasPrefix(auth, "Bearer ") {
			api.metrics.Errors.Inc()
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}

		token := strings.TrimPrefix(auth, "Bearer ")
		if !api.validateToken(token) {
			api.metrics.Errors.Inc()
			http.Error(w, "Invalid token", http.StatusUnauthorized)
			return
		}

		next.ServeHTTP(w, r)
	})
}

func (api *CamouflageAPI) corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Client-IP, X-Recon-Type")

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}

		next.ServeHTTP(w, r)
	})
}

func (api *CamouflageAPI) validateToken(token string) bool {
	// In production, validate against database or JWT
	expectedToken := getEnv("CAMOUFLAGE_API_KEY", "default_key")
	return token == expectedToken
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}