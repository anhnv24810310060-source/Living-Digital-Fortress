package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"shieldx/core/maze_engine"
	"shieldx/pkg/deception"
	"shieldx/pkg/ledger"
	"shieldx/pkg/metrics"

	"github.com/google/uuid"
)

type CamouflageService struct {
	dg   *deception.DeceptionGraph
	mu   sync.RWMutex
	mOps *metrics.LabeledCounter // labels: op,result
}

func NewCamouflageService() *CamouflageService {
	dg := deception.NewDeceptionGraph()
	// Seed with a few decoys
	dg.AddNode(deception.CreateWebServerDecoy())
	dg.AddNode(deception.CreateSSHHoneypot())
	dg.AddNode(deception.CreateDatabaseDecoy())
	return &CamouflageService{dg: dg}
}

type SelectResponse struct {
	Decoy deception.DeceptionNode `json:"decoy"`
	Token string                  `json:"token"`
}

type FeedbackRequest struct {
	NodeID string  `json:"node_id"`
	Reward float64 `json:"reward"`
}

func (cs *CamouflageService) handleSelect(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost && r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	ctx := r.Context()
	node, err := cs.dg.SelectOptimalDecoy(ctx)
	if err != nil {
		cs.mOps.Inc(map[string]string{"op": "select", "result": "empty"})
		http.Error(w, "no decoys", http.StatusServiceUnavailable)
		return
	}
	// One-time token for feedback pairing (no auth state here to keep simple)
	token := uuid.New().String()
	cs.mOps.Inc(map[string]string{"op": "select", "result": "ok"})
	writeJSON(w, http.StatusOK, SelectResponse{Decoy: *node, Token: token})
}

func (cs *CamouflageService) handleFeedback(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req FeedbackRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad json", http.StatusBadRequest)
		return
	}
	if req.NodeID == "" || !(req.Reward >= -1 && req.Reward <= 1) {
		http.Error(w, "invalid body", http.StatusBadRequest)
		return
	}
	cs.dg.UpdateReward(req.NodeID, req.Reward)
	cs.mOps.Inc(map[string]string{"op": "feedback", "result": "ok"})
	writeJSON(w, http.StatusOK, map[string]any{"success": true})
}

func (cs *CamouflageService) handleMetricsJSON(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	writeJSON(w, http.StatusOK, cs.dg.GetMetrics())
}

func main() {
	// Ports and auth
	port := getenv("PORT", "8089")
	apiKey := os.Getenv("CAMOUFLAGE_API_KEY")
	templatesPath := getenv("TEMPLATES_PATH", "./core/maze_engine/templates")

	// Metrics registry and HTTP metrics
	reg := metrics.NewRegistry()
	httpMetrics := metrics.NewHTTPMetrics(reg, "camouflage-api")

	// Bandit-based decoy selection
	svc := NewCamouflageService()
	svc.mOps = metrics.NewLabeledCounter("camouflage_ops_total", "operations by type and result", []string{"op", "result"})
	reg.RegisterLabeledCounter(svc.mOps)

	// Maze engine for template-based camouflage
	engine, err := maze_engine.NewCamouflageEngine(templatesPath)
	if err != nil {
		log.Fatalf("Failed to initialize camouflage engine: %v", err)
	}
	api := &CamouflageAPI{engine: engine, metrics: initAPIMetrics()}
	// Register API counters into registry for Prom scraping
	reg.Register(api.metrics.TemplateRequests)
	reg.Register(api.metrics.SessionCreated)
	reg.Register(api.metrics.LogRequests)
	reg.Register(api.metrics.Errors)

	mux := http.NewServeMux()
	// Bandit endpoints
	mux.HandleFunc("/select", svc.handleSelect)
	mux.HandleFunc("/feedback", svc.handleFeedback)
	mux.HandleFunc("/graph", svc.handleMetricsJSON)
	// Camouflage template endpoints
	mux.HandleFunc("/v1/camouflage/template/", api.getTemplateHandler)
	mux.HandleFunc("/v1/camouflage/templates", api.listTemplatesHandler)
	mux.HandleFunc("/v1/camouflage/session", api.createSessionHandler)
	mux.HandleFunc("/v1/camouflage/session/", api.getSessionHandler)
	mux.HandleFunc("/v1/camouflage/adaptive", api.adaptiveTemplateHandler)
	mux.HandleFunc("/v1/camouflage/log", api.logHandler)
	// Health and metrics
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(200)
		w.Write([]byte("{\"status\":\"healthy\",\"service\":\"camouflage-api\"}"))
	})
	mux.Handle("/metrics", reg)

	// Compose middlewares: auth then CORS then HTTP metrics
	handler := withAuth(apiKey, api.corsMiddleware(mux))
	handler = httpMetrics.Middleware(handler)
	srv := &http.Server{Addr: ":" + port, Handler: handler, ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 15 * time.Second, WriteTimeout: 15 * time.Second, IdleTimeout: 60 * time.Second, MaxHeaderBytes: 1 << 20}
	log.Printf("[camouflage-api] listening on :%s", port)
	log.Fatal(srv.ListenAndServe())
}

func withAuth(apiKey string, next http.Handler) http.Handler {
	if apiKey == "" {
		return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" || r.URL.Path == "/metrics" {
			next.ServeHTTP(w, r)
			return
		}
		const p = "Bearer "
		auth := r.Header.Get("Authorization")
		if len(auth) <= len(p) || auth[:len(p)] != p || auth[len(p):] != apiKey {
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func getenv(k, d string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return d
}

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
	SessionID string                `json:"session_id"`
	Template  *maze_engine.Template `json:"template"`
	Success   bool                  `json:"success"`
	Message   string                `json:"message"`
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

func (api *CamouflageAPI) getSessionHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		api.metrics.Errors.Inc()
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	// Path: /v1/camouflage/session/{id}
	parts := strings.Split(strings.TrimPrefix(r.URL.Path, "/v1/camouflage/session/"), "/")
	if len(parts) == 0 || parts[0] == "" {
		api.metrics.Errors.Inc()
		http.Error(w, "Session id required", http.StatusBadRequest)
		return
	}
	id := parts[0]
	if sess, ok := api.engine.GetSession(id); ok {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"session_id":  sess.ID,
			"template":    sess.Template.Name,
			"start_time":  sess.StartTime,
			"request_cnt": sess.RequestCount,
			"client_ip":   sess.ClientIP,
			"user_agent":  sess.UserAgent,
		})
		return
	}
	http.Error(w, "Not found", http.StatusNotFound)
}

// adaptiveTemplateHandler picks a template matching recon profile; it does not expose model weights/params
func (api *CamouflageAPI) adaptiveTemplateHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		api.metrics.Errors.Inc()
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req struct {
		ReconType string `json:"recon_type"`
		ClientIP  string `json:"client_ip"`
		UserAgent string `json:"user_agent"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		api.metrics.Errors.Inc()
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}
	// Simple mapping strategy; in future could use ML, but do not expose internals
	pick := "nginx"
	rt := strings.ToLower(req.ReconType)
	switch {
	case strings.Contains(rt, "dirb") || strings.Contains(rt, "bruteforce"):
		pick = "apache"
	case strings.Contains(rt, "tech") || strings.Contains(rt, "wapp"):
		pick = "nginx"
	case strings.Contains(rt, "iis") || strings.Contains(req.UserAgent, "Windows"):
		pick = "iis"
	}
	if _, err := api.engine.GetTemplate(pick); err != nil {
		// fallback first available
		names := api.engine.ListTemplates()
		if len(names) > 0 {
			pick = names[0]
		}
	}
	_ = ledger.AppendJSONLine("data/ledger-camouflage-api.log", "camouflage_api", "adaptive_pick", map[string]any{
		"recon_type": req.ReconType,
		"client_ip":  req.ClientIP,
		"user_agent": req.UserAgent,
		"template":   pick,
	})
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]string{"template": pick})
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

func (api *CamouflageAPI) metricsHandler(w http.ResponseWriter, r *http.Request) {
	// Expose minimal Prometheus text format for API-level metrics
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	api.metrics.TemplateRequests.Expose(w)
	api.metrics.SessionCreated.Expose(w)
	api.metrics.LogRequests.Expose(w)
	api.metrics.Errors.Expose(w)
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
