package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/tls"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"golang.org/x/time/rate"
    "shieldx/pkg/metrics"
    otelobs "shieldx/pkg/observability/otel"
)

// Production ShieldX Gateway - Central Orchestrator
type ShieldXGateway struct {
	// Service endpoints with load balancing
	services       map[string]*ServiceEndpoint
	servicesMutex  sync.RWMutex

	// Rate limiting per IP
	rateLimiters   map[string]*rate.Limiter
	limiterMutex   sync.RWMutex

	// Circuit breakers for fault tolerance
	circuitBreakers map[string]*CircuitBreaker

	// HTTP clients with connection pooling
	httpClients    map[string]*http.Client
	clientMutex    sync.RWMutex

	// Configuration
	config         *GatewayConfig

	// Graceful shutdown
	shutdownChan   chan struct{}
	wg            sync.WaitGroup
	server        *http.Server

	// Metrics counters
	requestCount   int64
	errorCount     int64
	activeRequests int64
}

type GatewayConfig struct {
	Port                  int           `json:"port"`
	MaxConcurrentRequests int           `json:"max_concurrent_requests"`
	RequestTimeout        time.Duration `json:"request_timeout"`
	RateLimitRPS         int           `json:"rate_limit_rps"`
	HealthCheckInterval  time.Duration `json:"health_check_interval"`
	
	Services map[string]ServiceConfig `json:"services"`
	
	TLS struct {
		Enabled  bool   `json:"enabled"`
		CertFile string `json:"cert_file"`
		KeyFile  string `json:"key_file"`
	} `json:"tls"`
}

type ServiceConfig struct {
	URLs    []string      `json:"urls"`
	Timeout time.Duration `json:"timeout"`
	Retries int          `json:"retries"`
}

type ServiceEndpoint struct {
	URLs        []string
	HealthyURLs []string
	Timeout     time.Duration
	Retries     int
	mutex       sync.RWMutex
	lastCheck   time.Time
}

type CircuitBreaker struct {
	mutex        sync.Mutex
	state        CircuitState
	failures     int
	lastFailTime time.Time
	maxFailures  int
	timeout      time.Duration
}

type CircuitState int

const (
	StateClosed CircuitState = iota
	StateOpen
	StateHalfOpen
)

// Request/Response structures
type GatewayRequest struct {
	ClientIP    string            `json:"client_ip"`
	UserAgent   string            `json:"user_agent"`
	Path        string            `json:"path"`
	Method      string            `json:"method"`
	Headers     map[string]string `json:"headers"`
	Body        string            `json:"body"`
	Timestamp   time.Time         `json:"timestamp"`
	RequestID   string            `json:"request_id"`
}

type GatewayResponse struct {
	Action         string                 `json:"action"`
	Destination    string                 `json:"destination"`
	Message        string                 `json:"message"`
	ThreatScore    float64               `json:"threat_score"`
	TrustScore     float64               `json:"trust_score"`
	ProcessingTime time.Duration         `json:"processing_time"`
	Metadata       map[string]interface{} `json:"metadata"`
	RequestID      string                `json:"request_id"`
}

type DecisionContext struct {
	TrustResult map[string]interface{} `json:"trust_result"`
	AIResult    map[string]interface{} `json:"ai_result"`
	Request     *GatewayRequest        `json:"request"`
}

func NewShieldXGateway() *ShieldXGateway {
	config := loadConfig()
	
	gateway := &ShieldXGateway{
		services:        make(map[string]*ServiceEndpoint),
		rateLimiters:    make(map[string]*rate.Limiter),
		circuitBreakers: make(map[string]*CircuitBreaker),
		httpClients:     make(map[string]*http.Client),
		config:          config,
		shutdownChan:    make(chan struct{}),
	}

	gateway.initServices()
	gateway.initCircuitBreakers()
	
	return gateway
}

func (sg *ShieldXGateway) initServices() {
	for serviceName, serviceConfig := range sg.config.Services {
		endpoint := &ServiceEndpoint{
			URLs:        serviceConfig.URLs,
			HealthyURLs: make([]string, len(serviceConfig.URLs)),
			Timeout:     serviceConfig.Timeout,
			Retries:     serviceConfig.Retries,
		}
		
		// Initially mark all URLs as healthy
		copy(endpoint.HealthyURLs, endpoint.URLs)
		
		sg.services[serviceName] = endpoint
		
		// Create HTTP client with connection pooling
		// Wrap transport with OpenTelemetry to propagate traces on client calls
		baseTransport := &http.Transport{
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 10,
				IdleConnTimeout:     90 * time.Second,
				TLSClientConfig: &tls.Config{
					InsecureSkipVerify: false,
				},
		}
		sg.httpClients[serviceName] = &http.Client{
			Timeout:   serviceConfig.Timeout,
			Transport: otelobs.WrapHTTPTransport(baseTransport),
		}
		
		// Start health checking
		go sg.healthCheckLoop(serviceName, endpoint)
	}
}

func (sg *ShieldXGateway) initCircuitBreakers() {
	for serviceName := range sg.services {
		sg.circuitBreakers[serviceName] = &CircuitBreaker{
			maxFailures: 5,
			timeout:     30 * time.Second,
			state:       StateClosed,
		}
	}
}

// Main request processing handler
func (sg *ShieldXGateway) processRequest(w http.ResponseWriter, r *http.Request) {
	startTime := time.Now()
	sg.activeRequests++
	defer func() { sg.activeRequests-- }()

	// Generate request ID
	requestID := generateRequestID()
	r.Header.Set("X-Request-ID", requestID)

	// Parse and validate request
	gatewayReq, err := sg.parseRequest(r, requestID)
	if err != nil {
		sg.handleError(w, "INVALID_REQUEST", err, startTime, requestID)
		return
	}

	// Rate limiting check
	if !sg.checkRateLimit(gatewayReq.ClientIP) {
		sg.handleError(w, "RATE_LIMITED", fmt.Errorf("rate limit exceeded"), startTime, requestID)
		return
	}

	// Process through decision pipeline
	response := sg.processDecisionPipeline(gatewayReq)
	response.ProcessingTime = time.Since(startTime)
	response.RequestID = requestID

	// Log request
	sg.logRequest(gatewayReq, response)

	// Record metrics
	sg.recordMetrics(gatewayReq, response, startTime)

	// Send response
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Request-ID", requestID)
	
	statusCode := sg.getStatusCode(response.Action)
	w.WriteHeader(statusCode)
	
	json.NewEncoder(w).Encode(response)
}

func (sg *ShieldXGateway) processDecisionPipeline(req *GatewayRequest) *GatewayResponse {
	ctx := &DecisionContext{Request: req}

	// Step 1: Zero Trust Evaluation
	trustResult, err := sg.callZeroTrust(req)
	if err != nil {
		log.Printf("Zero Trust service error: %v", err)
		trustResult = sg.getFallbackTrustResult()
	}
	ctx.TrustResult = trustResult

	// Step 2: AI Analysis
	aiResult, err := sg.callAIAnalyzer(req)
	if err != nil {
		log.Printf("AI Analyzer service error: %v", err)
		aiResult = sg.getFallbackAIResult()
	}
	ctx.AIResult = aiResult

	// Step 3: Decision Matrix
	decision := sg.makeDecision(ctx)

	// Step 4: Route Request
	return sg.routeRequest(decision, ctx)
}

func (sg *ShieldXGateway) makeDecision(ctx *DecisionContext) string {
	trustScore := getFloat64(ctx.TrustResult, "trust_score", 0.5)
	threatScore := getFloat64(ctx.AIResult, "threat_score", 0.5)
	isAnomaly := getBool(ctx.AIResult, "is_anomaly", false)
	requireDeception := getBool(ctx.TrustResult, "require_deception", false)

	// Production decision matrix
	if threatScore > 0.9 || trustScore < 0.1 {
		return "BLOCK"
	} else if threatScore > 0.7 || trustScore < 0.3 {
		return "ISOLATE"
	} else if threatScore > 0.5 || isAnomaly || requireDeception {
		return "DECEIVE"
	} else if trustScore < 0.7 {
		return "MAZE"
	}
	
	return "ALLOW"
}

func (sg *ShieldXGateway) routeRequest(decision string, ctx *DecisionContext) *GatewayResponse {
	response := &GatewayResponse{
		Action:      decision,
		ThreatScore: getFloat64(ctx.AIResult, "threat_score", 0.0),
		TrustScore:  getFloat64(ctx.TrustResult, "trust_score", 1.0),
		Metadata:    make(map[string]interface{}),
	}

	switch decision {
	case "BLOCK":
		response.Destination = "blocked"
		response.Message = "Request blocked by security policy"
		
	case "ISOLATE":
		isolationResult := sg.callIsolationVault(ctx.Request)
		response.Destination = "isolation_vault"
		response.Message = "Request routed to isolation vault"
		response.Metadata["isolation_config"] = isolationResult
		
	case "DECEIVE":
		deceptionResult := sg.callDeceptionEngine(ctx.Request)
		response.Destination = "deception_engine"
		response.Message = "Request routed to deception layer"
		response.Metadata["decoy_endpoint"] = deceptionResult
		
	case "MAZE":
		mazeResult := sg.callDynamicMaze(ctx.Request)
		response.Destination = "dynamic_maze"
		response.Message = "Request routed through dynamic maze"
		response.Metadata["maze_path"] = mazeResult
		
	default: // ALLOW
		response.Destination = "privacy_enclave"
		response.Message = "Request allowed to privacy enclave"
	}

	return response
}

// Service call methods with circuit breaker protection
func (sg *ShieldXGateway) callZeroTrust(req *GatewayRequest) (map[string]interface{}, error) {
	return sg.callServiceWithCircuitBreaker("zero_trust", map[string]interface{}{
		"client_ip":  req.ClientIP,
		"user_agent": req.UserAgent,
		"path":       req.Path,
		"method":     req.Method,
		"headers":    req.Headers,
		"timestamp":  req.Timestamp,
	})
}

func (sg *ShieldXGateway) callAIAnalyzer(req *GatewayRequest) (map[string]interface{}, error) {
	return sg.callServiceWithCircuitBreaker("ai_analyzer", map[string]interface{}{
		"timestamp":   req.Timestamp,
		"source":      "shieldx-gateway",
		"event_type":  "connection",
		"tenant_id":   extractTenant(req),
		"features":    sg.extractFeatures(req),
		"threat_score": sg.calculateBaseThreatScore(req),
	})
}

func (sg *ShieldXGateway) callIsolationVault(req *GatewayRequest) map[string]interface{} {
	result, err := sg.callServiceWithCircuitBreaker("isolation_vault", map[string]interface{}{
		"client_ip":    req.ClientIP,
		"threat_level": "HIGH",
		"request_id":   req.RequestID,
	})
	
	if err != nil {
		return map[string]interface{}{
			"isolation_level": "STANDARD",
			"network_restrictions": []string{"10.0.0.0/8"},
		}
	}
	
	return result
}

func (sg *ShieldXGateway) callDeceptionEngine(req *GatewayRequest) map[string]interface{} {
	result, err := sg.callServiceWithCircuitBreaker("deception_engine", map[string]interface{}{
		"client_ip":      req.ClientIP,
		"attack_vector":  sg.identifyAttackVector(req.Path),
		"threat_score":   sg.calculateBaseThreatScore(req),
	})
	
	if err != nil {
		return map[string]interface{}{
			"decoy_endpoint": "http://localhost:8083/decoy",
			"decoy_type":     "honeypot",
		}
	}
	
	return result
}

func (sg *ShieldXGateway) callDynamicMaze(req *GatewayRequest) map[string]interface{} {
	result, err := sg.callServiceWithCircuitBreaker("dynamic_maze", map[string]interface{}{
		"client_ip":   req.ClientIP,
		"path":        req.Path,
		"user_agent":  req.UserAgent,
	})
	
	if err != nil {
		return map[string]interface{}{
			"maze_path": "standard_route",
			"delay_ms":  100,
		}
	}
	
	return result
}

func (sg *ShieldXGateway) callServiceWithCircuitBreaker(serviceName string, payload map[string]interface{}) (map[string]interface{}, error) {
	cb := sg.circuitBreakers[serviceName]
	
	var result map[string]interface{}
	var err error
	
	cbErr := cb.Call(func() error {
		result, err = sg.callService(serviceName, payload)
		return err
	})
	
	if cbErr != nil {
		return nil, cbErr
	}
	
	return result, err
}

func (sg *ShieldXGateway) callService(serviceName string, payload map[string]interface{}) (map[string]interface{}, error) {
	service := sg.services[serviceName]
	client := sg.httpClients[serviceName]
	
	service.mutex.RLock()
	healthyURLs := make([]string, len(service.HealthyURLs))
	copy(healthyURLs, service.HealthyURLs)
	service.mutex.RUnlock()
	
	if len(healthyURLs) == 0 {
		return nil, fmt.Errorf("no healthy endpoints for service %s", serviceName)
	}
	
	// Try each healthy endpoint
	for _, url := range healthyURLs {
		body, _ := json.Marshal(payload)
		
		resp, err := client.Post(url+"/evaluate", "application/json", bytes.NewReader(body))
		if err != nil {
			sg.markUnhealthy(serviceName, url)
			continue
		}
		
		defer resp.Body.Close()
		
		if resp.StatusCode != http.StatusOK {
			sg.markUnhealthy(serviceName, url)
			continue
		}
		
		var result map[string]interface{}
		if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
			continue
		}
		
		return result, nil
	}
	
	return nil, fmt.Errorf("all endpoints failed for service %s", serviceName)
}

// Circuit breaker implementation
func (cb *CircuitBreaker) Call(fn func() error) error {
	cb.mutex.Lock()
	defer cb.mutex.Unlock()
	
	if cb.state == StateOpen {
		if time.Since(cb.lastFailTime) > cb.timeout {
			cb.state = StateHalfOpen
		} else {
			return fmt.Errorf("circuit breaker is open")
		}
	}
	
	err := fn()
	
	if err != nil {
		cb.failures++
		cb.lastFailTime = time.Now()
		
		if cb.failures >= cb.maxFailures {
			cb.state = StateOpen
		}
		return err
	}
	
	// Success - reset circuit breaker
	cb.failures = 0
	cb.state = StateClosed
	return nil
}

// Health checking
func (sg *ShieldXGateway) healthCheckLoop(serviceName string, endpoint *ServiceEndpoint) {
	ticker := time.NewTicker(sg.config.HealthCheckInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			sg.performHealthCheck(serviceName, endpoint)
		case <-sg.shutdownChan:
			return
		}
	}
}

func (sg *ShieldXGateway) performHealthCheck(serviceName string, endpoint *ServiceEndpoint) {
	client := sg.httpClients[serviceName]
	var healthyURLs []string
	
	for _, url := range endpoint.URLs {
		resp, err := client.Get(url + "/health")
		if err == nil && resp.StatusCode == http.StatusOK {
			healthyURLs = append(healthyURLs, url)
		}
		if resp != nil {
			resp.Body.Close()
		}
	}
	
	endpoint.mutex.Lock()
	endpoint.HealthyURLs = healthyURLs
	endpoint.lastCheck = time.Now()
	endpoint.mutex.Unlock()
}

func (sg *ShieldXGateway) markUnhealthy(serviceName, url string) {
	service := sg.services[serviceName]
	service.mutex.Lock()
	defer service.mutex.Unlock()
	
	for i, healthyURL := range service.HealthyURLs {
		if healthyURL == url {
			service.HealthyURLs = append(service.HealthyURLs[:i], service.HealthyURLs[i+1:]...)
			break
		}
	}
}

// Rate limiting
func (sg *ShieldXGateway) checkRateLimit(clientIP string) bool {
	sg.limiterMutex.Lock()
	defer sg.limiterMutex.Unlock()
	
	limiter, exists := sg.rateLimiters[clientIP]
	if !exists {
		limiter = rate.NewLimiter(rate.Limit(sg.config.RateLimitRPS), sg.config.RateLimitRPS)
		sg.rateLimiters[clientIP] = limiter
	}
	
	return limiter.Allow()
}

// Security middleware
func (sg *ShieldXGateway) securityMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Request size limiting
		r.Body = http.MaxBytesReader(w, r.Body, 10<<20) // 10MB max
		
		// Security headers
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("X-Frame-Options", "DENY")
		w.Header().Set("X-XSS-Protection", "1; mode=block")
		
		// Request validation
		if !sg.validateRequest(r) {
			http.Error(w, "Invalid request", http.StatusBadRequest)
			return
		}
		
		next.ServeHTTP(w, r)
	})
}

// Health handler
func (sg *ShieldXGateway) healthHandler(w http.ResponseWriter, r *http.Request) {
	health := map[string]interface{}{
		"status":          "healthy",
		"timestamp":       time.Now(),
		"active_requests": sg.activeRequests,
		"request_count":   sg.requestCount,
		"error_count":     sg.errorCount,
		"services":        sg.getServiceHealth(),
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(health)
}

func (sg *ShieldXGateway) getServiceHealth() map[string]interface{} {
	health := make(map[string]interface{})
	
	for serviceName, endpoint := range sg.services {
		endpoint.mutex.RLock()
		healthyCount := len(endpoint.HealthyURLs)
		totalCount := len(endpoint.URLs)
		endpoint.mutex.RUnlock()
		
		health[serviceName] = map[string]interface{}{
			"healthy_endpoints": healthyCount,
			"total_endpoints":   totalCount,
			"status":           func() string {
				if healthyCount == 0 {
					return "unhealthy"
				} else if healthyCount < totalCount {
					return "degraded"
				}
				return "healthy"
			}(),
		}
	}
	
	return health
}

// Graceful shutdown
func (sg *ShieldXGateway) Shutdown(ctx context.Context) error {
	log.Println("Starting graceful shutdown...")
	
	close(sg.shutdownChan)
	
	// Shutdown HTTP server
	if err := sg.server.Shutdown(ctx); err != nil {
		return err
	}
	
	// Wait for ongoing requests to complete
	done := make(chan struct{})
	go func() {
		sg.wg.Wait()
		close(done)
	}()
	
	select {
	case <-done:
		log.Println("Graceful shutdown completed")
		return nil
	case <-ctx.Done():
		log.Println("Shutdown timeout exceeded")
		return ctx.Err()
	}
}

func main() {
	log.Println("Starting ShieldX Gateway (Production)")
	
	gateway := NewShieldXGateway()
	// OpenTelemetry tracing (no-op unless built with otelotlp and endpoint set)
	shutdown := otelobs.InitTracer("shieldx_gateway")
	defer shutdown(context.Background())
	
	// Setup HTTP server
	mux := http.NewServeMux()
	reg := metrics.NewRegistry()
	
	// Main processing endpoint
	mux.HandleFunc("/", gateway.processRequest)
	
	// Health check endpoint
	mux.HandleFunc("/health", gateway.healthHandler)
	
	// Expose metrics
	mux.Handle("/metrics", reg)

	// Apply middleware with HTTP metrics and tracing wrapper
	httpMetrics := metrics.NewHTTPMetrics(reg, "shieldx_gateway")
	handler := httpMetrics.Middleware(gateway.securityMiddleware(mux))
	handler = otelobs.WrapHTTPHandler("shieldx_gateway", handler)
	
	// Configure server
	gateway.server = &http.Server{
		Addr:         fmt.Sprintf(":%d", gateway.config.Port),
		Handler:      handler,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  120 * time.Second,
	}
	
	// Start server
	go func() {
		log.Printf("ShieldX Gateway listening on port %d", gateway.config.Port)
		
		if gateway.config.TLS.Enabled {
			err := gateway.server.ListenAndServeTLS(
				gateway.config.TLS.CertFile,
				gateway.config.TLS.KeyFile,
			)
			if err != nil && err != http.ErrServerClosed {
				log.Fatalf("HTTPS server failed: %v", err)
			}
		} else {
			err := gateway.server.ListenAndServe()
			if err != nil && err != http.ErrServerClosed {
				log.Fatalf("HTTP server failed: %v", err)
			}
		}
	}()
	
	// Wait for shutdown signal
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	<-sigChan
	
	// Graceful shutdown
	log.Println("Shutting down ShieldX Gateway...")
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	
	if err := gateway.Shutdown(ctx); err != nil {
		log.Printf("Shutdown error: %v", err)
	}
	
	log.Println("ShieldX Gateway stopped")
}

func loadConfig() *GatewayConfig {
	// Default production configuration
	cfg := &GatewayConfig{
		Port:                  8080,
		MaxConcurrentRequests: 10000,
		RequestTimeout:        30 * time.Second,
		RateLimitRPS:         1000,
		HealthCheckInterval:  30 * time.Second,
		Services: map[string]ServiceConfig{
			"zero_trust": {
				URLs:    []string{"http://localhost:8091"},
				Timeout: 5 * time.Second,
				Retries: 2,
			},
			"ai_analyzer": {
				URLs:    []string{"http://localhost:8087"},
				Timeout: 10 * time.Second,
				Retries: 2,
			},
			"isolation_vault": {
				URLs:    []string{"http://localhost:8085"},
				Timeout: 5 * time.Second,
				Retries: 1,
			},
			"deception_engine": {
				URLs:    []string{"http://localhost:8084"},
				Timeout: 5 * time.Second,
				Retries: 1,
			},
			"dynamic_maze": {
				URLs:    []string{"http://localhost:8084"},
				Timeout: 5 * time.Second,
				Retries: 1,
			},
		},
	}
	if v := os.Getenv("GATEWAY_PORT"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 { cfg.Port = n }
	}
	return cfg
}

// getenvInt is a small helper kept for parity with other services (unused now but handy)
func getenvInt(key string, def int) int {
	v := os.Getenv(key)
	if v == "" { return def }
	if n, err := strconv.Atoi(v); err == nil { return n }
	return def
}

// Helper functions
func generateRequestID() string {
	bytes := make([]byte, 16)
	rand.Read(bytes)
	return hex.EncodeToString(bytes)
}

func (sg *ShieldXGateway) parseRequest(r *http.Request, requestID string) (*GatewayRequest, error) {
	headers := make(map[string]string)
	for k, v := range r.Header {
		if len(v) > 0 {
			headers[k] = v[0]
		}
	}
	
	return &GatewayRequest{
		ClientIP:  getClientIP(r),
		UserAgent: r.UserAgent(),
		Path:      r.URL.Path,
		Method:    r.Method,
		Headers:   headers,
		Timestamp: time.Now(),
		RequestID: requestID,
	}, nil
}

func getClientIP(r *http.Request) string {
	if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
		parts := strings.Split(xff, ",")
		if len(parts) > 0 {
			return strings.TrimSpace(parts[0])
		}
	}
	if rip := r.Header.Get("X-Real-IP"); rip != "" {
		return rip
	}
	return r.RemoteAddr
}

func (sg *ShieldXGateway) validateRequest(r *http.Request) bool {
	// Basic request validation
	if r.Method == "" || r.URL.Path == "" {
		return false
	}
	return true
}

func (sg *ShieldXGateway) handleError(w http.ResponseWriter, action string, err error, startTime time.Time, requestID string) {
	sg.errorCount++
	
	response := &GatewayResponse{
		Action:         action,
		Message:        err.Error(),
		ProcessingTime: time.Since(startTime),
		RequestID:      requestID,
	}
	
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Request-ID", requestID)
	w.WriteHeader(sg.getStatusCode(action))
	
	json.NewEncoder(w).Encode(response)
}

func (sg *ShieldXGateway) getStatusCode(action string) int {
	switch action {
	case "BLOCK", "INVALID_REQUEST":
		return http.StatusForbidden
	case "RATE_LIMITED":
		return http.StatusTooManyRequests
	default:
		return http.StatusOK
	}
}

func (sg *ShieldXGateway) logRequest(req *GatewayRequest, resp *GatewayResponse) {
	log.Printf("[%s] %s %s -> %s (threat: %.2f, trust: %.2f, time: %v)",
		req.RequestID, req.Method, req.Path, resp.Action,
		resp.ThreatScore, resp.TrustScore, resp.ProcessingTime)
}

func (sg *ShieldXGateway) recordMetrics(req *GatewayRequest, resp *GatewayResponse, startTime time.Time) {
	sg.requestCount++
	// In production, this would send to Prometheus/metrics system
}

func (sg *ShieldXGateway) getFallbackTrustResult() map[string]interface{} {
	return map[string]interface{}{
		"trust_score":        0.5,
		"allow":             true,
		"require_deception": false,
	}
}

func (sg *ShieldXGateway) getFallbackAIResult() map[string]interface{} {
	return map[string]interface{}{
		"threat_score": 0.3,
		"is_anomaly":   false,
	}
}

func (sg *ShieldXGateway) extractFeatures(req *GatewayRequest) []float64 {
	features := make([]float64, 18)
	features[0] = float64(len(req.Path))
	features[1] = float64(strings.Count(req.Path, "/"))
	features[2] = float64(len(req.UserAgent))
	// Add more feature extraction logic
	return features
}

func (sg *ShieldXGateway) calculateBaseThreatScore(req *GatewayRequest) float64 {
	score := 0.0
	if strings.Contains(req.Path, "../") { score += 0.3 }
	if strings.Contains(req.Path, "admin") { score += 0.2 }
	if len(req.Path) > 200 { score += 0.2 }
	return min(score, 1.0)
}

func (sg *ShieldXGateway) identifyAttackVector(path string) string {
	path = strings.ToLower(path)
	if strings.Contains(path, "union") || strings.Contains(path, "select") {
		return "SQL_INJECTION"
	}
	if strings.Contains(path, "script") {
		return "XSS"
	}
	if strings.Contains(path, "../") {
		return "PATH_TRAVERSAL"
	}
	return "UNKNOWN"
}

func extractTenant(req *GatewayRequest) string {
	if tenant := req.Headers["X-Tenant"]; tenant != "" {
		return tenant
	}
	return "default"
}

func getFloat64(m map[string]interface{}, key string, defaultVal float64) float64 {
	if val, ok := m[key]; ok {
		if f, ok := val.(float64); ok {
			return f
		}
	}
	return defaultVal
}

func getBool(m map[string]interface{}, key string, defaultVal bool) bool {
	if val, ok := m[key]; ok {
		if b, ok := val.(bool); ok {
			return b
		}
	}
	return defaultVal
}

func min(a, b float64) float64 {
	if a < b { return a }
	return b
}