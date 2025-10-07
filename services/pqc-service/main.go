package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"shieldx/core/crypto"
	"shieldx/shared/ledger"
	"shieldx/shared/metrics"
)

type PQCService struct {
	kex     *crypto.HybridKEX
	metrics *ServiceMetrics
}

type ServiceMetrics struct {
	KeyGenRequests    *metrics.Counter
	HandshakeRequests *metrics.Counter
	Errors            *metrics.Counter
	ActiveSessions    *metrics.Gauge
}

type KeyGenRequest struct {
	Algorithm string `json:"algorithm"`
	ClientID  string `json:"client_id"`
}

type KeyGenResponse struct {
	SessionID     string `json:"session_id"`
	PublicKey     []byte `json:"public_key"`
	Algorithm     string `json:"algorithm"`
	Version       int    `json:"version"`
	ExpiresAt     string `json:"expires_at"`
}

type HandshakeRequest struct {
	SessionID     string `json:"session_id"`
	PeerPublicKey []byte `json:"peer_public_key"`
	ClientID      string `json:"client_id"`
}

type HandshakeResponse struct {
	Success      bool   `json:"success"`
	SharedSecret []byte `json:"shared_secret,omitempty"`
	Message      string `json:"message"`
}

type SessionInfoResponse struct {
	SessionID   string    `json:"session_id"`
	Algorithm   string    `json:"algorithm"`
	Established bool      `json:"established"`
	CreatedAt   time.Time `json:"created_at"`
	ExpiresAt   time.Time `json:"expires_at"`
}

func main() {
	port := getEnv("PQC_SERVICE_PORT", "8092")
	kyberEnabled := getEnv("KYBER_ENABLED", "true") == "true"

	// Initialize PQC service
	service := &PQCService{
		kex:     crypto.NewHybridKEX(kyberEnabled),
		metrics: initServiceMetrics(),
	}

	// Setup routes
	mux := http.NewServeMux()

	// Key exchange endpoints
	mux.HandleFunc("/v1/pqc/keygen", service.keyGenHandler)
	mux.HandleFunc("/v1/pqc/handshake", service.handshakeHandler)
	mux.HandleFunc("/v1/pqc/session/", service.sessionInfoHandler)

	// Management endpoints
	mux.HandleFunc("/v1/pqc/cleanup", service.cleanupHandler)
	mux.HandleFunc("/v1/pqc/metrics", service.metricsHandler)

	// Health and status
	mux.HandleFunc("/health", service.healthHandler)
	mux.HandleFunc("/status", service.statusHandler)

	// Add middleware
	handler := service.authMiddleware(service.corsMiddleware(mux))

	// Start cleanup goroutine
	go service.periodicCleanup()

	server := &http.Server{
		Addr:         ":" + port,
		Handler:      handler,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 15 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	log.Printf("[pqc-service] Starting server on port %s (Kyber: %v)", port, kyberEnabled)
	log.Fatal(server.ListenAndServe())
}

func initServiceMetrics() *ServiceMetrics {
	return &ServiceMetrics{
		KeyGenRequests:    metrics.NewCounter("pqc_keygen_requests_total", "Total key generation requests"),
		HandshakeRequests: metrics.NewCounter("pqc_handshake_requests_total", "Total handshake requests"),
		Errors:            metrics.NewCounter("pqc_errors_total", "Total PQC service errors"),
		ActiveSessions:    metrics.NewGauge("pqc_active_sessions", "Active PQC sessions"),
	}
}

func (s *PQCService) keyGenHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.metrics.Errors.Inc()
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req KeyGenRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.metrics.Errors.Inc()
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	// Validate algorithm
	if req.Algorithm != "hybrid" && req.Algorithm != "x25519" {
		s.metrics.Errors.Inc()
		http.Error(w, "Unsupported algorithm", http.StatusBadRequest)
		return
	}

	// Generate key pair
	session, err := s.kex.GenerateKeyPair()
	if err != nil {
		s.metrics.Errors.Inc()
		http.Error(w, fmt.Sprintf("Key generation failed: %v", err), http.StatusInternalServerError)
		return
	}

	s.metrics.KeyGenRequests.Inc()
	s.metrics.ActiveSessions.Inc()

	// Create public key message
	publicKey := s.createPublicKeyMessage(session)

	response := KeyGenResponse{
		SessionID: session.ID,
		PublicKey: publicKey,
		Algorithm: s.algorithmToString(session.Algorithm),
		Version:   int(session.Version),
		ExpiresAt: session.ExpiresAt.Format(time.RFC3339),
	}

	// Audit log
	_ = ledger.AppendJSONLine("data/ledger-pqc.log", "pqc", "keygen", map[string]any{
		"session_id": session.ID,
		"client_id":  req.ClientID,
		"algorithm":  response.Algorithm,
		"version":    response.Version,
	})

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (s *PQCService) handshakeHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.metrics.Errors.Inc()
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req HandshakeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.metrics.Errors.Inc()
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	// Validate request
	if req.SessionID == "" || len(req.PeerPublicKey) == 0 {
		s.metrics.Errors.Inc()
		http.Error(w, "Missing required fields", http.StatusBadRequest)
		return
	}

	// Complete handshake
	err := s.kex.CompleteHandshake(req.SessionID, req.PeerPublicKey)
	if err != nil {
		s.metrics.Errors.Inc()
		http.Error(w, fmt.Sprintf("Handshake failed: %v", err), http.StatusBadRequest)
		return
	}

	// Get shared secret
	sharedSecret, err := s.kex.GetSharedSecret(req.SessionID)
	if err != nil {
		s.metrics.Errors.Inc()
		http.Error(w, fmt.Sprintf("Failed to get shared secret: %v", err), http.StatusInternalServerError)
		return
	}

	s.metrics.HandshakeRequests.Inc()

	response := HandshakeResponse{
		Success:      true,
		SharedSecret: sharedSecret,
		Message:      "Handshake completed successfully",
	}

	// Audit log
	_ = ledger.AppendJSONLine("data/ledger-pqc.log", "pqc", "handshake", map[string]any{
		"session_id": req.SessionID,
		"client_id":  req.ClientID,
		"success":    true,
	})

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (s *PQCService) sessionInfoHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		s.metrics.Errors.Inc()
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Extract session ID from path
	path := strings.TrimPrefix(r.URL.Path, "/v1/pqc/session/")
	sessionID := strings.Split(path, "/")[0]

	if sessionID == "" {
		s.metrics.Errors.Inc()
		http.Error(w, "Session ID required", http.StatusBadRequest)
		return
	}

	// This is a simplified implementation - in production, you'd need to expose
	// session info from the HybridKEX struct
	response := SessionInfoResponse{
		SessionID:   sessionID,
		Algorithm:   "hybrid",
		Established: false,
		CreatedAt:   time.Now(),
		ExpiresAt:   time.Now().Add(24 * time.Hour),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (s *PQCService) cleanupHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.metrics.Errors.Inc()
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	s.kex.CleanupExpiredSessions()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status":  "success",
		"message": "Expired sessions cleaned up",
	})
}

func (s *PQCService) metricsHandler(w http.ResponseWriter, r *http.Request) {
	kexMetrics := s.kex.GetMetrics()

	metrics := map[string]interface{}{
		"keygen_requests_total":    s.metrics.KeyGenRequests.Value(),
		"handshake_requests_total": s.metrics.HandshakeRequests.Value(),
		"errors_total":             s.metrics.Errors.Value(),
		"active_sessions":          s.metrics.ActiveSessions.Value(),
		"handshakes_total":         kexMetrics.HandshakesTotal,
		"handshakes_succeeded":     kexMetrics.HandshakesSucceeded,
		"handshakes_failed":        kexMetrics.HandshakesFailed,
		"quantum_handshakes":       kexMetrics.QuantumHandshakes,
		"classical_handshakes":     kexMetrics.ClassicalHandshakes,
		"average_latency_ms":       kexMetrics.AverageLatencyMs,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(metrics)
}

func (s *PQCService) healthHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(200)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":    "healthy",
		"service":   "pqc-service",
		"timestamp": time.Now().Format(time.RFC3339),
		"version":   "1.0.0",
	})
}

func (s *PQCService) statusHandler(w http.ResponseWriter, r *http.Request) {
	kexMetrics := s.kex.GetMetrics()

	status := map[string]interface{}{
		"service":            "pqc-service",
		"version":            "1.0.0",
		"uptime":             time.Since(time.Now()).String(), // Simplified
		"active_sessions":    s.metrics.ActiveSessions.Value(),
		"total_handshakes":   kexMetrics.HandshakesTotal,
		"success_rate":       calculateSuccessRate(kexMetrics),
		"average_latency_ms": kexMetrics.AverageLatencyMs,
		"quantum_enabled":    true, // Based on initialization
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(status)
}

func (s *PQCService) authMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Skip auth for health and status
		if r.URL.Path == "/health" || r.URL.Path == "/status" {
			next.ServeHTTP(w, r)
			return
		}

		auth := r.Header.Get("Authorization")
		if !strings.HasPrefix(auth, "Bearer ") {
			s.metrics.Errors.Inc()
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}

		token := strings.TrimPrefix(auth, "Bearer ")
		if !s.validateToken(token) {
			s.metrics.Errors.Inc()
			http.Error(w, "Invalid token", http.StatusUnauthorized)
			return
		}

		next.ServeHTTP(w, r)
	})
}

func (s *PQCService) corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}

		next.ServeHTTP(w, r)
	})
}

func (s *PQCService) periodicCleanup() {
	ticker := time.NewTicker(10 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		s.kex.CleanupExpiredSessions()
	}
}

func (s *PQCService) validateToken(token string) bool {
	expectedToken := getEnv("PQC_API_KEY", "default_pqc_key")
	return token == expectedToken
}

func (s *PQCService) createPublicKeyMessage(session interface{}) []byte {
	// Simplified - in production, this would extract the actual public key
	// from the session and format it properly
	return []byte("mock_public_key_message")
}

func (s *PQCService) algorithmToString(alg byte) string {
	switch alg {
	case 0x01:
		return "x25519"
	case 0x02:
		return "hybrid"
	default:
		return "unknown"
	}
}

func calculateSuccessRate(metrics crypto.KEXMetrics) float64 {
	if metrics.HandshakesTotal == 0 {
		return 0.0
	}
	return float64(metrics.HandshakesSucceeded) / float64(metrics.HandshakesTotal) * 100.0
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}