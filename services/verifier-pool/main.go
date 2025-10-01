package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"strconv"
	"time"

	"shieldx/pkg/verifier"
	"shieldx/pkg/metrics"
	otelobs "shieldx/pkg/observability/otel"
    logcorr "shieldx/pkg/observability/logcorr"
)

type Server struct {
	pool *verifier.Pool
	port string
}

func main() {
	port := getEnv("VERIFIER_POOL_PORT", "8087")
	minNodes, _ := strconv.Atoi(getEnv("MIN_VERIFIER_NODES", "3"))
	consensus, _ := strconv.ParseFloat(getEnv("CONSENSUS_THRESHOLD", "0.67"), 64)
	
	pool := verifier.NewPool(minNodes, consensus)
	
	server := &Server{
		pool: pool,
		port: port,
	}
	
	mux := http.NewServeMux()
	reg := metrics.NewRegistry()
	httpMetrics := metrics.NewHTTPMetrics(reg, "verifier_pool")
	mux.HandleFunc("/register", server.handleRegister)
	mux.HandleFunc("/validate", server.handleValidate)
	mux.HandleFunc("/nodes", server.handleListNodes)
	mux.HandleFunc("/health", server.handleHealth)
	mux.Handle("/metrics", reg)

	// OpenTelemetry tracing (no-op unless built with otelotlp and endpoint set)
	shutdown := otelobs.InitTracer("verifier_pool")
	defer shutdown(context.Background())

	// Wrap with metrics, log-correlation, then tracing
	h := httpMetrics.Middleware(mux)
	h = logcorr.Middleware(h)
	h = otelobs.WrapHTTPHandler("verifier_pool", h)

	log.Printf("Verifier Pool starting on port %s", port)
	log.Fatal(http.ListenAndServe(":"+port, h))
}

func (s *Server) handleRegister(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	
	var node verifier.VerifierNode
	if err := json.NewDecoder(r.Body).Decode(&node); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}
	
	node.LastSeen = time.Now()
	if node.Reputation == 0 {
		node.Reputation = 0.5
	}
	
	if err := s.pool.AddNode(&node); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status": "registered",
		"node_id": node.ID,
	})
}

func (s *Server) handleValidate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	
	var req verifier.ValidationRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}
	
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	
	result, err := s.pool.ValidatePack(ctx, &req)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

func (s *Server) handleListNodes(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status": "nodes_listed",
		"note": "Implementation pending - requires access control",
	})
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status": "healthy",
		"timestamp": time.Now(),
		"service": "verifier-pool",
	})
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}