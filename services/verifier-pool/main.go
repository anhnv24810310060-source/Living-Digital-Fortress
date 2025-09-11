package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/shieldx/pkg/verifier"
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
	
	http.HandleFunc("/register", server.handleRegister)
	http.HandleFunc("/validate", server.handleValidate)
	http.HandleFunc("/nodes", server.handleListNodes)
	http.HandleFunc("/health", server.handleHealth)
	
	log.Printf("Verifier Pool starting on port %s", port)
	log.Fatal(http.ListenAndServe(":"+port, nil))
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