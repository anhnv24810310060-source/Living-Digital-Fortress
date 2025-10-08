package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"time"

	"shieldx/shared/shieldx-common/pkg/metrics"
	logcorr "shieldx/shared/shieldx-common/pkg/observability/logcorr"
	otelobs "shieldx/shared/shieldx-common/pkg/observability/otel"
	"shieldx/shared/shieldx-common/pkg/ratls"
	"shieldx/shared/shieldx-common/pkg/verifier"
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
	mux.HandleFunc("/whoami", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		info := map[string]interface{}{
			"service":       "verifier-pool",
			"ratls_enabled": os.Getenv("RATLS_ENABLE") == "true",
		}
		if os.Getenv("RATLS_ENABLE") == "true" {
			td := getenvDefault("RATLS_TRUST_DOMAIN", "shieldx.local")
			info["trust_domain"] = td
		}
		json.NewEncoder(w).Encode(info)
	})
	mux.Handle("/metrics", reg)

	// OpenTelemetry tracing (no-op unless built with otelotlp and endpoint set)
	shutdown := otelobs.InitTracer("verifier_pool")
	defer shutdown(context.Background())

	// Wrap with metrics, log-correlation, then tracing
	h := httpMetrics.Middleware(mux)
	h = logcorr.Middleware(h)
	h = otelobs.WrapHTTPHandler("verifier_pool", h)

	// RA-TLS optional enablement
	gCertExpiry := metrics.NewGauge("ratls_cert_expiry_seconds", "Seconds until current RA-TLS cert expiry")
	reg.RegisterGauge(gCertExpiry)
	var issuer *ratls.AutoIssuer
	if os.Getenv("RATLS_ENABLE") == "true" {
		td := getenvDefault("RATLS_TRUST_DOMAIN", "shieldx.local")
		ns := getenvDefault("RATLS_NAMESPACE", "default")
		svc := getenvDefault("RATLS_SERVICE", "verifier-pool")
		rotate := parseDurationDefault("RATLS_ROTATE_EVERY", 45*time.Minute)
		valid := parseDurationDefault("RATLS_VALIDITY", 60*time.Minute)
		ai, err := ratls.NewDevIssuer(ratls.Identity{TrustDomain: td, Namespace: ns, Service: svc}, rotate, valid)
		if err != nil {
			log.Fatalf("[verifier-pool] RA-TLS init: %v", err)
		}
		issuer = ai
		go func() {
			for {
				if t, err := issuer.LeafNotAfter(); err == nil {
					gCertExpiry.Set(uint64(time.Until(t).Seconds()))
				}
				time.Sleep(1 * time.Minute)
			}
		}()
	}
	addr := fmt.Sprintf(":%s", port)
	srv := &http.Server{Addr: addr, Handler: h}
	if issuer != nil {
		srv.TLSConfig = issuer.ServerTLSConfig(true, getenvDefault("RATLS_TRUST_DOMAIN", "shieldx.local"))
		log.Printf("Verifier Pool (RA-TLS) starting on %s", addr)
		log.Fatal(srv.ListenAndServeTLS("", ""))
	} else {
		log.Printf("Verifier Pool starting on %s", addr)
		log.Fatal(srv.ListenAndServe())
	}
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
		"status":  "registered",
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
		"note":   "Implementation pending - requires access control",
	})
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":    "healthy",
		"timestamp": time.Now(),
		"service":   "verifier-pool",
	})
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getenvDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
func parseDurationDefault(key string, def time.Duration) time.Duration {
	if v := os.Getenv(key); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			return d
		}
	}
	return def
}
