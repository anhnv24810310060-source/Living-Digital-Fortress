package main

import (
	"log"
	"net/http"
	"os"
	"time"

	"shieldx/pkg/metrics"
)

func main() {
	dbURL := getEnv("DATABASE_URL", "postgres://credits_user:credits_pass2024@localhost:5432/credits")
	port := getEnv("PORT", "5004")
	serviceName := "credits"

	ledger, err := NewCreditLedger(dbURL)
	if err != nil {
		log.Fatalf("Failed to initialize ledger: %v", err)
	}
	defer ledger.Close()

	// Optional Redis cache for hot balance lookups
	if addr := os.Getenv("REDIS_ADDR"); addr != "" {
		if err := ledger.initRedis(addr, os.Getenv("REDIS_PASSWORD")); err != nil {
			log.Printf("[credits] Redis disabled: %v", err)
		} else {
			log.Printf("[credits] Redis enabled at %s for balance cache", addr)
		}
	}

	// Metrics setup
	reg := metrics.NewRegistry()
	httpMetrics := metrics.NewHTTPMetrics(reg, serviceName)
	ledger.metrics = initCreditsMetrics(reg)

	go ledger.cleanupExpiredKeys()
	ledger.startAlertWatcher()

	// API per spec
	mux := http.NewServeMux()
	mux.HandleFunc("/credits/purchase", ledger.PurchaseCredits)
	mux.HandleFunc("/credits/topup", ledger.PurchaseCredits)
	mux.HandleFunc("/credits/consume", ledger.ConsumeCredits)
	mux.HandleFunc("/credits/reserve", ledger.ReserveCredits)
	mux.HandleFunc("/credits/commit", ledger.CommitReservation)
	mux.HandleFunc("/credits/cancel", ledger.CancelReservation)
	mux.HandleFunc("/credits/balance/", ledger.GetBalance)
	mux.HandleFunc("/credits/history", ledger.GetHistory)
	mux.HandleFunc("/credits/threshold", ledger.SetAlertThreshold)
	mux.HandleFunc("/credits/report", ledger.GetUsageReport)
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(200)
		w.Write([]byte(`{"status":"healthy","service":"credits"}`))
	})
	// Prometheus metrics
	mux.Handle("/metrics", reg)

	// Wrap with auth (optional) and HTTP metrics middleware
	handler := withAuth(os.Getenv("CREDITS_API_KEY"), httpMetrics.Middleware(mux))

	// Hardened HTTP server config
	srv := &http.Server{
		Addr:              ":" + port,
		Handler:           handler,
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       15 * time.Second,
		WriteTimeout:      15 * time.Second,
		IdleTimeout:       60 * time.Second,
		MaxHeaderBytes:    1 << 20,
	}

	log.Printf("[credits] service starting on %s", srv.Addr)
	// In production, fronted by TLS terminator (HAProxy/Cloudflare) to enforce TLS1.3
	log.Fatal(srv.ListenAndServe())
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

// Credits-specific metrics
type CreditsMetrics struct {
	Ops *metrics.LabeledCounter // labels: op, result
}

func initCreditsMetrics(reg *metrics.Registry) *CreditsMetrics {
	m := &CreditsMetrics{
		Ops: metrics.NewLabeledCounter("credits_operations_total", "Credits operations by type and result", []string{"op", "result"}),
	}
	if reg != nil {
		reg.RegisterLabeledCounter(m.Ops)
	}
	return m
}

// withAuth enforces a simple bearer token check if apiKey is set. Health and metrics are always public.
func withAuth(apiKey string, next http.Handler) http.Handler {
	if apiKey == "" { // auth disabled
		return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Public endpoints
		if r.URL.Path == "/health" || r.URL.Path == "/metrics" {
			next.ServeHTTP(w, r)
			return
		}
		auth := r.Header.Get("Authorization")
		const prefix = "Bearer "
		if len(auth) <= len(prefix) || auth[:len(prefix)] != prefix || auth[len(prefix):] != apiKey {
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}
		next.ServeHTTP(w, r)
	})
}
