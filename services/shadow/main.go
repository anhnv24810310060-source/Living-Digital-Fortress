package main

import (
	"log"
	"net/http"
	"os"
	"time"

	"shieldx/shared/metrics"
)

func main() {
	dbURL := getEnv("DATABASE_URL", "postgres://shadow_user:shadow_pass2024@localhost:5432/shadow")
	port := getEnv("PORT", "5005")
	serviceName := "shadow"

	evaluator, err := NewShadowEvaluator(dbURL)
	if err != nil {
		log.Fatalf("Failed to initialize shadow evaluator: %v", err)
	}
	defer evaluator.Close()

	reg := metrics.NewRegistry()
	httpMetrics := metrics.NewHTTPMetrics(reg, serviceName)

	// API per spec
	mux := http.NewServeMux()
	mux.HandleFunc("/shadow/evaluate", evaluator.CreateShadowEval)
	mux.HandleFunc("/shadow/results/", evaluator.GetShadowEvalByPath)
	mux.HandleFunc("/shadow/deploy", evaluator.DeployRule)
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(200)
		w.Write([]byte(`{"status":"healthy","service":"shadow"}`))
	})
	mux.Handle("/metrics", reg)

	handler := withAuth(os.Getenv("SHADOW_API_KEY"), httpMetrics.Middleware(mux))

	srv := &http.Server{
		Addr:              ":" + port,
		Handler:           handler,
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       15 * time.Second,
		WriteTimeout:      15 * time.Second,
		IdleTimeout:       60 * time.Second,
		MaxHeaderBytes:    1 << 20,
	}

	log.Printf("Shadow evaluation service starting on port %s", port)
	log.Fatal(srv.ListenAndServe())
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

// withAuth protects endpoints with a Bearer token if SHADOW_API_KEY is set; health/metrics remain public
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
