package main

import (
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"shieldx/services/cdefnet/store"
	"shieldx/shared/shieldx-common/pkg/metrics"
)

func main() {
	// Configuration
	port := getEnv("CDEFNET_PORT", "8090")
	dbURL := getEnv("DATABASE_URL", "postgres://user:pass@localhost/cdefnet?sslmode=disable")

	// Initialize store
	st, err := store.NewStore(dbURL)
	if err != nil {
		log.Fatalf("Failed to initialize store: %v", err)
	}
	defer st.Close()

	// Initialize API server
	apiServer := NewAPIServer(st)

	// Setup metrics
	reg := metrics.NewRegistry()
	mSubmit := metrics.NewCounter("cdefnet_ioc_submit_total", "Total IOC submissions")
	mQuery := metrics.NewCounter("cdefnet_ioc_query_total", "Total IOC queries")
	mErrors := metrics.NewCounter("cdefnet_errors_total", "Total errors")

	reg.Register(mSubmit)
	reg.Register(mQuery)
	reg.Register(mErrors)

	// Setup HTTP routes
	mux := http.NewServeMux()

	// API endpoints
	mux.HandleFunc("/v1/submit-ioc", func(w http.ResponseWriter, r *http.Request) {
		mSubmit.Inc()
		apiServer.submitIOCHandler(w, r)
	})

	mux.HandleFunc("/v1/query-ioc", func(w http.ResponseWriter, r *http.Request) {
		mQuery.Inc()
		apiServer.queryIOCHandler(w, r)
	})

	mux.HandleFunc("/v1/feed", func(w http.ResponseWriter, r *http.Request) {
		apiServer.feedHandler(w, r)
	})

	// Health and metrics
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		w.Write([]byte(`{"status":"healthy","service":"cdefnet","timestamp":"` + time.Now().Format(time.RFC3339) + `"}`))
	})

	mux.Handle("/metrics", reg)

	// Graceful shutdown
	server := &http.Server{
		Addr:         ":" + port,
		Handler:      mux,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 15 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	// Start server in goroutine
	go func() {
		log.Printf("[cdefnet] Starting server on port %s", port)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Server failed to start: %v", err)
		}
	}()

	// Background cleanup task
	go func() {
		ticker := time.NewTicker(1 * time.Hour)
		defer ticker.Stop()

		for range ticker.C {
			if err := st.CleanupExpired(); err != nil {
				log.Printf("Cleanup failed: %v", err)
				mErrors.Inc()
			} else {
				log.Println("Expired IOCs cleaned up")
			}
		}
	}()

	// Wait for interrupt signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("[cdefnet] Shutting down server...")
	server.Close()
	log.Println("[cdefnet] Server stopped")
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}
