package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"shieldx/pkg/metrics"
	otelobs "shieldx/pkg/observability/otel"
)

func main() {
	dbURL := getEnv("DATABASE_URL", "postgres://contauth_user:contauth_pass2024@localhost:5432/contauth")
	port := getEnv("PORT", "5002")

	var collector interface{ Close() error; CollectTelemetry(http.ResponseWriter, *http.Request); CalculateRiskScore(http.ResponseWriter, *http.Request); GetAuthDecision(http.ResponseWriter, *http.Request) }
	if os.Getenv("DISABLE_DB") == "true" {
		log.Printf("DISABLE_DB=true detected; using DummyCollector (no database)")
		collector = NewDummyCollector()
	} else {
		c, err := NewContAuthCollector(dbURL)
		if err != nil {
			log.Fatalf("Failed to initialize collector: %v", err)
		}
		collector = c
	}
	defer collector.Close()

	// Setup routes with metrics registry
	mux := http.NewServeMux()
	reg := metrics.NewRegistry()
	httpMetrics := metrics.NewHTTPMetrics(reg, "contauth")
	mux.HandleFunc("/contauth/telemetry", collector.CollectTelemetry)
	mux.HandleFunc("/contauth/score", collector.CalculateRiskScore)
	mux.HandleFunc("/contauth/decision", collector.GetAuthDecision)
	mux.Handle("/metrics", reg)
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(200)
		w.Write([]byte(`{"status":"healthy","service":"contauth"}`))
	})

	// OpenTelemetry tracing (no-op unless built with otelotlp and endpoint set)
	shutdown := otelobs.InitTracer("contauth")
	defer shutdown(context.Background())

	// Wrap with tracing + metrics middleware
	h := httpMetrics.Middleware(mux)
	h = otelobs.WrapHTTPHandler("contauth", h)

	log.Printf("Continuous Auth service starting on port %s", port)
	log.Fatal(http.ListenAndServe(":"+port, h))
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}