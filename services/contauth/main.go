package main

import (
	"log"
	"net/http"
	"os"
)

func main() {
	dbURL := getEnv("DATABASE_URL", "postgres://contauth_user:contauth_pass2024@localhost:5432/contauth")
	port := getEnv("PORT", "5002")

	collector, err := NewContAuthCollector(dbURL)
	if err != nil {
		log.Fatalf("Failed to initialize collector: %v", err)
	}
	defer collector.Close()

	// Setup routes
	http.HandleFunc("/contauth/telemetry", collector.CollectTelemetry)
	http.HandleFunc("/contauth/score", collector.CalculateRiskScore)
	http.HandleFunc("/contauth/decision", collector.GetAuthDecision)
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(200)
		w.Write([]byte(`{"status":"healthy","service":"contauth"}`))
	})

	log.Printf("Continuous Auth service starting on port %s", port)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}