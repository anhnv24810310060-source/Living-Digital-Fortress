package main

import (
	"log"
	"net/http"
	"os"
)

func main() {
	dbURL := getEnv("DATABASE_URL", "postgres://shadow_user:shadow_pass2024@localhost:5432/shadow")
	port := getEnv("PORT", "5005")

	evaluator, err := NewShadowEvaluator(dbURL)
	if err != nil {
		log.Fatalf("Failed to initialize shadow evaluator: %v", err)
	}
	defer evaluator.Close()

	http.HandleFunc("/shadow/eval", evaluator.CreateShadowEval)
	http.HandleFunc("/shadow/result", evaluator.GetShadowEval)
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(200)
		w.Write([]byte(`{"status":"healthy","service":"shadow"}`))
	})

	log.Printf("Shadow evaluation service starting on port %s", port)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}