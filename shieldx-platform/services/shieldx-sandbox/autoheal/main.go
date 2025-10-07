package main

import (
	"log"
	"net/http"
	"os"

	"shieldx/core/autoheal"
)

func main() {
	port := getEnv("PORT", "8080")

	// Initialize mesh controller
	controller := autoheal.NewMeshController()

	// Setup routes
	http.HandleFunc("/autoheal/incident", controller.HandleIncident)
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(200)
		w.Write([]byte(`{"status":"healthy","service":"autoheal"}`))
	})

	log.Printf("Autoheal service starting on port %s", port)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}