package main

import (
	"log"
	"net/http"
	"os"
)

func main() {
	dbURL := getEnv("DATABASE_URL", "postgres://credits_user:credits_pass2024@localhost:5432/credits")
	port := getEnv("PORT", "5004")

	ledger, err := NewCreditLedger(dbURL)
	if err != nil {
		log.Fatalf("Failed to initialize ledger: %v", err)
	}
	defer ledger.Close()

	go ledger.cleanupExpiredKeys()

	http.HandleFunc("/credits/purchase", ledger.PurchaseCredits)
	http.HandleFunc("/credits/consume", ledger.ConsumeCredits)
	http.HandleFunc("/credits/balance/", ledger.GetBalance)
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(200)
		w.Write([]byte(`{"status":"healthy","service":"credits"}`))
	})

	log.Printf("Credits service starting on port %s", port)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}