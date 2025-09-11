package main

import (
	"encoding/json"
	"log"
	"net/http"
	"os"
	"time"
)

type WebAPI struct {
	shadowURL string
}

type AttackData struct {
	Attacks []Attack `json:"attacks"`
	Stats   Stats    `json:"stats"`
}

type Attack struct {
	ID            string      `json:"id"`
	SourceIP      string      `json:"source_ip"`
	DestIP        string      `json:"dest_ip"`
	SourceCountry string      `json:"source_country"`
	AttackType    string      `json:"attack_type"`
	Severity      string      `json:"severity"`
	Timestamp     time.Time   `json:"timestamp"`
	Blocked       bool        `json:"blocked"`
	Coordinates   Coordinates `json:"coordinates"`
}

type Coordinates struct {
	Source []float64 `json:"source"`
	Dest   []float64 `json:"dest"`
}

type Stats struct {
	TotalAttacks   int           `json:"total_attacks"`
	BlockedAttacks int           `json:"blocked_attacks"`
	ActiveThreats  int           `json:"active_threats"`
	TopCountries   []CountryData `json:"top_countries"`
}

type CountryData struct {
	Country string `json:"country"`
	Count   int    `json:"count"`
}

func NewWebAPI(shadowURL string) *WebAPI {
	return &WebAPI{shadowURL: shadowURL}
}

func (api *WebAPI) GetLiveAttacks(w http.ResponseWriter, r *http.Request) {
	attacks := []Attack{
		{
			ID:            "attack_001",
			SourceIP:      "185.220.101.182",
			DestIP:        "10.0.1.100",
			SourceCountry: "Russia",
			AttackType:    "Brute Force",
			Severity:      "High",
			Timestamp:     time.Now().Add(-time.Minute * 2),
			Blocked:       true,
			Coordinates: Coordinates{
				Source: []float64{55.7558, 37.6176},
				Dest:   []float64{39.7392, -104.9903},
			},
		},
		{
			ID:            "attack_002",
			SourceIP:      "103.224.182.251",
			DestIP:        "10.0.1.101",
			SourceCountry: "China",
			AttackType:    "SQL Injection",
			Severity:      "Critical",
			Timestamp:     time.Now().Add(-time.Minute * 1),
			Blocked:       false,
			Coordinates: Coordinates{
				Source: []float64{39.9042, 116.4074},
				Dest:   []float64{37.7749, -122.4194},
			},
		},
	}

	stats := Stats{
		TotalAttacks:   1247,
		BlockedAttacks: 1089,
		ActiveThreats:  23,
		TopCountries: []CountryData{
			{Country: "China", Count: 342},
			{Country: "Russia", Count: 298},
			{Country: "North Korea", Count: 156},
			{Country: "Iran", Count: 134},
		},
	}

	response := AttackData{Attacks: attacks, Stats: stats}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (api *WebAPI) ProxyShadowEval(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodPost {
		response := map[string]interface{}{
			"success": true,
			"eval_id": "eval_" + time.Now().Format("20060102150405"),
			"message": "Shadow evaluation started",
			"status":  "pending",
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
		return
	}

	evaluations := []map[string]interface{}{
		{
			"eval_id":             "eval_001",
			"rule_name":           "IP Blacklist Test",
			"rule_type":           "ip_blacklist",
			"status":              "completed",
			"sample_size":         1000,
			"true_positives":      85,
			"false_positives":     12,
			"true_negatives":      890,
			"false_negatives":     13,
			"precision":           0.876,
			"recall":              0.867,
			"f1_score":            0.871,
			"estimated_fp_rate":   0.012,
			"estimated_tp_rate":   0.085,
			"recommendations":     []string{"Rule performance looks good - ready for production"},
			"execution_time_ms":   2340,
			"created_at":          time.Now().Add(-time.Hour * 2).Format(time.RFC3339),
		},
	}

	response := map[string]interface{}{"evaluations": evaluations}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (api *WebAPI) ProxyShadowResult(w http.ResponseWriter, r *http.Request) {
	evalID := r.URL.Query().Get("eval_id")
	
	result := map[string]interface{}{
		"eval_id":             evalID,
		"status":              "completed",
		"sample_size":         1000,
		"true_positives":      85,
		"false_positives":     12,
		"true_negatives":      890,
		"false_negatives":     13,
		"precision":           0.876,
		"recall":              0.867,
		"f1_score":            0.871,
		"estimated_fp_rate":   0.012,
		"estimated_tp_rate":   0.085,
		"recommendations":     []string{"Rule performance looks good - ready for production"},
		"execution_time_ms":   2340,
		"created_at":          time.Now().Add(-time.Hour * 2).Format(time.RFC3339),
		"completed_at":        time.Now().Add(-time.Hour * 1).Format(time.RFC3339),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

func main() {
	shadowURL := getEnv("SHADOW_URL", "http://localhost:5005")
	port := getEnv("PORT", "5006")

	api := NewWebAPI(shadowURL)

	http.HandleFunc("/api/attacks/live", api.GetLiveAttacks)
	http.HandleFunc("/api/shadow/eval", api.ProxyShadowEval)
	http.HandleFunc("/api/shadow/result", api.ProxyShadowResult)
	http.HandleFunc("/api/shadow/evaluations", api.ProxyShadowEval)
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(200)
		w.Write([]byte(`{"status":"healthy","service":"webapi"}`))
	})

	log.Printf("Web API server starting on port %s", port)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}