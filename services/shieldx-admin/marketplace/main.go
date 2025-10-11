package main

import (
	"encoding/json"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"

	"shieldx/shared/shieldx-common/pkg/marketplace"
)

type Server struct {
	registry *marketplace.Registry
	bounty   *marketplace.BountyManager
	port     string
}

func main() {
	port := getEnv("MARKETPLACE_PORT", "8088")
	authorPct, _ := strconv.ParseFloat(getEnv("AUTHOR_REVENUE_PCT", "0.7"), 64)
	platformPct, _ := strconv.ParseFloat(getEnv("PLATFORM_REVENUE_PCT", "0.3"), 64)

	registry := marketplace.NewRegistry(authorPct, platformPct)
	bountyMgr := marketplace.NewBountyManager()

	server := &Server{
		registry: registry,
		bounty:   bountyMgr,
		port:     port,
	}

	// Registry endpoints
	http.HandleFunc("/packages", server.handlePackages)
	http.HandleFunc("/packages/publish", server.handlePublish)
	http.HandleFunc("/packages/search", server.handleSearch)
	http.HandleFunc("/packages/purchase", server.handlePurchase)

	// Bounty endpoints
	http.HandleFunc("/bounties", server.handleBounties)
	http.HandleFunc("/bounties/create", server.handleCreateBounty)
	http.HandleFunc("/bounties/submit", server.handleSubmitSolution)
	http.HandleFunc("/decoy-jam", server.handleDecoyJam)

	http.HandleFunc("/health", server.handleHealth)

	log.Printf("Marketplace starting on port %s", port)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}

func (s *Server) handlePackages(w http.ResponseWriter, r *http.Request) {
    if r.Method == http.MethodGet {
        packages := s.registry.GetTopPackages(20)
        if packages == nil {
            // Ensure an empty slice of pointers is returned instead of nil for JSON encoding
            packages = []*marketplace.Package{}
        }
        w.Header().Set("Content-Type", "application/json")
        json.NewEncoder(w).Encode(packages)
        return
    }

    http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
}

func (s *Server) handlePublish(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var pkg marketplace.Package
	if err := json.NewDecoder(r.Body).Decode(&pkg); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	if err := s.registry.PublishPackage(&pkg); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status":     "published",
		"package_id": pkg.ID,
	})
}

func (s *Server) handleSearch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	query := r.URL.Query().Get("q")
	tagsParam := r.URL.Query().Get("tags")

	var tags []string
	if tagsParam != "" {
		tags = strings.Split(tagsParam, ",")
	}

	results := s.registry.SearchPackages(query, tags)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(results)
}

func (s *Server) handlePurchase(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		PackageID string `json:"package_id"`
		Buyer     string `json:"buyer"`
		TxHash    string `json:"tx_hash"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	purchase, err := s.registry.PurchasePackage(req.PackageID, req.Buyer, req.TxHash)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(purchase)
}

func (s *Server) handleBounties(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	bountyType := marketplace.BountyType(r.URL.Query().Get("type"))
	status := marketplace.BountyStatus(r.URL.Query().Get("status"))

	bounties := s.bounty.ListBounties(bountyType, status)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(bounties)
}

func (s *Server) handleCreateBounty(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var bounty marketplace.Bounty
	if err := json.NewDecoder(r.Body).Decode(&bounty); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	if err := s.bounty.CreateBounty(&bounty); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status":    "created",
		"bounty_id": bounty.ID,
	})
}

func (s *Server) handleSubmitSolution(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var submission marketplace.Submission
	if err := json.NewDecoder(r.Body).Decode(&submission); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	if err := s.bounty.SubmitSolution(&submission); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status":        "submitted",
		"submission_id": submission.ID,
	})
}

func (s *Server) handleDecoyJam(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodPost {
		var entry marketplace.DecoyJamEntry
		if err := json.NewDecoder(r.Body).Decode(&entry); err != nil {
			http.Error(w, "Invalid JSON", http.StatusBadRequest)
			return
		}

		if err := s.bounty.SubmitDecoyJam(&entry); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{
			"status":   "submitted",
			"entry_id": entry.ID,
		})
		return
	}

	if r.Method == http.MethodGet {
		leaderboard := s.bounty.GetDecoyJamLeaderboard(10)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(leaderboard)
		return
	}

	http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":  "healthy",
		"service": "marketplace",
	})
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}
