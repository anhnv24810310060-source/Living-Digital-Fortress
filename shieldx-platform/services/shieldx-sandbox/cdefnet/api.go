package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"shieldx/services/cdefnet/store"
	"shieldx/services/cdefnet/privacy"
	"shieldx/pkg/ledger"
)

type SubmitIOCRequest struct {
	TenantID   string  `json:"tenant_id"`
	IOCType    string  `json:"ioc_type"`
	Value      string  `json:"value"`
	Confidence float64 `json:"confidence"`
	TTL        int     `json:"ttl"`
	Signature  string  `json:"signature"`
}

type SubmitIOCResponse struct {
	Success         bool   `json:"success"`
	IOCHash         string `json:"ioc_hash,omitempty"`
	Message         string `json:"message"`
	AggregatedCount int    `json:"aggregated_count,omitempty"`
}

type QueryIOCRequest struct {
	IOCType string `json:"ioc_type"`
	Value   string `json:"value"`
}

type QueryIOCResponse struct {
	Found       bool    `json:"found"`
	Confidence  float64 `json:"confidence,omitempty"`
	FirstSeen   string  `json:"first_seen,omitempty"`
	ThreatLevel string  `json:"threat_level,omitempty"`
}

type APIServer struct {
	store       *store.Store
	anonymizer  *privacy.Anonymizer
	rateLimiter map[string]*RateLimiter
}

type RateLimiter struct {
	requests    int
	windowStart time.Time
	limit       int
	window      time.Duration
}

func NewAPIServer(store *store.Store) *APIServer {
	return &APIServer{
		store:       store,
		anonymizer:  privacy.NewAnonymizer(),
		rateLimiter: make(map[string]*RateLimiter),
	}
}

func (s *APIServer) rateLimit(clientIP string) bool {
	now := time.Now()
	limiter, exists := s.rateLimiter[clientIP]

	if !exists || now.Sub(limiter.windowStart) > limiter.window {
		s.rateLimiter[clientIP] = &RateLimiter{
			requests:    1,
			windowStart: now,
			limit:       100, // 100 requests per minute
			window:      time.Minute,
		}
		return true
	}

	if limiter.requests >= limiter.limit {
		return false
	}

	limiter.requests++
	return true
}

func (s *APIServer) authenticate(r *http.Request) (string, error) {
	auth := r.Header.Get("Authorization")
	if !strings.HasPrefix(auth, "Bearer ") {
		return "", fmt.Errorf("missing or invalid authorization header")
	}

	token := strings.TrimPrefix(auth, "Bearer ")
	if len(token) < 32 {
		return "", fmt.Errorf("invalid token format")
	}

	// Extract tenant from token (simplified for demo)
	return "tenant_" + token[:8], nil
}

func (s *APIServer) submitIOCHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Rate limiting
	clientIP := strings.Split(r.RemoteAddr, ":")[0]
	if !s.rateLimit(clientIP) {
		http.Error(w, "Rate limit exceeded", http.StatusTooManyRequests)
		return
	}

	// Authentication
	tenantID, err := s.authenticate(r)
	if err != nil {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}

	var req SubmitIOCRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	// Validation
	if !s.anonymizer.IsValidIOC(req.IOCType, req.Value) {
		http.Error(w, "Invalid IOC format", http.StatusBadRequest)
		return
	}

	if req.Confidence < 0 || req.Confidence > 1 {
		http.Error(w, "Confidence must be between 0 and 1", http.StatusBadRequest)
		return
	}

	if req.TTL <= 0 || req.TTL > 86400*30 { // Max 30 days
		http.Error(w, "TTL must be between 1 and 2592000 seconds", http.StatusBadRequest)
		return
	}

	// Scrub PII and hash
	cleanValue := s.anonymizer.ScrubPII(req.Value)
	valueHash := store.HashValue(cleanValue)
	tenantHash := store.HashTenant(tenantID)

	// Store IOC
	ioc := &store.IOC{
		TenantIDHash: tenantHash,
		IOCType:      strings.ToLower(req.IOCType),
		ValueHash:    valueHash,
		Confidence:   req.Confidence,
		TTL:          req.TTL,
	}

	if err := s.store.SubmitIOC(ioc); err != nil {
		log.Printf("Failed to store IOC: %v", err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}

	// Audit log
	_ = ledger.AppendJSONLine("data/ledger-cdefnet.log", "cdefnet", "ioc_submit", map[string]any{
		"tenant_hash": tenantHash,
		"ioc_type":    req.IOCType,
		"value_hash":  valueHash,
		"confidence":  req.Confidence,
		"ttl":         req.TTL,
	})

	resp := SubmitIOCResponse{
		Success:         true,
		IOCHash:         valueHash,
		Message:         "IOC stored successfully",
		AggregatedCount: ioc.AggregatedCount,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *APIServer) queryIOCHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	iocType := r.URL.Query().Get("type")
	value := r.URL.Query().Get("value")

	if iocType == "" || value == "" {
		http.Error(w, "Missing type or value parameter", http.StatusBadRequest)
		return
	}

	// Rate limiting
	clientIP := strings.Split(r.RemoteAddr, ":")[0]
	if !s.rateLimit(clientIP) {
		http.Error(w, "Rate limit exceeded", http.StatusTooManyRequests)
		return
	}

	// Scrub and hash
	cleanValue := s.anonymizer.ScrubPII(value)
	valueHash := store.HashValue(cleanValue)

	// Query IOC
	ioc, err := s.store.QueryIOC(valueHash, strings.ToLower(iocType))
	if err != nil {
		log.Printf("Failed to query IOC: %v", err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}

	resp := QueryIOCResponse{
		Found: ioc != nil,
	}

	if ioc != nil {
		// Add differential privacy noise to confidence
		noisyConfidence := s.anonymizer.AddDifferentialPrivacyNoise(ioc.Confidence, 0.1)
		if noisyConfidence < 0 {
			noisyConfidence = 0
		}
		if noisyConfidence > 1 {
			noisyConfidence = 1
		}

		resp.Confidence = noisyConfidence
		resp.FirstSeen = ioc.FirstSeen.Format(time.RFC3339)

		// Threat level based on confidence
		switch {
		case noisyConfidence >= 0.8:
			resp.ThreatLevel = "high"
		case noisyConfidence >= 0.5:
			resp.ThreatLevel = "medium"
		default:
			resp.ThreatLevel = "low"
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *APIServer) feedHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Rate limiting
	clientIP := strings.Split(r.RemoteAddr, ":")[0]
	if !s.rateLimit(clientIP) {
		http.Error(w, "Rate limit exceeded", http.StatusTooManyRequests)
		return
	}

	// Placeholder for feed implementation
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status":  "not_implemented",
		"message": "Feed endpoint will be implemented in Week 2",
	})
}