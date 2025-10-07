// Phase 1 Enhanced Handlers for Orchestrator
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"shieldx/pkg/metrics"
)

// Phase 1 global instance and metrics (only PQC specific metrics retained here to avoid duplication)
var (
	phase1         *Phase1Enhancement
	mPQCOperations = metrics.NewCounter("orchestrator_pqc_operations_total", "Total PQC operations")
)

// initPhase1 initializes Phase 1 enhancements
func initPhase1() error {
	config := LoadPhase1Config()
	var err error
	phase1, err = NewPhase1Enhancement(config)
	if err != nil {
		return fmt.Errorf("Phase 1 init: %w", err)
	}
	phase1.Start()
	if reg != nil { // register only PQC metric to avoid duplicate names
		reg.Register(mPQCOperations)
	}
	return nil
}

// NOTE: Route and health enhanced handlers are implemented in enhanced_handlers.go.
// This file keeps only Phase1-specific endpoints to avoid symbol redeclaration.

// handlePQCEndpoint provides PQC key exchange endpoint
func handlePQCEndpoint(w http.ResponseWriter, r *http.Request) {
	if phase1 == nil || !phase1.config.EnablePQC {
		http.Error(w, "PQC not enabled", http.StatusNotImplemented)
		return
	}

	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Perform key encapsulation
	ciphertext, sharedSecret, err := phase1.PQCEncapsulate()
	if err != nil {
		http.Error(w, "encapsulation failed", http.StatusInternalServerError)
		return
	}

	mPQCOperations.Inc()

	// In production, use sharedSecret for session key derivation
	// For now, just return ciphertext
	writeJSON(w, 200, map[string]any{
		"ciphertext":    fmt.Sprintf("%x", ciphertext[:64]),   // Truncated for display
		"shared_secret": fmt.Sprintf("%x", sharedSecret[:16]), // Truncated
		"algorithm":     phase1.config.PQCAlgorithm,
	})
}

// handleGraphQLEndpoint provides GraphQL with security validation
func handleGraphQLEndpoint(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var gqlReq struct {
		Query         string                 `json:"query"`
		Variables     map[string]interface{} `json:"variables"`
		OperationName string                 `json:"operationName"`
	}

	if err := json.NewDecoder(r.Body).Decode(&gqlReq); err != nil {
		http.Error(w, "bad request", 400)
		return
	}

	clientID := clientIP(r)

	// Phase 1 GraphQL security validation
	if phase1 != nil && phase1.config.EnableGraphQLSec {
		ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
		defer cancel()

		if err := phase1.ValidateGraphQLQuery(ctx, clientID, gqlReq.Query, gqlReq.Variables); err != nil {
			http.Error(w, fmt.Sprintf("GraphQL security: %s", err), http.StatusBadRequest)
			return
		}
	}

	// Forward to GraphQL backend (simplified - in production use proper GraphQL library)
	writeJSON(w, 200, map[string]any{
		"data": map[string]any{
			"message": "GraphQL query validated successfully",
		},
	})
}

// newSecurityMiddleware wraps handlers with security logging
// (Security helpers moved to enhanced_handlers.go to avoid duplication)
