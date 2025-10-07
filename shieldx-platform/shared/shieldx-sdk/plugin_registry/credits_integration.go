package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"

	"shieldx/core"
)

type PluginRegistryWithCredits struct {
	*PluginRegistry
	creditsClient *core.CreditsClient
}

func NewPluginRegistryWithCredits(dbURL, creditsURL string) (*PluginRegistryWithCredits, error) {
	registry, err := NewPluginRegistry(dbURL)
	if err != nil {
		return nil, err
	}

	creditsClient := core.NewCreditsClient(creditsURL)

	return &PluginRegistryWithCredits{
		PluginRegistry: registry,
		creditsClient:  creditsClient,
	}, nil
}

func (pr *PluginRegistryWithCredits) runPluginWithCredits(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req RunRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	if req.PluginID == "" || req.ArtifactID == "" || req.TenantID == "" {
		http.Error(w, "Missing required fields", http.StatusBadRequest)
		return
	}

	creditsRequired := int64(10)

	hasCredits, err := pr.creditsClient.HasSufficientCredits(req.TenantID, creditsRequired)
	if err != nil {
		log.Printf("Failed to check credits: %v", err)
		http.Error(w, "Failed to verify credits", http.StatusInternalServerError)
		return
	}

	if !hasCredits {
		response := map[string]interface{}{
			"success": false,
			"error":   "Insufficient credits",
			"message": fmt.Sprintf("Plugin execution requires %d credits", creditsRequired),
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusPaymentRequired)
		json.NewEncoder(w).Encode(response)
		return
	}

	consumeReq := core.ConsumeRequest{
		TenantID:    req.TenantID,
		Amount:      creditsRequired,
		Description: fmt.Sprintf("Plugin execution: %s", req.PluginID),
		Reference:   fmt.Sprintf("plugin_run_%s_%s", req.PluginID, req.ArtifactID),
	}

	if err := pr.creditsClient.ConsumeCredits(consumeReq); err != nil {
		log.Printf("Failed to consume credits: %v", err)
		http.Error(w, "Failed to consume credits", http.StatusInternalServerError)
		return
	}

	plugin, err := pr.validator.GetPlugin(req.PluginID)
	if err != nil {
		pr.refundCredits(req.TenantID, creditsRequired, "Plugin not found")
		log.Printf("Failed to get plugin: %v", err)
		http.Error(w, "Failed to get plugin", http.StatusInternalServerError)
		return
	}

	if plugin == nil {
		pr.refundCredits(req.TenantID, creditsRequired, "Plugin not found")
		http.Error(w, "Plugin not found", http.StatusNotFound)
		return
	}

	if !plugin.Verified {
		pr.refundCredits(req.TenantID, creditsRequired, "Plugin not verified")
		http.Error(w, "Plugin not verified", http.StatusForbidden)
		return
	}

	mockWasmData := []byte("\x00asm\x01\x00\x00\x00")
	
	input := PluginInput{
		ArtifactID:   req.ArtifactID,
		ArtifactType: "unknown",
		Metadata:     req.Metadata,
		S3URL:        fmt.Sprintf("s3://artifacts/%s", req.ArtifactID),
		Timestamp:    time.Now(),
	}

	var policy SandboxPolicy
	if err := json.Unmarshal([]byte(plugin.SandboxPolicy), &policy); err != nil {
		pr.refundCredits(req.TenantID, creditsRequired, "Invalid sandbox policy")
		log.Printf("Failed to parse sandbox policy: %v", err)
		http.Error(w, "Invalid sandbox policy", http.StatusInternalServerError)
		return
	}

	output, err := pr.runner.ExecutePlugin(mockWasmData, input, policy)
	if err != nil {
		pr.refundCredits(req.TenantID, creditsRequired, "Plugin execution failed")
		log.Printf("Plugin execution failed: %v", err)
		http.Error(w, fmt.Sprintf("Plugin execution failed: %v", err), http.StatusInternalServerError)
		return
	}

	balance, _ := pr.creditsClient.GetBalance(req.TenantID)
	
	response := map[string]interface{}{
		"success":           output.Success,
		"plugin_id":         req.PluginID,
		"artifact_id":       req.ArtifactID,
		"results":           output.Results,
		"confidence":        output.Confidence,
		"tags":              output.Tags,
		"indicators":        output.Indicators,
		"execution_time":    output.ExecutionTime,
		"credits_consumed":  creditsRequired,
		"remaining_balance": balance,
		"executed_at":       time.Now().Format(time.RFC3339),
	}

	if output.Error != "" {
		response["error"] = output.Error
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(response); err != nil {
		log.Printf("Failed to encode response: %v", err)
	}
}

func (pr *PluginRegistryWithCredits) refundCredits(tenantID string, amount int64, reason string) {
	log.Printf("Refunding %d credits to tenant %s: %s", amount, tenantID, reason)
}

func (pr *PluginRegistryWithCredits) getCreditsPricing(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	pricing := map[string]interface{}{
		"plugin_execution": map[string]interface{}{
			"credits": 10,
			"description": "Execute a plugin on an artifact",
		},
		"digital_twin_simulation": map[string]interface{}{
			"credits": 50,
			"description": "Run a digital twin simulation",
		},
		"ml_model_training": map[string]interface{}{
			"credits": 100,
			"description": "Train an ML model",
		},
		"credit_value": map[string]interface{}{
			"usd_per_credit": 0.10,
			"minimum_purchase": 100,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(pricing)
}

func (pr *PluginRegistryWithCredits) getCreditsUsage(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	tenantID := r.URL.Query().Get("tenant_id")
	if tenantID == "" {
		http.Error(w, "Missing tenant_id parameter", http.StatusBadRequest)
		return
	}

	balance, err := pr.creditsClient.GetBalance(tenantID)
	if err != nil {
		log.Printf("Failed to get balance: %v", err)
		http.Error(w, "Failed to get balance", http.StatusInternalServerError)
		return
	}

	usage := map[string]interface{}{
		"tenant_id":       tenantID,
		"current_balance": balance,
		"last_updated":    time.Now().Format(time.RFC3339),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(usage)
}