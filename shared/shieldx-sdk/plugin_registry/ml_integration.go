package main

import (
	"log"
	"time"

	"shieldx/core"
	"shieldx/sandbox/runner"
)

// Update PluginRegistry to include ML client
func (pr *PluginRegistry) initMLClient() {
	mlURL := getEnv("ML_FEATURE_STORE_URL", "http://localhost:5000")
	pr.mlClient = core.NewMLClient(mlURL)
	
	log.Printf("ML client initialized with feature store: %s", mlURL)
}

// Enhanced runPlugin with ML integration
func (pr *PluginRegistry) runPluginWithML(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req RunRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	if req.PluginID == "" || req.ArtifactID == "" {
		http.Error(w, "Missing required fields", http.StatusBadRequest)
		return
	}

	// Get plugin from database
	plugin, err := pr.validator.GetPlugin(req.PluginID)
	if err != nil {
		log.Printf("Failed to get plugin: %v", err)
		http.Error(w, "Failed to get plugin", http.StatusInternalServerError)
		return
	}

	if plugin == nil {
		http.Error(w, "Plugin not found", http.StatusNotFound)
		return
	}

	if !plugin.Verified {
		http.Error(w, "Plugin not verified", http.StatusForbidden)
		return
	}

	// Parse sandbox policy
	var policy runner.SandboxPolicy
	if err := json.Unmarshal([]byte(plugin.SandboxPolicy), &policy); err != nil {
		log.Printf("Failed to parse sandbox policy: %v", err)
		http.Error(w, "Invalid sandbox policy", http.StatusInternalServerError)
		return
	}

	// Create plugin input
	input := runner.PluginInput{
		ArtifactID:   req.ArtifactID,
		ArtifactType: determineArtifactType(req.Metadata),
		Metadata:     req.Metadata,
		S3URL:        fmt.Sprintf("s3://artifacts/%s", req.ArtifactID),
		Timestamp:    time.Now(),
	}

	// Execute plugin
	mockWasmData := []byte("\x00asm\x01\x00\x00\x00") // In production, load from secure storage
	output, err := pr.runner.ExecutePlugin(mockWasmData, input, policy)
	if err != nil {
		log.Printf("Plugin execution failed: %v", err)
		http.Error(w, fmt.Sprintf("Plugin execution failed: %v", err), http.StatusInternalServerError)
		return
	}

	// Convert to ML format and send to ML pipeline
	mlOutput, err := core.ConvertToMLOutput(req.PluginID, req.ArtifactID, map[string]interface{}{
		"success":        output.Success,
		"results":        output.Results,
		"confidence":     output.Confidence,
		"tags":           output.Tags,
		"indicators":     convertIndicatorsToML(output.Indicators),
		"execution_time": output.ExecutionTime,
	})
	
	if err != nil {
		log.Printf("Failed to convert to ML format: %v", err)
	} else {
		// Send to ML pipeline asynchronously
		go func() {
			if err := pr.mlClient.SendPluginOutput(mlOutput); err != nil {
				log.Printf("Failed to send to ML pipeline: %v", err)
			} else {
				log.Printf("Plugin output sent to ML pipeline for artifact %s", req.ArtifactID)
			}
		}()
	}

	// Prepare response
	response := map[string]interface{}{
		"success":        output.Success,
		"plugin_id":      req.PluginID,
		"artifact_id":    req.ArtifactID,
		"results":        output.Results,
		"confidence":     output.Confidence,
		"tags":           output.Tags,
		"indicators":     output.Indicators,
		"execution_time": output.ExecutionTime,
		"executed_at":    time.Now().Format(time.RFC3339),
		"ml_processed":   err == nil, // Indicate if ML processing was successful
	}

	if output.Error != "" {
		response["error"] = output.Error
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(response); err != nil {
		log.Printf("Failed to encode response: %v", err)
	}
}

// Batch processing endpoint for ML training
func (pr *PluginRegistry) batchProcessForML(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var batchReq struct {
		PluginID    string   `json:"plugin_id"`
		ArtifactIDs []string `json:"artifact_ids"`
	}

	if err := json.NewDecoder(r.Body).Decode(&batchReq); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	if batchReq.PluginID == "" || len(batchReq.ArtifactIDs) == 0 {
		http.Error(w, "Missing required fields", http.StatusBadRequest)
		return
	}

	// Process artifacts in batch
	var mlOutputs []core.PluginOutputML
	successCount := 0

	for _, artifactID := range batchReq.ArtifactIDs {
		// Simulate plugin execution for batch processing
		mockOutput := &runner.PluginOutput{
			Success:       true,
			Results:       map[string]interface{}{"batch_processed": true},
			Confidence:    0.7,
			Tags:          []string{"batch", "processed"},
			Indicators:    []runner.Indicator{},
			ExecutionTime: 100,
		}

		mlOutput, err := core.ConvertToMLOutput(batchReq.PluginID, artifactID, map[string]interface{}{
			"success":        mockOutput.Success,
			"results":        mockOutput.Results,
			"confidence":     mockOutput.Confidence,
			"tags":           mockOutput.Tags,
			"indicators":     []interface{}{},
			"execution_time": mockOutput.ExecutionTime,
		})

		if err != nil {
			log.Printf("Failed to convert artifact %s: %v", artifactID, err)
			continue
		}

		mlOutputs = append(mlOutputs, mlOutput)
		successCount++
	}

	// Send batch to ML pipeline
	go func() {
		if err := pr.mlClient.SendBatchOutputs(mlOutputs); err != nil {
			log.Printf("Batch ML processing failed: %v", err)
		}
	}()

	response := map[string]interface{}{
		"success":       true,
		"processed":     successCount,
		"total":         len(batchReq.ArtifactIDs),
		"plugin_id":     batchReq.PluginID,
		"processed_at":  time.Now().Format(time.RFC3339),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// ML metrics endpoint
func (pr *PluginRegistry) getMLMetrics(w http.ResponseWriter, r *http.Request) {
	metrics := pr.mlClient.GetMetrics()
	
	response := map[string]interface{}{
		"ml_metrics": metrics,
		"timestamp": time.Now().Format(time.RFC3339),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// Helper functions
func convertIndicatorsToML(indicators []runner.Indicator) []interface{} {
	result := make([]interface{}, len(indicators))
	for i, ind := range indicators {
		result[i] = map[string]interface{}{
			"type":       ind.Type,
			"value":      ind.Value,
			"confidence": ind.Confidence,
			"context":    ind.Context,
		}
	}
	return result
}

func determineArtifactType(metadata map[string]interface{}) string {
	if fileType, ok := metadata["file_type"].(string); ok {
		return fileType
	}
	if extension, ok := metadata["extension"].(string); ok {
		switch extension {
		case ".exe", ".dll", ".so":
			return "executable"
		case ".pdf", ".doc", ".docx":
			return "document"
		case ".zip", ".rar", ".tar":
			return "archive"
		default:
			return "unknown"
		}
	}
	return "unknown"
}