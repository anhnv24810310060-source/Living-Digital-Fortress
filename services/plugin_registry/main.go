package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"shieldx/core/fortress_bridge"
	"shieldx/sandbox/runner"
)

type PluginRegistry struct {
	validator *fortress_bridge.PluginValidator
	runner    *runner.WasmRunner
}

type RunRequest struct {
	PluginID   string                 `json:"plugin_id"`
	ArtifactID string                 `json:"artifact_id"`
	Metadata   map[string]interface{} `json:"metadata"`
}

func main() {
	dbURL := getEnv("DATABASE_URL", "postgres://user:pass@localhost/plugins?sslmode=disable")
	port := getEnv("PORT", "8082")
	
	validator, err := fortress_bridge.NewPluginValidator(dbURL)
	if err != nil {
		log.Fatalf("Failed to initialize validator: %v", err)
	}
	defer validator.Close()

	registry := &PluginRegistry{
		validator: validator,
		runner:    runner.NewWasmRunner(),
	}
	defer registry.runner.Close()

	http.HandleFunc("/plugins/publish", registry.publishPlugin)
	http.HandleFunc("/plugins/", registry.getPluginStatus)
	http.HandleFunc("/plugins/run", registry.runPlugin)
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(200)
		w.Write([]byte(`{"status":"healthy","service":"plugin-registry"}`))
	})

	log.Printf("Plugin Registry starting on port %s", port)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}

func (pr *PluginRegistry) publishPlugin(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	err := r.ParseMultipartForm(32 << 20)
	if err != nil {
		http.Error(w, "Failed to parse form", http.StatusBadRequest)
		return
	}

	wasmFile, wasmHeader, err := r.FormFile("wasm")
	if err != nil {
		http.Error(w, "Missing WASM file", http.StatusBadRequest)
		return
	}
	defer wasmFile.Close()

	if !strings.HasSuffix(wasmHeader.Filename, ".wasm") {
		http.Error(w, "File must be a .wasm file", http.StatusBadRequest)
		return
	}

	wasmData, err := io.ReadAll(wasmFile)
	if err != nil {
		http.Error(w, "Failed to read WASM file", http.StatusInternalServerError)
		return
	}

	if len(wasmData) < 4 || string(wasmData[:4]) != "\x00asm" {
		http.Error(w, "Invalid WASM file format", http.StatusBadRequest)
		return
	}

	cosignSig := r.FormValue("cosign_signature")
	sbom := r.FormValue("sbom")
	owner := r.FormValue("owner")
	version := r.FormValue("version")

	if cosignSig == "" || sbom == "" || owner == "" || version == "" {
		http.Error(w, "Missing required fields", http.StatusBadRequest)
		return
	}

	result, err := pr.validator.ValidatePlugin(wasmData, cosignSig, sbom, owner, version)
	if err != nil {
		http.Error(w, fmt.Sprintf("Validation failed: %v", err), http.StatusInternalServerError)
		return
	}

	if result.Valid {
		if err := pr.runner.TestPluginIsolation(wasmData); err != nil {
			result.Valid = false
			result.Errors = append(result.Errors, fmt.Sprintf("Isolation test failed: %v", err))
		}
	}

	response := map[string]interface{}{
		"success":      result.Valid,
		"validation":   result,
		"message":      getPublishMessage(result),
		"published_at": time.Now().Format(time.RFC3339),
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(response); err != nil {
		log.Printf("Failed to encode response: %v", err)
	}
}

func (pr *PluginRegistry) getPluginStatus(w http.ResponseWriter, r *http.Request) {
	path := strings.TrimPrefix(r.URL.Path, "/plugins/")
	pluginID := strings.Split(path, "/")[0]

	if pluginID == "" {
		http.Error(w, "Plugin ID required", http.StatusBadRequest)
		return
	}

	plugin, err := pr.validator.GetPlugin(pluginID)
	if err != nil {
		http.Error(w, "Failed to get plugin", http.StatusInternalServerError)
		return
	}

	if plugin == nil {
		http.Error(w, "Plugin not found", http.StatusNotFound)
		return
	}

	response := map[string]interface{}{
		"id":             plugin.ID,
		"owner":          plugin.Owner,
		"version":        plugin.Version,
		"verified":       plugin.Verified,
		"status":         plugin.Status,
		"created_at":     plugin.CreatedAt.Format(time.RFC3339),
		"sandbox_policy": plugin.SandboxPolicy,
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(response); err != nil {
		log.Printf("Failed to encode response: %v", err)
	}
}

func (pr *PluginRegistry) runPlugin(w http.ResponseWriter, r *http.Request) {
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

	plugin, err := pr.validator.GetPlugin(req.PluginID)
	if err != nil {
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

	var policy runner.SandboxPolicy
	if err := json.Unmarshal([]byte(plugin.SandboxPolicy), &policy); err != nil {
		http.Error(w, "Invalid sandbox policy", http.StatusInternalServerError)
		return
	}

	input := runner.PluginInput{
		ArtifactID:   req.ArtifactID,
		ArtifactType: "unknown",
		Metadata:     req.Metadata,
		S3URL:        fmt.Sprintf("s3://artifacts/%s", req.ArtifactID),
		Timestamp:    time.Now(),
	}

	mockWasmData := []byte("\x00asm\x01\x00\x00\x00")

	output, err := pr.runner.ExecutePlugin(mockWasmData, input, policy)
	if err != nil {
		http.Error(w, fmt.Sprintf("Plugin execution failed: %v", err), http.StatusInternalServerError)
		return
	}

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
	}

	if output.Error != "" {
		response["error"] = output.Error
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(response); err != nil {
		log.Printf("Failed to encode response: %v", err)
	}
}

func getPublishMessage(result *fortress_bridge.ValidationResult) string {
	if result.Valid {
		return "Plugin published and verified successfully"
	}
	if len(result.Errors) > 0 {
		return fmt.Sprintf("Plugin rejected: %s", strings.Join(result.Errors, ", "))
	}
	return "Plugin submitted for review"
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}