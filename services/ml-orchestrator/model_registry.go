package main
package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// ModelRegistry manages model versions with rollback capability
type ModelRegistry struct {
	mu              sync.RWMutex
	models          map[string]*ModelVersion // version -> model
	currentVersion  string
	previousVersion string
	metadata        map[string]*ModelMetadata
	storagePath     string
}

type ModelVersion struct {
	Version     string    `json:"version"`
	Data        []byte    `json:"data"`
	CreatedAt   time.Time `json:"created_at"`
	Checksum    string    `json:"checksum"`
	Size        int64     `json:"size"`
	Description string    `json:"description"`
}

type ModelMetadata struct {
	Version      string            `json:"version"`
	Algorithm    string            `json:"algorithm"`
	Accuracy     float64           `json:"accuracy"`
	Precision    float64           `json:"precision"`
	Recall       float64           `json:"recall"`
	F1Score      float64           `json:"f1_score"`
	TrainingDate time.Time         `json:"training_date"`
	SampleCount  int               `json:"sample_count"`
	Features     []string          `json:"features"`
	Hyperparams  map[string]string `json:"hyperparameters"`
	Status       string            `json:"status"` // active, archived, deprecated
}

// NewModelRegistry creates a new model registry with persistent storage
func NewModelRegistry(storagePath string) (*ModelRegistry, error) {
	// Ensure storage directory exists
	if err := os.MkdirAll(storagePath, 0750); err != nil {
		return nil, fmt.Errorf("create storage dir: %w", err)
	}

	registry := &ModelRegistry{
		models:      make(map[string]*ModelVersion),
		metadata:    make(map[string]*ModelMetadata),
		storagePath: storagePath,
	}

	// Load existing models from disk
	if err := registry.loadFromDisk(); err != nil {
		log.Printf("[ModelRegistry] Warning: failed to load models: %v", err)
	}

	return registry, nil
}

// RegisterModel registers a new model version
func (mr *ModelRegistry) RegisterModel(version string, data []byte, meta *ModelMetadata) error {
	mr.mu.Lock()
	defer mr.mu.Unlock()

	if version == "" {
		return fmt.Errorf("version cannot be empty")
	}

	if _, exists := mr.models[version]; exists {
		return fmt.Errorf("version %s already exists", version)
	}

	// Create model version
	model := &ModelVersion{
		Version:     version,
		Data:        data,
		CreatedAt:   time.Now(),
		Checksum:    calculateChecksum(data),
		Size:        int64(len(data)),
		Description: meta.Algorithm,
	}

	// Update metadata
	if meta != nil {
		meta.Version = version
		meta.Status = "active"
		mr.metadata[version] = meta
	}

	// Store in memory
	mr.models[version] = model

	// Persist to disk
	if err := mr.saveToDisk(version, model, meta); err != nil {
		log.Printf("[ModelRegistry] Warning: failed to persist model: %v", err)
	}

	// Update current version if this is the first model
	if mr.currentVersion == "" {
		mr.previousVersion = ""
		mr.currentVersion = version
	}

	log.Printf("[ModelRegistry] Registered model version %s (size: %d bytes, checksum: %s)",
		version, model.Size, model.Checksum[:8])

	return nil
}

// PromoteModel promotes a model version to current
func (mr *ModelRegistry) PromoteModel(version string) error {
	mr.mu.Lock()
	defer mr.mu.Unlock()

	if _, exists := mr.models[version]; !exists {
		return fmt.Errorf("version %s not found", version)
	}

	// Keep track of previous version for rollback
	mr.previousVersion = mr.currentVersion
	mr.currentVersion = version

	// Update metadata status
	if mr.previousVersion != "" {
		if meta, ok := mr.metadata[mr.previousVersion]; ok {
			meta.Status = "archived"
		}
	}
	if meta, ok := mr.metadata[version]; ok {
		meta.Status = "active"
	}

	log.Printf("[ModelRegistry] Promoted version %s to current (previous: %s)",
		version, mr.previousVersion)

	return nil
}

// Rollback rolls back to the previous model version
func (mr *ModelRegistry) Rollback() error {
	mr.mu.Lock()
	defer mr.mu.Unlock()

	if mr.previousVersion == "" {
		return fmt.Errorf("no previous version available for rollback")
	}

	log.Printf("[ModelRegistry] Rolling back from %s to %s",
		mr.currentVersion, mr.previousVersion)

	// Swap current and previous
	temp := mr.currentVersion
	mr.currentVersion = mr.previousVersion
	mr.previousVersion = temp

	// Update metadata
	if meta, ok := mr.metadata[mr.currentVersion]; ok {
		meta.Status = "active"
	}
	if meta, ok := mr.metadata[mr.previousVersion]; ok {
		meta.Status = "archived"
	}

	return nil
}

// GetCurrentModel returns the current active model
func (mr *ModelRegistry) GetCurrentModel() (*ModelVersion, error) {
	mr.mu.RLock()
	defer mr.mu.RUnlock()

	if mr.currentVersion == "" {
		return nil, fmt.Errorf("no current model version set")
	}

	model, exists := mr.models[mr.currentVersion]
	if !exists {
		return nil, fmt.Errorf("current version %s not found", mr.currentVersion)
	}

	return model, nil
}

// GetModel returns a specific model version
func (mr *ModelRegistry) GetModel(version string) (*ModelVersion, error) {
	mr.mu.RLock()
	defer mr.mu.RUnlock()

	model, exists := mr.models[version]
	if !exists {
		return nil, fmt.Errorf("version %s not found", version)
	}

	return model, nil
}

// ListVersions returns all available model versions
func (mr *ModelRegistry) ListVersions() []string {
	mr.mu.RLock()
	defer mr.mu.RUnlock()

	versions := make([]string, 0, len(mr.models))
	for v := range mr.models {
		versions = append(versions, v)
	}

	return versions
}

// GetMetadata returns metadata for a specific version
func (mr *ModelRegistry) GetMetadata(version string) (*ModelMetadata, error) {
	mr.mu.RLock()
	defer mr.mu.RUnlock()

	meta, exists := mr.metadata[version]
	if !exists {
		return nil, fmt.Errorf("metadata for version %s not found", version)
	}

	return meta, nil
}

// GetCurrentVersion returns the current active version identifier
func (mr *ModelRegistry) GetCurrentVersion() string {
	mr.mu.RLock()
	defer mr.mu.RUnlock()
	return mr.currentVersion
}

// GetPreviousVersion returns the previous version identifier
func (mr *ModelRegistry) GetPreviousVersion() string {
	mr.mu.RLock()
	defer mr.mu.RUnlock()
	return mr.previousVersion
}

// DeleteVersion removes a model version (cannot delete current or previous)
func (mr *ModelRegistry) DeleteVersion(version string) error {
	mr.mu.Lock()
	defer mr.mu.Unlock()

	if version == mr.currentVersion {
		return fmt.Errorf("cannot delete current version")
	}

	if version == mr.previousVersion {
		return fmt.Errorf("cannot delete previous version (needed for rollback)")
	}

	delete(mr.models, version)
	delete(mr.metadata, version)

	// Delete from disk
	modelPath := filepath.Join(mr.storagePath, fmt.Sprintf("model_%s.bin", version))
	metaPath := filepath.Join(mr.storagePath, fmt.Sprintf("meta_%s.json", version))
	os.Remove(modelPath)
	os.Remove(metaPath)

	log.Printf("[ModelRegistry] Deleted version %s", version)

	return nil
}

// saveToDisk persists model to disk
func (mr *ModelRegistry) saveToDisk(version string, model *ModelVersion, meta *ModelMetadata) error {
	// Save model binary
	modelPath := filepath.Join(mr.storagePath, fmt.Sprintf("model_%s.bin", version))
	if err := os.WriteFile(modelPath, model.Data, 0640); err != nil {
		return fmt.Errorf("write model: %w", err)
	}

	// Save metadata
	if meta != nil {
		metaPath := filepath.Join(mr.storagePath, fmt.Sprintf("meta_%s.json", version))
		metaJSON, err := json.MarshalIndent(meta, "", "  ")
		if err != nil {
			return fmt.Errorf("marshal metadata: %w", err)
		}
		if err := os.WriteFile(metaPath, metaJSON, 0640); err != nil {
			return fmt.Errorf("write metadata: %w", err)
		}
	}

	return nil
}

// loadFromDisk loads all models from storage directory
func (mr *ModelRegistry) loadFromDisk() error {
	entries, err := os.ReadDir(mr.storagePath)
	if err != nil {
		return err
	}

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}

		name := entry.Name()
		
		// Load model binaries
		if filepath.Ext(name) == ".bin" && len(name) > 10 {
			version := name[6 : len(name)-4] // Extract from "model_VERSION.bin"
			
			modelPath := filepath.Join(mr.storagePath, name)
			data, err := os.ReadFile(modelPath)
			if err != nil {
				log.Printf("[ModelRegistry] Failed to load model %s: %v", version, err)
				continue
			}

			model := &ModelVersion{
				Version:   version,
				Data:      data,
				CreatedAt: time.Now(), // Would need to store this
				Checksum:  calculateChecksum(data),
				Size:      int64(len(data)),
			}

			mr.models[version] = model

			// Try to load metadata
			metaPath := filepath.Join(mr.storagePath, fmt.Sprintf("meta_%s.json", version))
			if metaData, err := os.ReadFile(metaPath); err == nil {
				var meta ModelMetadata
				if err := json.Unmarshal(metaData, &meta); err == nil {
					mr.metadata[version] = &meta
				}
			}

			log.Printf("[ModelRegistry] Loaded model version %s from disk", version)
		}
	}

	return nil
}

// ExportModel exports a model version to a writer (for backup/transfer)
func (mr *ModelRegistry) ExportModel(version string, w io.Writer) error {
	mr.mu.RLock()
	model, exists := mr.models[version]
	meta, hasMeta := mr.metadata[version]
	mr.mu.RUnlock()

	if !exists {
		return fmt.Errorf("version %s not found", version)
	}

	// Create export package
	export := struct {
		Model    *ModelVersion    `json:"model"`
		Metadata *ModelMetadata   `json:"metadata,omitempty"`
	}{
		Model:    model,
		Metadata: meta,
	}

	if !hasMeta {
		export.Metadata = nil
	}

	return json.NewEncoder(w).Encode(export)
}

// ImportModel imports a model from a reader
func (mr *ModelRegistry) ImportModel(r io.Reader) error {
	var export struct {
		Model    *ModelVersion  `json:"model"`
		Metadata *ModelMetadata `json:"metadata"`
	}

	if err := json.NewDecoder(r).Decode(&export); err != nil {
		return fmt.Errorf("decode import: %w", err)
	}

	return mr.RegisterModel(export.Model.Version, export.Model.Data, export.Metadata)
}

// calculateChecksum generates a checksum for model data
func calculateChecksum(data []byte) string {
	// Simple FNV-1a hash for checksumming
	const (
		offset64 = 14695981039346656037
		prime64  = 1099511628211
	)
	hash := uint64(offset64)
	for _, b := range data {
		hash ^= uint64(b)
		hash *= prime64
	}
	return fmt.Sprintf("%016x", hash)
}
