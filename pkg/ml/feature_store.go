package ml

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

// FeatureVersion represents a versioned feature set
type FeatureVersion struct {
	Version     int                    `json:"version"`
	Features    map[string]interface{} `json:"features"`
	Timestamp   time.Time              `json:"timestamp"`
	Hash        string                 `json:"hash"`        // SHA256 of features
	Lineage     *FeatureLineage        `json:"lineage"`     // Provenance tracking
	Schema      *FeatureSchema         `json:"schema"`      // Feature schema
	Tags        map[string]string      `json:"tags"`        // Metadata tags
	IsArchived  bool                   `json:"is_archived"` // Soft delete
}

// FeatureLineage tracks feature provenance
type FeatureLineage struct {
	SourceID      string            `json:"source_id"`       // Original data source
	Transformers  []string          `json:"transformers"`    // Applied transformations
	ParentVersion int               `json:"parent_version"`  // Previous version
	CreatedBy     string            `json:"created_by"`      // User/service
	CreatedAt     time.Time         `json:"created_at"`
	Dependencies  []string          `json:"dependencies"`    // Dependent features
	Metadata      map[string]string `json:"metadata"`
}

// FeatureSchema defines feature types and constraints
type FeatureSchema struct {
	Name        string                 `json:"name"`
	Version     string                 `json:"version"`
	Fields      map[string]FieldSchema `json:"fields"`
	CreatedAt   time.Time              `json:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at"`
}

// FieldSchema defines individual field schema
type FieldSchema struct {
	Type        string      `json:"type"`         // int, float, string, bool
	Required    bool        `json:"required"`
	DefaultVal  interface{} `json:"default"`
	MinValue    *float64    `json:"min_value,omitempty"`
	MaxValue    *float64    `json:"max_value,omitempty"`
	Description string      `json:"description"`
}

// AdvancedFeatureStore manages versioned features
type AdvancedFeatureStore struct {
	mu              sync.RWMutex
	features        map[string][]*FeatureVersion // key -> versions (sorted by timestamp)
	schemas         map[string]*FeatureSchema
	currentVersion  map[string]int // key -> current version number
	enableLineage   bool
	enableVersioning bool
	maxVersions     int // Max versions to keep per key
	ctx             context.Context
	cancel          context.CancelFunc
}

// FeatureStoreConfig configures the feature store
type FeatureStoreConfig struct {
	EnableLineage    bool
	EnableVersioning bool
	MaxVersions      int
}

// NewAdvancedFeatureStore creates a new feature store
func NewAdvancedFeatureStore(config FeatureStoreConfig) *AdvancedFeatureStore {
	if config.MaxVersions <= 0 {
		config.MaxVersions = 100 // Default
	}

	ctx, cancel := context.WithCancel(context.Background())

	store := &AdvancedFeatureStore{
		features:         make(map[string][]*FeatureVersion),
		schemas:          make(map[string]*FeatureSchema),
		currentVersion:   make(map[string]int),
		enableLineage:    config.EnableLineage,
		enableVersioning: config.EnableVersioning,
		maxVersions:      config.MaxVersions,
		ctx:              ctx,
		cancel:           cancel,
	}

	return store
}

// RegisterSchema registers a feature schema
func (fs *AdvancedFeatureStore) RegisterSchema(schema *FeatureSchema) error {
	fs.mu.Lock()
	defer fs.mu.Unlock()

	if schema.Name == "" {
		return fmt.Errorf("schema name cannot be empty")
	}

	schema.UpdatedAt = time.Now()
	if schema.CreatedAt.IsZero() {
		schema.CreatedAt = time.Now()
	}

	fs.schemas[schema.Name] = schema
	return nil
}

// GetSchema retrieves a feature schema
func (fs *AdvancedFeatureStore) GetSchema(name string) (*FeatureSchema, error) {
	fs.mu.RLock()
	defer fs.mu.RUnlock()

	schema, exists := fs.schemas[name]
	if !exists {
		return nil, fmt.Errorf("schema not found: %s", name)
	}

	return schema, nil
}

// Set stores features with versioning and lineage
func (fs *AdvancedFeatureStore) Set(key string, features map[string]interface{}, lineage *FeatureLineage) error {
	fs.mu.Lock()
	defer fs.mu.Unlock()

	// Validate against schema if exists
	if schema, exists := fs.schemas[key]; exists {
		if err := fs.validateFeatures(features, schema); err != nil {
			return fmt.Errorf("schema validation failed: %w", err)
		}
	}

	// Get next version number
	version := 1
	if fs.enableVersioning {
		version = fs.currentVersion[key] + 1
	}

	// Calculate hash
	hash := fs.calculateHash(features)

	// Create feature version
	fv := &FeatureVersion{
		Version:   version,
		Features:  features,
		Timestamp: time.Now(),
		Hash:      hash,
		Tags:      make(map[string]string),
	}

	if fs.enableLineage && lineage != nil {
		lineage.ParentVersion = fs.currentVersion[key]
		lineage.CreatedAt = time.Now()
		fv.Lineage = lineage
	}

	// Add to versions
	if _, exists := fs.features[key]; !exists {
		fs.features[key] = make([]*FeatureVersion, 0)
	}
	fs.features[key] = append(fs.features[key], fv)

	// Update current version
	fs.currentVersion[key] = version

	// Cleanup old versions if needed
	if len(fs.features[key]) > fs.maxVersions {
		fs.features[key] = fs.features[key][len(fs.features[key])-fs.maxVersions:]
	}

	return nil
}

// Get retrieves the latest features
func (fs *AdvancedFeatureStore) Get(key string) (map[string]interface{}, error) {
	fs.mu.RLock()
	defer fs.mu.RUnlock()

	versions, exists := fs.features[key]
	if !exists || len(versions) == 0 {
		return nil, fmt.Errorf("key not found: %s", key)
	}

	// Return latest non-archived version
	for i := len(versions) - 1; i >= 0; i-- {
		if !versions[i].IsArchived {
			return versions[i].Features, nil
		}
	}

	return nil, fmt.Errorf("no active version found for key: %s", key)
}

// GetVersion retrieves a specific version
func (fs *AdvancedFeatureStore) GetVersion(key string, version int) (*FeatureVersion, error) {
	fs.mu.RLock()
	defer fs.mu.RUnlock()

	versions, exists := fs.features[key]
	if !exists {
		return nil, fmt.Errorf("key not found: %s", key)
	}

	for _, fv := range versions {
		if fv.Version == version {
			return fv, nil
		}
	}

	return nil, fmt.Errorf("version %d not found for key %s", version, key)
}

// GetPointInTime retrieves features as they were at a specific time
func (fs *AdvancedFeatureStore) GetPointInTime(key string, timestamp time.Time) (map[string]interface{}, error) {
	fs.mu.RLock()
	defer fs.mu.RUnlock()

	versions, exists := fs.features[key]
	if !exists || len(versions) == 0 {
		return nil, fmt.Errorf("key not found: %s", key)
	}

	// Find the latest version before or at the timestamp
	var selectedVersion *FeatureVersion
	for i := len(versions) - 1; i >= 0; i-- {
		if versions[i].Timestamp.Before(timestamp) || versions[i].Timestamp.Equal(timestamp) {
			if !versions[i].IsArchived {
				selectedVersion = versions[i]
				break
			}
		}
	}

	if selectedVersion == nil {
		return nil, fmt.Errorf("no version found at timestamp %s for key %s", timestamp, key)
	}

	return selectedVersion.Features, nil
}

// GetVersionHistory retrieves all versions for a key
func (fs *AdvancedFeatureStore) GetVersionHistory(key string) ([]*FeatureVersion, error) {
	fs.mu.RLock()
	defer fs.mu.RUnlock()

	versions, exists := fs.features[key]
	if !exists {
		return nil, fmt.Errorf("key not found: %s", key)
	}

	// Return copy to prevent external modification
	result := make([]*FeatureVersion, len(versions))
	copy(result, versions)
	return result, nil
}

// GetLineage retrieves the lineage for a specific version
func (fs *AdvancedFeatureStore) GetLineage(key string, version int) (*FeatureLineage, error) {
	fv, err := fs.GetVersion(key, version)
	if err != nil {
		return nil, err
	}

	if fv.Lineage == nil {
		return nil, fmt.Errorf("no lineage information for version %d", version)
	}

	return fv.Lineage, nil
}

// TraceLineage traces the full lineage chain back to origin
func (fs *AdvancedFeatureStore) TraceLineage(key string, version int) ([]*FeatureLineage, error) {
	fs.mu.RLock()
	defer fs.mu.RUnlock()

	lineages := make([]*FeatureLineage, 0)
	currentVersion := version

	for currentVersion > 0 {
		fv, err := fs.getVersionUnsafe(key, currentVersion)
		if err != nil {
			break
		}

		if fv.Lineage != nil {
			lineages = append(lineages, fv.Lineage)
			currentVersion = fv.Lineage.ParentVersion
		} else {
			break
		}
	}

	return lineages, nil
}

// AddTag adds metadata tags to a version
func (fs *AdvancedFeatureStore) AddTag(key string, version int, tagKey, tagValue string) error {
	fs.mu.Lock()
	defer fs.mu.Unlock()

	fv, err := fs.getVersionUnsafe(key, version)
	if err != nil {
		return err
	}

	if fv.Tags == nil {
		fv.Tags = make(map[string]string)
	}
	fv.Tags[tagKey] = tagValue

	return nil
}

// Archive marks a version as archived (soft delete)
func (fs *AdvancedFeatureStore) Archive(key string, version int) error {
	fs.mu.Lock()
	defer fs.mu.Unlock()

	fv, err := fs.getVersionUnsafe(key, version)
	if err != nil {
		return err
	}

	fv.IsArchived = true
	return nil
}

// Exists checks if a key exists
func (fs *AdvancedFeatureStore) Exists(key string) bool {
	fs.mu.RLock()
	defer fs.mu.RUnlock()

	versions, exists := fs.features[key]
	if !exists || len(versions) == 0 {
		return false
	}

	// Check if any non-archived version exists
	for i := len(versions) - 1; i >= 0; i-- {
		if !versions[i].IsArchived {
			return true
		}
	}

	return false
}

// Delete removes all versions of a key
func (fs *AdvancedFeatureStore) Delete(key string) error {
	fs.mu.Lock()
	defer fs.mu.Unlock()

	delete(fs.features, key)
	delete(fs.currentVersion, key)
	return nil
}

// GetStats returns store statistics
func (fs *AdvancedFeatureStore) GetStats() map[string]interface{} {
	fs.mu.RLock()
	defer fs.mu.RUnlock()

	totalVersions := 0
	for _, versions := range fs.features {
		totalVersions += len(versions)
	}

	return map[string]interface{}{
		"total_keys":        len(fs.features),
		"total_versions":    totalVersions,
		"total_schemas":     len(fs.schemas),
		"lineage_enabled":   fs.enableLineage,
		"versioning_enabled": fs.enableVersioning,
		"max_versions":      fs.maxVersions,
	}
}

// Close shuts down the feature store
func (fs *AdvancedFeatureStore) Close() error {
	fs.cancel()
	return nil
}

// Helper methods

func (fs *AdvancedFeatureStore) calculateHash(features map[string]interface{}) string {
	data, _ := json.Marshal(features)
	hash := sha256.Sum256(data)
	return fmt.Sprintf("%x", hash)
}

func (fs *AdvancedFeatureStore) validateFeatures(features map[string]interface{}, schema *FeatureSchema) error {
	for fieldName, fieldSchema := range schema.Fields {
		value, exists := features[fieldName]

		if !exists && fieldSchema.Required {
			return fmt.Errorf("required field missing: %s", fieldName)
		}

		if exists {
			if err := fs.validateField(value, fieldSchema); err != nil {
				return fmt.Errorf("field %s: %w", fieldName, err)
			}
		}
	}

	return nil
}

func (fs *AdvancedFeatureStore) validateField(value interface{}, schema FieldSchema) error {
	// Type validation
	switch schema.Type {
	case "int":
		if _, ok := value.(int); !ok {
			if _, ok := value.(float64); !ok { // JSON numbers are float64
				return fmt.Errorf("expected int, got %T", value)
			}
		}
	case "float":
		if _, ok := value.(float64); !ok {
			return fmt.Errorf("expected float, got %T", value)
		}
	case "string":
		if _, ok := value.(string); !ok {
			return fmt.Errorf("expected string, got %T", value)
		}
	case "bool":
		if _, ok := value.(bool); !ok {
			return fmt.Errorf("expected bool, got %T", value)
		}
	}

	// Range validation for numeric types
	if schema.MinValue != nil || schema.MaxValue != nil {
		var numValue float64
		switch v := value.(type) {
		case int:
			numValue = float64(v)
		case float64:
			numValue = v
		default:
			return fmt.Errorf("cannot apply range validation to non-numeric type")
		}

		if schema.MinValue != nil && numValue < *schema.MinValue {
			return fmt.Errorf("value %f below minimum %f", numValue, *schema.MinValue)
		}
		if schema.MaxValue != nil && numValue > *schema.MaxValue {
			return fmt.Errorf("value %f above maximum %f", numValue, *schema.MaxValue)
		}
	}

	return nil
}

func (fs *AdvancedFeatureStore) getVersionUnsafe(key string, version int) (*FeatureVersion, error) {
	versions, exists := fs.features[key]
	if !exists {
		return nil, fmt.Errorf("key not found: %s", key)
	}

	for _, fv := range versions {
		if fv.Version == version {
			return fv, nil
		}
	}

	return nil, fmt.Errorf("version %d not found for key %s", version, key)
}
