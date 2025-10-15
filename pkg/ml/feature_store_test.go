package ml

import (
	"testing"
	"time"
)

func TestNewAdvancedFeatureStore(t *testing.T) {
	config := FeatureStoreConfig{
		EnableLineage:    true,
		EnableVersioning: true,
		MaxVersions:      50,
	}
	
	store := NewAdvancedFeatureStore(config)
	if store == nil {
		t.Fatal("NewAdvancedFeatureStore returned nil")
	}
	
	if !store.enableLineage {
		t.Error("Lineage should be enabled")
	}
	if !store.enableVersioning {
		t.Error("Versioning should be enabled")
	}
	if store.maxVersions != 50 {
		t.Errorf("Expected maxVersions=50, got %d", store.maxVersions)
	}
	
	defer store.Close()
}

func TestFeatureStore_SetAndGet(t *testing.T) {
	store := NewAdvancedFeatureStore(FeatureStoreConfig{
		EnableVersioning: true,
		MaxVersions:      10,
	})
	defer store.Close()
	
	features := map[string]interface{}{
		"packet_count": 100,
		"byte_count":   5000,
		"duration":     1.5,
		"protocol":     "tcp",
	}
	
	err := store.Set("flow_123", features, nil)
	if err != nil {
		t.Fatalf("Set failed: %v", err)
	}
	
	retrieved, err := store.Get("flow_123")
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}
	
	if retrieved["packet_count"] != 100 {
		t.Errorf("Expected packet_count=100, got %v", retrieved["packet_count"])
	}
	if retrieved["protocol"] != "tcp" {
		t.Errorf("Expected protocol=tcp, got %v", retrieved["protocol"])
	}
}

func TestFeatureStore_Versioning(t *testing.T) {
	store := NewAdvancedFeatureStore(FeatureStoreConfig{
		EnableVersioning: true,
		MaxVersions:      10,
	})
	defer store.Close()
	
	// Set version 1
	features1 := map[string]interface{}{"count": 10}
	err := store.Set("key1", features1, nil)
	if err != nil {
		t.Fatalf("Set v1 failed: %v", err)
	}
	
	// Set version 2
	features2 := map[string]interface{}{"count": 20}
	err = store.Set("key1", features2, nil)
	if err != nil {
		t.Fatalf("Set v2 failed: %v", err)
	}
	
	// Set version 3
	features3 := map[string]interface{}{"count": 30}
	err = store.Set("key1", features3, nil)
	if err != nil {
		t.Fatalf("Set v3 failed: %v", err)
	}
	
	// Get latest (should be v3)
	latest, err := store.Get("key1")
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}
	if latest["count"] != 30 {
		t.Errorf("Expected count=30, got %v", latest["count"])
	}
	
	// Get version 1
	v1, err := store.GetVersion("key1", 1)
	if err != nil {
		t.Fatalf("GetVersion 1 failed: %v", err)
	}
	if v1.Features["count"] != 10 {
		t.Errorf("Expected count=10 for v1, got %v", v1.Features["count"])
	}
	
	// Get version 2
	v2, err := store.GetVersion("key1", 2)
	if err != nil {
		t.Fatalf("GetVersion 2 failed: %v", err)
	}
	if v2.Features["count"] != 20 {
		t.Errorf("Expected count=20 for v2, got %v", v2.Features["count"])
	}
	
	// Get version history
	history, err := store.GetVersionHistory("key1")
	if err != nil {
		t.Fatalf("GetVersionHistory failed: %v", err)
	}
	if len(history) != 3 {
		t.Errorf("Expected 3 versions, got %d", len(history))
	}
}

func TestFeatureStore_PointInTime(t *testing.T) {
	store := NewAdvancedFeatureStore(FeatureStoreConfig{
		EnableVersioning: true,
		MaxVersions:      10,
	})
	defer store.Close()
	
	baseTime := time.Now()
	
	// Set features at different times
	store.Set("key1", map[string]interface{}{"value": 10}, nil)
	time.Sleep(10 * time.Millisecond)
	
	middleTime := time.Now()
	time.Sleep(10 * time.Millisecond)
	
	store.Set("key1", map[string]interface{}{"value": 20}, nil)
	time.Sleep(10 * time.Millisecond)
	
	store.Set("key1", map[string]interface{}{"value": 30}, nil)
	
	// Get features at middle time (should get value=10)
	features, err := store.GetPointInTime("key1", middleTime)
	if err != nil {
		t.Fatalf("GetPointInTime failed: %v", err)
	}
	
	if features["value"] != 10 {
		t.Errorf("Expected value=10 at middle time, got %v", features["value"])
	}
	
	// Get features at base time (should get value=10)
	features2, err := store.GetPointInTime("key1", baseTime.Add(5*time.Millisecond))
	if err != nil {
		t.Fatalf("GetPointInTime failed: %v", err)
	}
	
	if features2["value"] != 10 {
		t.Errorf("Expected value=10 at base time, got %v", features2["value"])
	}
}

func TestFeatureStore_Lineage(t *testing.T) {
	store := NewAdvancedFeatureStore(FeatureStoreConfig{
		EnableLineage:    true,
		EnableVersioning: true,
		MaxVersions:      10,
	})
	defer store.Close()
	
	// Set v1 with lineage
	lineage1 := &FeatureLineage{
		SourceID:     "raw_packet_stream",
		Transformers: []string{"packet_parser", "feature_extractor"},
		CreatedBy:    "ml-service",
		Dependencies: []string{"network_interface"},
		Metadata:     map[string]string{"env": "production"},
	}
	
	err := store.Set("flow_123", map[string]interface{}{"count": 10}, lineage1)
	if err != nil {
		t.Fatalf("Set v1 failed: %v", err)
	}
	
	// Set v2 with lineage (child of v1)
	lineage2 := &FeatureLineage{
		SourceID:     "raw_packet_stream",
		Transformers: []string{"packet_parser", "feature_extractor", "aggregator"},
		CreatedBy:    "ml-service",
		Dependencies: []string{"network_interface", "time_window"},
		Metadata:     map[string]string{"env": "production"},
	}
	
	err = store.Set("flow_123", map[string]interface{}{"count": 20}, lineage2)
	if err != nil {
		t.Fatalf("Set v2 failed: %v", err)
	}
	
	// Get lineage for v2
	retrievedLineage, err := store.GetLineage("flow_123", 2)
	if err != nil {
		t.Fatalf("GetLineage failed: %v", err)
	}
	
	if retrievedLineage.ParentVersion != 1 {
		t.Errorf("Expected parent_version=1, got %d", retrievedLineage.ParentVersion)
	}
	if retrievedLineage.SourceID != "raw_packet_stream" {
		t.Errorf("Expected source_id=raw_packet_stream, got %s", retrievedLineage.SourceID)
	}
	if len(retrievedLineage.Transformers) != 3 {
		t.Errorf("Expected 3 transformers, got %d", len(retrievedLineage.Transformers))
	}
	
	// Trace full lineage chain
	lineageChain, err := store.TraceLineage("flow_123", 2)
	if err != nil {
		t.Fatalf("TraceLineage failed: %v", err)
	}
	
	if len(lineageChain) != 2 {
		t.Errorf("Expected lineage chain length=2, got %d", len(lineageChain))
	}
}

func TestFeatureStore_Schema(t *testing.T) {
	store := NewAdvancedFeatureStore(FeatureStoreConfig{
		EnableVersioning: true,
		MaxVersions:      10,
	})
	defer store.Close()
	
	// Register schema
	minVal := 0.0
	maxVal := 1000000.0
	
	schema := &FeatureSchema{
		Name:    "network_flow",
		Version: "1.0",
		Fields: map[string]FieldSchema{
			"packet_count": {
				Type:        "int",
				Required:    true,
				MinValue:    &minVal,
				Description: "Total packet count",
			},
			"byte_count": {
				Type:        "int",
				Required:    true,
				MinValue:    &minVal,
				Description: "Total bytes",
			},
			"duration": {
				Type:        "float",
				Required:    true,
				MinValue:    &minVal,
				MaxValue:    &maxVal,
				Description: "Flow duration in seconds",
			},
			"protocol": {
				Type:        "string",
				Required:    false,
				DefaultVal:  "tcp",
				Description: "Network protocol",
			},
		},
	}
	
	err := store.RegisterSchema(schema)
	if err != nil {
		t.Fatalf("RegisterSchema failed: %v", err)
	}
	
	// Get schema
	retrieved, err := store.GetSchema("network_flow")
	if err != nil {
		t.Fatalf("GetSchema failed: %v", err)
	}
	
	if retrieved.Name != "network_flow" {
		t.Errorf("Expected schema name=network_flow, got %s", retrieved.Name)
	}
	if len(retrieved.Fields) != 4 {
		t.Errorf("Expected 4 fields, got %d", len(retrieved.Fields))
	}
	
	// Test valid features
	validFeatures := map[string]interface{}{
		"packet_count": 100.0,
		"byte_count":   5000.0,
		"duration":     1.5,
		"protocol":     "tcp",
	}
	
	err = store.Set("network_flow", validFeatures, nil)
	if err != nil {
		t.Fatalf("Set with valid schema failed: %v", err)
	}
	
	// Test invalid features (missing required field)
	invalidFeatures := map[string]interface{}{
		"packet_count": 100.0,
		// missing byte_count
		"duration": 1.5,
	}
	
	err = store.Set("network_flow", invalidFeatures, nil)
	if err == nil {
		t.Error("Expected error for missing required field, got nil")
	}
}

func TestFeatureStore_Archive(t *testing.T) {
	store := NewAdvancedFeatureStore(FeatureStoreConfig{
		EnableVersioning: true,
		MaxVersions:      10,
	})
	defer store.Close()
	
	// Set features
	store.Set("key1", map[string]interface{}{"value": 10}, nil)
	store.Set("key1", map[string]interface{}{"value": 20}, nil)
	
	// Archive version 1
	err := store.Archive("key1", 1)
	if err != nil {
		t.Fatalf("Archive failed: %v", err)
	}
	
	// Get version 1 should still work
	v1, err := store.GetVersion("key1", 1)
	if err != nil {
		t.Fatalf("GetVersion failed: %v", err)
	}
	
	if !v1.IsArchived {
		t.Error("Version 1 should be archived")
	}
	
	// Latest should still be v2
	latest, err := store.Get("key1")
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}
	
	if latest["value"] != 20 {
		t.Errorf("Expected value=20, got %v", latest["value"])
	}
}

func TestFeatureStore_Tags(t *testing.T) {
	store := NewAdvancedFeatureStore(FeatureStoreConfig{
		EnableVersioning: true,
		MaxVersions:      10,
	})
	defer store.Close()
	
	// Set features
	store.Set("key1", map[string]interface{}{"value": 10}, nil)
	
	// Add tags
	err := store.AddTag("key1", 1, "environment", "production")
	if err != nil {
		t.Fatalf("AddTag failed: %v", err)
	}
	
	err = store.AddTag("key1", 1, "model", "anomaly_detector")
	if err != nil {
		t.Fatalf("AddTag failed: %v", err)
	}
	
	// Get version and check tags
	v1, err := store.GetVersion("key1", 1)
	if err != nil {
		t.Fatalf("GetVersion failed: %v", err)
	}
	
	if v1.Tags["environment"] != "production" {
		t.Errorf("Expected environment=production, got %s", v1.Tags["environment"])
	}
	if v1.Tags["model"] != "anomaly_detector" {
		t.Errorf("Expected model=anomaly_detector, got %s", v1.Tags["model"])
	}
}

func TestFeatureStore_Exists(t *testing.T) {
	store := NewAdvancedFeatureStore(FeatureStoreConfig{
		EnableVersioning: true,
		MaxVersions:      10,
	})
	defer store.Close()
	
	// Should not exist initially
	if store.Exists("key1") {
		t.Error("key1 should not exist")
	}
	
	// Add features
	store.Set("key1", map[string]interface{}{"value": 10}, nil)
	
	// Should exist now
	if !store.Exists("key1") {
		t.Error("key1 should exist")
	}
	
	// Archive the only version
	store.Archive("key1", 1)
	
	// Should not exist (all versions archived)
	if store.Exists("key1") {
		t.Error("key1 should not exist after archiving")
	}
}

func TestFeatureStore_Delete(t *testing.T) {
	store := NewAdvancedFeatureStore(FeatureStoreConfig{
		EnableVersioning: true,
		MaxVersions:      10,
	})
	defer store.Close()
	
	// Add features
	store.Set("key1", map[string]interface{}{"value": 10}, nil)
	store.Set("key1", map[string]interface{}{"value": 20}, nil)
	
	// Delete
	err := store.Delete("key1")
	if err != nil {
		t.Fatalf("Delete failed: %v", err)
	}
	
	// Should not exist
	if store.Exists("key1") {
		t.Error("key1 should not exist after deletion")
	}
	
	// Get should fail
	_, err = store.Get("key1")
	if err == nil {
		t.Error("Expected error getting deleted key, got nil")
	}
}

func TestFeatureStore_MaxVersions(t *testing.T) {
	store := NewAdvancedFeatureStore(FeatureStoreConfig{
		EnableVersioning: true,
		MaxVersions:      3, // Keep only 3 versions
	})
	defer store.Close()
	
	// Add 5 versions
	for i := 1; i <= 5; i++ {
		store.Set("key1", map[string]interface{}{"value": i * 10}, nil)
		time.Sleep(5 * time.Millisecond)
	}
	
	// Should only have 3 versions (v3, v4, v5)
	history, err := store.GetVersionHistory("key1")
	if err != nil {
		t.Fatalf("GetVersionHistory failed: %v", err)
	}
	
	if len(history) != 3 {
		t.Errorf("Expected 3 versions (max), got %d", len(history))
	}
	
	// Oldest should be v3
	if history[0].Version != 3 {
		t.Errorf("Expected oldest version=3, got %d", history[0].Version)
	}
	
	// Latest should be v5
	if history[2].Version != 5 {
		t.Errorf("Expected latest version=5, got %d", history[2].Version)
	}
}

func TestFeatureStore_GetStats(t *testing.T) {
	store := NewAdvancedFeatureStore(FeatureStoreConfig{
		EnableLineage:    true,
		EnableVersioning: true,
		MaxVersions:      10,
	})
	defer store.Close()
	
	// Add some data
	store.Set("key1", map[string]interface{}{"value": 10}, nil)
	store.Set("key1", map[string]interface{}{"value": 20}, nil)
	store.Set("key2", map[string]interface{}{"value": 30}, nil)
	
	// Register schema
	schema := &FeatureSchema{
		Name:   "test_schema",
		Fields: map[string]FieldSchema{},
	}
	store.RegisterSchema(schema)
	
	// Get stats
	stats := store.GetStats()
	
	if stats["total_keys"] != 2 {
		t.Errorf("Expected 2 keys, got %v", stats["total_keys"])
	}
	if stats["total_versions"] != 3 {
		t.Errorf("Expected 3 versions, got %v", stats["total_versions"])
	}
	if stats["total_schemas"] != 1 {
		t.Errorf("Expected 1 schema, got %v", stats["total_schemas"])
	}
	if stats["lineage_enabled"] != true {
		t.Error("Lineage should be enabled")
	}
}

func TestFeatureStore_ConcurrentAccess(t *testing.T) {
	store := NewAdvancedFeatureStore(FeatureStoreConfig{
		EnableVersioning: true,
		MaxVersions:      100,
	})
	defer store.Close()
	
	// Concurrent writes
	done := make(chan bool)
	for i := 0; i < 10; i++ {
		go func(id int) {
			for j := 0; j < 10; j++ {
				key := "concurrent_key"
				features := map[string]interface{}{
					"worker_id": id,
					"iteration": j,
				}
				store.Set(key, features, nil)
			}
			done <- true
		}(i)
	}
	
	// Wait for all goroutines
	for i := 0; i < 10; i++ {
		<-done
	}
	
	// Should have 100 versions
	history, err := store.GetVersionHistory("concurrent_key")
	if err != nil {
		t.Fatalf("GetVersionHistory failed: %v", err)
	}
	
	if len(history) != 100 {
		t.Errorf("Expected 100 versions, got %d", len(history))
	}
}

func BenchmarkFeatureStore_Set(b *testing.B) {
	store := NewAdvancedFeatureStore(FeatureStoreConfig{
		EnableVersioning: true,
		MaxVersions:      1000,
	})
	defer store.Close()
	
	features := map[string]interface{}{
		"packet_count": 100,
		"byte_count":   5000,
		"duration":     1.5,
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		store.Set("benchmark_key", features, nil)
	}
}

func BenchmarkFeatureStore_Get(b *testing.B) {
	store := NewAdvancedFeatureStore(FeatureStoreConfig{
		EnableVersioning: true,
		MaxVersions:      1000,
	})
	defer store.Close()
	
	features := map[string]interface{}{
		"packet_count": 100,
		"byte_count":   5000,
		"duration":     1.5,
	}
	store.Set("benchmark_key", features, nil)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		store.Get("benchmark_key")
	}
}

func BenchmarkFeatureStore_GetPointInTime(b *testing.B) {
	store := NewAdvancedFeatureStore(FeatureStoreConfig{
		EnableVersioning: true,
		MaxVersions:      1000,
	})
	defer store.Close()
	
	// Add 100 versions
	for i := 0; i < 100; i++ {
		features := map[string]interface{}{"value": i}
		store.Set("benchmark_key", features, nil)
		time.Sleep(1 * time.Millisecond)
	}
	
	timestamp := time.Now()
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		store.GetPointInTime("benchmark_key", timestamp)
	}
}
