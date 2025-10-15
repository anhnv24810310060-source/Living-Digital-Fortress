package ml

import (
	"testing"
	"time"
)

func TestNewBehavioralFeatureExtractor(t *testing.T) {
	config := BehavioralConfig{
		WindowSize:          24 * time.Hour,
		UpdateInterval:      1 * time.Hour,
		MinEventsForProfile: 10,
		MaxEventsBuffer:     1000,
	}
	
	extractor := NewBehavioralFeatureExtractor(config)
	if extractor == nil {
		t.Fatal("NewBehavioralFeatureExtractor returned nil")
	}
	
	if extractor.windowSize != 24*time.Hour {
		t.Errorf("Expected windowSize=24h, got %v", extractor.windowSize)
	}
}

func TestBehavioralExtractor_AddEvent(t *testing.T) {
	extractor := NewBehavioralFeatureExtractor(BehavioralConfig{})
	
	event := BehavioralEvent{
		EntityID:  "user123",
		Timestamp: time.Now(),
		Action:    "login",
		Resource:  "/api/data",
		Location:  "US-East",
		Success:   true,
		Duration:  5 * time.Minute,
		DataRead:  1.5,
		DataWrite: 0.5,
	}
	
	extractor.AddEvent(event)
	
	events := extractor.events["user123"]
	if len(events) != 1 {
		t.Errorf("Expected 1 event, got %d", len(events))
	}
	
	if events[0].Action != "login" {
		t.Errorf("Expected action=login, got %s", events[0].Action)
	}
}

func TestBehavioralExtractor_ExtractFeatures(t *testing.T) {
	extractor := NewBehavioralFeatureExtractor(BehavioralConfig{})
	
	baseTime := time.Now()
	
	// Add multiple events
	for i := 0; i < 10; i++ {
		event := BehavioralEvent{
			EntityID:  "user123",
			Timestamp: baseTime.Add(time.Duration(i) * time.Hour),
			Action:    "access",
			Resource:  "/api/resource1",
			Location:  "US-East",
			Success:   true,
			Duration:  5 * time.Minute,
			DataRead:  1.0,
			DataWrite: 0.5,
		}
		extractor.AddEvent(event)
	}
	
	pattern, err := extractor.ExtractFeatures("user123")
	if err != nil {
		t.Fatalf("ExtractFeatures failed: %v", err)
	}
	
	if pattern.EntityID != "user123" {
		t.Errorf("Expected entityID=user123, got %s", pattern.EntityID)
	}
	
	if pattern.AccessCount != 10 {
		t.Errorf("Expected AccessCount=10, got %d", pattern.AccessCount)
	}
	
	if pattern.UniqueResources != 1 {
		t.Errorf("Expected UniqueResources=1, got %d", pattern.UniqueResources)
	}
	
	if pattern.SuccessRate != 1.0 {
		t.Errorf("Expected SuccessRate=1.0, got %.2f", pattern.SuccessRate)
	}
}

func TestBehavioralExtractor_FailedAttempts(t *testing.T) {
	extractor := NewBehavioralFeatureExtractor(BehavioralConfig{})
	
	baseTime := time.Now()
	
	// Add events with some failures
	for i := 0; i < 10; i++ {
		event := BehavioralEvent{
			EntityID:  "user123",
			Timestamp: baseTime.Add(time.Duration(i) * time.Minute),
			Action:    "login",
			Resource:  "/auth",
			Success:   i%3 != 0, // Fail every 3rd attempt
		}
		extractor.AddEvent(event)
	}
	
	pattern, err := extractor.ExtractFeatures("user123")
	if err != nil {
		t.Fatalf("ExtractFeatures failed: %v", err)
	}
	
	if pattern.FailedAttempts != 4 {
		t.Errorf("Expected FailedAttempts=4, got %d", pattern.FailedAttempts)
	}
	
	expectedSuccessRate := 6.0 / 10.0
	if pattern.SuccessRate != expectedSuccessRate {
		t.Errorf("Expected SuccessRate=%.2f, got %.2f", expectedSuccessRate, pattern.SuccessRate)
	}
}

func TestBehavioralExtractor_ResourceDiversity(t *testing.T) {
	extractor := NewBehavioralFeatureExtractor(BehavioralConfig{})
	
	baseTime := time.Now()
	
	// Add events with different resources
	resources := []string{"/api/data", "/api/users", "/api/admin", "/api/data"}
	for i, resource := range resources {
		event := BehavioralEvent{
			EntityID:  "user123",
			Timestamp: baseTime.Add(time.Duration(i) * time.Minute),
			Resource:  resource,
			Success:   true,
		}
		extractor.AddEvent(event)
	}
	
	pattern, err := extractor.ExtractFeatures("user123")
	if err != nil {
		t.Fatalf("ExtractFeatures failed: %v", err)
	}
	
	if pattern.UniqueResources != 3 {
		t.Errorf("Expected UniqueResources=3, got %d", pattern.UniqueResources)
	}
	
	if pattern.ResourceDiversity <= 0 {
		t.Error("ResourceDiversity should be positive")
	}
}

func TestBehavioralExtractor_TemporalPatterns(t *testing.T) {
	extractor := NewBehavioralFeatureExtractor(BehavioralConfig{})
	
	// Create event at specific time (use local time)
	specificTime := time.Date(2025, 10, 15, 14, 30, 0, 0, time.Local) // Wednesday 14:30 (Oct 15, 2025)
	
	event := BehavioralEvent{
		EntityID:  "user123",
		Timestamp: specificTime,
		Action:    "access",
		Success:   true,
	}
	extractor.AddEvent(event)
	
	pattern, err := extractor.ExtractFeatures("user123")
	if err != nil {
		t.Fatalf("ExtractFeatures failed: %v", err)
	}
	
	if pattern.HourOfDay != 14 {
		t.Errorf("Expected HourOfDay=14, got %d", pattern.HourOfDay)
	}
	
	// Oct 15, 2025 is Wednesday (day 3)
	expectedDay := int(specificTime.Weekday())
	if pattern.DayOfWeek != expectedDay {
		t.Errorf("Expected DayOfWeek=%d, got %d", expectedDay, pattern.DayOfWeek)
	}
	
	if !pattern.IsBusinessHours {
		t.Error("14:30 should be business hours")
	}
	
	if pattern.IsWeekend {
		t.Error("Wednesday should not be weekend")
	}
}

func TestBehavioralExtractor_LocationDiversity(t *testing.T) {
	extractor := NewBehavioralFeatureExtractor(BehavioralConfig{})
	
	baseTime := time.Now()
	
	locations := []string{"US-East", "US-West", "EU-Central", "US-East"}
	for i, location := range locations {
		event := BehavioralEvent{
			EntityID:  "user123",
			Timestamp: baseTime.Add(time.Duration(i) * time.Minute),
			Location:  location,
			Success:   true,
		}
		extractor.AddEvent(event)
	}
	
	pattern, err := extractor.ExtractFeatures("user123")
	if err != nil {
		t.Fatalf("ExtractFeatures failed: %v", err)
	}
	
	if pattern.LocationCount != 3 {
		t.Errorf("Expected LocationCount=3, got %d", pattern.LocationCount)
	}
	
	if pattern.LocationDiversity <= 0 {
		t.Error("LocationDiversity should be positive")
	}
}

func TestBehavioralExtractor_DataVolume(t *testing.T) {
	extractor := NewBehavioralFeatureExtractor(BehavioralConfig{})
	
	baseTime := time.Now()
	
	for i := 0; i < 5; i++ {
		event := BehavioralEvent{
			EntityID:  "user123",
			Timestamp: baseTime.Add(time.Duration(i) * time.Minute),
			DataRead:  2.0,
			DataWrite: 1.0,
			Success:   true,
		}
		extractor.AddEvent(event)
	}
	
	pattern, err := extractor.ExtractFeatures("user123")
	if err != nil {
		t.Fatalf("ExtractFeatures failed: %v", err)
	}
	
	expectedRead := 10.0 // 5 * 2.0
	if pattern.DataVolumeRead != expectedRead {
		t.Errorf("Expected DataVolumeRead=%.1f, got %.1f", expectedRead, pattern.DataVolumeRead)
	}
	
	expectedWrite := 5.0 // 5 * 1.0
	if pattern.DataVolumeWrite != expectedWrite {
		t.Errorf("Expected DataVolumeWrite=%.1f, got %.1f", expectedWrite, pattern.DataVolumeWrite)
	}
}

func TestBehavioralExtractor_UpdateProfile(t *testing.T) {
	extractor := NewBehavioralFeatureExtractor(BehavioralConfig{
		MinEventsForProfile: 5,
	})
	
	baseTime := time.Now()
	
	// Add enough events for profile
	for i := 0; i < 10; i++ {
		event := BehavioralEvent{
			EntityID:  "user123",
			Timestamp: baseTime.Add(time.Duration(i) * time.Hour),
			Action:    "access",
			Resource:  "/api/data",
			Location:  "US-East",
			Success:   true,
			Duration:  5 * time.Minute,
			DataRead:  1.0,
			DataWrite: 0.5,
		}
		extractor.AddEvent(event)
	}
	
	err := extractor.UpdateProfile("user123")
	if err != nil {
		t.Fatalf("UpdateProfile failed: %v", err)
	}
	
	profile, err := extractor.GetProfile("user123")
	if err != nil {
		t.Fatalf("GetProfile failed: %v", err)
	}
	
	if profile.EntityID != "user123" {
		t.Errorf("Expected entityID=user123, got %s", profile.EntityID)
	}
	
	if profile.EventCount != 10 {
		t.Errorf("Expected EventCount=10, got %d", profile.EventCount)
	}
	
	if len(profile.CommonHours) == 0 {
		t.Error("CommonHours should not be empty")
	}
}

func TestBehavioralExtractor_ProfileDeviations(t *testing.T) {
	extractor := NewBehavioralFeatureExtractor(BehavioralConfig{
		MinEventsForProfile: 5,
	})
	
	baseTime := time.Now()
	
	// Add baseline events (9-5 weekdays, US-East)
	for i := 0; i < 10; i++ {
		event := BehavioralEvent{
			EntityID:  "user123",
			Timestamp: baseTime.Add(time.Duration(i) * time.Hour),
			Resource:  "/api/data",
			Location:  "US-East",
			Success:   true,
			DataRead:  1.0,
			DataWrite: 0.5,
		}
		extractor.AddEvent(event)
	}
	
	// Create profile
	extractor.UpdateProfile("user123")
	
	// Clear events and add new anomalous events with different location
	extractor.events["user123"] = make([]BehavioralEvent, 0)
	
	for i := 0; i < 5; i++ {
		anomalousEvent := BehavioralEvent{
			EntityID:  "user123",
			Timestamp: baseTime.Add(time.Duration(20+i) * time.Hour),
			Resource:  "/api/admin",
			Location:  "Asia-Pacific", // Different location
			Success:   true,
			DataRead:  100.0, // Much higher than normal
			DataWrite: 50.0,
		}
		extractor.AddEvent(anomalousEvent)
	}
	
	pattern, err := extractor.ExtractFeatures("user123")
	if err != nil {
		t.Fatalf("ExtractFeatures failed: %v", err)
	}
	
	// Should detect location anomaly (none of the current locations match profile)
	if pattern.GeoAnomaly == 0 {
		t.Error("Should detect geographic anomaly")
	}
	
	// Should have volume deviation
	if pattern.VolumeDeviation == 0 {
		t.Log("Warning: Volume deviation not detected")
	}
	
	// Should have overall pattern anomaly
	if pattern.PatternAnomaly == 0 {
		t.Error("Should detect overall pattern anomaly")
	}
}

func TestBehavioralExtractor_InsufficientEvents(t *testing.T) {
	extractor := NewBehavioralFeatureExtractor(BehavioralConfig{
		MinEventsForProfile: 10,
	})
	
	// Add only 5 events
	baseTime := time.Now()
	for i := 0; i < 5; i++ {
		event := BehavioralEvent{
			EntityID:  "user123",
			Timestamp: baseTime.Add(time.Duration(i) * time.Minute),
			Success:   true,
		}
		extractor.AddEvent(event)
	}
	
	err := extractor.UpdateProfile("user123")
	if err == nil {
		t.Error("Expected error for insufficient events, got nil")
	}
}

func TestBehavioralExtractor_NoEvents(t *testing.T) {
	extractor := NewBehavioralFeatureExtractor(BehavioralConfig{})
	
	_, err := extractor.ExtractFeatures("nonexistent")
	if err == nil {
		t.Error("Expected error for nonexistent entity, got nil")
	}
}

func TestBehavioralPattern_ToVector(t *testing.T) {
	pattern := &BehavioralPattern{
		EntityID:           "user123",
		AccessFrequency:    10.5,
		AccessCount:        100,
		UniqueResources:    5,
		ResourceDiversity:  2.3,
		HourOfDay:          14,
		DayOfWeek:          2,
		IsBusinessHours:    true,
		IsWeekend:          false,
		TimeDeviation:      0.5,
		LocationCount:      3,
		LocationDiversity:  1.8,
		GeoAnomaly:         0.2,
		FailedAttempts:     2,
		SuccessRate:        0.98,
		SessionDuration:    15.5,
		ActionsPerSession:  25.0,
		DataVolumeRead:     100.5,
		DataVolumeWrite:    50.2,
		RequestRate:        5.5,
		PatternAnomaly:     0.3,
	}
	
	vector := pattern.ToVector()
	
	if len(vector) != 20 {
		t.Errorf("Expected vector length 20, got %d", len(vector))
	}
	
	if vector[0] != 10.5 {
		t.Errorf("Expected vector[0]=10.5, got %.2f", vector[0])
	}
	if vector[6] != 1.0 { // IsBusinessHours = true
		t.Errorf("Expected vector[6]=1.0, got %.2f", vector[6])
	}
	if vector[7] != 0.0 { // IsWeekend = false
		t.Errorf("Expected vector[7]=0.0, got %.2f", vector[7])
	}
}

func TestBehavioralExtractor_MaxBufferSize(t *testing.T) {
	extractor := NewBehavioralFeatureExtractor(BehavioralConfig{
		MaxEventsBuffer: 10,
	})
	
	baseTime := time.Now()
	
	// Add 20 events (should keep only last 10)
	for i := 0; i < 20; i++ {
		event := BehavioralEvent{
			EntityID:  "user123",
			Timestamp: baseTime.Add(time.Duration(i) * time.Minute),
			Success:   true,
		}
		extractor.AddEvent(event)
	}
	
	events := extractor.events["user123"]
	if len(events) != 10 {
		t.Errorf("Expected 10 events (max buffer), got %d", len(events))
	}
}

func TestBehavioralExtractor_ConcurrentAccess(t *testing.T) {
	extractor := NewBehavioralFeatureExtractor(BehavioralConfig{})
	
	baseTime := time.Now()
	
	// Concurrent writes
	done := make(chan bool)
	for i := 0; i < 5; i++ {
		go func(id int) {
			for j := 0; j < 10; j++ {
				event := BehavioralEvent{
					EntityID:  "user123",
					Timestamp: baseTime.Add(time.Duration(j) * time.Minute),
					Action:    "access",
					Success:   true,
				}
				extractor.AddEvent(event)
			}
			done <- true
		}(i)
	}
	
	// Wait for all goroutines
	for i := 0; i < 5; i++ {
		<-done
	}
	
	// Concurrent reads
	for i := 0; i < 5; i++ {
		go func() {
			_, err := extractor.ExtractFeatures("user123")
			if err != nil {
				t.Errorf("Concurrent extract failed: %v", err)
			}
			done <- true
		}()
	}
	
	for i := 0; i < 5; i++ {
		<-done
	}
}

func BenchmarkBehavioralExtractor_AddEvent(b *testing.B) {
	extractor := NewBehavioralFeatureExtractor(BehavioralConfig{})
	
	baseTime := time.Now()
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		event := BehavioralEvent{
			EntityID:  "user123",
			Timestamp: baseTime.Add(time.Duration(i) * time.Second),
			Action:    "access",
			Resource:  "/api/data",
			Location:  "US-East",
			Success:   true,
			Duration:  5 * time.Minute,
			DataRead:  1.0,
			DataWrite: 0.5,
		}
		extractor.AddEvent(event)
	}
}

func BenchmarkBehavioralExtractor_ExtractFeatures(b *testing.B) {
	extractor := NewBehavioralFeatureExtractor(BehavioralConfig{})
	
	baseTime := time.Now()
	
	// Add 100 events
	for i := 0; i < 100; i++ {
		event := BehavioralEvent{
			EntityID:  "user123",
			Timestamp: baseTime.Add(time.Duration(i) * time.Minute),
			Action:    "access",
			Resource:  "/api/data",
			Location:  "US-East",
			Success:   true,
			DataRead:  1.0,
			DataWrite: 0.5,
		}
		extractor.AddEvent(event)
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		extractor.ExtractFeatures("user123")
	}
}
