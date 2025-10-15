package ml

import (
	"math"
	"testing"
	"time"
)

func TestNewDriftDetector(t *testing.T) {
	detector := NewDriftDetector(DriftConfig{
		Name:   "test",
		Method: "ks",
	})
	
	if detector == nil {
		t.Fatal("NewDriftDetector returned nil")
	}
	
	if detector.method != "ks" {
		t.Errorf("Method = %s, want ks", detector.method)
	}
	
	if detector.threshold != 0.05 {
		t.Errorf("Default threshold = %f, want 0.05", detector.threshold)
	}
}

func TestDriftDetector_SetReference(t *testing.T) {
	detector := NewDriftDetector(DriftConfig{Name: "test"})
	
	reference := []float64{1, 2, 3, 4, 5}
	err := detector.SetReference(reference)
	
	if err != nil {
		t.Fatalf("SetReference failed: %v", err)
	}
	
	if len(detector.reference) != 5 {
		t.Errorf("Reference length = %d, want 5", len(detector.reference))
	}
	
	// Test empty reference
	err = detector.SetReference([]float64{})
	if err == nil {
		t.Error("Should fail with empty reference")
	}
}

func TestDriftDetector_AddObservation(t *testing.T) {
	detector := NewDriftDetector(DriftConfig{
		Name:       "test",
		WindowSize: 10,
	})
	
	// Add observations
	for i := 0; i < 15; i++ {
		detector.AddObservation(float64(i))
	}
	
	// Should maintain window size
	if len(detector.current) != 10 {
		t.Errorf("Current length = %d, want 10", len(detector.current))
	}
}

func TestDriftDetector_KSTest_NoDrift(t *testing.T) {
	detector := NewDriftDetector(DriftConfig{
		Name:      "test",
		Method:    "ks",
		Threshold: 0.1,
	})
	
	// Set reference: normal distribution
	reference := make([]float64, 100)
	for i := 0; i < 100; i++ {
		reference[i] = float64(i)
	}
	detector.SetReference(reference)
	
	// Add similar current data
	for i := 0; i < 100; i++ {
		detector.AddObservation(float64(i) + 0.1)
	}
	
	event, err := detector.CheckDrift()
	if err != nil {
		t.Fatalf("CheckDrift failed: %v", err)
	}
	
	// Should not detect drift
	if event.Detected {
		t.Errorf("Should not detect drift, score = %f", event.Score)
	}
}

func TestDriftDetector_KSTest_WithDrift(t *testing.T) {
	detector := NewDriftDetector(DriftConfig{
		Name:      "test",
		Method:    "ks",
		Threshold: 0.1,
	})
	
	// Set reference: 0-99
	reference := make([]float64, 100)
	for i := 0; i < 100; i++ {
		reference[i] = float64(i)
	}
	detector.SetReference(reference)
	
	// Add drifted current data: 100-199
	for i := 0; i < 100; i++ {
		detector.AddObservation(float64(i + 100))
	}
	
	event, err := detector.CheckDrift()
	if err != nil {
		t.Fatalf("CheckDrift failed: %v", err)
	}
	
	// Should detect drift
	if !event.Detected {
		t.Errorf("Should detect drift, score = %f", event.Score)
	}
	
	if event.Method != "ks" {
		t.Errorf("Method = %s, want ks", event.Method)
	}
}

func TestDriftDetector_PSITest(t *testing.T) {
	detector := NewDriftDetector(DriftConfig{
		Name:      "test",
		Method:    "psi",
		Threshold: 0.1,
	})
	
	// Set reference
	reference := make([]float64, 100)
	for i := 0; i < 100; i++ {
		reference[i] = float64(i)
	}
	detector.SetReference(reference)
	
	// Add similar data
	for i := 0; i < 100; i++ {
		detector.AddObservation(float64(i))
	}
	
	event, err := detector.CheckDrift()
	if err != nil {
		t.Fatalf("CheckDrift failed: %v", err)
	}
	
	if event.Method != "psi" {
		t.Errorf("Method = %s, want psi", event.Method)
	}
}

func TestDriftDetector_WassersteinDistance(t *testing.T) {
	detector := NewDriftDetector(DriftConfig{
		Name:      "test",
		Method:    "wasserstein",
		Threshold: 10.0,
	})
	
	// Set reference
	reference := make([]float64, 50)
	for i := 0; i < 50; i++ {
		reference[i] = float64(i)
	}
	detector.SetReference(reference)
	
	// Add similar data
	for i := 0; i < 50; i++ {
		detector.AddObservation(float64(i + 1))
	}
	
	event, err := detector.CheckDrift()
	if err != nil {
		t.Fatalf("CheckDrift failed: %v", err)
	}
	
	if event.Method != "wasserstein" {
		t.Errorf("Method = %s, want wasserstein", event.Method)
	}
}

func TestDriftDetector_InsufficientData(t *testing.T) {
	detector := NewDriftDetector(DriftConfig{Name: "test"})
	
	detector.SetReference([]float64{1, 2, 3, 4, 5})
	
	// Add too few observations
	for i := 0; i < 10; i++ {
		detector.AddObservation(float64(i))
	}
	
	_, err := detector.CheckDrift()
	if err == nil {
		t.Error("Should fail with insufficient data")
	}
}

func TestDriftDetector_NoReference(t *testing.T) {
	detector := NewDriftDetector(DriftConfig{Name: "test"})
	
	for i := 0; i < 50; i++ {
		detector.AddObservation(float64(i))
	}
	
	_, err := detector.CheckDrift()
	if err == nil {
		t.Error("Should fail without reference")
	}
}

func TestDriftDetector_Severity(t *testing.T) {
	detector := NewDriftDetector(DriftConfig{
		Name:      "test",
		Method:    "ks",
		Threshold: 0.1,
	})
	
	reference := make([]float64, 100)
	for i := 0; i < 100; i++ {
		reference[i] = float64(i)
	}
	detector.SetReference(reference)
	
	// Add highly drifted data
	for i := 0; i < 100; i++ {
		detector.AddObservation(float64(i + 200))
	}
	
	event, _ := detector.CheckDrift()
	
	// High drift should have high severity
	if event.Severity != "high" {
		t.Errorf("Severity = %s, want high for large drift", event.Severity)
	}
}

func TestDriftDetector_GetDriftHistory(t *testing.T) {
	detector := NewDriftDetector(DriftConfig{Name: "test"})
	
	reference := make([]float64, 50)
	for i := 0; i < 50; i++ {
		reference[i] = float64(i)
	}
	detector.SetReference(reference)
	
	// Run checks multiple times
	for run := 0; run < 3; run++ {
		for i := 0; i < 50; i++ {
			detector.AddObservation(float64(i))
		}
		detector.CheckDrift()
	}
	
	history := detector.GetDriftHistory()
	if len(history) != 3 {
		t.Errorf("History length = %d, want 3", len(history))
	}
}

func TestNewModelMonitor(t *testing.T) {
	monitor := NewModelMonitor(MonitorConfig{
		ModelName:    "test-model",
		ModelVersion: "v1.0",
	})
	
	if monitor == nil {
		t.Fatal("NewModelMonitor returned nil")
	}
	
	if monitor.modelName != "test-model" {
		t.Errorf("ModelName = %s, want test-model", monitor.modelName)
	}
}

func TestModelMonitor_RecordMetrics(t *testing.T) {
	monitor := NewModelMonitor(MonitorConfig{
		ModelName: "test",
	})
	
	monitor.RecordMetrics(0.95, 0.92, 0.93, 0.925, 1000, 10*time.Millisecond)
	
	metrics := monitor.GetMetrics()
	
	if math.Abs(metrics["accuracy"]-0.95) > 0.01 {
		t.Errorf("Accuracy = %f, want 0.95", metrics["accuracy"])
	}
	
	if math.Abs(metrics["precision"]-0.92) > 0.01 {
		t.Errorf("Precision = %f, want 0.92", metrics["precision"])
	}
}

func TestModelMonitor_GetMetricHistory(t *testing.T) {
	monitor := NewModelMonitor(MonitorConfig{
		ModelName: "test",
	})
	
	// Record multiple times
	for i := 0; i < 5; i++ {
		monitor.RecordMetrics(0.95, 0.92, 0.93, 0.925, 1000, 10*time.Millisecond)
	}
	
	history := monitor.GetMetricHistory()
	if len(history) != 5 {
		t.Errorf("History length = %d, want 5", len(history))
	}
}

func TestModelMonitor_PerformanceDegradation(t *testing.T) {
	monitor := NewModelMonitor(MonitorConfig{
		ModelName:      "test",
		AlertThreshold: 0.1,
	})
	
	// Record baseline good performance
	for i := 0; i < 10; i++ {
		monitor.RecordMetrics(0.95, 0.92, 0.93, 0.925, 1000, 10*time.Millisecond)
	}
	
	// Record degraded performance
	monitor.RecordMetrics(0.80, 0.80, 0.80, 0.80, 1000, 10*time.Millisecond)
	
	alerts := monitor.GetAlerts()
	if len(alerts) == 0 {
		t.Error("Should generate alert for performance degradation")
	}
	
	if alerts[0].Type != "performance" {
		t.Errorf("Alert type = %s, want performance", alerts[0].Type)
	}
}

func TestModelMonitor_CriticalAlert(t *testing.T) {
	monitor := NewModelMonitor(MonitorConfig{
		ModelName:      "test",
		AlertThreshold: 0.05,
	})
	
	// Baseline
	for i := 0; i < 10; i++ {
		monitor.RecordMetrics(0.95, 0.92, 0.93, 0.925, 1000, 10*time.Millisecond)
	}
	
	// Severe degradation
	monitor.RecordMetrics(0.70, 0.70, 0.70, 0.70, 1000, 10*time.Millisecond)
	
	alerts := monitor.GetAlerts()
	if len(alerts) == 0 {
		t.Fatal("Should generate alert")
	}
	
	if alerts[0].Severity != "critical" {
		t.Errorf("Alert severity = %s, want critical for severe degradation", alerts[0].Severity)
	}
}

func TestModelMonitor_ResolveAlert(t *testing.T) {
	monitor := NewModelMonitor(MonitorConfig{
		ModelName:      "test",
		AlertThreshold: 0.1,
	})
	
	// Generate alert
	for i := 0; i < 10; i++ {
		monitor.RecordMetrics(0.95, 0.92, 0.93, 0.925, 1000, 10*time.Millisecond)
	}
	monitor.RecordMetrics(0.80, 0.80, 0.80, 0.80, 1000, 10*time.Millisecond)
	
	alerts := monitor.GetAlerts()
	if len(alerts) == 0 {
		t.Fatal("Should have alert")
	}
	
	alertID := alerts[0].ID
	err := monitor.ResolveAlert(alertID)
	if err != nil {
		t.Fatalf("ResolveAlert failed: %v", err)
	}
	
	// Should not appear in active alerts
	activeAlerts := monitor.GetAlerts()
	if len(activeAlerts) != 0 {
		t.Error("Resolved alert should not appear in active alerts")
	}
}

func TestModelMonitor_ResolveInvalidAlert(t *testing.T) {
	monitor := NewModelMonitor(MonitorConfig{
		ModelName: "test",
	})
	
	err := monitor.ResolveAlert("invalid-id")
	if err == nil {
		t.Error("Should fail for invalid alert ID")
	}
}

func TestModelMonitor_GetDriftDetectors(t *testing.T) {
	monitor := NewModelMonitor(MonitorConfig{
		ModelName:   "test",
		DriftMethod: "psi",
	})
	
	inputDrift := monitor.GetInputDrift()
	if inputDrift == nil {
		t.Error("InputDrift should not be nil")
	}
	if inputDrift.method != "psi" {
		t.Errorf("InputDrift method = %s, want psi", inputDrift.method)
	}
	
	outputDrift := monitor.GetOutputDrift()
	if outputDrift == nil {
		t.Error("OutputDrift should not be nil")
	}
	
	perfDrift := monitor.GetPerformanceDrift()
	if perfDrift == nil {
		t.Error("PerformanceDrift should not be nil")
	}
}

func TestDriftDetector_UnknownMethod(t *testing.T) {
	detector := NewDriftDetector(DriftConfig{
		Name:   "test",
		Method: "unknown",
	})
	
	reference := make([]float64, 50)
	for i := 0; i < 50; i++ {
		reference[i] = float64(i)
	}
	detector.SetReference(reference)
	
	for i := 0; i < 50; i++ {
		detector.AddObservation(float64(i))
	}
	
	_, err := detector.CheckDrift()
	if err == nil {
		t.Error("Should fail with unknown method")
	}
}

func TestMetricSnapshot_Fields(t *testing.T) {
	monitor := NewModelMonitor(MonitorConfig{
		ModelName: "test",
	})
	
	latency := 50 * time.Millisecond
	monitor.RecordMetrics(0.95, 0.92, 0.93, 0.925, 1000, latency)
	
	history := monitor.GetMetricHistory()
	if len(history) == 0 {
		t.Fatal("Should have history")
	}
	
	snapshot := history[0]
	if snapshot.Predictions != 1000 {
		t.Errorf("Predictions = %d, want 1000", snapshot.Predictions)
	}
	
	if snapshot.Latency != latency {
		t.Errorf("Latency = %v, want %v", snapshot.Latency, latency)
	}
}

func TestDriftDetector_ConstantData(t *testing.T) {
	detector := NewDriftDetector(DriftConfig{
		Name:   "test",
		Method: "psi",
	})
	
	// All same values
	reference := make([]float64, 50)
	for i := 0; i < 50; i++ {
		reference[i] = 5.0
	}
	detector.SetReference(reference)
	
	for i := 0; i < 50; i++ {
		detector.AddObservation(5.0)
	}
	
	event, err := detector.CheckDrift()
	if err != nil {
		t.Fatalf("CheckDrift failed: %v", err)
	}
	
	// No drift for identical distributions
	if event.Detected {
		t.Error("Should not detect drift for identical constant distributions")
	}
}

func BenchmarkDriftDetector_KSTest(b *testing.B) {
	detector := NewDriftDetector(DriftConfig{
		Name:   "test",
		Method: "ks",
	})
	
	reference := make([]float64, 1000)
	for i := 0; i < 1000; i++ {
		reference[i] = float64(i)
	}
	detector.SetReference(reference)
	
	for i := 0; i < 1000; i++ {
		detector.AddObservation(float64(i))
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		detector.CheckDrift()
	}
}

func BenchmarkModelMonitor_RecordMetrics(b *testing.B) {
	monitor := NewModelMonitor(MonitorConfig{
		ModelName: "test",
	})
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		monitor.RecordMetrics(0.95, 0.92, 0.93, 0.925, 1000, 10*time.Millisecond)
	}
}
