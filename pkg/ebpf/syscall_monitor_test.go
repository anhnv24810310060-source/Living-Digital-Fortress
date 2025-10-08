package ebpf

import (
	"testing"
	"time"
)

// TestSyscallMonitor tests basic syscall monitoring functionality
func TestSyscallMonitor(t *testing.T) {
	monitor := NewSyscallMonitor(1234, 1024)

	if monitor == nil {
		t.Fatal("Failed to create syscall monitor")
	}

	if err := monitor.Start(); err != nil {
		t.Fatalf("Failed to start monitor: %v", err)
	}
	defer monitor.Stop()

	// Let it capture some events
	time.Sleep(500 * time.Millisecond)

	features := monitor.ExtractFeatures()

	if features == nil {
		t.Fatal("Failed to extract features")
	}

	if features.EventCount == 0 {
		t.Error("Expected some events to be captured")
	}

	if features.EventsPerSecond <= 0 {
		t.Error("Expected positive events per second")
	}

	t.Logf("Captured %d events (%.2f events/sec)",
		features.EventCount, features.EventsPerSecond)
}

// TestThreatFeatures tests feature extraction
func TestThreatFeatures(t *testing.T) {
	monitor := MockSyscallMonitor(5678, 50) // 50 dangerous syscalls

	features := monitor.ExtractFeatures()

	if features.DangerousSyscalls != 50 {
		t.Errorf("Expected 50 dangerous syscalls, got %d",
			features.DangerousSyscalls)
	}

	if features.ShellExecution != 5 {
		t.Errorf("Expected 5 shell executions, got %d",
			features.ShellExecution)
	}
}

// TestCalculateThreatScore tests threat scoring
func TestCalculateThreatScore(t *testing.T) {
	tests := []struct {
		name           string
		dangerousCalls uint64
		expectedRange  [2]int // [min, max]
	}{
		{
			name:           "low_threat",
			dangerousCalls: 5,
			expectedRange:  [2]int{0, 30},
		},
		{
			name:           "medium_threat",
			dangerousCalls: 20,
			expectedRange:  [2]int{30, 70},
		},
		{
			name:           "high_threat",
			dangerousCalls: 50,
			expectedRange:  [2]int{70, 100},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			monitor := MockSyscallMonitor(1234, tt.dangerousCalls)
			score := monitor.CalculateThreatScore()

			if score < tt.expectedRange[0] || score > tt.expectedRange[1] {
				t.Errorf("Score %d not in expected range [%d, %d]",
					score, tt.expectedRange[0], tt.expectedRange[1])
			}

			t.Logf("%s: score = %d", tt.name, score)
		})
	}
}

// TestPatternDetection tests unusual pattern detection
func TestPatternDetection(t *testing.T) {
	monitor := NewSyscallMonitor(1234, 1024)

	// Inject ptrace + mmap pattern (should be detected)
	monitor.patternMu.Lock()
	monitor.recentSyscalls[0] = "ptrace"
	monitor.recentSyscalls[1] = "mmap"
	monitor.recentSyscalls[2] = "write"
	monitor.patternMu.Unlock()

	features := monitor.ExtractFeatures()

	if features.UnusualPatterns == 0 {
		t.Error("Expected ptrace+mmap pattern to be detected")
	}

	t.Logf("Detected %d unusual patterns", features.UnusualPatterns)
}

// TestGetMetrics tests metrics collection
func TestGetMetrics(t *testing.T) {
	monitor := MockSyscallMonitor(1234, 30)

	metrics := monitor.GetMetrics()

	requiredMetrics := []string{
		"ebpf_syscall_total",
		"ebpf_dangerous_syscalls",
		"ebpf_network_calls",
		"ebpf_file_operations",
		"ebpf_process_operations",
		"ebpf_shell_executions",
	}

	for _, key := range requiredMetrics {
		if _, ok := metrics[key]; !ok {
			t.Errorf("Missing required metric: %s", key)
		}
	}

	t.Logf("Metrics: %+v", metrics)
}

// BenchmarkSyscallCapture benchmarks syscall capture performance
func BenchmarkSyscallCapture(b *testing.B) {
	monitor := NewSyscallMonitor(1234, 8192)
	monitor.Start()
	defer monitor.Stop()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		monitor.simulateCapture()
	}
}

// BenchmarkFeatureExtraction benchmarks feature extraction
func BenchmarkFeatureExtraction(b *testing.B) {
	monitor := MockSyscallMonitor(1234, 50)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = monitor.ExtractFeatures()
	}
}

// BenchmarkThreatScoring benchmarks threat score calculation
func BenchmarkThreatScoring(b *testing.B) {
	monitor := MockSyscallMonitor(1234, 30)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = monitor.CalculateThreatScore()
	}
}
