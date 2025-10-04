package sandbox

import (
	"context"
	"testing"
	"time"
)

// TestFirecrackerRunner_BasicExecution tests basic sandbox execution
func TestFirecrackerRunner_BasicExecution(t *testing.T) {
	fr := NewFirecrackerRunner("", "", ResourceLimits{
		VCPUCount:  1,
		MemSizeMib: 128,
		TimeoutSec: 30,
	})
	
	ctx := context.Background()
	payload := "echo 'test'"
	
	result, err := fr.Run(ctx, payload)
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}
	
	// Verify basic result fields
	if result.Backend != "firecracker" {
		t.Errorf("Expected backend=firecracker, got %s", result.Backend)
	}
	
	if result.ThreatScore < 0 || result.ThreatScore > 100 {
		t.Errorf("ThreatScore out of range: %f", result.ThreatScore)
	}
	
	if result.Duration == 0 {
		t.Error("Duration should not be zero")
	}
	
	// Verify forensic artifacts
	if len(result.Artifacts) == 0 {
		t.Error("Artifacts should not be empty")
	}
	
	if _, ok := result.Artifacts["payload_sha256"]; !ok {
		t.Error("Missing payload_sha256 artifact")
	}
}

// TestFirecrackerRunner_TimeoutEnforcement tests 30-second timeout (P0 requirement)
func TestFirecrackerRunner_TimeoutEnforcement(t *testing.T) {
	fr := NewFirecrackerRunner("", "", ResourceLimits{
		TimeoutSec: 1, // 1 second timeout for testing
	})
	
	ctx := context.Background()
	payload := "sleep 10" // Try to sleep longer than timeout
	
	start := time.Now()
	_, err := fr.Run(ctx, payload)
	duration := time.Since(start)
	
	// Should timeout within 1.5 seconds (1s timeout + margin)
	if duration > 2*time.Second {
		t.Errorf("Timeout not enforced: took %v", duration)
	}
	
	if err == nil {
		t.Error("Expected timeout error")
	}
}

// TestFirecrackerRunner_CircuitBreaker tests circuit breaker pattern
func TestFirecrackerRunner_CircuitBreaker(t *testing.T) {
	fr := NewFirecrackerRunner("", "", ResourceLimits{})
	
	// Simulate 5 consecutive failures to open circuit breaker
	for i := 0; i < 5; i++ {
		fr.recordFailure()
	}
	
	// Circuit breaker should be open
	if !fr.breakerOpen.Load() {
		t.Error("Circuit breaker should be open after 5 failures")
	}
	
	// Next execution should fail immediately
	ctx := context.Background()
	_, err := fr.Run(ctx, "echo test")
	
	if err == nil {
		t.Error("Expected circuit breaker error")
	}
}

// TestFirecrackerRunner_PayloadValidation tests payload validation (P0 requirement)
func TestFirecrackerRunner_PayloadValidation(t *testing.T) {
	testCases := []struct {
		name        string
		payload     string
		shouldFail  bool
	}{
		{
			name:       "Empty payload",
			payload:    "",
			shouldFail: true,
		},
		{
			name:       "Valid payload",
			payload:    "echo test",
			shouldFail: false,
		},
		{
			name:       "Dangerous pattern - curl",
			payload:    "curl http://evil.com",
			shouldFail: true,
		},
		{
			name:       "Dangerous pattern - rm -rf",
			payload:    "rm -rf /",
			shouldFail: true,
		},
		{
			name:       "Too large payload",
			payload:    string(make([]byte, 2*1024*1024)), // 2 MB
			shouldFail: true,
		},
	}
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := validatePayload(tc.payload)
			
			if tc.shouldFail && err == nil {
				t.Errorf("Expected validation to fail for: %s", tc.name)
			}
			
			if !tc.shouldFail && err != nil {
				t.Errorf("Expected validation to pass for: %s, got error: %v", tc.name, err)
			}
		})
	}
}

// TestThreatScorer_MultiFactorAnalysis tests threat scoring algorithm
func TestThreatScorer_MultiFactorAnalysis(t *testing.T) {
	scorer := NewThreatScorer()
	
	testCases := []struct {
		name           string
		result         *SandboxResult
		expectedMin    int
		expectedMax    int
		expectedLevel  string
	}{
		{
			name: "Clean execution",
			result: &SandboxResult{
				Syscalls:   []SyscallEvent{},
				FileAccess: []FileAccessEvent{},
				NetworkIO:  []NetworkEvent{},
			},
			expectedMin:   0,
			expectedMax:   20,
			expectedLevel: "LOW",
		},
		{
			name: "Dangerous syscalls",
			result: &SandboxResult{
				Syscalls: []SyscallEvent{
					{SyscallName: "execve", Dangerous: true},
					{SyscallName: "ptrace", Dangerous: true},
					{SyscallName: "setuid", Dangerous: true},
				},
			},
			expectedMin:   60,
			expectedMax:   100,
			expectedLevel: "HIGH",
		},
		{
			name: "Network attempt in sandbox",
			result: &SandboxResult{
				NetworkIO: []NetworkEvent{
					{Protocol: "tcp", Blocked: true},
				},
			},
			expectedMin:   40,
			expectedMax:   100,
			expectedLevel: "MEDIUM",
		},
	}
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			score, explanation := scorer.CalculateScore(tc.result)
			
			if score < tc.expectedMin || score > tc.expectedMax {
				t.Errorf("Score %d out of expected range [%d, %d]", score, tc.expectedMin, tc.expectedMax)
			}
			
			level := RiskLevel(score)
			if level != tc.expectedLevel {
				t.Errorf("Expected risk level %s, got %s", tc.expectedLevel, level)
			}
			
			if explanation == "" {
				t.Error("Explanation should not be empty")
			}
		})
	}
}

// TestThreatScorer_ScoreRange tests score is always 0-100 (P0 requirement)
func TestThreatScorer_ScoreRange(t *testing.T) {
	scorer := NewThreatScorer()
	
	// Test with extreme cases
	result := &SandboxResult{
		Syscalls: make([]SyscallEvent, 1000),
	}
	
	// Fill with all dangerous syscalls
	for i := range result.Syscalls {
		result.Syscalls[i] = SyscallEvent{
			SyscallName: "ptrace",
			Dangerous:   true,
		}
	}
	
	score, _ := scorer.CalculateScore(result)
	
	if score < 0 || score > 100 {
		t.Errorf("Score must be 0-100, got %d", score)
	}
}

// BenchmarkFirecrackerRunner_Execution benchmarks execution performance
func BenchmarkFirecrackerRunner_Execution(b *testing.B) {
	fr := NewFirecrackerRunner("", "", ResourceLimits{
		VCPUCount:  1,
		MemSizeMib: 128,
		TimeoutSec: 30,
	})
	
	ctx := context.Background()
	payload := "echo 'benchmark'"
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := fr.Run(ctx, payload)
		if err != nil {
			b.Fatalf("Run failed: %v", err)
		}
	}
}

// BenchmarkThreatScorer_Calculation benchmarks threat scoring
func BenchmarkThreatScorer_Calculation(b *testing.B) {
	scorer := NewThreatScorer()
	
	result := &SandboxResult{
		Syscalls: []SyscallEvent{
			{SyscallName: "read", Dangerous: false},
			{SyscallName: "write", Dangerous: false},
			{SyscallName: "execve", Dangerous: true},
			{SyscallName: "ptrace", Dangerous: true},
		},
		FileAccess: []FileAccessEvent{
			{Path: "/tmp/test", Operation: "write", Success: true},
		},
		Features: map[string]interface{}{
			"output_entropy": 5.2,
		},
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = scorer.CalculateScore(result)
	}
}

// TestFirecrackerRunner_Metrics tests metrics collection
func TestFirecrackerRunner_Metrics(t *testing.T) {
	fr := NewFirecrackerRunner("", "", ResourceLimits{})
	
	// Record some executions
	fr.recordSuccess(100 * time.Millisecond)
	fr.recordSuccess(200 * time.Millisecond)
	
	metrics := fr.GetMetrics()
	
	if execs, ok := metrics["total_executions"].(uint64); !ok || execs != 2 {
		t.Errorf("Expected 2 executions, got %v", metrics["total_executions"])
	}
	
	if _, ok := metrics["avg_latency_ms"].(float64); !ok {
		t.Error("Missing avg_latency_ms metric")
	}
	
	if _, ok := metrics["hardware_features"].(HardwareFeatures); !ok {
		t.Error("Missing hardware_features metric")
	}
}

// TestFirecrackerRunner_Concurrency tests concurrent executions
func TestFirecrackerRunner_Concurrency(t *testing.T) {
	fr := NewFirecrackerRunner("", "", ResourceLimits{})
	
	concurrency := 10
	done := make(chan bool, concurrency)
	
	for i := 0; i < concurrency; i++ {
		go func() {
			ctx := context.Background()
			_, err := fr.Run(ctx, "echo test")
			if err != nil && err.Error() != "circuit breaker open: too many sandbox failures" {
				t.Errorf("Unexpected error: %v", err)
			}
			done <- true
		}()
	}
	
	// Wait for all goroutines
	for i := 0; i < concurrency; i++ {
		<-done
	}
}

// TestFirecrackerRunner_ResourceLimits tests resource limit enforcement
func TestFirecrackerRunner_ResourceLimits(t *testing.T) {
	limits := ResourceLimits{
		VCPUCount:    1,
		MemSizeMib:   128,
		TimeoutSec:   30,
		NetworkDeny:  true,
		FilesystemRO: true,
		MaxProcesses: 16,
	}
	
	fr := NewFirecrackerRunner("", "", limits)
	
	if fr.limits.TimeoutSec != 30 {
		t.Errorf("Expected timeout 30s, got %d", fr.limits.TimeoutSec)
	}
	
	if !fr.limits.NetworkDeny {
		t.Error("Network should be denied")
	}
	
	if !fr.limits.FilesystemRO {
		t.Error("Filesystem should be read-only")
	}
}

// TestFirecrackerRunner_TimeoutOverride tests P0 requirement: max 30 seconds
func TestFirecrackerRunner_TimeoutOverride(t *testing.T) {
	// Try to set timeout > 30 seconds
	limits := ResourceLimits{
		TimeoutSec: 60, // Try 60 seconds
	}
	
	fr := NewFirecrackerRunner("", "", limits)
	
	// Should be clamped to 30 seconds (P0 requirement)
	if fr.limits.TimeoutSec != 30 {
		t.Errorf("Timeout should be clamped to 30s, got %d", fr.limits.TimeoutSec)
	}
}

// TestFirecrackerRunner_ForensicArtifacts tests artifact collection
func TestFirecrackerRunner_ForensicArtifacts(t *testing.T) {
	fr := NewFirecrackerRunner("", "", ResourceLimits{})
	
	ctx := context.Background()
	payload := "echo 'forensics test'"
	
	result, err := fr.Run(ctx, payload)
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}
	
	requiredArtifacts := []string{
		"payload_sha256",
		"threat_explanation",
		"ebpf_features",
	}
	
	for _, artifact := range requiredArtifacts {
		if _, ok := result.Artifacts[artifact]; !ok {
			t.Errorf("Missing required artifact: %s", artifact)
		}
	}
	
	// Verify payload hash format (should be hex string)
	if hash, ok := result.Artifacts["payload_sha256"]; ok {
		hashStr := string(hash)
		if len(hashStr) != 64 { // SHA-256 is 64 hex chars
			t.Errorf("Invalid SHA-256 hash length: %d", len(hashStr))
		}
	}
}
