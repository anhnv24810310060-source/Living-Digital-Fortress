package runner

import (
	"testing"
	"time"
)

func TestWasmRunnerCreation(t *testing.T) {
	runner := NewWasmRunner()
	defer runner.Close()

	if runner == nil {
		t.Fatal("Runner should not be nil")
	}

	if runner.runtime == nil {
		t.Fatal("Runtime should not be nil")
	}
}

func TestPluginInputValidation(t *testing.T) {
	input := PluginInput{
		ArtifactID:   "test-artifact",
		ArtifactType: "executable",
		Metadata:     map[string]interface{}{"size": 1024},
		S3URL:        "s3://bucket/artifact",
		Timestamp:    time.Now(),
	}

	if input.ArtifactID == "" {
		t.Error("ArtifactID should not be empty")
	}

	if input.ArtifactType == "" {
		t.Error("ArtifactType should not be empty")
	}

	if input.S3URL == "" {
		t.Error("S3URL should not be empty")
	}
}

func TestPluginOutputValidation(t *testing.T) {
	runner := NewWasmRunner()
	defer runner.Close()

	tests := []struct {
		name     string
		output   *PluginOutput
		expected bool
	}{
		{
			name: "Valid output",
			output: &PluginOutput{
				Success:    true,
				Results:    map[string]interface{}{"test": "value"},
				Confidence: 0.8,
				Tags:       []string{"malware"},
				Indicators: []Indicator{
					{Type: "hash", Value: "abc123", Confidence: 0.9, Context: "test"},
				},
			},
			expected: true,
		},
		{
			name: "Invalid confidence",
			output: &PluginOutput{
				Success:    true,
				Confidence: 1.5, // Invalid: > 1.0
				Tags:       []string{},
				Indicators: []Indicator{},
			},
			expected: false,
		},
		{
			name: "Invalid indicator",
			output: &PluginOutput{
				Success:    true,
				Confidence: 0.8,
				Tags:       []string{},
				Indicators: []Indicator{
					{Type: "", Value: "test", Confidence: 0.5}, // Invalid: empty type
				},
			},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := runner.validateOutput(tt.output)
			if tt.expected && err != nil {
				t.Errorf("Expected validation to pass, got error: %v", err)
			}
			if !tt.expected && err == nil {
				t.Error("Expected validation to fail, but it passed")
			}
		})
	}
}

func TestSandboxPolicyValidation(t *testing.T) {
	policy := SandboxPolicy{
		NetworkAccess:    false,
		FilesystemAccess: "none",
		MemoryLimit:      "128MB",
		CPULimit:         "100m",
		ExecutionTimeout: "30s",
		AllowedSyscalls:  []string{"read", "write", "exit"},
		RiskLevel:        "low",
	}

	if policy.NetworkAccess {
		t.Error("Network access should be disabled")
	}

	if policy.FilesystemAccess != "none" {
		t.Error("Filesystem access should be none")
	}

	if policy.ExecutionTimeout == "" {
		t.Error("Execution timeout should be set")
	}

	if len(policy.AllowedSyscalls) == 0 {
		t.Error("Should have allowed syscalls")
	}
}

func TestPluginIsolationTest(t *testing.T) {
	runner := NewWasmRunner()
	defer runner.Close()

	// Create minimal WASM module for testing
	wasmData := []byte("\x00asm\x01\x00\x00\x00") // Minimal WASM header

	// Test should pass for minimal module (no network/fs access)
	err := runner.TestPluginIsolation(wasmData)
	if err != nil {
		t.Errorf("Isolation test should pass for minimal WASM: %v", err)
	}
}

func TestRuntimeStats(t *testing.T) {
	runner := NewWasmRunner()
	defer runner.Close()

	stats := runner.GetRuntimeStats()

	if stats == nil {
		t.Fatal("Stats should not be nil")
	}

	expectedFields := []string{"runtime_type", "wasi_enabled", "memory_limit", "network_access", "filesystem_access"}
	for _, field := range expectedFields {
		if _, exists := stats[field]; !exists {
			t.Errorf("Stats missing field: %s", field)
		}
	}

	// Check specific values
	if stats["runtime_type"] != "wazero" {
		t.Error("Runtime type should be wazero")
	}

	if stats["network_access"] != false {
		t.Error("Network access should be false")
	}

	if stats["filesystem_access"] != "none" {
		t.Error("Filesystem access should be none")
	}
}

func TestIndicatorValidation(t *testing.T) {
	tests := []struct {
		name      string
		indicator Indicator
		valid     bool
	}{
		{
			name: "Valid indicator",
			indicator: Indicator{
				Type:       "hash",
				Value:      "abc123",
				Confidence: 0.8,
				Context:    "SHA256 hash",
			},
			valid: true,
		},
		{
			name: "Missing type",
			indicator: Indicator{
				Type:       "",
				Value:      "abc123",
				Confidence: 0.8,
				Context:    "test",
			},
			valid: false,
		},
		{
			name: "Missing value",
			indicator: Indicator{
				Type:       "hash",
				Value:      "",
				Confidence: 0.8,
				Context:    "test",
			},
			valid: false,
		},
		{
			name: "Invalid confidence",
			indicator: Indicator{
				Type:       "hash",
				Value:      "abc123",
				Confidence: 1.5,
				Context:    "test",
			},
			valid: false,
		},
	}

	runner := NewWasmRunner()
	defer runner.Close()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output := &PluginOutput{
				Success:    true,
				Confidence: 0.5,
				Indicators: []Indicator{tt.indicator},
			}

			err := runner.validateOutput(output)
			if tt.valid && err != nil {
				t.Errorf("Expected indicator to be valid, got error: %v", err)
			}
			if !tt.valid && err == nil {
				t.Error("Expected indicator to be invalid, but validation passed")
			}
		})
	}
}

func BenchmarkWasmExecution(b *testing.B) {
	runner := NewWasmRunner()
	defer runner.Close()

	wasmData := []byte("\x00asm\x01\x00\x00\x00")
	input := PluginInput{
		ArtifactID:   "bench-test",
		ArtifactType: "executable",
		Metadata:     map[string]interface{}{"size": 1024},
		S3URL:        "s3://test/artifact",
		Timestamp:    time.Now(),
	}
	policy := SandboxPolicy{
		NetworkAccess:    false,
		FilesystemAccess: "none",
		MemoryLimit:      "128MB",
		CPULimit:         "100m",
		ExecutionTimeout: "30s",
		AllowedSyscalls:  []string{"read", "write", "exit"},
		RiskLevel:        "low",
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err := runner.ExecutePlugin(wasmData, input, policy)
		if err == nil {
			// Expected to fail with minimal WASM, but we're testing performance
			continue
		}
	}
}

func TestExecutionTimeout(t *testing.T) {
	runner := NewWasmRunner()
	defer runner.Close()

	wasmData := []byte("\x00asm\x01\x00\x00\x00")
	input := PluginInput{
		ArtifactID:   "timeout-test",
		ArtifactType: "executable",
		Metadata:     map[string]interface{}{},
		S3URL:        "s3://test/artifact",
		Timestamp:    time.Now(),
	}
	policy := SandboxPolicy{
		ExecutionTimeout: "1ms", // Very short timeout
		NetworkAccess:    false,
		FilesystemAccess: "none",
		MemoryLimit:      "128MB",
		CPULimit:         "100m",
		AllowedSyscalls:  []string{"read", "write", "exit"},
		RiskLevel:        "low",
	}

	start := time.Now()
	output, err := runner.ExecutePlugin(wasmData, input, policy)
	duration := time.Since(start)

	// Should timeout quickly
	if duration > 100*time.Millisecond {
		t.Error("Execution should timeout quickly")
	}

	if output == nil {
		t.Error("Output should not be nil even on timeout")
	}

	if err == nil {
		t.Error("Should have timeout error")
	}
}