package fortress_bridge

import (
	"testing"
	"time"
)

func TestPluginValidation(t *testing.T) {
	// Skip if no database available
	validator, err := NewPluginValidator("postgres://test:test@localhost/test_plugins?sslmode=disable")
	if err != nil {
		t.Skip("Database not available for testing")
	}
	defer validator.Close()

	wasmData := []byte("\x00asm\x01\x00\x00\x00") // Minimal WASM header
	cosignSig := "mock_signature_data"
	sbom := `{
		"bomFormat": "CycloneDX",
		"specVersion": "1.4",
		"components": [
			{
				"name": "test-component",
				"version": "1.0.0",
				"type": "library"
			}
		]
	}`

	result, err := validator.ValidatePlugin(wasmData, cosignSig, sbom, "test_owner", "1.0.0")

	if err != nil {
		t.Fatalf("Validation failed: %v", err)
	}

	if result == nil {
		t.Fatal("Result should not be nil")
	}

	// Check that SBOM validation passes
	if !result.SBOMValid {
		t.Error("SBOM validation should pass")
	}

	// Check risk score calculation
	if result.RiskScore < 0.0 || result.RiskScore > 1.0 {
		t.Errorf("Risk score should be between 0 and 1, got %f", result.RiskScore)
	}
}

func TestSBOMValidation(t *testing.T) {
	validator := &PluginValidator{}

	tests := []struct {
		name     string
		sbom     string
		expected bool
	}{
		{
			name: "Valid SBOM",
			sbom: `{
				"bomFormat": "CycloneDX",
				"specVersion": "1.4",
				"components": [{"name": "test", "version": "1.0"}]
			}`,
			expected: true,
		},
		{
			name:     "Invalid JSON",
			sbom:     `{invalid json}`,
			expected: false,
		},
		{
			name: "Missing required field",
			sbom: `{
				"bomFormat": "CycloneDX",
				"components": []
			}`,
			expected: false,
		},
		{
			name: "Empty components",
			sbom: `{
				"bomFormat": "CycloneDX",
				"specVersion": "1.4",
				"components": []
			}`,
			expected: false,
		},
		{
			name: "Blacklisted component",
			sbom: `{
				"bomFormat": "CycloneDX",
				"specVersion": "1.4",
				"components": [{"name": "malicious-lib", "version": "1.0"}]
			}`,
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := validator.validateSBOM(tt.sbom)
			if tt.expected && err != nil {
				t.Errorf("Expected validation to pass, got error: %v", err)
			}
			if !tt.expected && err == nil {
				t.Error("Expected validation to fail, but it passed")
			}
			if result != tt.expected {
				t.Errorf("Expected %v, got %v", tt.expected, result)
			}
		})
	}
}

func TestRiskScoreCalculation(t *testing.T) {
	validator := &PluginValidator{}

	tests := []struct {
		name     string
		result   *ValidationResult
		expected float64
	}{
		{
			name: "All valid",
			result: &ValidationResult{
				CosignValid: true,
				SBOMValid:   true,
				TrivyClean:  true,
			},
			expected: 0.0,
		},
		{
			name: "Invalid cosign",
			result: &ValidationResult{
				CosignValid: false,
				SBOMValid:   true,
				TrivyClean:  true,
			},
			expected: 0.4,
		},
		{
			name: "All invalid",
			result: &ValidationResult{
				CosignValid: false,
				SBOMValid:   false,
				TrivyClean:  false,
			},
			expected: 1.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			score := validator.calculateRiskScore(tt.result)
			if score != tt.expected {
				t.Errorf("Expected risk score %f, got %f", tt.expected, score)
			}
		})
	}
}

func TestBlacklistValidation(t *testing.T) {
	validator := &PluginValidator{}

	blacklistedComponents := []string{
		"malicious-lib",
		"crypto-miner",
		"backdoor-util",
		"keylogger",
		"rootkit",
	}

	for _, component := range blacklistedComponents {
		if !validator.isBlacklistedComponent(component) {
			t.Errorf("Component %s should be blacklisted", component)
		}
	}

	safeComponents := []string{
		"safe-lib",
		"crypto-utils",
		"logger",
	}

	for _, component := range safeComponents {
		if validator.isBlacklistedComponent(component) {
			t.Errorf("Component %s should not be blacklisted", component)
		}
	}
}

func TestSandboxPolicyGeneration(t *testing.T) {
	validator := &PluginValidator{}

	result := &ValidationResult{
		CosignValid: true,
		SBOMValid:   true,
		TrivyClean:  true,
		RiskScore:   0.2,
	}

	policy := validator.generateSandboxPolicy(result)

	if policy == "" {
		t.Error("Policy should not be empty")
	}

	// Verify it's valid JSON
	var policyData map[string]interface{}
	if err := json.Unmarshal([]byte(policy), &policyData); err != nil {
		t.Errorf("Policy should be valid JSON: %v", err)
	}

	// Check required fields
	requiredFields := []string{"network_access", "filesystem_access", "memory_limit", "cpu_limit", "execution_timeout"}
	for _, field := range requiredFields {
		if _, exists := policyData[field]; !exists {
			t.Errorf("Policy missing required field: %s", field)
		}
	}

	// Network access should be false
	if networkAccess, ok := policyData["network_access"].(bool); !ok || networkAccess {
		t.Error("Network access should be false")
	}
}

func BenchmarkPluginValidation(b *testing.B) {
	validator, err := NewPluginValidator("postgres://test:test@localhost/test_plugins?sslmode=disable")
	if err != nil {
		b.Skip("Database not available for benchmarking")
	}
	defer validator.Close()

	wasmData := []byte("\x00asm\x01\x00\x00\x00")
	cosignSig := "mock_signature"
	sbom := `{"bomFormat": "CycloneDX", "specVersion": "1.4", "components": [{"name": "test", "version": "1.0"}]}`

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err := validator.ValidatePlugin(wasmData, cosignSig, sbom, "bench_owner", "1.0.0")
		if err != nil {
			b.Fatalf("Validation failed: %v", err)
		}
	}
}

func TestGetStatusFromResult(t *testing.T) {
	tests := []struct {
		name     string
		result   *ValidationResult
		expected string
	}{
		{
			name: "Valid result",
			result: &ValidationResult{
				Valid:  true,
				Errors: []string{},
			},
			expected: "verified",
		},
		{
			name: "Invalid with errors",
			result: &ValidationResult{
				Valid:  false,
				Errors: []string{"cosign failed"},
			},
			expected: "rejected",
		},
		{
			name: "Invalid without errors",
			result: &ValidationResult{
				Valid:  false,
				Errors: []string{},
			},
			expected: "pending",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			status := getStatusFromResult(tt.result)
			if status != tt.expected {
				t.Errorf("Expected status %s, got %s", tt.expected, status)
			}
		})
	}
}

func TestGetRiskLevel(t *testing.T) {
	tests := []struct {
		score    float64
		expected string
	}{
		{0.1, "low"},
		{0.5, "medium"},
		{0.8, "high"},
		{0.0, "low"},
		{1.0, "high"},
	}

	for _, tt := range tests {
		result := getRiskLevel(tt.score)
		if result != tt.expected {
			t.Errorf("Score %f: expected %s, got %s", tt.score, tt.expected, result)
		}
	}
}