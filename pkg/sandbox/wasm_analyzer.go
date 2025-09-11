package sandbox

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"time"
)

type AnalyzerManifest struct {
	Name         string            `json:"name"`
	Version      string            `json:"version"`
	Author       string            `json:"author"`
	License      string            `json:"license"`
	Capabilities CapabilitySet     `json:"capabilities"`
	Resources    ResourceLimits    `json:"resources"`
	SBOM         string            `json:"sbom"`
	Signature    string            `json:"signature"`
}

type CapabilitySet struct {
	NetworkEgress   bool     `json:"network_egress"`
	FilesystemWrite bool     `json:"filesystem_write"`
	Syscalls        []string `json:"allowed_syscalls"`
	MemoryMB        int      `json:"max_memory_mb"`
	CPUTimeMS       int      `json:"max_cpu_time_ms"`
}

type AnalyzerRegistry struct {
	analyzers map[string]*AnalyzerPackage
	verifier  *PackageVerifier
}

type AnalyzerPackage struct {
	Manifest    AnalyzerManifest
	WASMBytes   []byte
	Verified    bool
	Rating      float64
	Downloads   int
	LastUpdated time.Time
}

type PackageVerifier struct {
	trustedKeys [][]byte
}

func NewAnalyzerRegistry() *AnalyzerRegistry {
	return &AnalyzerRegistry{
		analyzers: make(map[string]*AnalyzerPackage),
		verifier:  &PackageVerifier{},
	}
}

func (r *AnalyzerRegistry) Install(pkg *AnalyzerPackage) error {
	// Verify signature
	if !r.verifier.VerifySignature(pkg) {
		return fmt.Errorf("invalid signature")
	}

	// Check capabilities
	if err := r.validateCapabilities(pkg.Manifest.Capabilities); err != nil {
		return fmt.Errorf("invalid capabilities: %w", err)
	}

	// Sandbox test run
	if err := r.testAnalyzer(pkg); err != nil {
		return fmt.Errorf("test failed: %w", err)
	}

	r.analyzers[pkg.Manifest.Name] = pkg
	return nil
}

func (r *AnalyzerRegistry) Get(name string) (*AnalyzerPackage, bool) {
	pkg, exists := r.analyzers[name]
	return pkg, exists
}

func (v *PackageVerifier) VerifySignature(pkg *AnalyzerPackage) bool {
	// Verify SBOM and signature
	hash := sha256.Sum256(pkg.WASMBytes)
	expectedHash := fmt.Sprintf("%x", hash)
	
	// Simple verification - production would use proper crypto
	return len(pkg.Manifest.Signature) > 0 && len(expectedHash) > 0
}

func (r *AnalyzerRegistry) validateCapabilities(caps CapabilitySet) error {
	// Enforce security constraints
	if caps.NetworkEgress {
		return fmt.Errorf("network egress not allowed")
	}
	
	if caps.FilesystemWrite {
		return fmt.Errorf("filesystem write not allowed")
	}
	
	if caps.MemoryMB > 64 {
		return fmt.Errorf("memory limit too high")
	}
	
	if caps.CPUTimeMS > 5000 {
		return fmt.Errorf("CPU time limit too high")
	}
	
	return nil
}

func (r *AnalyzerRegistry) testAnalyzer(pkg *AnalyzerPackage) error {
	// Create test runner with strict limits
	runner := NewWASMRunner(pkg.WASMBytes, "analyze", 1*time.Second)
	
	// Test with benign payload
	ctx := context.Background()
	result, err := runner.Run(ctx, "test payload")
	if err != nil {
		return err
	}
	
	// Verify result is valid JSON
	var output map[string]interface{}
	if err := json.Unmarshal([]byte(result), &output); err != nil {
		return fmt.Errorf("invalid output format")
	}
	
	return nil
}