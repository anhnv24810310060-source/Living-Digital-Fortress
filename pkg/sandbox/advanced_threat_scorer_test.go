package sandbox
package sandbox

import (
	"testing"
	"time"
)

func TestAdvancedThreatScorer_CleanExecution(t *testing.T) {
	scorer := NewAdvancedThreatScorer()

	result := &SandboxResult{
		Syscalls: []SyscallEvent{
			{SyscallName: "open", Dangerous: false},
			{SyscallName: "read", Dangerous: false},
			{SyscallName: "close", Dangerous: false},
		},
		Duration: 100 * time.Millisecond,
		Stdout:   "clean output",
	}

	score, explanation, metrics := scorer.CalculateAdvancedScore(result)

	if score > 30 {
		t.Errorf("Expected low score for clean execution, got %d", score)
	}

	if metrics == nil {
		t.Error("Expected metrics to be non-nil")
	}

	t.Logf("Clean execution score: %d, explanation: %s", score, explanation)
}

func TestAdvancedThreatScorer_ShellcodeInjection(t *testing.T) {
	scorer := NewAdvancedThreatScorer()

	result := &SandboxResult{
		Syscalls: []SyscallEvent{
			{SyscallName: "mmap", Dangerous: true},
			{SyscallName: "mprotect", Dangerous: true},
			{SyscallName: "execve", Dangerous: true},
		},
		Duration: 50 * time.Millisecond,
		Stdout:   "exploit payload",
	}

	score, explanation, metrics := scorer.CalculateAdvancedScore(result)

	if score < 70 {
		t.Errorf("Expected high score for shellcode injection, got %d", score)
	}

	if explanation == "clean" {
		t.Error("Expected threat explanation, got clean")
	}

	confidence := metrics["confidence"].(float64)
	if confidence < 0.5 {
		t.Errorf("Expected high confidence, got %.2f", confidence)
	}

	t.Logf("Shellcode injection score: %d, explanation: %s, confidence: %.2f", score, explanation, confidence)
}

func TestIsolationForest_AnomalyDetection(t *testing.T) {
	forest := NewIsolationForest(50, 8, 256)

	// Normal data should score low
	normalFeatures := []float64{10, 2, 0.2, 1, 0.3, 1, 1, 0.2, 100000, 5000, 10, 0.1, 0.5}
	normalScore := forest.AnomalyScore(normalFeatures)

	if normalScore > 0.6 {
		t.Errorf("Normal data scored as anomaly: %.2f", normalScore)
	}

	t.Logf("Normal data anomaly score: %.2f", normalScore)

	// Anomalous data should score high  
	anomalousFeatures := []float64{1000, 500, 0.9, 100, 5.0, 500, 500, 0.95, 1000000000, 100000, 1000, 10.0, 0.9}
	anomalousScore := forest.AnomalyScore(anomalousFeatures)

	t.Logf("Anomalous data score: %.2f", anomalousScore)
}

func TestSyscallSequenceAnalyzer_MaliciousPattern(t *testing.T) {
	analyzer := NewSyscallSequenceAnalyzer()

	syscalls := []SyscallEvent{
		{SyscallName: "mmap"},
		{SyscallName: "mprotect"},
		{SyscallName: "execve"},
	}

	score, patterns := analyzer.AnalyzeSequence(syscalls)

	if score < 0.7 {
		t.Errorf("Expected high score for malicious pattern, got %.2f", score)
	}

	if len(patterns) == 0 {
		t.Error("Expected pattern detection")
	}

	t.Logf("Malicious pattern score: %.2f, patterns: %v", score, patterns)
}

func TestBayesianThreatModel_Learning(t *testing.T) {
	model := NewBayesianThreatModel()

	// Train with threat examples
	for i := 0; i < 50; i++ {
		threatFeatures := []float64{100, 50, 0.8, 10, 2.0}
		model.UpdateModel(threatFeatures, true)
	}

	// Train with benign examples
	for i := 0; i < 50; i++ {
		benignFeatures := []float64{10, 1, 0.1, 1, 0.3}
		model.UpdateModel(benignFeatures, false)
	}

	// Test threat-like features
	threatTest := []float64{95, 45, 0.75, 9, 1.8}
	threatProb := model.ThreatProbability(threatTest)

	// Test benign-like features
	benignTest := []float64{12, 2, 0.15, 1, 0.4}
	benignProb := model.ThreatProbability(benignTest)

	t.Logf("After training - Threat prob: %.2f, Benign prob: %.2f", threatProb, benignProb)

	if threatProb <= benignProb {
		t.Error("Model failed to distinguish threat from benign")
	}
}

func BenchmarkAdvancedScorer(b *testing.B) {
	scorer := NewAdvancedThreatScorer()

	result := &SandboxResult{
		Syscalls: []SyscallEvent{
			{SyscallName: "open", Dangerous: false},
			{SyscallName: "mmap", Dangerous: true},
			{SyscallName: "mprotect", Dangerous: true},
			{SyscallName: "read", Dangerous: false},
			{SyscallName: "write", Dangerous: false},
		},
		NetworkIO: []NetworkEvent{{Protocol: "tcp", DstPort: 443}},
		FileAccess: []FileEvent{
			{Path: "/tmp/test", Operation: "write"},
		},
		Duration: 100 * time.Millisecond,
		Stdout:   "benchmark output",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		scorer.CalculateAdvancedScore(result)
	}
}
