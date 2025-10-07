package main

import (
	"shieldx/pkg/ebpf"
	gts "shieldx/services/honeypot-service/internal/guardian"
	"testing"
)

// TestGuardianThreatScorerBasicPatterns ensures static pattern detection elevates score & risk.
func TestGuardianThreatScorerBasicPatterns(t *testing.T) {
	ts := gts.NewThreatScorer()
	defer ts.Stop()
	res := ts.AnalyzeThreat("/bin/sh -c echo hi", nil)
	if res.Score < 60 {
		t.Fatalf("expected score >=60 got %d", res.Score)
	}
	if res.RiskLevel == "low" {
		t.Fatalf("expected elevated risk, got %s", res.RiskLevel)
	}
	// Call again to exercise cache hit
	res2 := ts.AnalyzeThreat("/bin/sh -c echo hi", nil)
	if res2.Hash != res.Hash {
		t.Fatalf("expected same hash, got %s vs %s", res2.Hash, res.Hash)
	}
	stats := ts.GetStats()
	if stats["cache_hits"].(uint64) == 0 {
		t.Fatalf("expected cache hit recorded, stats=%v", stats)
	}
}

// TestGuardianThreatScorerEBPFIntegration ensures eBPF features can boost score above static.
func TestGuardianThreatScorerEBPFIntegration(t *testing.T) {
	ts := gts.NewThreatScorer()
	defer ts.Stop()
	// Payload mild but runtime shows dangerous behaviour
	feats := &ebpf.ThreatFeatures{DangerousSyscalls: 5, ShellExecution: 1, EventCount: 40, UnusualPatterns: 2}
	res := ts.AnalyzeThreat("echo harmless", feats)
	if res.Score < 50 {
		t.Fatalf("expected boosted score >=50, got %d", res.Score)
	}
	hasBehavior := false
	for _, ind := range res.Indicators {
		if ind == "EBPF_BEHAVIOR" {
			hasBehavior = true
			break
		}
	}
	if !hasBehavior {
		t.Fatalf("expected EBPF_BEHAVIOR indicator, got %v", res.Indicators)
	}
}

// TestGuardianThreatScorerHighEntropy triggers HIGH_ENTROPY indicator via random-like payload.
func TestGuardianThreatScorerHighEntropy(t *testing.T) {
	ts := gts.NewThreatScorer()
	defer ts.Stop()
	high := "d1f0a9bb37c2e4f5aa11ccbbddeeff0099aabbccddeeff1122334455667788"
	res := ts.AnalyzeThreat(high, nil)
	found := false
	for _, ind := range res.Indicators {
		if ind == "HIGH_ENTROPY" || ind == "OBFUSCATED_CODE" {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected entropy/obfuscation indicator, got %v", res.Indicators)
	}
}
