package guardian

import (
	"crypto/sha256"
	"encoding/hex"
	"math"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"shieldx/pkg/ebpf"
)

// ThreatResult represents the outcome of a threat analysis.
type ThreatResult struct {
	Score           int
	RiskLevel       string
	Indicators      []string
	Recommendations []string
	Details         map[string]interface{}
	Hash            string
}

type ThreatScorer struct {
	cache      map[string]*ThreatResult
	cacheMutex sync.RWMutex

	cacheHits   uint64
	cacheMisses uint64
	totalScored uint64

	stopped atomic.Bool
}

// NewThreatScorer constructs a thread-safe threat scoring engine with in-memory caching.
func NewThreatScorer() *ThreatScorer {
	return &ThreatScorer{
		cache: make(map[string]*ThreatResult),
	}
}

// Stop releases resources associated with the scorer.
func (ts *ThreatScorer) Stop() {
	if ts.stopped.CompareAndSwap(false, true) {
		ts.cacheMutex.Lock()
		ts.cache = make(map[string]*ThreatResult)
		ts.cacheMutex.Unlock()
	}
}

// AnalyzeThreat scores a payload using static heuristics and optional eBPF behaviour features.
func (ts *ThreatScorer) AnalyzeThreat(payload string, features *ebpf.ThreatFeatures) *ThreatResult {
	hash := hashPayload(payload)

	atomic.AddUint64(&ts.totalScored, 1)

	// Fast path: cached result
	ts.cacheMutex.RLock()
	cached, ok := ts.cache[hash]
	ts.cacheMutex.RUnlock()
	if ok {
		atomic.AddUint64(&ts.cacheHits, 1)
		return cloneResult(cached)
	}

	atomic.AddUint64(&ts.cacheMisses, 1)

	// Static pattern analysis
	patternScore, indicators := ts.analyzeStaticPatterns(payload)

	// Obfuscation / entropy detection
	entropy := calculateEntropy(payload)
	if entropy > 3.5 {
		indicators = appendUnique(indicators, "HIGH_ENTROPY")
	}
	if looksHex(payload) {
		indicators = appendUnique(indicators, "OBFUSCATED_CODE")
	}

	score := patternScore

	// Incorporate eBPF runtime behaviour
	details := map[string]interface{}{
		"analyzed_at": time.Now().UTC(),
		"entropy":     entropy,
	}

	if features != nil {
		runtimeScore := scoreEBPF(features)
		if runtimeScore > 0 {
			indicators = appendUnique(indicators, "EBPF_BEHAVIOR")
			score = maxScore(score, runtimeScore)
		}
		details["dynamic_behavior"] = map[string]any{
			"dangerous_syscalls": features.DangerousSyscalls,
			"event_count":        features.EventCount,
			"shell_execution":    features.ShellExecution,
			"unusual_patterns":   features.UnusualPatterns,
		}
	}

	// Clamp score and derive risk level
	if score < 0 {
		score = 0
	}
	if score > 100 {
		score = 100
	}

	risk := deriveRisk(score)
	recs := buildRecommendations(score, indicators)

	result := &ThreatResult{
		Score:           score,
		RiskLevel:       risk,
		Indicators:      indicators,
		Recommendations: recs,
		Details:         details,
		Hash:            hash,
	}

	ts.cacheMutex.Lock()
	ts.cache[hash] = cloneResult(result)
	ts.cacheMutex.Unlock()

	return cloneResult(result)
}

func (ts *ThreatScorer) analyzeStaticPatterns(payload string) (int, []string) {
	normalized := strings.ToLower(payload)
	score := 0
	indicators := []string{}

	patternScores := map[string]int{
		"execve(":    65,
		"/bin/sh":    70,
		"bash -i":    80,
		"eval(":      75,
		"system(":    70,
		"'||'":       60,
		" or '1'='1": 65,
		"setuid(0":   85,
		"nc -e":      85,
		"powershell": 70,
		"wget http":  60,
		"curl http":  55,
		"0x":         45,
	}

	for pattern, value := range patternScores {
		if strings.Contains(normalized, pattern) {
			score = maxScore(score, value)
			indicators = appendUnique(indicators, strings.ToUpper(pattern))
		}
	}

	if len(payload) > 256 {
		score = maxScore(score, 40)
		indicators = appendUnique(indicators, "LARGE_PAYLOAD")
	}

	return score, indicators
}

// scoreEBPF converts eBPF telemetry into a threat score on the 0-100 scale.
func scoreEBPF(features *ebpf.ThreatFeatures) int {
	if features == nil {
		return 0
	}

	weighted := float64(features.DangerousSyscalls)*1.2 +
		float64(features.ShellExecution)*10 +
		float64(features.UnusualPatterns)*7 +
		float64(features.EventCount)/2 +
		float64(features.NetworkCalls)/10 +
		float64(features.FileCalls)/12 +
		float64(features.ProcessCalls)/8

	if weighted <= 0 {
		return 0
	}

	return int(math.Min(100, weighted))
}

func deriveRisk(score int) string {
	switch {
	case score >= 90:
		return "critical"
	case score >= 70:
		return "high"
	case score >= 50:
		return "medium"
	case score >= 0:
		return "low"
	default:
		return "low"
	}
}

func buildRecommendations(score int, indicators []string) []string {
	recs := []string{}
	switch {
	case score >= 80:
		recs = append(recs, "BLOCK: terminate session")
	case score >= 60:
		recs = append(recs, "QUARANTINE: require manual review")
	default:
		recs = append(recs, "ALLOW: monitor only")
	}

	if containsIndicator(indicators, "HIGH_ENTROPY") {
		recs = append(recs, "INSPECT: high entropy payload detected")
	}

	return recs
}

// GetStats returns runtime statistics for the scorer.
func (ts *ThreatScorer) GetStats() map[string]interface{} {
	hits := atomic.LoadUint64(&ts.cacheHits)
	misses := atomic.LoadUint64(&ts.cacheMisses)
	total := atomic.LoadUint64(&ts.totalScored)
	rate := 0.0
	if hits+misses > 0 {
		rate = float64(hits) / float64(hits+misses)
	}

	return map[string]interface{}{
		"cache_hits":     hits,
		"cache_misses":   misses,
		"cache_hit_rate": rate,
		"total_scored":   total,
	}
}

func cloneResult(result *ThreatResult) *ThreatResult {
	if result == nil {
		return nil
	}
	clone := &ThreatResult{
		Score:           result.Score,
		RiskLevel:       result.RiskLevel,
		Hash:            result.Hash,
		Recommendations: append([]string(nil), result.Recommendations...),
		Indicators:      append([]string(nil), result.Indicators...),
		Details:         make(map[string]interface{}, len(result.Details)),
	}
	for k, v := range result.Details {
		clone.Details[k] = v
	}
	return clone
}

func hashPayload(payload string) string {
	digest := sha256.Sum256([]byte(payload))
	return hex.EncodeToString(digest[:])
}

func calculateEntropy(payload string) float64 {
	if len(payload) == 0 {
		return 0
	}

	freq := map[rune]float64{}
	total := 0.0
	for _, r := range payload {
		freq[r]++
		total++
	}

	entropy := 0.0
	for _, count := range freq {
		p := count / total
		entropy += -p * math.Log2(p)
	}
	return entropy
}

func looksHex(payload string) bool {
	if len(payload) < 16 {
		return false
	}
	for _, r := range payload {
		if !((r >= '0' && r <= '9') || (r >= 'a' && r <= 'f') || (r >= 'A' && r <= 'F')) {
			return false
		}
	}
	return true
}

func appendUnique(list []string, value string) []string {
	for _, existing := range list {
		if existing == value {
			return list
		}
	}
	return append(list, value)
}

func containsIndicator(indicators []string, indicator string) bool {
	for _, value := range indicators {
		if value == indicator {
			return true
		}
	}
	return false
}

func maxScore(values ...int) int {
	max := 0
	for _, v := range values {
		if v > max {
			max = v
		}
	}
	return max
}
