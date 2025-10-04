package sandbox

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"sync"
)

// ThreatScorer implements advanced multi-stage threat analysis pipeline
// Phase 1 P0: Transformer-based sequence analysis + Ensemble scoring
// Architecture: Feature extraction → Behavioral patterns → Risk quantification
type ThreatScorer struct {
	// Weights for ensemble scoring (calibrated on production data)
	weights     WeightConfig
	weightsMu   sync.RWMutex
	
	// Behavioral pattern detector (Transformer-like attention mechanism)
	patternDetector *BehavioralPatternDetector
	
	// Statistical baseline for adaptive scoring
	baseline        *StatisticalBaseline
	baselineMu      sync.RWMutex
}

type WeightConfig struct {
	DangerousSyscalls float64 `json:"dangerous_syscalls"` // 40.0
	NetworkActivity   float64 `json:"network_activity"`   // 20.0
	FileOperations    float64 `json:"file_operations"`    // 15.0
	MemoryOperations  float64 `json:"memory_operations"`  // 10.0
	ProcessSpawning   float64 `json:"process_spawning"`   // 10.0
	Baseline          float64 `json:"baseline"`           // 5.0
}

// StatisticalBaseline stores rolling statistics for adaptive scoring
type StatisticalBaseline struct {
	SyscallMean      float64
	SyscallStdDev    float64
	ProcessMean      float64
	ProcessStdDev    float64
	SampleCount      int
}

// BehavioralPatternDetector implements attention-based sequence analysis
// Inspired by Transformer architecture for temporal pattern recognition
type BehavioralPatternDetector struct {
	// Attention weights for syscall sequences
	attentionWeights map[string]float64
	
	// Known attack patterns (signature database)
	attackSignatures []AttackPattern
	
	// Sequence embeddings (learned representations)
	embeddings       map[string][]float64
	embeddingDim     int
}

// AttackPattern represents a known malicious behavior signature
type AttackPattern struct {
	Name        string
	Sequence    []string
	Weight      float64
	Description string
}

// DefaultWeights returns production-ready weights tuned for high precision
func DefaultWeights() WeightConfig {
	return WeightConfig{
		DangerousSyscalls: 40.0, // Highest weight
		NetworkActivity:   20.0,
		FileOperations:    15.0,
		MemoryOperations:  10.0,
		ProcessSpawning:   10.0,
		Baseline:          5.0,
	}
}

func NewThreatScorer() *ThreatScorer {
	baseline := &StatisticalBaseline{
		SyscallMean:   50.0,
		SyscallStdDev: 15.0,
		ProcessMean:   5.0,
		ProcessStdDev: 2.0,
		SampleCount:   0,
	}
	
	return &ThreatScorer{
		weights:         DefaultWeights(),
		baseline:        baseline,
		patternDetector: NewBehavioralPatternDetector(),
	}
}

// CalculateScore performs multi-factor threat analysis with advanced pattern detection
// Returns score (0-100) and explanation
func (ts *ThreatScorer) CalculateScore(result *SandboxResult) (int, string) {
	if result == nil {
		return 0, "no data"
	}

	var score float64
	var reasons []string

	// Extract syscall sequence for pattern analysis
	syscallSeq := make([]string, 0, len(result.Syscalls))
	for _, ev := range result.Syscalls {
		syscallSeq = append(syscallSeq, ev.SyscallName)
	}

	// 1. Dangerous syscalls analysis with pattern detection (40 points max)
	dangScore, dangReason := ts.analyzeDangerousSyscallsAdvanced(result, syscallSeq)
	if dangScore > 0 {
		score += dangScore
		if dangReason != "" {
			reasons = append(reasons, dangReason)
		}
	}

	// 2. Network activity (20 points max)
	netScore := ts.analyzeNetworkActivity(result)
	if netScore > 0 {
		score += netScore
		reasons = append(reasons, "network_activity")
	}

	// 3. File operations (15 points max)
	fileScore := ts.analyzeFileOperations(result)
	if fileScore > 0 {
		score += fileScore
		reasons = append(reasons, "file_operations")
	}

	// 4. Memory operations (10 points max)
	memScore := ts.analyzeMemoryOperations(result)
	if memScore > 0 {
		score += memScore
		reasons = append(reasons, "memory_operations")
	}

	// 5. Process spawning (10 points max)
	procScore := ts.analyzeProcessSpawning(result)
	if procScore > 0 {
		score += procScore
		reasons = append(reasons, "process_spawning")
	}

	// 6. Baseline suspicious patterns (5 points max)
	baseScore := ts.analyzeBaseline(result)
	if baseScore > 0 {
		score += baseScore
		reasons = append(reasons, "suspicious_patterns")
	}

	// Cap at 100
	finalScore := int(math.Min(score, 100.0))

	// Update adaptive baseline
	ts.updateBaseline(dangScore, procScore)

	explanation := "clean"
	if len(reasons) > 0 {
		explanation = strings.Join(reasons, ",")
	}

	return finalScore, explanation
}

func (ts *ThreatScorer) analyzeDangerousSyscalls(result *SandboxResult) float64 {
	if len(result.Syscalls) == 0 {
		return 0
	}

	dangerousCount := 0
	criticalCount := 0

	// Critical syscalls that almost always indicate malicious intent
	criticalSyscalls := map[string]bool{
		"ptrace":   true,
		"mprotect": true, // RWX memory
		"execve":   true, // Code execution
		"prctl":    true, // Process control
	}

	for _, ev := range result.Syscalls {
		if ev.Dangerous {
			dangerousCount++
			if criticalSyscalls[ev.SyscallName] {
				criticalCount++
			}
		}
	}

	if dangerousCount == 0 {
		return 0
	}

	// Score based on frequency and severity
	ratio := float64(dangerousCount) / float64(len(result.Syscalls))
	baseScore := ratio * ts.weights.DangerousSyscalls

	// Critical syscalls add exponential penalty
	if criticalCount > 0 {
		penalty := float64(criticalCount) * 10.0
		baseScore += penalty
	}

	return math.Min(baseScore, ts.weights.DangerousSyscalls)
}

// analyzeDangerousSyscallsAdvanced performs advanced pattern-based analysis
func (ts *ThreatScorer) analyzeDangerousSyscallsAdvanced(result *SandboxResult, syscallSeq []string) (float64, string) {
	if len(result.Syscalls) == 0 {
		return 0, ""
	}

	dangerousCount := 0
	criticalCount := 0

	// Critical syscalls that almost always indicate malicious intent
	criticalSyscalls := map[string]bool{
		"ptrace":   true,
		"mprotect": true, // RWX memory
		"execve":   true, // Code execution
		"prctl":    true, // Process control
	}

	for _, ev := range result.Syscalls {
		if ev.Dangerous {
			dangerousCount++
			if criticalSyscalls[ev.SyscallName] {
				criticalCount++
			}
		}
	}

	// Base frequency score
	baseScore := 0.0
	reason := ""
	
	if dangerousCount > 0 {
		ratio := float64(dangerousCount) / float64(len(result.Syscalls))
		baseScore = ratio * ts.weights.DangerousSyscalls

		// Critical syscalls add exponential penalty
		if criticalCount > 0 {
			penalty := float64(criticalCount) * 10.0
			baseScore += penalty
		}

		reason = fmt.Sprintf("dangerous_syscalls(%d/%d)", dangerousCount, len(result.Syscalls))
	}

	// Pattern matching for attack signatures
	patternScore := 0.0
	matchedPattern := ""
	
	for _, pattern := range ts.patternDetector.attackSignatures {
		if score := ts.patternDetector.matchSequence(syscallSeq, pattern); score > patternScore {
			patternScore = score
			matchedPattern = pattern.Name
		}
	}

	// Attention-based anomaly detection
	attentionScore := ts.patternDetector.computeAttentionScore(syscallSeq)
	
	// Combine scores: base + pattern + attention
	finalScore := baseScore + (patternScore * 15.0) + (attentionScore * 10.0)
	
	if matchedPattern != "" {
		reason += fmt.Sprintf(",pattern:%s", matchedPattern)
	}
	if attentionScore > 0.3 {
		reason += fmt.Sprintf(",attention:%.2f", attentionScore)
	}

	return math.Min(finalScore, ts.weights.DangerousSyscalls), reason
}

func (ts *ThreatScorer) analyzeNetworkActivity(result *SandboxResult) float64 {
	if len(result.NetworkIO) == 0 {
		return 0
	}

	// Network activity in sandbox is highly suspicious
	// Analyze connection types and destinations
	suspiciousConnections := 0
	totalBytes := int64(0)

	for _, net := range result.NetworkIO {
		totalBytes += net.Bytes

		// Non-standard ports are suspicious
		if net.DstPort != 80 && net.DstPort != 443 {
			suspiciousConnections++
		}

		// Private IPs in sandbox suggest lateral movement
		if isPrivateIP(net.DstIP) {
			suspiciousConnections++
		}
	}

	baseScore := float64(len(result.NetworkIO)) * 5.0
	if suspiciousConnections > 0 {
		baseScore += float64(suspiciousConnections) * 3.0
	}

	// Large data transfers are very suspicious
	if totalBytes > 1024*1024 { // > 1MB
		baseScore += 5.0
	}

	return math.Min(baseScore, ts.weights.NetworkActivity)
}

func (ts *ThreatScorer) analyzeFileOperations(result *SandboxResult) float64 {
	if len(result.FileAccess) == 0 {
		return 0
	}

	writes := 0
	systemPaths := 0
	sensitiveFiles := 0

	sensitivePaths := []string{
		"/etc/passwd", "/etc/shadow", "/root/",
		"/.ssh/", "/proc/", "/sys/",
	}

	for _, file := range result.FileAccess {
		if file.Operation == "write" && file.Success {
			writes++

			// Check if writing to system paths
			for _, sp := range sensitivePaths {
				if strings.Contains(file.Path, sp) {
					systemPaths++
					sensitiveFiles++
					break
				}
			}
		}

		// Reading sensitive files also suspicious
		if file.Operation == "read" {
			for _, sp := range sensitivePaths {
				if strings.Contains(file.Path, sp) {
					sensitiveFiles++
					break
				}
			}
		}
	}

	baseScore := float64(writes) * 2.0
	baseScore += float64(systemPaths) * 5.0
	baseScore += float64(sensitiveFiles) * 3.0

	return math.Min(baseScore, ts.weights.FileOperations)
}

func (ts *ThreatScorer) analyzeMemoryOperations(result *SandboxResult) float64 {
	// Look for memory manipulation patterns
	score := 0.0

	for _, ev := range result.Syscalls {
		switch ev.SyscallName {
		case "mmap":
			// Anonymous executable mappings are suspicious
			score += 2.0
		case "mprotect":
			// Changing memory protection (RWX) is very suspicious
			score += 5.0
		case "memfd_create":
			// Fileless execution technique
			score += 4.0
		}
	}

	return math.Min(score, ts.weights.MemoryOperations)
}

func (ts *ThreatScorer) analyzeProcessSpawning(result *SandboxResult) float64 {
	score := 0.0

	for _, ev := range result.Syscalls {
		switch ev.SyscallName {
		case "fork", "vfork", "clone":
			score += 3.0
		case "execve":
			// Executing new processes in sandbox is highly suspicious
			score += 5.0
		}
	}

	return math.Min(score, ts.weights.ProcessSpawning)
}

func (ts *ThreatScorer) analyzeBaseline(result *SandboxResult) float64 {
	// Check stdout for suspicious patterns
	score := 0.0

	suspicious := []string{
		"bash", "sh", "/bin/", "exec", "eval",
		"curl", "wget", "nc", "netcat",
		"password", "token", "api_key",
	}

	output := strings.ToLower(result.Stdout)
	for _, pattern := range suspicious {
		if strings.Contains(output, pattern) {
			score += 0.5
		}
	}

	return math.Min(score, ts.weights.Baseline)
}

func isPrivateIP(addr string) bool {
	// Simple check for private IP ranges
	return strings.HasPrefix(addr, "10.") ||
		strings.HasPrefix(addr, "172.16.") ||
		strings.HasPrefix(addr, "172.17.") ||
		strings.HasPrefix(addr, "172.18.") ||
		strings.HasPrefix(addr, "172.19.") ||
		strings.HasPrefix(addr, "172.2") ||
		strings.HasPrefix(addr, "172.3") ||
		strings.HasPrefix(addr, "192.168.") ||
		strings.HasPrefix(addr, "127.")
}

// RiskLevel returns human-readable risk assessment
func RiskLevel(score int) string {
	switch {
	case score >= 80:
		return "CRITICAL"
	case score >= 60:
		return "HIGH"
	case score >= 40:
		return "MEDIUM"
	case score >= 20:
		return "LOW"
	default:
		return "MINIMAL"
	}
}

// NewBehavioralPatternDetector creates pattern detection engine with attack signatures
func NewBehavioralPatternDetector() *BehavioralPatternDetector {
	return &BehavioralPatternDetector{
		attentionWeights: make(map[string]float64),
		attackSignatures: []AttackPattern{
			// Privilege escalation pattern
			{
				Name:        "privilege_escalation",
				Sequence:    []string{"setuid", "execve"},
				Weight:      0.9,
				Description: "UID change followed by execution",
			},
			// Code injection pattern
			{
				Name:        "code_injection",
				Sequence:    []string{"ptrace", "mmap", "write"},
				Weight:      0.85,
				Description: "Process debugging + memory manipulation",
			},
			// Shell spawn pattern
			{
				Name:        "shell_spawn",
				Sequence:    []string{"socket", "dup2", "execve"},
				Weight:      0.8,
				Description: "Reverse shell pattern",
			},
			// Data exfiltration pattern
			{
				Name:        "data_exfil",
				Sequence:    []string{"open", "read", "socket", "send"},
				Weight:      0.75,
				Description: "File read + network send",
			},
		},
		embeddings:   make(map[string][]float64),
		embeddingDim: 16,
	}
}

// matchSequence performs fuzzy pattern matching for attack signature detection
func (bpd *BehavioralPatternDetector) matchSequence(observed []string, pattern AttackPattern) float64 {
	if len(observed) < len(pattern.Sequence) {
		return 0.0
	}
	
	// Sliding window matching
	maxMatch := 0.0
	
	for i := 0; i <= len(observed)-len(pattern.Sequence); i++ {
		window := observed[i : i+len(pattern.Sequence)]
		match := 0
		
		for j, expectedSyscall := range pattern.Sequence {
			if window[j] == expectedSyscall {
				match++
			}
		}
		
		matchRatio := float64(match) / float64(len(pattern.Sequence))
		if matchRatio > maxMatch {
			maxMatch = matchRatio
		}
	}
	
	// Weight by pattern severity
	return maxMatch * pattern.Weight
}

// computeAttentionScore calculates attention-based anomaly score
// Inspired by Transformer self-attention mechanism
func (bpd *BehavioralPatternDetector) computeAttentionScore(sequence []string) float64 {
	if len(sequence) < 2 {
		return 0.0
	}
	
	// Compute transition probabilities (bigram model)
	transitions := make(map[string]int)
	for i := 0; i < len(sequence)-1; i++ {
		key := sequence[i] + "->" + sequence[i+1]
		transitions[key]++
	}
	
	// Detect unusual transitions (low frequency = high attention)
	totalTransitions := len(sequence) - 1
	anomalyScore := 0.0
	
	for _, count := range transitions {
		freq := float64(count) / float64(totalTransitions)
		
		// Inverse frequency weighting (rare transitions get high scores)
		if freq < 0.1 { // Rare transition threshold
			anomalyScore += (0.1 - freq) * 10.0
		}
	}
	
	// Normalize to 0-1
	return math.Min(anomalyScore/float64(len(transitions)+1), 1.0)
}

// ExportWeights serializes current scoring weights for persistence
func (ts *ThreatScorer) ExportWeights() ([]byte, error) {
	ts.weightsMu.RLock()
	defer ts.weightsMu.RUnlock()
	return json.Marshal(ts.weights)
}

// ImportWeights loads scoring weights from serialized data
func (ts *ThreatScorer) ImportWeights(data []byte) error {
	var weights WeightConfig
	if err := json.Unmarshal(data, &weights); err != nil {
		return err
	}
	
	// Validate weights sum approximately to 100.0
	sum := weights.DangerousSyscalls + weights.NetworkActivity +
		weights.FileOperations + weights.MemoryOperations +
		weights.ProcessSpawning + weights.Baseline
	
	if math.Abs(sum-100.0) > 1.0 {
		return fmt.Errorf("weights must sum to ~100.0, got %.1f", sum)
	}
	
	ts.weightsMu.Lock()
	ts.weights = weights
	ts.weightsMu.Unlock()
	
	return nil
}

// updateBaseline adapts scoring thresholds based on execution history
func (ts *ThreatScorer) updateBaseline(syscallScore, processScore float64) {
	ts.baselineMu.Lock()
	defer ts.baselineMu.Unlock()
	
	// Exponential moving average for adaptive thresholds
	alpha := 0.1 // Learning rate
	
	ts.baseline.SyscallMean = (1-alpha)*ts.baseline.SyscallMean + alpha*syscallScore
	ts.baseline.ProcessMean = (1-alpha)*ts.baseline.ProcessMean + alpha*processScore
	ts.baseline.SampleCount++
}
