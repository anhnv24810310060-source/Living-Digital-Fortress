package sandbox

import (
	"math"
	"strings"
)

// ThreatScorer calculates threat score (0-100) from sandbox execution results
type ThreatScorer struct {
	weights WeightConfig
}

type WeightConfig struct {
	DangerousSyscalls float64
	NetworkActivity   float64
	FileOperations    float64
	MemoryOperations  float64
	ProcessSpawning   float64
	Baseline          float64
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
	return &ThreatScorer{
		weights: DefaultWeights(),
	}
}

// CalculateScore performs multi-factor threat analysis
// Returns score (0-100) and explanation
func (ts *ThreatScorer) CalculateScore(result *SandboxResult) (int, string) {
	if result == nil {
		return 0, "no data"
	}

	var score float64
	var reasons []string

	// 1. Dangerous syscalls analysis (40 points max)
	dangScore := ts.analyzeDangerousSyscalls(result)
	if dangScore > 0 {
		score += dangScore
		reasons = append(reasons, "dangerous_syscalls")
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
