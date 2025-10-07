package sandbox

import (
	"testing"
	"time"
)

func TestThreatScorer_Clean(t *testing.T) {
	scorer := NewThreatScorer()
	result := &SandboxResult{
		Stdout:   "hello world",
		Duration: time.Second,
		Syscalls: []SyscallEvent{
			{SyscallName: "read", Dangerous: false},
			{SyscallName: "write", Dangerous: false},
		},
	}

	score, explanation := scorer.CalculateScore(result)
	if score > 10 {
		t.Errorf("Expected clean score <= 10, got %d", score)
	}
	if explanation != "clean" {
		t.Errorf("Expected 'clean', got '%s'", explanation)
	}
}

func TestThreatScorer_DangerousSyscalls(t *testing.T) {
	scorer := NewThreatScorer()
	result := &SandboxResult{
		Stdout:   "test",
		Duration: time.Second,
		Syscalls: []SyscallEvent{
			{SyscallName: "ptrace", Dangerous: true},
			{SyscallName: "execve", Dangerous: true},
			{SyscallName: "mprotect", Dangerous: true},
			{SyscallName: "read", Dangerous: false},
		},
	}

	score, explanation := scorer.CalculateScore(result)
	if score < 40 {
		t.Errorf("Expected high threat score >= 40 for dangerous syscalls, got %d", score)
	}
	if explanation == "clean" {
		t.Errorf("Expected threat reasons, got '%s'", explanation)
	}
}

func TestThreatScorer_NetworkActivity(t *testing.T) {
	scorer := NewThreatScorer()
	result := &SandboxResult{
		Stdout:   "test",
		Duration: time.Second,
		NetworkIO: []NetworkEvent{
			{DstIP: "192.168.1.1", DstPort: 4444, Bytes: 1024},
			{DstIP: "10.0.0.1", DstPort: 8080, Bytes: 2048},
		},
	}

	score, explanation := scorer.CalculateScore(result)
	if score < 10 {
		t.Errorf("Expected network threat score >= 10, got %d", score)
	}
	if explanation == "clean" {
		t.Errorf("Expected network_activity in reasons, got '%s'", explanation)
	}
}

func TestThreatScorer_FileOperations(t *testing.T) {
	scorer := NewThreatScorer()
	result := &SandboxResult{
		Stdout:   "test",
		Duration: time.Second,
		FileAccess: []FileEvent{
			{Path: "/etc/passwd", Operation: "read", Success: true},
			{Path: "/etc/shadow", Operation: "read", Success: true},
			{Path: "/root/.ssh/id_rsa", Operation: "read", Success: true},
		},
	}

	score, explanation := scorer.CalculateScore(result)
	if score < 9 {
		t.Errorf("Expected file threat score >= 9, got %d", score)
	}
	if explanation == "clean" {
		t.Errorf("Expected file_operations in reasons, got '%s'", explanation)
	}
}

func TestThreatScorer_MaxCap(t *testing.T) {
	scorer := NewThreatScorer()
	result := &SandboxResult{
		Stdout:   "exec bash curl wget",
		Duration: time.Second,
		Syscalls: []SyscallEvent{
			{SyscallName: "ptrace", Dangerous: true},
			{SyscallName: "execve", Dangerous: true},
			{SyscallName: "mprotect", Dangerous: true},
			{SyscallName: "fork", Dangerous: true},
			{SyscallName: "clone", Dangerous: true},
		},
		NetworkIO: []NetworkEvent{
			{DstIP: "10.0.0.1", DstPort: 4444, Bytes: 2000000},
		},
		FileAccess: []FileEvent{
			{Path: "/etc/passwd", Operation: "write", Success: true},
			{Path: "/etc/shadow", Operation: "write", Success: true},
		},
	}

	score, _ := scorer.CalculateScore(result)
	if score > 100 {
		t.Errorf("Expected score capped at 100, got %d", score)
	}
	if score < 80 {
		t.Errorf("Expected critical threat score >= 80, got %d", score)
	}
}

func TestRiskLevel(t *testing.T) {
	tests := []struct {
		score    int
		expected string
	}{
		{0, "MINIMAL"},
		{15, "LOW"},
		{25, "LOW"},
		{45, "MEDIUM"},
		{65, "HIGH"},
		{85, "CRITICAL"},
		{100, "CRITICAL"},
	}

	for _, tt := range tests {
		result := RiskLevel(tt.score)
		if result != tt.expected {
			t.Errorf("RiskLevel(%d) = %s, want %s", tt.score, result, tt.expected)
		}
	}
}
