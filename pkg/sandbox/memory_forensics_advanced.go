package sandbox

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"regexp"
	"sync"
)

// AdvancedMemoryForensics performs deep memory analysis with YARA-like rules
type AdvancedMemoryForensics struct {
	rules         []*ForensicRule
	stringScanner *StringScanner
	artifactDB    map[string]*Artifact
	mu            sync.RWMutex
}

// ForensicRule defines pattern matching rule for memory analysis
type ForensicRule struct {
	Name        string
	Description string
	Severity    string // critical, high, medium, low
	Patterns    []Pattern
	Condition   string
	ThreatScore int
}

// Pattern defines search pattern (hex, string, regex)
type Pattern struct {
	Type    string // hex, string, regex
	Value   string
	Offset  int64 // Optional fixed offset
	MaxHits int   // Stop after N hits
}

// Artifact represents discovered memory artifact
type Artifact struct {
	RuleName    string
	Description string
	Offset      int64
	Data        []byte
	Hash        string
	Severity    string
	Category    string // shellcode, credential, encryption_key, malware_signature
	Timestamp   int64
}

// ForensicMemoryRegion represents analyzed memory section (different from sandbox.MemoryRegion)
type ForensicMemoryRegion struct {
	Start      uint64
	End        uint64
	Size       uint64
	Protection string // rwx, rw-, r-x, etc.
	Type       string // heap, stack, code, data
	Artifacts  []*Artifact
}

// StringScanner efficiently extracts strings from memory
type StringScanner struct {
	minLen     int
	maxLen     int
	patterns   []*regexp.Regexp
	entropyMin float64 // Detect encrypted/packed data
}

// NewAdvancedMemoryForensics creates production-grade memory forensics engine
func NewAdvancedMemoryForensics() *AdvancedMemoryForensics {
	amf := &AdvancedMemoryForensics{
		rules:      make([]*ForensicRule, 0),
		artifactDB: make(map[string]*Artifact),
		stringScanner: &StringScanner{
			minLen:     4,
			maxLen:     1024,
			entropyMin: 6.0,
			patterns:   make([]*regexp.Regexp, 0),
		},
	}

	// Load built-in detection rules
	amf.loadBuiltinRules()

	return amf
}

// AnalyzeMemoryDump performs comprehensive memory forensics
func (amf *AdvancedMemoryForensics) AnalyzeMemoryDump(memory []byte) (*MemoryForensicsReport, error) {
	report := &MemoryForensicsReport{
		TotalSize:   len(memory),
		Artifacts:   make([]*Artifact, 0),
		Regions:     make([]*ForensicMemoryRegion, 0),
		Strings:     make([]string, 0),
		ThreatScore: 0,
		Timestamp:   currentTimestamp(),
	}

	// 1. Scan for known malicious patterns
	artifacts := amf.scanWithRules(memory)
	report.Artifacts = append(report.Artifacts, artifacts...)

	// 2. Extract suspicious strings
	strings := amf.stringScanner.ExtractStrings(memory)
	report.Strings = append(report.Strings, strings...)

	// 3. Analyze memory regions for code injection
	regions := amf.analyzeMemoryRegions(memory)
	report.Regions = append(report.Regions, regions...)

	// 4. Detect shellcode patterns
	shellcode := amf.detectShellcode(memory)
	if shellcode != nil {
		report.Artifacts = append(report.Artifacts, shellcode...)
		report.ShellcodeDetected = len(shellcode) > 0
	}

	// 5. Search for credentials/secrets
	secrets := amf.detectSecrets(memory)
	report.Artifacts = append(report.Artifacts, secrets...)

	// 6. Analyze entropy for packed/encrypted sections
	highEntropyRegions := amf.findHighEntropyRegions(memory)
	report.HighEntropyRegions = highEntropyRegions

	// Calculate overall threat score
	report.ThreatScore = amf.calculateThreatScore(report)

	return report, nil
}

// loadBuiltinRules loads production-grade detection rules
func (amf *AdvancedMemoryForensics) loadBuiltinRules() {
	rules := []*ForensicRule{
		{
			Name:        "Metasploit_Shellcode_Pattern",
			Description: "Detects common Metasploit shellcode signatures",
			Severity:    "critical",
			ThreatScore: 95,
			Patterns: []Pattern{
				{Type: "hex", Value: "fc4883e4f0e8c0000000"}, // push rbp; and rsp, -0x10; call
				{Type: "hex", Value: "4989d14989d8"},         // mov r9, rdx; mov r8, rbx
			},
		},
		{
			Name:        "WannaCry_Encryption_Marker",
			Description: "WannaCry ransomware encryption marker",
			Severity:    "critical",
			ThreatScore: 98,
			Patterns: []Pattern{
				{Type: "string", Value: "WANACRY!"},
				{Type: "string", Value: ".WNCRYT"},
			},
		},
		{
			Name:        "Mimikatz_Pattern",
			Description: "Mimikatz credential dumper signatures",
			Severity:    "critical",
			ThreatScore: 90,
			Patterns: []Pattern{
				{Type: "string", Value: "sekurlsa::logonpasswords"},
				{Type: "string", Value: "mimikatz"},
				{Type: "string", Value: "gentilkiwi"},
			},
		},
		{
			Name:        "SSH_Private_Key",
			Description: "Unencrypted SSH private key in memory",
			Severity:    "high",
			ThreatScore: 70,
			Patterns: []Pattern{
				{Type: "string", Value: "-----BEGIN RSA PRIVATE KEY-----"},
				{Type: "string", Value: "-----BEGIN OPENSSH PRIVATE KEY-----"},
			},
		},
		{
			Name:        "AWS_Credentials",
			Description: "AWS access keys in memory",
			Severity:    "high",
			ThreatScore: 75,
			Patterns: []Pattern{
				{Type: "regex", Value: "AKIA[0-9A-Z]{16}"},
			},
		},
		{
			Name:        "Linux_Rootkit_LKM",
			Description: "Linux rootkit loadable kernel module patterns",
			Severity:    "critical",
			ThreatScore: 92,
			Patterns: []Pattern{
				{Type: "string", Value: "hide_pid"},
				{Type: "string", Value: "give_root"},
				{Type: "hex", Value: "48c7c0010000004889c7"}, // syscall hooking pattern
			},
		},
		{
			Name:        "Code_Injection_Marker",
			Description: "Common code injection techniques",
			Severity:    "high",
			ThreatScore: 80,
			Patterns: []Pattern{
				{Type: "hex", Value: "eb14909090"}, // jmp short; nop; nop; nop
				{Type: "hex", Value: "558bec83ec"}, // push ebp; mov ebp, esp; sub esp
			},
		},
		{
			Name:        "Cryptocurrency_Miner",
			Description: "Cryptocurrency mining software patterns",
			Severity:    "medium",
			ThreatScore: 60,
			Patterns: []Pattern{
				{Type: "string", Value: "xmrig"},
				{Type: "string", Value: "stratum+tcp"},
				{Type: "string", Value: "cryptonight"},
			},
		},
	}

	amf.rules = append(amf.rules, rules...)

	// Compile regex patterns
	for _, r := range rules {
		for _, p := range r.Patterns {
			if p.Type == "regex" {
				compiled, err := regexp.Compile(p.Value)
				if err == nil {
					amf.stringScanner.patterns = append(amf.stringScanner.patterns, compiled)
				}
			}
		}
	}
}

// scanWithRules applies all forensic rules to memory
func (amf *AdvancedMemoryForensics) scanWithRules(memory []byte) []*Artifact {
	artifacts := make([]*Artifact, 0)

	for _, rule := range amf.rules {
		matches := amf.applyRule(rule, memory)
		artifacts = append(artifacts, matches...)
	}

	return artifacts
}

// applyRule checks if rule patterns match in memory
func (amf *AdvancedMemoryForensics) applyRule(rule *ForensicRule, memory []byte) []*Artifact {
	artifacts := make([]*Artifact, 0)
	matchedPatterns := 0

	for _, pattern := range rule.Patterns {
		var matches []int64

		switch pattern.Type {
		case "hex":
			matches = amf.searchHex(memory, pattern.Value)
		case "string":
			matches = amf.searchString(memory, pattern.Value)
		case "regex":
			matches = amf.searchRegex(memory, pattern.Value)
		}

		if len(matches) > 0 {
			matchedPatterns++

			// Create artifact for each match
			for _, offset := range matches {
				// Extract surrounding context (32 bytes before/after)
				start := int(offset) - 32
				if start < 0 {
					start = 0
				}
				end := int(offset) + 32 + len(pattern.Value)
				if end > len(memory) {
					end = len(memory)
				}

				data := memory[start:end]
				h := sha256.Sum256(data)

				artifact := &Artifact{
					RuleName:    rule.Name,
					Description: rule.Description,
					Offset:      offset,
					Data:        data,
					Hash:        hex.EncodeToString(h[:]),
					Severity:    rule.Severity,
					Category:    "pattern_match",
					Timestamp:   currentTimestamp(),
				}

				artifacts = append(artifacts, artifact)

				// Limit hits per pattern
				if pattern.MaxHits > 0 && len(artifacts) >= pattern.MaxHits {
					break
				}
			}
		}
	}

	// Rule condition evaluation (simple: require all patterns OR any critical pattern)
	if matchedPatterns < len(rule.Patterns) && rule.Severity != "critical" {
		return nil // Rule not satisfied
	}

	return artifacts
}

// searchHex finds hex pattern in memory
func (amf *AdvancedMemoryForensics) searchHex(memory []byte, hexPattern string) []int64 {
	pattern, err := hex.DecodeString(hexPattern)
	if err != nil {
		return nil
	}

	offsets := make([]int64, 0)
	for i := 0; i <= len(memory)-len(pattern); i++ {
		if bytes.Equal(memory[i:i+len(pattern)], pattern) {
			offsets = append(offsets, int64(i))
		}
	}

	return offsets
}

// searchString finds string pattern in memory
func (amf *AdvancedMemoryForensics) searchString(memory []byte, str string) []int64 {
	pattern := []byte(str)
	offsets := make([]int64, 0)

	for i := 0; i <= len(memory)-len(pattern); i++ {
		if bytes.Equal(memory[i:i+len(pattern)], pattern) {
			offsets = append(offsets, int64(i))
		}
	}

	return offsets
}

// searchRegex finds regex matches in memory
func (amf *AdvancedMemoryForensics) searchRegex(memory []byte, regexStr string) []int64 {
	re, err := regexp.Compile(regexStr)
	if err != nil {
		return nil
	}

	offsets := make([]int64, 0)
	matches := re.FindAllIndex(memory, -1)

	for _, match := range matches {
		offsets = append(offsets, int64(match[0]))
	}

	return offsets
}

// detectShellcode uses heuristics to identify shellcode
func (amf *AdvancedMemoryForensics) detectShellcode(memory []byte) []*Artifact {
	artifacts := make([]*Artifact, 0)

	// Look for common shellcode characteristics:
	// 1. High frequency of NOP sleds (0x90)
	// 2. Suspicious instruction sequences
	// 3. Executable sections with unusual patterns

	windowSize := 256
	nopThreshold := 10 // Consecutive NOPs

	for i := 0; i <= len(memory)-windowSize; i++ {
		window := memory[i : i+windowSize]

		// Count NOPs
		nopCount := 0
		maxConsecutiveNops := 0
		currentNops := 0

		for _, b := range window {
			if b == 0x90 { // NOP instruction
				currentNops++
				if currentNops > maxConsecutiveNops {
					maxConsecutiveNops = currentNops
				}
				nopCount++
			} else {
				currentNops = 0
			}
		}

		// Suspicious NOP sled detected
		if maxConsecutiveNops >= nopThreshold {
			h := sha256.Sum256(window)
			artifact := &Artifact{
				RuleName:    "Shellcode_NOP_Sled",
				Description: fmt.Sprintf("NOP sled detected (%d consecutive NOPs)", maxConsecutiveNops),
				Offset:      int64(i),
				Data:        window,
				Hash:        hex.EncodeToString(h[:]),
				Severity:    "critical",
				Category:    "shellcode",
				Timestamp:   currentTimestamp(),
			}
			artifacts = append(artifacts, artifact)

			i += windowSize // Skip analyzed region
		}
	}

	return artifacts
}

// detectSecrets searches for credentials and secrets
func (amf *AdvancedMemoryForensics) detectSecrets(memory []byte) []*Artifact {
	artifacts := make([]*Artifact, 0)

	// Common secret patterns
	secretPatterns := []struct {
		name    string
		pattern string
	}{
		{"AWS_Access_Key", "AKIA[0-9A-Z]{16}"},
		{"GitHub_Token", "gh[pousr]_[A-Za-z0-9]{36}"},
		{"Slack_Token", "xox[baprs]-[0-9]{10,13}-[0-9]{10,13}-[A-Za-z0-9]{24}"},
		{"Private_Key_Header", "-----BEGIN.*PRIVATE KEY-----"},
		{"JWT_Token", "eyJ[A-Za-z0-9_-]+\\.[A-Za-z0-9_-]+\\.[A-Za-z0-9_-]+"},
		{"Password_Field", "password[\"']?\\s*[:=]\\s*[\"']([^\"']+)"},
	}

	for _, sp := range secretPatterns {
		re, err := regexp.Compile(sp.pattern)
		if err != nil {
			continue
		}

		matches := re.FindAllIndex(memory, -1)
		for _, match := range matches {
			start := match[0]
			end := match[1]
			if end > len(memory) {
				end = len(memory)
			}

			data := memory[start:end]
			h := sha256.Sum256(data)

			artifact := &Artifact{
				RuleName:    sp.name,
				Description: fmt.Sprintf("Potential %s found in memory", sp.name),
				Offset:      int64(start),
				Data:        data,
				Hash:        hex.EncodeToString(h[:]),
				Severity:    "high",
				Category:    "credential",
				Timestamp:   currentTimestamp(),
			}

			artifacts = append(artifacts, artifact)
		}
	}

	return artifacts
}

// findHighEntropyRegions detects packed/encrypted sections
func (amf *AdvancedMemoryForensics) findHighEntropyRegions(memory []byte) []EntropyRegion {
	regions := make([]EntropyRegion, 0)
	chunkSize := 1024
	entropyThreshold := 7.5 // High entropy indicates encryption/packing

	for i := 0; i <= len(memory)-chunkSize; i += chunkSize {
		chunk := memory[i : i+chunkSize]
		entropy := calculateShannonEntropy(chunk)

		if entropy >= entropyThreshold {
			regions = append(regions, EntropyRegion{
				Offset:  int64(i),
				Size:    int64(chunkSize),
				Entropy: entropy,
			})
		}
	}

	return regions
}

// analyzeMemoryRegions identifies memory segments and their properties
func (amf *AdvancedMemoryForensics) analyzeMemoryRegions(memory []byte) []*ForensicMemoryRegion {
	// Simplified: divide into fixed-size regions
	// In production: parse actual memory map from /proc/[pid]/maps
	regionSize := 64 * 1024 // 64KB regions
	regions := make([]*ForensicMemoryRegion, 0)

	for i := 0; i < len(memory); i += regionSize {
		end := i + regionSize
		if end > len(memory) {
			end = len(memory)
		}

		region := &ForensicMemoryRegion{
			Start:      uint64(i),
			End:        uint64(end),
			Size:       uint64(end - i),
			Protection: "rw-", // Assume read-write
			Type:       "heap",
			Artifacts:  make([]*Artifact, 0),
		}

		regions = append(regions, region)
	}

	return regions
}

// ExtractStrings finds ASCII/Unicode strings in memory
func (ss *StringScanner) ExtractStrings(memory []byte) []string {
	strings := make([]string, 0)
	currentString := make([]byte, 0, ss.maxLen)

	for i := 0; i < len(memory); i++ {
		b := memory[i]

		// Printable ASCII
		if b >= 32 && b <= 126 {
			currentString = append(currentString, b)
			if len(currentString) >= ss.maxLen {
				if len(currentString) >= ss.minLen {
					strings = append(strings, string(currentString))
				}
				currentString = currentString[:0]
			}
		} else {
			// End of string
			if len(currentString) >= ss.minLen {
				str := string(currentString)
				// Filter out high-entropy strings (likely binary data)
				if calculateShannonEntropy(currentString) < ss.entropyMin {
					strings = append(strings, str)
				}
			}
			currentString = currentString[:0]
		}
	}

	// Final string
	if len(currentString) >= ss.minLen {
		strings = append(strings, string(currentString))
	}

	return strings
}

// calculateThreatScore aggregates threat indicators
func (amf *AdvancedMemoryForensics) calculateThreatScore(report *MemoryForensicsReport) int {
	score := 0

	// Weight artifacts by severity
	for _, artifact := range report.Artifacts {
		switch artifact.Severity {
		case "critical":
			score += 25
		case "high":
			score += 15
		case "medium":
			score += 5
		case "low":
			score += 1
		}
	}

	// Shellcode detection is critical
	if report.ShellcodeDetected {
		score += 30
	}

	// High entropy regions (packed/encrypted)
	score += len(report.HighEntropyRegions) * 2

	// Cap at 100
	if score > 100 {
		score = 100
	}

	return score
}

// MemoryForensicsReport contains analysis results
type MemoryForensicsReport struct {
	TotalSize          int
	Artifacts          []*Artifact
	Regions            []*ForensicMemoryRegion
	Strings            []string
	HighEntropyRegions []EntropyRegion
	ShellcodeDetected  bool
	ThreatScore        int
	Timestamp          int64
}

// EntropyRegion represents high-entropy memory section
type EntropyRegion struct {
	Offset  int64
	Size    int64
	Entropy float64
}

// calculateShannonEntropy measures randomness (0-8 for bytes)
func calculateShannonEntropy(data []byte) float64 {
	if len(data) == 0 {
		return 0
	}

	freq := make(map[byte]int)
	for _, b := range data {
		freq[b]++
	}

	entropy := 0.0
	dataLen := float64(len(data))

	for _, count := range freq {
		p := float64(count) / dataLen
		if p > 0 {
			entropy -= p * math.Log2(p)
		}
	}

	return entropy
}

// AddCustomRule allows adding user-defined detection rules
func (amf *AdvancedMemoryForensics) AddCustomRule(rule *ForensicRule) {
	amf.mu.Lock()
	defer amf.mu.Unlock()

	amf.rules = append(amf.rules, rule)
}

// ExportReport generates JSON report for SIEM integration
func (report *MemoryForensicsReport) ExportJSON() ([]byte, error) {
	return json.Marshal(report)
}
