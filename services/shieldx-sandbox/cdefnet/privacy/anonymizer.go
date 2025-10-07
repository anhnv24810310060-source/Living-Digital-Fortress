package privacy

import (
	"crypto/rand"
	"fmt"
	"math"
	"regexp"
	"strings"
)

type Anonymizer struct {
	piiPatterns   []*regexp.Regexp
	epsilonBudget float64
	deltaBudget   float64
}

func NewAnonymizer() *Anonymizer {
	patterns := []*regexp.Regexp{
		regexp.MustCompile(`\b(?:\d{1,3}\.){3}\d{1,3}\b`),                                    // IPv4
		regexp.MustCompile(`\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b`),          // Email
		regexp.MustCompile(`\b(?:\d{4}[-\s]?){3}\d{4}\b`),                                   // Credit card
		regexp.MustCompile(`\b\d{3}-\d{2}-\d{4}\b`),                                         // SSN
	}

	return &Anonymizer{
		piiPatterns:   patterns,
		epsilonBudget: 1.0,
		deltaBudget:   0.001,
	}
}

func (a *Anonymizer) ScrubPII(data string) string {
	result := data
	for _, pattern := range a.piiPatterns {
		result = pattern.ReplaceAllStringFunc(result, func(match string) string {
			return fmt.Sprintf("[REDACTED_%d]", len(match))
		})
	}
	return result
}

func (a *Anonymizer) AddDifferentialPrivacyNoise(value float64, sensitivity float64) float64 {
	if a.epsilonBudget <= 0 {
		return value
	}

	scale := sensitivity / a.epsilonBudget
	noise := a.sampleLaplace(scale)

	return value + noise
}

func (a *Anonymizer) sampleLaplace(scale float64) float64 {
	bytes := make([]byte, 8)
	rand.Read(bytes)

	u := float64(int64(bytes[0])<<56|int64(bytes[1])<<48|int64(bytes[2])<<40|int64(bytes[3])<<32|
		int64(bytes[4])<<24|int64(bytes[5])<<16|int64(bytes[6])<<8|int64(bytes[7])) / float64(1<<63) - 0.5

	if u < 0 {
		return scale * math.Log(1+2*u)
	}
	return -scale * math.Log(1-2*u)
}

func (a *Anonymizer) IsValidIOC(iocType, value string) bool {
	value = strings.TrimSpace(value)
	if len(value) < 4 || len(value) > 1024 {
		return false
	}

	switch strings.ToLower(iocType) {
	case "hash":
		return regexp.MustCompile(`^[a-fA-F0-9]{32,128}$`).MatchString(value)
	case "domain":
		return regexp.MustCompile(`^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$`).MatchString(value)
	case "ip":
		return regexp.MustCompile(`^(?:\d{1,3}\.){3}\d{1,3}$`).MatchString(value)
	case "url":
		return strings.HasPrefix(value, "http://") || strings.HasPrefix(value, "https://")
	case "file_path":
		return len(value) > 0 && !strings.Contains(value, "\x00")
	default:
		return false
	}
}

func (a *Anonymizer) KAnonymize(data []string, k int) []string {
	if len(data) < k {
		return []string{}
	}

	result := make([]string, 0, len(data))
	for i := 0; i < len(data); i += k {
		end := i + k
		if end > len(data) {
			end = len(data)
		}

		if end-i >= k {
			generalized := fmt.Sprintf("[GROUP_%d_SIZE_%d]", i/k, end-i)
			for j := i; j < end; j++ {
				result = append(result, generalized)
			}
		}
	}

	return result
}