package privacy

import (
	"strings"
	"testing"
)

func TestScrubPII(t *testing.T) {
	anonymizer := NewAnonymizer()

	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "IPv4 address",
			input:    "Malicious IP: 192.168.1.1 detected",
			expected: "Malicious IP: [REDACTED_11] detected",
		},
		{
			name:     "Email address",
			input:    "Contact: admin@evil.com for details",
			expected: "Contact: [REDACTED_13] for details",
		},
		{
			name:     "Credit card",
			input:    "Card: 1234-5678-9012-3456",
			expected: "Card: [REDACTED_19]",
		},
		{
			name:     "SSN",
			input:    "SSN: 123-45-6789",
			expected: "SSN: [REDACTED_11]",
		},
		{
			name:     "Clean data",
			input:    "This is clean threat data",
			expected: "This is clean threat data",
		},
		{
			name:     "Multiple PII",
			input:    "IP 10.0.0.1 and email test@domain.com",
			expected: "IP [REDACTED_8] and email [REDACTED_15]",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			result := anonymizer.ScrubPII(test.input)
			if result != test.expected {
				t.Errorf("Expected %q, got %q", test.expected, result)
			}
		})
	}
}

func TestIsValidIOC(t *testing.T) {
	anonymizer := NewAnonymizer()

	tests := []struct {
		name    string
		iocType string
		value   string
		valid   bool
	}{
		{
			name:    "Valid MD5 hash",
			iocType: "hash",
			value:   "5d41402abc4b2a76b9719d911017c592",
			valid:   true,
		},
		{
			name:    "Valid SHA256 hash",
			iocType: "hash",
			value:   "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
			valid:   true,
		},
		{
			name:    "Invalid hash",
			iocType: "hash",
			value:   "not_a_hash",
			valid:   false,
		},
		{
			name:    "Valid domain",
			iocType: "domain",
			value:   "evil.com",
			valid:   true,
		},
		{
			name:    "Valid subdomain",
			iocType: "domain",
			value:   "malware.evil.com",
			valid:   true,
		},
		{
			name:    "Invalid domain",
			iocType: "domain",
			value:   "not..valid",
			valid:   false,
		},
		{
			name:    "Valid IP",
			iocType: "ip",
			value:   "192.168.1.1",
			valid:   true,
		},
		{
			name:    "Invalid IP",
			iocType: "ip",
			value:   "999.999.999.999",
			valid:   false,
		},
		{
			name:    "Valid HTTP URL",
			iocType: "url",
			value:   "http://evil.com/malware",
			valid:   true,
		},
		{
			name:    "Valid HTTPS URL",
			iocType: "url",
			value:   "https://evil.com/malware",
			valid:   true,
		},
		{
			name:    "Invalid URL",
			iocType: "url",
			value:   "ftp://evil.com",
			valid:   false,
		},
		{
			name:    "Empty value",
			iocType: "hash",
			value:   "",
			valid:   false,
		},
		{
			name:    "Too short value",
			iocType: "hash",
			value:   "abc",
			valid:   false,
		},
		{
			name:    "Unknown IOC type",
			iocType: "unknown",
			value:   "some_value",
			valid:   false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			result := anonymizer.IsValidIOC(test.iocType, test.value)
			if result != test.valid {
				t.Errorf("IOC %s:%s expected %v, got %v", test.iocType, test.value, test.valid, result)
			}
		})
	}
}

func TestKAnonymize(t *testing.T) {
	anonymizer := NewAnonymizer()

	tests := []struct {
		name     string
		data     []string
		k        int
		expected int // expected result length
	}{
		{
			name:     "Sufficient data for k=3",
			data:     []string{"a", "b", "c", "d", "e", "f"},
			k:        3,
			expected: 6,
		},
		{
			name:     "Insufficient data for k=5",
			data:     []string{"a", "b", "c"},
			k:        5,
			expected: 0,
		},
		{
			name:     "Exact k size",
			data:     []string{"a", "b", "c"},
			k:        3,
			expected: 3,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			result := anonymizer.KAnonymize(test.data, test.k)
			if len(result) != test.expected {
				t.Errorf("Expected length %d, got %d", test.expected, len(result))
			}

			// Check that all results in a group are the same
			if len(result) > 0 {
				for i := 0; i < len(result); i += test.k {
					end := i + test.k
					if end > len(result) {
						end = len(result)
					}

					if end-i >= test.k {
						groupValue := result[i]
						for j := i + 1; j < end; j++ {
							if result[j] != groupValue {
								t.Errorf("Group values not consistent: %s != %s", result[j], groupValue)
							}
						}

						if !strings.Contains(groupValue, "GROUP_") {
							t.Errorf("Expected generalized value, got %s", groupValue)
						}
					}
				}
			}
		})
	}
}

func TestDifferentialPrivacyNoise(t *testing.T) {
	anonymizer := NewAnonymizer()

	value := 100.0
	sensitivity := 1.0

	// Test that noise is added
	noisyValue1 := anonymizer.AddDifferentialPrivacyNoise(value, sensitivity)
	noisyValue2 := anonymizer.AddDifferentialPrivacyNoise(value, sensitivity)

	// Values should be different due to noise
	if noisyValue1 == noisyValue2 {
		t.Error("Expected different noisy values, got same values")
	}

	// Test with zero epsilon (no noise)
	anonymizer.epsilonBudget = 0
	noisyValue3 := anonymizer.AddDifferentialPrivacyNoise(value, sensitivity)
	if noisyValue3 != value {
		t.Errorf("Expected no noise with epsilon=0, got %f instead of %f", noisyValue3, value)
	}
}