package validation

import (
	"encoding/json"
	"fmt"
	"net"
	"net/url"
	"regexp"
	"strings"
	"unicode/utf8"
)

// RequestValidator provides comprehensive input validation for production security
type RequestValidator struct {
	maxBodySize    int64
	allowedSchemes []string
	allowedHosts   []string // if set, restrict URLs to these hosts
}

// NewRequestValidator creates a production-grade input validator
func NewRequestValidator(maxBodySize int64) *RequestValidator {
	return &RequestValidator{
		maxBodySize:    maxBodySize,
		allowedSchemes: []string{"http", "https"},
	}
}

// ValidateJSON checks if input is valid JSON and within size limits
func (v *RequestValidator) ValidateJSON(data []byte) error {
	if int64(len(data)) > v.maxBodySize {
		return fmt.Errorf("request body exceeds maximum size %d bytes", v.maxBodySize)
	}
	var temp interface{}
	if err := json.Unmarshal(data, &temp); err != nil {
		return fmt.Errorf("invalid JSON: %w", err)
	}
	return nil
}

// ValidateString performs comprehensive string validation
func (v *RequestValidator) ValidateString(s string, maxLen int, allowEmpty bool) error {
	if !allowEmpty && strings.TrimSpace(s) == "" {
		return fmt.Errorf("string cannot be empty")
	}
	if !utf8.ValidString(s) {
		return fmt.Errorf("string contains invalid UTF-8")
	}
	if len(s) > maxLen {
		return fmt.Errorf("string exceeds maximum length %d", maxLen)
	}
	// Detect null bytes (common in injection attacks)
	if strings.Contains(s, "\x00") {
		return fmt.Errorf("string contains null byte")
	}
	return nil
}

// ValidateTenantID checks tenant identifier format (alphanumeric + hyphens/underscores)
func (v *RequestValidator) ValidateTenantID(tenant string) error {
	if err := v.ValidateString(tenant, 128, false); err != nil {
		return err
	}
	// Tenant IDs must be alphanumeric with hyphens/underscores
	matched, _ := regexp.MatchString(`^[a-zA-Z0-9_-]+$`, tenant)
	if !matched {
		return fmt.Errorf("invalid tenant ID format (alphanumeric, hyphens, underscores only)")
	}
	return nil
}

// ValidateScope checks scope format (e.g., "api", "wch", "admin")
func (v *RequestValidator) ValidateScope(scope string) error {
	if err := v.ValidateString(scope, 64, false); err != nil {
		return err
	}
	matched, _ := regexp.MatchString(`^[a-z][a-z0-9-]*$`, scope)
	if !matched {
		return fmt.Errorf("invalid scope format (lowercase alphanumeric with hyphens)")
	}
	return nil
}

// ValidatePath checks URL path for injection attacks
func (v *RequestValidator) ValidatePath(path string) error {
	if err := v.ValidateString(path, 2048, true); err != nil {
		return err
	}
	// Detect path traversal attempts
	if strings.Contains(path, "..") {
		return fmt.Errorf("path contains directory traversal")
	}
	// Check for encoded null bytes and other suspicious patterns
	suspicious := []string{
		"%00", "%0a", "%0d", // null, newline, carriage return
		"<script", "javascript:", "onerror=", // XSS patterns
		"union", "select", "drop", "insert", "update", "delete", // SQL injection (basic)
	}
	lowerPath := strings.ToLower(path)
	for _, pattern := range suspicious {
		if strings.Contains(lowerPath, pattern) {
			return fmt.Errorf("path contains suspicious pattern: %s", pattern)
		}
	}
	return nil
}

// ValidateURL checks URL format and scheme
func (v *RequestValidator) ValidateURL(rawURL string) error {
	if err := v.ValidateString(rawURL, 4096, false); err != nil {
		return err
	}
	parsed, err := url.Parse(rawURL)
	if err != nil {
		return fmt.Errorf("invalid URL: %w", err)
	}
	// Check scheme whitelist
	schemeAllowed := false
	for _, allowed := range v.allowedSchemes {
		if parsed.Scheme == allowed {
			schemeAllowed = true
			break
		}
	}
	if !schemeAllowed {
		return fmt.Errorf("URL scheme %s not allowed", parsed.Scheme)
	}
	// Check host whitelist if configured
	if len(v.allowedHosts) > 0 {
		hostAllowed := false
		for _, allowed := range v.allowedHosts {
			if parsed.Host == allowed || strings.HasSuffix(parsed.Host, "."+allowed) {
				hostAllowed = true
				break
			}
		}
		if !hostAllowed {
			return fmt.Errorf("URL host %s not allowed", parsed.Host)
		}
	}
	return nil
}

// ValidateIPAddress checks if string is valid IPv4 or IPv6
func (v *RequestValidator) ValidateIPAddress(ip string) error {
	if net.ParseIP(ip) == nil {
		return fmt.Errorf("invalid IP address")
	}
	return nil
}

// ValidatePort checks if port number is valid
func (v *RequestValidator) ValidatePort(port int) error {
	if port < 1 || port > 65535 {
		return fmt.Errorf("invalid port number (must be 1-65535)")
	}
	return nil
}

// DetectSQLInjection performs heuristic SQL injection detection
func (v *RequestValidator) DetectSQLInjection(input string) bool {
	// Common SQL injection patterns
	sqlPatterns := []string{
		`(?i)\bunion\b.*\bselect\b`,
		`(?i)\bor\b.*=.*`,
		`(?i)\band\b.*=.*`,
		`(?i);\s*drop\b`,
		`(?i);\s*delete\b`,
		`(?i);\s*update\b`,
		`(?i);\s*insert\b`,
		`(?i)--`,  // SQL comments
		`(?i)/\*`, // SQL block comments
		`(?i)\bexec\b.*\(`,
		`(?i)\bexecute\b.*\(`,
		`'.*or.*'.*=.*'`, // Classic: ' or '1'='1
	}
	for _, pattern := range sqlPatterns {
		matched, _ := regexp.MatchString(pattern, input)
		if matched {
			return true
		}
	}
	return false
}

// DetectXSS performs heuristic XSS detection
func (v *RequestValidator) DetectXSS(input string) bool {
	// Common XSS patterns
	xssPatterns := []string{
		`(?i)<script`,
		`(?i)javascript:`,
		`(?i)onerror\s*=`,
		`(?i)onload\s*=`,
		`(?i)onclick\s*=`,
		`(?i)<iframe`,
		`(?i)<object`,
		`(?i)<embed`,
		`(?i)eval\s*\(`,
		`(?i)expression\s*\(`, // IE CSS expression
	}
	for _, pattern := range xssPatterns {
		matched, _ := regexp.MatchString(pattern, input)
		if matched {
			return true
		}
	}
	return false
}

// DetectCommandInjection performs heuristic command injection detection
func (v *RequestValidator) DetectCommandInjection(input string) bool {
	// Common command injection patterns
	cmdPatterns := []string{
		`[;&|]\s*\w+`, // command chaining
		`\$\(`,        // command substitution
		"``",          // backticks
		`>\s*/`,       // output redirection
		`<\s*/`,       // input redirection
	}
	for _, pattern := range cmdPatterns {
		matched, _ := regexp.MatchString(pattern, input)
		if matched {
			return true
		}
	}
	return false
}

// SanitizeForLog removes sensitive data and dangerous characters from log output
func (v *RequestValidator) SanitizeForLog(input string, maxLen int) string {
	if maxLen <= 0 {
		maxLen = 256
	}
	// Truncate
	if len(input) > maxLen {
		input = input[:maxLen] + "..."
	}
	// Remove control characters except space, tab, newline
	sanitized := strings.Map(func(r rune) rune {
		if r == ' ' || r == '\t' || r == '\n' {
			return r
		}
		if r < 32 || r == 127 {
			return -1 // remove
		}
		return r
	}, input)

	// Mask common sensitive patterns
	// Credit card (basic): replace middle digits
	ccRegex := regexp.MustCompile(`\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b`)
	sanitized = ccRegex.ReplaceAllStringFunc(sanitized, func(s string) string {
		if len(s) >= 8 {
			return s[:4] + "********" + s[len(s)-4:]
		}
		return "****"
	})

	// Email: keep domain, mask username
	emailRegex := regexp.MustCompile(`\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b`)
	sanitized = emailRegex.ReplaceAllStringFunc(sanitized, func(s string) string {
		parts := strings.Split(s, "@")
		if len(parts) == 2 && len(parts[0]) > 2 {
			return parts[0][:2] + "***@" + parts[1]
		}
		return "***@" + parts[len(parts)-1]
	})

	return sanitized
}

// ValidateRouteRequest validates orchestrator /route request
func (v *RequestValidator) ValidateRouteRequest(service, tenant, scope, path, hashKey string) error {
	if service != "" {
		if err := v.ValidateString(service, 128, false); err != nil {
			return fmt.Errorf("invalid service: %w", err)
		}
		// Service names must be alphanumeric with hyphens
		matched, _ := regexp.MatchString(`^[a-z][a-z0-9-]*$`, service)
		if !matched {
			return fmt.Errorf("invalid service name format")
		}
	}

	if tenant != "" {
		if err := v.ValidateTenantID(tenant); err != nil {
			return fmt.Errorf("invalid tenant: %w", err)
		}
	}

	if scope != "" {
		if err := v.ValidateScope(scope); err != nil {
			return fmt.Errorf("invalid scope: %w", err)
		}
	}

	if path != "" {
		if err := v.ValidatePath(path); err != nil {
			return fmt.Errorf("invalid path: %w", err)
		}
	}

	if hashKey != "" {
		if err := v.ValidateString(hashKey, 256, true); err != nil {
			return fmt.Errorf("invalid hashKey: %w", err)
		}
	}

	// Check for injection attacks
	allFields := service + tenant + scope + path + hashKey
	if v.DetectSQLInjection(allFields) {
		return fmt.Errorf("potential SQL injection detected")
	}
	if v.DetectXSS(allFields) {
		return fmt.Errorf("potential XSS detected")
	}
	if v.DetectCommandInjection(allFields) {
		return fmt.Errorf("potential command injection detected")
	}

	return nil
}
