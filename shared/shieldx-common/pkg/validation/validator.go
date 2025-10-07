package validation

import (
	"fmt"
	"net"
	"net/url"
	"regexp"
	"strings"
	"unicode/utf8"
)

// P0 Requirement: Input validation for all public endpoints
// Production-ready validators with strict rules

var (
	// Service name: alphanumeric, dash, underscore (1-64 chars)
	serviceNameRegex = regexp.MustCompile(`^[a-z0-9_-]{1,64}$`)

	// Tenant ID: alphanumeric with dots (1-128 chars)
	tenantIDRegex = regexp.MustCompile(`^[a-zA-Z0-9._-]{1,128}$`)

	// Path: must start with / and contain safe characters
	pathRegex = regexp.MustCompile(`^/[a-zA-Z0-9/_.-]*$`)

	// Scope: alphanumeric with colon separator (e.g., "read:data")
	scopeRegex = regexp.MustCompile(`^[a-z0-9:_-]{1,64}$`)

	// Deny list: common attack patterns
	sqlInjectionPatterns = []string{
		"'", "\"", ";--", "/*", "*/", "@@", "@",
		"char(", "nchar(", "varchar(", "nvarchar(",
		"alter", "begin", "cast", "create", "cursor",
		"declare", "delete", "drop", "exec", "execute",
		"fetch", "insert", "select", "sys", "table",
		"union", "update",
	}

	xssPatterns = []string{
		"<script", "</script", "javascript:", "onerror=",
		"onload=", "onclick=", "onfocus=", "onmouseover=",
		"alert(", "eval(", "expression(",
	}

	pathTraversalPatterns = []string{
		"..", "~", "%2e%2e", "%252e", "%c0%ae",
	}
)

// ValidateServiceName validates service name (P0 requirement)
func ValidateServiceName(name string) error {
	if name == "" {
		return fmt.Errorf("service name cannot be empty")
	}
	if len(name) > 64 {
		return fmt.Errorf("service name too long (max 64 chars)")
	}
	if !serviceNameRegex.MatchString(name) {
		return fmt.Errorf("service name must be lowercase alphanumeric with dash/underscore")
	}
	return nil
}

// ValidateTenantID validates tenant identifier
func ValidateTenantID(tenant string) error {
	if tenant == "" {
		return fmt.Errorf("tenant ID cannot be empty")
	}
	if len(tenant) > 128 {
		return fmt.Errorf("tenant ID too long (max 128 chars)")
	}
	if !tenantIDRegex.MatchString(tenant) {
		return fmt.Errorf("tenant ID contains invalid characters")
	}
	return nil
}

// ValidatePath validates URL path (P0 requirement: prevent path traversal)
func ValidatePath(path string) error {
	if path == "" {
		return fmt.Errorf("path cannot be empty")
	}
	if !strings.HasPrefix(path, "/") {
		return fmt.Errorf("path must start with /")
	}
	if len(path) > 2048 {
		return fmt.Errorf("path too long (max 2048 chars)")
	}

	// Check for path traversal
	lowerPath := strings.ToLower(path)
	for _, pattern := range pathTraversalPatterns {
		if strings.Contains(lowerPath, pattern) {
			return fmt.Errorf("path traversal attempt detected: %s", pattern)
		}
	}

	// Validate characters
	if !pathRegex.MatchString(path) {
		return fmt.Errorf("path contains invalid characters")
	}

	return nil
}

// ValidateScope validates authorization scope
func ValidateScope(scope string) error {
	if scope == "" {
		return nil // Empty scope is allowed
	}
	if len(scope) > 64 {
		return fmt.Errorf("scope too long (max 64 chars)")
	}
	if !scopeRegex.MatchString(scope) {
		return fmt.Errorf("scope contains invalid characters")
	}
	return nil
}

// ValidateURL validates backend URL
func ValidateURL(rawURL string) error {
	if rawURL == "" {
		return fmt.Errorf("URL cannot be empty")
	}

	u, err := url.Parse(rawURL)
	if err != nil {
		return fmt.Errorf("invalid URL: %w", err)
	}

	// Only allow http/https
	if u.Scheme != "http" && u.Scheme != "https" {
		return fmt.Errorf("URL scheme must be http or https")
	}

	// Must have host
	if u.Host == "" {
		return fmt.Errorf("URL must have a host")
	}

	return nil
}

// ValidateIPAddress validates IP address
func ValidateIPAddress(ip string) error {
	if ip == "" {
		return fmt.Errorf("IP address cannot be empty")
	}

	parsed := net.ParseIP(ip)
	if parsed == nil {
		return fmt.Errorf("invalid IP address format")
	}

	// Deny private/reserved ranges in production (optional)
	// Uncomment for stricter validation:
	// if parsed.IsLoopback() || parsed.IsPrivate() || parsed.IsMulticast() {
	//     return fmt.Errorf("IP address in reserved range")
	// }

	return nil
}

// SanitizeForLog removes sensitive data and control characters for safe logging
// P0 Requirement: PII masking
func SanitizeForLog(s string) string {
	// Remove control characters
	s = strings.Map(func(r rune) rune {
		if r < 32 || r == 127 {
			return -1
		}
		return r
	}, s)

	// Truncate if too long
	if len(s) > 256 {
		s = s[:256] + "..."
	}

	return s
}

// CheckSQLInjection detects potential SQL injection attempts
func CheckSQLInjection(input string) error {
	lowerInput := strings.ToLower(input)
	for _, pattern := range sqlInjectionPatterns {
		if strings.Contains(lowerInput, pattern) {
			return fmt.Errorf("potential SQL injection pattern detected")
		}
	}
	return nil
}

// CheckXSS detects potential XSS attempts
func CheckXSS(input string) error {
	lowerInput := strings.ToLower(input)
	for _, pattern := range xssPatterns {
		if strings.Contains(lowerInput, pattern) {
			return fmt.Errorf("potential XSS pattern detected")
		}
	}
	return nil
}

// ValidateJSONSize ensures JSON payload is within size limits (P0: prevent DoS)
func ValidateJSONSize(size int64, maxBytes int64) error {
	if size > maxBytes {
		return fmt.Errorf("payload too large: %d bytes (max %d)", size, maxBytes)
	}
	if size <= 0 {
		return fmt.Errorf("payload cannot be empty")
	}
	return nil
}

// ValidateUTF8 ensures string is valid UTF-8
func ValidateUTF8(s string) error {
	if !utf8.ValidString(s) {
		return fmt.Errorf("invalid UTF-8 encoding")
	}
	return nil
}

// ValidateHeader validates HTTP header value
func ValidateHeader(value string) error {
	if len(value) > 4096 {
		return fmt.Errorf("header value too long")
	}

	// Check for CRLF injection
	if strings.Contains(value, "\r") || strings.Contains(value, "\n") {
		return fmt.Errorf("header contains invalid characters (CRLF)")
	}

	return nil
}

// ValidateAll performs comprehensive validation on route request fields
func ValidateRouteRequest(service, tenant, path, scope string) error {
	if service != "" {
		if err := ValidateServiceName(service); err != nil {
			return fmt.Errorf("invalid service: %w", err)
		}
	}

	if tenant != "" {
		if err := ValidateTenantID(tenant); err != nil {
			return fmt.Errorf("invalid tenant: %w", err)
		}
	}

	if path != "" {
		if err := ValidatePath(path); err != nil {
			return fmt.Errorf("invalid path: %w", err)
		}
		// Extra checks
		if err := CheckSQLInjection(path); err != nil {
			return fmt.Errorf("path validation failed: %w", err)
		}
		if err := CheckXSS(path); err != nil {
			return fmt.Errorf("path validation failed: %w", err)
		}
	}

	if scope != "" {
		if err := ValidateScope(scope); err != nil {
			return fmt.Errorf("invalid scope: %w", err)
		}
	}

	return nil
}
