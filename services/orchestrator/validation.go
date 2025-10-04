package main

import (
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"unicode/utf8"
)

// Production-grade input validation and sanitization
// Defense against: injection attacks, path traversal, XSS, malformed input

var (
	// Service name must be alphanumeric + hyphens/underscores only
	serviceNameRegex = regexp.MustCompile(`^[a-zA-Z0-9_-]{1,64}$`)

	// Tenant ID: alphanumeric + hyphens, 1-128 chars
	tenantIDRegex = regexp.MustCompile(`^[a-zA-Z0-9_-]{1,128}$`)

	// Scope: alphanumeric + dots/colons, 1-256 chars (for namespace-like scopes)
	scopeRegex = regexp.MustCompile(`^[a-zA-Z0-9._:-]{1,256}$`)

	// Path: must start with /, no path traversal patterns
	pathTraversalRegex = regexp.MustCompile(`\.\.|//|\\`)

	// SQL injection patterns (basic defense in depth)
	sqlInjectionRegex = regexp.MustCompile(`(?i)(union|select|insert|update|delete|drop|exec|script|javascript|onerror|alert)[\s\(]`)

	// Deny list: suspicious paths
	denyPaths = []string{
		"/.env",
		"/.git",
		"/.aws",
		"/.ssh",
		"/admin",
		"/phpmyadmin",
		"/wp-admin",
		"/.well-known/security.txt",
	}

	// Deny list: suspicious query parameters
	denyQueryKeys = []string{
		"__proto__",
		"constructor",
		"prototype",
		"exec",
		"eval",
	}
)

// ValidationError represents input validation failure
type ValidationError struct {
	Field   string `json:"field"`
	Message string `json:"message"`
	Value   string `json:"value,omitempty"`
}

func (e ValidationError) Error() string {
	return fmt.Sprintf("validation failed for %s: %s", e.Field, e.Message)
}

// ValidateServiceName validates service name format
func ValidateServiceName(service string) error {
	if service == "" {
		return ValidationError{Field: "service", Message: "service name is required"}
	}

	if !serviceNameRegex.MatchString(service) {
		return ValidationError{
			Field:   "service",
			Message: "service name must be alphanumeric with hyphens/underscores only",
			Value:   sanitizeForLog(service),
		}
	}

	return nil
}

// ValidateTenantID validates tenant ID format
func ValidateTenantID(tenant string) error {
	if tenant == "" {
		return ValidationError{Field: "tenant", Message: "tenant ID is required"}
	}

	if !tenantIDRegex.MatchString(tenant) {
		return ValidationError{
			Field:   "tenant",
			Message: "tenant ID must be alphanumeric with hyphens/underscores",
			Value:   sanitizeForLog(tenant),
		}
	}

	return nil
}

// ValidateScope validates scope format
func ValidateScope(scope string) error {
	if scope == "" {
		return nil // scope is optional
	}

	if !scopeRegex.MatchString(scope) {
		return ValidationError{
			Field:   "scope",
			Message: "scope contains invalid characters",
			Value:   sanitizeForLog(scope),
		}
	}

	return nil
}

// ValidatePath validates URL path and checks for security issues
func ValidatePath(path string) error {
	if path == "" {
		path = "/"
	}

	// Must start with /
	if !strings.HasPrefix(path, "/") {
		return ValidationError{
			Field:   "path",
			Message: "path must start with /",
			Value:   sanitizeForLog(path),
		}
	}

	// Check path traversal
	if pathTraversalRegex.MatchString(path) {
		return ValidationError{
			Field:   "path",
			Message: "path contains traversal patterns",
			Value:   sanitizeForLog(path),
		}
	}

	// Check for SQL injection patterns (defense in depth)
	if sqlInjectionRegex.MatchString(path) {
		return ValidationError{
			Field:   "path",
			Message: "path contains suspicious patterns",
			Value:   sanitizeForLog(path),
		}
	}

	// Length check
	if len(path) > 2048 {
		return ValidationError{
			Field:   "path",
			Message: "path exceeds maximum length of 2048 characters",
		}
	}

	// UTF-8 validation
	if !utf8.ValidString(path) {
		return ValidationError{
			Field:   "path",
			Message: "path contains invalid UTF-8",
		}
	}

	return nil
}

// ValidateHashKey validates optional hash key for consistent hashing
func ValidateHashKey(hashKey string) error {
	if hashKey == "" {
		return nil // optional field
	}

	// Length check
	if len(hashKey) > 512 {
		return ValidationError{
			Field:   "hashKey",
			Message: "hashKey exceeds maximum length of 512 characters",
		}
	}

	// UTF-8 validation
	if !utf8.ValidString(hashKey) {
		return ValidationError{
			Field:   "hashKey",
			Message: "hashKey contains invalid UTF-8",
		}
	}

	return nil
}

// ValidateURL validates backend URL format
func ValidateURL(rawURL string) error {
	if rawURL == "" {
		return ValidationError{Field: "url", Message: "URL is required"}
	}

	parsed, err := url.Parse(rawURL)
	if err != nil {
		return ValidationError{
			Field:   "url",
			Message: fmt.Sprintf("invalid URL format: %v", err),
			Value:   sanitizeForLog(rawURL),
		}
	}

	// Must be HTTP or HTTPS
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return ValidationError{
			Field:   "url",
			Message: "URL scheme must be http or https",
			Value:   sanitizeForLog(rawURL),
		}
	}

	// Must have host
	if parsed.Host == "" {
		return ValidationError{
			Field:   "url",
			Message: "URL must have a host",
			Value:   sanitizeForLog(rawURL),
		}
	}

	return nil
}

// CheckDenyList checks if path or query matches deny list patterns
func CheckDenyList(r *http.Request) error {
	path := r.URL.Path

	// Check path deny list
	for _, denied := range denyPaths {
		if strings.HasPrefix(strings.ToLower(path), strings.ToLower(denied)) {
			// Metrics will be tracked by caller
			return fmt.Errorf("path %s is denied", sanitizeForLog(path))
		}
	}

	// Check query parameter deny list
	query := r.URL.Query()
	for key := range query {
		for _, denied := range denyQueryKeys {
			if strings.EqualFold(key, denied) {
				// Metrics will be tracked by caller
				return fmt.Errorf("query parameter %s is denied", sanitizeForLog(key))
			}
		}
	}

	return nil
}

// sanitizeForLog sanitizes strings before logging to prevent log injection
func sanitizeForLog(s string) string {
	// Truncate long strings
	if len(s) > 100 {
		s = s[:100] + "..."
	}

	// Replace newlines and control characters
	s = strings.Map(func(r rune) rune {
		if r < 32 || r == 127 {
			return '?'
		}
		return r
	}, s)

	return s
}

// ValidateJSON validates JSON request body structure
func ValidateJSON(r *http.Request, maxBytes int64, target interface{}) error {
	// Limit request body size
	if maxBytes > 0 {
		r.Body = http.MaxBytesReader(nil, r.Body, maxBytes)
	}

	// Decode with strict settings
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields() // Reject unknown fields

	if err := dec.Decode(target); err != nil {
		return ValidationError{
			Field:   "body",
			Message: fmt.Sprintf("invalid JSON: %v", err),
		}
	}

	// Ensure no trailing data
	if dec.More() {
		return ValidationError{
			Field:   "body",
			Message: "request body contains multiple JSON objects",
		}
	}

	return nil
}

// ValidateClientIP validates and normalizes client IP address
func ValidateClientIP(ip string) (string, error) {
	if ip == "" {
		return "", ValidationError{Field: "ip", Message: "IP address is required"}
	}

	// Try to parse as IP
	parsed := net.ParseIP(ip)
	if parsed == nil {
		// Try to parse as IP:port
		host, _, err := net.SplitHostPort(ip)
		if err != nil {
			return "", ValidationError{
				Field:   "ip",
				Message: "invalid IP address format",
				Value:   sanitizeForLog(ip),
			}
		}
		parsed = net.ParseIP(host)
		if parsed == nil {
			return "", ValidationError{
				Field:   "ip",
				Message: "invalid IP address",
				Value:   sanitizeForLog(host),
			}
		}
		return parsed.String(), nil
	}

	return parsed.String(), nil
}

// ValidateRouteRequest validates complete route request
func ValidateRouteRequest(req *routeRequest) []ValidationError {
	var errors []ValidationError

	// Service validation
	if err := ValidateServiceName(req.Service); err != nil {
		if ve, ok := err.(ValidationError); ok {
			errors = append(errors, ve)
		}
	}

	// Tenant validation
	if err := ValidateTenantID(req.Tenant); err != nil {
		if ve, ok := err.(ValidationError); ok {
			errors = append(errors, ve)
		}
	}

	// Scope validation
	if err := ValidateScope(req.Scope); err != nil {
		if ve, ok := err.(ValidationError); ok {
			errors = append(errors, ve)
		}
	}

	// Path validation
	if err := ValidatePath(req.Path); err != nil {
		if ve, ok := err.(ValidationError); ok {
			errors = append(errors, ve)
		}
	}

	// HashKey validation
	if err := ValidateHashKey(req.HashKey); err != nil {
		if ve, ok := err.(ValidationError); ok {
			errors = append(errors, ve)
		}
	}

	// Validate candidate URLs if provided
	for i, candidate := range req.Candidates {
		if err := ValidateURL(candidate); err != nil {
			if ve, ok := err.(ValidationError); ok {
				ve.Field = fmt.Sprintf("candidates[%d]", i)
				errors = append(errors, ve)
			}
		}
	}

	return errors
}

// WriteValidationErrors writes validation errors as JSON response
func WriteValidationErrors(w http.ResponseWriter, errors []ValidationError) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusBadRequest)

	response := map[string]interface{}{
		"error":  "validation_failed",
		"errors": errors,
	}

	json.NewEncoder(w).Encode(response)
}
