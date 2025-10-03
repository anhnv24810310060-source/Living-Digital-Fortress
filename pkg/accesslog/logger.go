package accesslog

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

// P0 Requirement: Structured access logging with PII masking
// All logs must be JSON format with correlation ID

type LogLevel string

const (
	LogLevelInfo     LogLevel = "INFO"
	LogLevelWarn     LogLevel = "WARN"
	LogLevelError    LogLevel = "ERROR"
	LogLevelSecurity LogLevel = "SECURITY"
)

type AccessLogEntry struct {
	Timestamp     string            `json:"timestamp"`
	Level         LogLevel          `json:"level"`
	Service       string            `json:"service"`
	CorrelationID string            `json:"correlation_id"`
	Method        string            `json:"method,omitempty"`
	Path          string            `json:"path,omitempty"`
	ClientIP      string            `json:"client_ip,omitempty"`
	UserAgent     string            `json:"user_agent,omitempty"`
	StatusCode    int               `json:"status_code,omitempty"`
	Duration      int64             `json:"duration_ms,omitempty"`
	BytesIn       int64             `json:"bytes_in,omitempty"`
	BytesOut      int64             `json:"bytes_out,omitempty"`
	Error         string            `json:"error,omitempty"`
	Extra         map[string]string `json:"extra,omitempty"`
}

type SecurityEventEntry struct {
	Timestamp     string            `json:"timestamp"`
	Service       string            `json:"service"`
	CorrelationID string            `json:"correlation_id"`
	EventType     string            `json:"event_type"` // rate_limit, auth_fail, injection_attempt, etc.
	Severity      string            `json:"severity"`   // low, medium, high, critical
	ClientIP      string            `json:"client_ip,omitempty"`
	Details       map[string]string `json:"details,omitempty"`
	Action        string            `json:"action"` // allow, deny, block, alert
}

type Logger struct {
	accessFile   *os.File
	securityFile *os.File
	mu           sync.Mutex
	service      string
}

var (
	// Sensitive headers that should be masked
	sensitiveHeaders = map[string]bool{
		"authorization":       true,
		"cookie":              true,
		"x-api-key":           true,
		"x-auth-token":        true,
		"proxy-authorization": true,
	}
)

// NewLogger creates a new structured logger
func NewLogger(service, accessLogPath, securityLogPath string) (*Logger, error) {
	af, err := os.OpenFile(accessLogPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		return nil, fmt.Errorf("failed to open access log: %w", err)
	}

	sf, err := os.OpenFile(securityLogPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		af.Close()
		return nil, fmt.Errorf("failed to open security log: %w", err)
	}

	return &Logger{
		accessFile:   af,
		securityFile: sf,
		service:      service,
	}, nil
}

// Close closes log files
func (l *Logger) Close() error {
	l.mu.Lock()
	defer l.mu.Unlock()

	var errs []error
	if err := l.accessFile.Close(); err != nil {
		errs = append(errs, err)
	}
	if err := l.securityFile.Close(); err != nil {
		errs = append(errs, err)
	}

	if len(errs) > 0 {
		return fmt.Errorf("errors closing log files: %v", errs)
	}
	return nil
}

// LogAccess logs an HTTP request/response (P0: mask PII)
func (l *Logger) LogAccess(entry AccessLogEntry) error {
	entry.Timestamp = time.Now().UTC().Format(time.RFC3339Nano)
	entry.Service = l.service

	// Mask sensitive data
	entry.UserAgent = maskUserAgent(entry.UserAgent)
	entry.Path = maskQueryParams(entry.Path)

	l.mu.Lock()
	defer l.mu.Unlock()

	data, err := json.Marshal(entry)
	if err != nil {
		return err
	}

	_, err = l.accessFile.Write(append(data, '\n'))
	return err
}

// LogSecurity logs a security event (P0: immutable audit trail)
func (l *Logger) LogSecurity(event SecurityEventEntry) error {
	event.Timestamp = time.Now().UTC().Format(time.RFC3339Nano)
	event.Service = l.service

	l.mu.Lock()
	defer l.mu.Unlock()

	data, err := json.Marshal(event)
	if err != nil {
		return err
	}

	_, err = l.securityFile.Write(append(data, '\n'))
	return err
}

// LogHTTPRequest is a convenience method for logging HTTP requests
func (l *Logger) LogHTTPRequest(r *http.Request, statusCode int, duration time.Duration, bytesOut int64, err error, corrID string) error {
	entry := AccessLogEntry{
		Level:         LogLevelInfo,
		CorrelationID: corrID,
		Method:        r.Method,
		Path:          r.URL.Path,
		ClientIP:      extractClientIP(r),
		UserAgent:     r.UserAgent(),
		StatusCode:    statusCode,
		Duration:      duration.Milliseconds(),
		BytesIn:       r.ContentLength,
		BytesOut:      bytesOut,
	}

	if err != nil {
		entry.Level = LogLevelError
		entry.Error = err.Error()
	}

	return l.LogAccess(entry)
}

// LogSecurityEvent is a convenience method for logging security events
func (l *Logger) LogSecurityEvent(eventType, severity, action, clientIP, corrID string, details map[string]string) error {
	return l.LogSecurity(SecurityEventEntry{
		CorrelationID: corrID,
		EventType:     eventType,
		Severity:      severity,
		ClientIP:      clientIP,
		Details:       details,
		Action:        action,
	})
}

// LogRateLimitExceeded logs a rate limit event
func (l *Logger) LogRateLimitExceeded(clientIP, path, corrID string) error {
	return l.LogSecurityEvent(
		"rate_limit_exceeded",
		"medium",
		"deny",
		clientIP,
		corrID,
		map[string]string{
			"path": path,
		},
	)
}

// LogAuthenticationFailure logs an authentication failure
func (l *Logger) LogAuthenticationFailure(clientIP, reason, corrID string) error {
	return l.LogSecurityEvent(
		"authentication_failure",
		"high",
		"deny",
		clientIP,
		corrID,
		map[string]string{
			"reason": reason,
		},
	)
}

// LogInjectionAttempt logs a potential injection attack
func (l *Logger) LogInjectionAttempt(clientIP, attackType, path, corrID string) error {
	return l.LogSecurityEvent(
		"injection_attempt",
		"critical",
		"block",
		clientIP,
		corrID,
		map[string]string{
			"attack_type": attackType,
			"path":        maskSensitiveData(path),
		},
	)
}

// LogPolicyDeny logs a policy-based denial
func (l *Logger) LogPolicyDeny(clientIP, tenant, path, reason, corrID string) error {
	return l.LogSecurityEvent(
		"policy_deny",
		"medium",
		"deny",
		clientIP,
		corrID,
		map[string]string{
			"tenant": tenant,
			"path":   path,
			"reason": reason,
		},
	)
}

// Helper functions for PII masking

// maskUserAgent truncates user agent to first 100 chars
func maskUserAgent(ua string) string {
	if len(ua) > 100 {
		return ua[:100] + "..."
	}
	return ua
}

// maskQueryParams removes sensitive query parameters
func maskQueryParams(path string) string {
	idx := strings.Index(path, "?")
	if idx == -1 {
		return path
	}

	// Keep path, mask query string
	basePath := path[:idx]
	queryString := path[idx+1:]

	// List of sensitive parameter names to mask
	sensitiveParams := []string{"token", "secret", "password", "api_key", "apikey", "auth"}

	for _, param := range sensitiveParams {
		// Simple replacement (not production-quality URL parsing)
		for _, prefix := range []string{param + "=", param + "%3D"} {
			if strings.Contains(strings.ToLower(queryString), strings.ToLower(prefix)) {
				return basePath + "?[MASKED]"
			}
		}
	}

	return basePath + "?" + queryString
}

// maskSensitiveData truncates and masks data for security logs
func maskSensitiveData(s string) string {
	if len(s) > 200 {
		s = s[:200] + "..."
	}
	return s
}

// extractClientIP extracts client IP from request (considering proxies)
func extractClientIP(r *http.Request) string {
	// Check X-Forwarded-For header first
	if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
		parts := strings.Split(xff, ",")
		if len(parts) > 0 {
			return strings.TrimSpace(parts[0])
		}
	}

	// Check X-Real-IP header
	if xri := r.Header.Get("X-Real-IP"); xri != "" {
		return xri
	}

	// Fall back to RemoteAddr
	ip := r.RemoteAddr
	if idx := strings.LastIndex(ip, ":"); idx != -1 {
		ip = ip[:idx]
	}
	return ip
}

// IsSensitiveHeader checks if a header should be masked in logs
func IsSensitiveHeader(name string) bool {
	return sensitiveHeaders[strings.ToLower(name)]
}
