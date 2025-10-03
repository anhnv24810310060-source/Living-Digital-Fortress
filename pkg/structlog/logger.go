package structlog
package structlog

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"runtime"
	"sync"
	"time"
)

// Level represents log severity
type Level int

const (
	LevelDebug Level = iota
	LevelInfo
	LevelWarn
	LevelError
	LevelFatal
)

func (l Level) String() string {
	switch l {
	case LevelDebug:
		return "DEBUG"
	case LevelInfo:
		return "INFO"
	case LevelWarn:
		return "WARN"
	case LevelError:
		return "ERROR"
	case LevelFatal:
		return "FATAL"
	default:
		return "UNKNOWN"
	}
}

// ContextKey for correlation ID
type ctxKeyCorrID struct{}

// Fields represents structured log fields
type Fields map[string]interface{}

// Logger provides structured logging with correlation ID support
type Logger struct {
	service   string
	level     Level
	output    io.Writer
	mu        sync.Mutex
	fields    Fields // base fields for all logs
	sanitizer *Sanitizer
}

// Sanitizer masks sensitive data in logs
type Sanitizer struct {
	maskPatterns map[string]string // field name -> mask type
}

// NewSanitizer creates a log sanitizer
func NewSanitizer() *Sanitizer {
	return &Sanitizer{
		maskPatterns: map[string]string{
			"password":     "MASKED",
			"secret":       "MASKED",
			"token":        "MASKED",
			"apikey":       "MASKED",
			"credit_card":  "MASKED",
			"ssn":          "MASKED",
			"authorization": "MASKED",
		},
	}
}

// Sanitize masks sensitive fields
func (s *Sanitizer) Sanitize(fields Fields) Fields {
	cleaned := make(Fields, len(fields))
	for k, v := range fields {
		// Check if field name matches sensitive pattern
		masked := false
		for pattern, maskValue := range s.maskPatterns {
			if containsIgnoreCase(k, pattern) {
				cleaned[k] = maskValue
				masked = true
				break
			}
		}
		if !masked {
			cleaned[k] = v
		}
	}
	return cleaned
}

// NewLogger creates a structured logger for a service
func NewLogger(serviceName string, level Level, output io.Writer) *Logger {
	if output == nil {
		output = os.Stdout
	}
	return &Logger{
		service:   serviceName,
		level:     level,
		output:    output,
		fields:    Fields{},
		sanitizer: NewSanitizer(),
	}
}

// WithFields returns a logger with additional base fields
func (l *Logger) WithFields(fields Fields) *Logger {
	newLogger := &Logger{
		service:   l.service,
		level:     l.level,
		output:    l.output,
		sanitizer: l.sanitizer,
		fields:    make(Fields, len(l.fields)+len(fields)),
	}
	for k, v := range l.fields {
		newLogger.fields[k] = v
	}
	for k, v := range fields {
		newLogger.fields[k] = v
	}
	return newLogger
}

// WithContext extracts correlation ID from context and adds to logger
func (l *Logger) WithContext(ctx context.Context) *Logger {
	if corrID := GetCorrelationID(ctx); corrID != "" {
		return l.WithFields(Fields{"correlation_id": corrID})
	}
	return l
}

// Debug logs debug message
func (l *Logger) Debug(message string, fields Fields) {
	l.log(LevelDebug, message, fields)
}

// Info logs info message
func (l *Logger) Info(message string, fields Fields) {
	l.log(LevelInfo, message, fields)
}

// Warn logs warning message
func (l *Logger) Warn(message string, fields Fields) {
	l.log(LevelWarn, message, fields)
}

// Error logs error message
func (l *Logger) Error(message string, fields Fields) {
	l.log(LevelError, message, fields)
}

// Fatal logs fatal message and exits
func (l *Logger) Fatal(message string, fields Fields) {
	l.log(LevelFatal, message, fields)
	os.Exit(1)
}

// SecurityEvent logs security event with special marker
func (l *Logger) SecurityEvent(event string, fields Fields) {
	if fields == nil {
		fields = Fields{}
	}
	fields["event_type"] = "security"
	fields["security_event"] = event
	l.log(LevelWarn, fmt.Sprintf("SECURITY: %s", event), fields)
}

// AuditLog logs audit trail with immutable marker
func (l *Logger) AuditLog(action string, fields Fields) {
	if fields == nil {
		fields = Fields{}
	}
	fields["event_type"] = "audit"
	fields["audit_action"] = action
	l.log(LevelInfo, fmt.Sprintf("AUDIT: %s", action), fields)
}

// log is the core logging function
func (l *Logger) log(level Level, message string, fields Fields) {
	if level < l.level {
		return // skip logs below threshold
	}

	// Merge base fields with log-specific fields
	allFields := make(Fields, len(l.fields)+len(fields)+5)
	for k, v := range l.fields {
		allFields[k] = v
	}
	for k, v := range fields {
		allFields[k] = v
	}

	// Add standard fields
	allFields["timestamp"] = time.Now().UTC().Format(time.RFC3339Nano)
	allFields["level"] = level.String()
	allFields["service"] = l.service
	allFields["message"] = message

	// Add caller info for errors
	if level >= LevelError {
		if pc, file, line, ok := runtime.Caller(2); ok {
			allFields["caller"] = fmt.Sprintf("%s:%d", file, line)
			if fn := runtime.FuncForPC(pc); fn != nil {
				allFields["function"] = fn.Name()
			}
		}
	}

	// Sanitize sensitive data
	allFields = l.sanitizer.Sanitize(allFields)

	// Marshal to JSON
	l.mu.Lock()
	defer l.mu.Unlock()
	
	encoder := json.NewEncoder(l.output)
	if err := encoder.Encode(allFields); err != nil {
		// Fallback to stderr if encoding fails
		fmt.Fprintf(os.Stderr, "LOG_ERROR: failed to encode log: %v\n", err)
	}
}

// SetLevel changes log level
func (l *Logger) SetLevel(level Level) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.level = level
}

// GetLevel returns current log level
func (l *Logger) GetLevel() Level {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.level
}

// Correlation ID helpers

// NewCorrelationID generates a new correlation ID
func NewCorrelationID() string {
	// Use timestamp + random suffix for uniqueness and sortability
	return fmt.Sprintf("%d-%d", time.Now().UnixNano(), randomInt63())
}

// ContextWithCorrelationID returns context with correlation ID
func ContextWithCorrelationID(ctx context.Context, corrID string) context.Context {
	return context.WithValue(ctx, ctxKeyCorrID{}, corrID)
}

// GetCorrelationID extracts correlation ID from context
func GetCorrelationID(ctx context.Context) string {
	if corrID, ok := ctx.Value(ctxKeyCorrID{}).(string); ok {
		return corrID
	}
	return ""
}

// GetOrCreateCorrelationID gets existing or creates new correlation ID
func GetOrCreateCorrelationID(ctx context.Context) (context.Context, string) {
	if corrID := GetCorrelationID(ctx); corrID != "" {
		return ctx, corrID
	}
	corrID := NewCorrelationID()
	return ContextWithCorrelationID(ctx, corrID), corrID
}

// Helper functions

func containsIgnoreCase(s, substr string) bool {
	s = toLower(s)
	substr = toLower(substr)
	return contains(s, substr)
}

func toLower(s string) string {
	// Simple ASCII lowercase (avoid importing strings)
	result := make([]byte, len(s))
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= 'A' && c <= 'Z' {
			c += 'a' - 'A'
		}
		result[i] = c
	}
	return string(result)
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(substr) == 0 || indexString(s, substr) >= 0)
}

func indexString(s, substr string) int {
	n := len(substr)
	if n == 0 {
		return 0
	}
	for i := 0; i <= len(s)-n; i++ {
		if s[i:i+n] == substr {
			return i
		}
	}
	return -1
}

// Simple random for correlation ID (not cryptographic)
var rndState uint64 = uint64(time.Now().UnixNano())

func randomInt63() int64 {
	// Simple LCG (Linear Congruential Generator) for correlation IDs
	rndState = rndState*1103515245 + 12345
	return int64(rndState & 0x7FFFFFFFFFFFFFFF)
}

// Global logger instance (can be replaced)
var defaultLogger = NewLogger("default", LevelInfo, os.Stdout)

// Package-level convenience functions

func Debug(message string, fields Fields) {
	defaultLogger.Debug(message, fields)
}

func Info(message string, fields Fields) {
	defaultLogger.Info(message, fields)
}

func Warn(message string, fields Fields) {
	defaultLogger.Warn(message, fields)
}

func Error(message string, fields Fields) {
	defaultLogger.Error(message, fields)
}

func Fatal(message string, fields Fields) {
	defaultLogger.Fatal(message, fields)
}

func SecurityEvent(event string, fields Fields) {
	defaultLogger.SecurityEvent(event, fields)
}

func AuditLog(action string, fields Fields) {
	defaultLogger.AuditLog(action, fields)
}

// SetDefaultLogger replaces the global logger
func SetDefaultLogger(logger *Logger) {
	defaultLogger = logger
}
