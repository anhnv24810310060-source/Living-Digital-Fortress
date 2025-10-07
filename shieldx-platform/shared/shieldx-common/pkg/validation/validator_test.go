package validation

import (
	"strings"
	"testing"
)

// P0 Requirement: Unit test coverage >= 80%

func TestValidateServiceName(t *testing.T) {
	tests := []struct {
		name      string
		input     string
		wantError bool
	}{
		{"valid lowercase", "my-service", false},
		{"valid with underscore", "my_service", false},
		{"valid with numbers", "service123", false},
		{"empty string", "", true},
		{"too long", strings.Repeat("a", 65), true},
		{"uppercase", "MyService", true},
		{"special chars", "my@service", true},
		{"spaces", "my service", true},
		{"dot", "my.service", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateServiceName(tt.input)
			if (err != nil) != tt.wantError {
				t.Errorf("ValidateServiceName(%q) error = %v, wantError %v", tt.input, err, tt.wantError)
			}
		})
	}
}

func TestValidatePath(t *testing.T) {
	tests := []struct {
		name      string
		input     string
		wantError bool
	}{
		{"valid root", "/", false},
		{"valid path", "/api/v1/users", false},
		{"valid with dash", "/api-v1/users", false},
		{"empty", "", true},
		{"no leading slash", "api/users", true},
		{"path traversal ..", "/api/../etc/passwd", true},
		{"path traversal encoded", "/api/%2e%2e/etc/passwd", true},
		{"too long", "/" + strings.Repeat("a", 2048), true},
		{"invalid chars", "/api/<script>", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidatePath(tt.input)
			if (err != nil) != tt.wantError {
				t.Errorf("ValidatePath(%q) error = %v, wantError %v", tt.input, err, tt.wantError)
			}
		})
	}
}

func TestValidateTenantID(t *testing.T) {
	tests := []struct {
		name      string
		input     string
		wantError bool
	}{
		{"valid simple", "tenant1", false},
		{"valid with dash", "tenant-1", false},
		{"valid with dot", "tenant.1", false},
		{"valid mixed case", "Tenant1", false},
		{"empty", "", true},
		{"too long", strings.Repeat("a", 129), true},
		{"special chars", "tenant@1", true},
		{"spaces", "tenant 1", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateTenantID(tt.input)
			if (err != nil) != tt.wantError {
				t.Errorf("ValidateTenantID(%q) error = %v, wantError %v", tt.input, err, tt.wantError)
			}
		})
	}
}

func TestValidateScope(t *testing.T) {
	tests := []struct {
		name      string
		input     string
		wantError bool
	}{
		{"valid simple", "read", false},
		{"valid with colon", "read:data", false},
		{"valid complex", "read:user:profile", false},
		{"empty", "", false}, // Empty is allowed
		{"too long", strings.Repeat("a", 65), true},
		{"uppercase", "READ", true},
		{"special chars", "read@data", true},
		{"spaces", "read data", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateScope(tt.input)
			if (err != nil) != tt.wantError {
				t.Errorf("ValidateScope(%q) error = %v, wantError %v", tt.input, err, tt.wantError)
			}
		})
	}
}

func TestValidateURL(t *testing.T) {
	tests := []struct {
		name      string
		input     string
		wantError bool
	}{
		{"valid http", "http://example.com", false},
		{"valid https", "https://example.com:8080", false},
		{"valid with path", "http://example.com/api", false},
		{"empty", "", true},
		{"no scheme", "example.com", true},
		{"invalid scheme", "ftp://example.com", true},
		{"no host", "http://", true},
		{"malformed", "http://[invalid", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateURL(tt.input)
			if (err != nil) != tt.wantError {
				t.Errorf("ValidateURL(%q) error = %v, wantError %v", tt.input, err, tt.wantError)
			}
		})
	}
}

func TestCheckSQLInjection(t *testing.T) {
	tests := []struct {
		name      string
		input     string
		wantError bool
	}{
		{"safe string", "normal text", false},
		{"safe path", "/api/users/123", false},
		{"single quote", "'; DROP TABLE users;--", true},
		{"union select", "1 UNION SELECT * FROM users", true},
		{"exec", "'; EXEC sp_executesql", true},
		{"comment", "/* malicious */", true},
		{"double dash", "test;--", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := CheckSQLInjection(tt.input)
			if (err != nil) != tt.wantError {
				t.Errorf("CheckSQLInjection(%q) error = %v, wantError %v", tt.input, err, tt.wantError)
			}
		})
	}
}

func TestCheckXSS(t *testing.T) {
	tests := []struct {
		name      string
		input     string
		wantError bool
	}{
		{"safe string", "normal text", false},
		{"safe html", "hello world", false},
		{"script tag", "<script>alert('xss')</script>", true},
		{"javascript", "javascript:alert(1)", true},
		{"onerror", "<img onerror='alert(1)'>", true},
		{"onload", "<body onload=alert(1)>", true},
		{"eval", "eval('malicious')", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := CheckXSS(tt.input)
			if (err != nil) != tt.wantError {
				t.Errorf("CheckXSS(%q) error = %v, wantError %v", tt.input, err, tt.wantError)
			}
		})
	}
}

func TestValidateRouteRequest(t *testing.T) {
	tests := []struct {
		name      string
		service   string
		tenant    string
		path      string
		scope     string
		wantError bool
	}{
		{"all valid", "my-service", "tenant1", "/api/v1", "read:data", false},
		{"valid minimal", "service", "", "/", "", false},
		{"invalid service", "My-Service", "tenant1", "/api", "read", true},
		{"invalid tenant", "service", "tenant@1", "/api", "read", true},
		{"invalid path", "service", "tenant", "api", "read", true},   // no leading slash
		{"invalid scope", "service", "tenant", "/api", "READ", true}, // uppercase
		{"sql injection in path", "service", "tenant", "/api'; DROP TABLE", "", true},
		{"xss in path", "service", "tenant", "/api<script>", "", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateRouteRequest(tt.service, tt.tenant, tt.path, tt.scope)
			if (err != nil) != tt.wantError {
				t.Errorf("ValidateRouteRequest() error = %v, wantError %v", err, tt.wantError)
			}
		})
	}
}

func TestSanitizeForLog(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		maxLen   int
		contains string
	}{
		{"normal string", "hello world", 100, "hello world"},
		{"with control chars", "hello\x00world\n\r", 100, "helloworld"},
		{"long string", strings.Repeat("a", 300), 260, "..."},
		{"empty", "", 10, ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := SanitizeForLog(tt.input)
			if !strings.Contains(result, tt.contains) {
				t.Errorf("SanitizeForLog() = %q, want to contain %q", result, tt.contains)
			}
			if len(result) > 260 {
				t.Errorf("SanitizeForLog() result too long: %d chars", len(result))
			}
		})
	}
}

func BenchmarkValidateServiceName(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = ValidateServiceName("my-service-123")
	}
}

func BenchmarkValidatePath(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = ValidatePath("/api/v1/users/123")
	}
}

func BenchmarkCheckSQLInjection(b *testing.B) {
	input := "/api/users?id=123&name=test"
	for i := 0; i < b.N; i++ {
		_ = CheckSQLInjection(input)
	}
}

func BenchmarkValidateRouteRequest(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = ValidateRouteRequest("my-service", "tenant1", "/api/v1/users", "read:data")
	}
}
