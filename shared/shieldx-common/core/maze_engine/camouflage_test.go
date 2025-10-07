package maze_engine

import (
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestCamouflageEngine_LoadTemplates(t *testing.T) {
	// Create temporary template directory
	tempDir := t.TempDir()

	// Create test template
	testTemplate := `{
		"name": "test_apache",
		"version": "2.4.54",
		"fingerprint_id": "test_apache_2454",
		"headers": {
			"Server": "Apache/2.4.54 (Test)",
			"X-Powered-By": "PHP/8.1.2"
		},
		"error_pages": {
			"404": {
				"title": "404 Not Found",
				"body": "<html><body><h1>Not Found</h1></body></html>",
				"content_type": "text/html"
			}
		},
		"behavioral_patterns": {
			"response_timing": {
				"min_ms": 50,
				"max_ms": 200,
				"distribution": "normal",
				"jitter_factor": 0.1
			}
		}
	}`

	templatePath := filepath.Join(tempDir, "test_apache.json")
	err := os.WriteFile(templatePath, []byte(testTemplate), 0644)
	if err != nil {
		t.Fatalf("Failed to write test template: %v", err)
	}

	// Initialize engine
	engine, err := NewCamouflageEngine(tempDir)
	if err != nil {
		t.Fatalf("Failed to create camouflage engine: %v", err)
	}

	// Test template loading
	template, err := engine.GetTemplate("test_apache")
	if err != nil {
		t.Fatalf("Failed to get template: %v", err)
	}

	if template.Name != "test_apache" {
		t.Errorf("Expected template name 'test_apache', got '%s'", template.Name)
	}

	if template.Version != "2.4.54" {
		t.Errorf("Expected version '2.4.54', got '%s'", template.Version)
	}

	if template.Headers["Server"] != "Apache/2.4.54 (Test)" {
		t.Errorf("Expected Server header 'Apache/2.4.54 (Test)', got '%s'", template.Headers["Server"])
	}
}

func TestCamouflageEngine_CreateSession(t *testing.T) {
	tempDir := t.TempDir()

	// Create minimal template
	testTemplate := `{
		"name": "test_nginx",
		"version": "1.22.1",
		"headers": {
			"Server": "nginx/1.22.1"
		},
		"behavioral_patterns": {
			"response_timing": {
				"min_ms": 30,
				"max_ms": 150
			}
		}
	}`

	templatePath := filepath.Join(tempDir, "test_nginx.json")
	os.WriteFile(templatePath, []byte(testTemplate), 0644)

	engine, err := NewCamouflageEngine(tempDir)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}

	// Test session creation
	session, err := engine.CreateSession("test_nginx", "192.168.1.100", "Mozilla/5.0 Test")
	if err != nil {
		t.Fatalf("Failed to create session: %v", err)
	}

	if session.Template.Name != "test_nginx" {
		t.Errorf("Expected template name 'test_nginx', got '%s'", session.Template.Name)
	}

	if session.ClientIP != "192.168.1.100" {
		t.Errorf("Expected client IP '192.168.1.100', got '%s'", session.ClientIP)
	}

	if session.UserAgent != "Mozilla/5.0 Test" {
		t.Errorf("Expected user agent 'Mozilla/5.0 Test', got '%s'", session.UserAgent)
	}

	// Test session retrieval
	retrievedSession, exists := engine.GetSession(session.ID)
	if !exists {
		t.Error("Session should exist")
	}

	if retrievedSession.ID != session.ID {
		t.Errorf("Expected session ID '%s', got '%s'", session.ID, retrievedSession.ID)
	}
}

func TestCamouflageEngine_ApplyTemplate(t *testing.T) {
	tempDir := t.TempDir()

	// Create template with error pages
	testTemplate := `{
		"name": "test_apache",
		"version": "2.4.54",
		"headers": {
			"Server": "Apache/2.4.54 (Ubuntu)",
			"X-Powered-By": "PHP/8.1.2"
		},
		"error_pages": {
			"403": {
				"title": "403 Forbidden",
				"body": "<html><body><h1>Forbidden</h1><p>Access denied to {{path}}</p></body></html>",
				"content_type": "text/html"
			},
			"404": {
				"title": "404 Not Found",
				"body": "<html><body><h1>Not Found</h1><p>{{path}} not found</p></body></html>",
				"content_type": "text/html"
			}
		},
		"behavioral_patterns": {
			"response_timing": {
				"min_ms": 10,
				"max_ms": 50,
				"distribution": "uniform"
			}
		}
	}`

	templatePath := filepath.Join(tempDir, "test_apache.json")
	os.WriteFile(templatePath, []byte(testTemplate), 0644)

	engine, err := NewCamouflageEngine(tempDir)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}

	session, err := engine.CreateSession("test_apache", "192.168.1.100", "nmap scanner")
	if err != nil {
		t.Fatalf("Failed to create session: %v", err)
	}

	tests := []struct {
		name           string
		path           string
		expectedStatus int
		expectedBody   string
	}{
		{
			name:           "Admin path should return 403",
			path:           "/admin/config.php",
			expectedStatus: 403,
			expectedBody:   "Forbidden",
		},
		{
			name:           "Normal path should return 200",
			path:           "/index.html",
			expectedStatus: 200,
			expectedBody:   "Apache/2.4.54",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			req := httptest.NewRequest("GET", "http://example.com"+test.path, nil)
			req.Header.Set("User-Agent", "nmap scanner")

			w := httptest.NewRecorder()

			start := time.Now()
			engine.ApplyTemplate(w, req, session)
			duration := time.Since(start)

			// Check status code
			if w.Code != test.expectedStatus {
				t.Errorf("Expected status %d, got %d", test.expectedStatus, w.Code)
			}

			// Check body contains expected content
			body := w.Body.String()
			if !strings.Contains(body, test.expectedBody) {
				t.Errorf("Expected body to contain '%s', got '%s'", test.expectedBody, body)
			}

			// Check headers
			serverHeader := w.Header().Get("Server")
			if serverHeader != "Apache/2.4.54 (Ubuntu)" {
				t.Errorf("Expected Server header 'Apache/2.4.54 (Ubuntu)', got '%s'", serverHeader)
			}

			// Check response timing (should be within configured range)
			if duration < 10*time.Millisecond || duration > 100*time.Millisecond {
				t.Errorf("Response time %v outside expected range", duration)
			}
		})
	}
}

func TestCamouflageEngine_VulnerabilityDetection(t *testing.T) {
	tempDir := t.TempDir()

	testTemplate := `{
		"name": "test_server",
		"headers": {
			"Server": "TestServer/1.0"
		},
		"error_pages": {
			"403": {
				"body": "<html><body><h1>Access Denied</h1></body></html>"
			}
		},
		"vulnerability_simulation": {
			"directory_traversal": {
				"enabled": true,
				"detection_patterns": ["../", "..\\", "%2e%2e"],
				"response": "403 Forbidden"
			}
		},
		"behavioral_patterns": {
			"response_timing": {
				"min_ms": 1,
				"max_ms": 10
			}
		}
	}`

	templatePath := filepath.Join(tempDir, "test_server.json")
	os.WriteFile(templatePath, []byte(testTemplate), 0644)

	engine, err := NewCamouflageEngine(tempDir)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}

	session, err := engine.CreateSession("test_server", "192.168.1.100", "nikto scanner")
	if err != nil {
		t.Fatalf("Failed to create session: %v", err)
	}

	vulnerabilityPaths := []string{
		"/../../etc/passwd",
		"/admin/../../../etc/shadow",
		"/test%2e%2e%2fetc%2fpasswd",
		"/.git/config",
		"/backup/database.sql",
	}

	for _, path := range vulnerabilityPaths {
		t.Run("Vulnerability path: "+path, func(t *testing.T) {
			req := httptest.NewRequest("GET", "http://example.com"+path, nil)
			w := httptest.NewRecorder()

			engine.ApplyTemplate(w, req, session)

			// Should return 403 or 400 for vulnerability probes
			if w.Code != 403 && w.Code != 400 {
				t.Errorf("Expected status 403 or 400 for vulnerability probe, got %d", w.Code)
			}
		})
	}
}

func TestCamouflageEngine_ResponseTiming(t *testing.T) {
	tempDir := t.TempDir()

	testTemplate := `{
		"name": "timing_test",
		"headers": {
			"Server": "TimingTest/1.0"
		},
		"behavioral_patterns": {
			"response_timing": {
				"min_ms": 100,
				"max_ms": 200,
				"distribution": "normal",
				"jitter_factor": 0.1
			}
		}
	}`

	templatePath := filepath.Join(tempDir, "timing_test.json")
	os.WriteFile(templatePath, []byte(testTemplate), 0644)

	engine, err := NewCamouflageEngine(tempDir)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}

	template, _ := engine.GetTemplate("timing_test")

	// Test multiple timing calculations
	timings := make([]time.Duration, 10)
	for i := 0; i < 10; i++ {
		delay := engine.calculateResponseDelay(template.BehavioralPatterns.ResponseTiming)
		timings[i] = delay

		// Should be within configured range (with some tolerance for jitter)
		if delay < 80*time.Millisecond || delay > 250*time.Millisecond {
			t.Errorf("Timing %v outside expected range", delay)
		}
	}

	// Check that timings are not all identical (randomness)
	allSame := true
	for i := 1; i < len(timings); i++ {
		if timings[i] != timings[0] {
			allSame = false
			break
		}
	}

	if allSame {
		t.Error("All timings are identical, expected randomness")
	}
}

func TestCamouflageEngine_VariableInterpolation(t *testing.T) {
	tempDir := t.TempDir()

	testTemplate := `{
		"name": "interpolation_test",
		"headers": {
			"Server": "InterpolationTest/1.0"
		},
		"error_pages": {
			"404": {
				"body": "<html><body><h1>Not Found</h1><p>{{path}} not found on {{host}}</p></body></html>"
			}
		},
		"behavioral_patterns": {
			"response_timing": {
				"min_ms": 1,
				"max_ms": 5
			}
		}
	}`

	templatePath := filepath.Join(tempDir, "interpolation_test.json")
	os.WriteFile(templatePath, []byte(testTemplate), 0644)

	engine, err := NewCamouflageEngine(tempDir)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}

	session, err := engine.CreateSession("interpolation_test", "192.168.1.100", "test agent")
	if err != nil {
		t.Fatalf("Failed to create session: %v", err)
	}

	req := httptest.NewRequest("GET", "http://example.com/nonexistent/file.php", nil)
	w := httptest.NewRecorder()

	engine.ApplyTemplate(w, req, session)

	body := w.Body.String()

	// Check that variables were interpolated
	if !strings.Contains(body, "/nonexistent/file.php") {
		t.Error("Path variable not interpolated correctly")
	}

	if !strings.Contains(body, "example.com") {
		t.Error("Host variable not interpolated correctly")
	}
}

func BenchmarkCamouflageEngine_ApplyTemplate(b *testing.B) {
	tempDir := b.TempDir()

	testTemplate := `{
		"name": "benchmark_test",
		"headers": {
			"Server": "BenchmarkTest/1.0",
			"X-Powered-By": "PHP/8.1.2"
		},
		"behavioral_patterns": {
			"response_timing": {
				"min_ms": 1,
				"max_ms": 2
			}
		}
	}`

	templatePath := filepath.Join(tempDir, "benchmark_test.json")
	os.WriteFile(templatePath, []byte(testTemplate), 0644)

	engine, err := NewCamouflageEngine(tempDir)
	if err != nil {
		b.Fatalf("Failed to create engine: %v", err)
	}

	session, err := engine.CreateSession("benchmark_test", "192.168.1.100", "benchmark agent")
	if err != nil {
		b.Fatalf("Failed to create session: %v", err)
	}

	req := httptest.NewRequest("GET", "http://example.com/test", nil)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		w := httptest.NewRecorder()
		engine.ApplyTemplate(w, req, session)
	}
}
