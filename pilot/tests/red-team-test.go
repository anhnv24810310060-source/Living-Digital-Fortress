package main

import (
	"bytes"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

type RedTeamTest struct {
	baseURL    string
	client     *http.Client
	results    []TestResult
	mu         sync.Mutex
}

type TestResult struct {
	TestName    string    `json:"test_name"`
	Status      string    `json:"status"`
	Response    string    `json:"response"`
	Duration    int64     `json:"duration_ms"`
	Timestamp   time.Time `json:"timestamp"`
	Severity    string    `json:"severity"`
	Description string    `json:"description"`
}

type AttackVector struct {
	Name        string
	Method      string
	Path        string
	Headers     map[string]string
	Body        string
	Expected    string
	Severity    string
	Description string
}

func NewRedTeamTest(baseURL string) *RedTeamTest {
	tr := &http.Transport{
		TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
	}
	
	return &RedTeamTest{
		baseURL: baseURL,
		client: &http.Client{
			Transport: tr,
			Timeout:   10 * time.Second,
		},
		results: make([]TestResult, 0),
	}
}

func (rt *RedTeamTest) RunAllTests() {
	log.Println("Starting Red Team Security Assessment...")
	
	// SQL Injection Tests
	rt.runSQLInjectionTests()
	
	// XSS Tests
	rt.runXSSTests()
	
	// Authentication Bypass Tests
	rt.runAuthBypassTests()
	
	// Rate Limiting Tests
	rt.runRateLimitTests()
	
	// Directory Traversal Tests
	rt.runDirectoryTraversalTests()
	
	// Command Injection Tests
	rt.runCommandInjectionTests()
	
	// CSRF Tests
	rt.runCSRFTests()
	
	// Container Escape Tests
	rt.runContainerEscapeTests()
	
	// Generate Report
	rt.generateReport()
}

func (rt *RedTeamTest) runSQLInjectionTests() {
	log.Println("Running SQL Injection Tests...")
	
	vectors := []AttackVector{
		{
			Name:        "Basic SQL Injection",
			Method:      "POST",
			Path:        "/api/login",
			Body:        `{"username":"admin' OR '1'='1","password":"test"}`,
			Expected:    "blocked",
			Severity:    "Critical",
			Description: "Basic SQL injection attempt in login form",
		},
		{
			Name:        "Union-based SQL Injection",
			Method:      "GET",
			Path:        "/api/users?id=1 UNION SELECT password FROM users--",
			Expected:    "blocked",
			Severity:    "Critical",
			Description: "Union-based SQL injection to extract passwords",
		},
		{
			Name:        "Time-based Blind SQL Injection",
			Method:      "GET",
			Path:        "/api/search?q=test'; WAITFOR DELAY '00:00:05'--",
			Expected:    "blocked",
			Severity:    "High",
			Description: "Time-based blind SQL injection",
		},
	}
	
	for _, vector := range vectors {
		rt.executeTest(vector)
	}
}

func (rt *RedTeamTest) runXSSTests() {
	log.Println("Running XSS Tests...")
	
	vectors := []AttackVector{
		{
			Name:        "Reflected XSS",
			Method:      "GET",
			Path:        "/search?q=<script>alert('XSS')</script>",
			Expected:    "blocked",
			Severity:    "High",
			Description: "Reflected XSS in search parameter",
		},
		{
			Name:        "Stored XSS",
			Method:      "POST",
			Path:        "/api/comments",
			Body:        `{"comment":"<img src=x onerror=alert('XSS')>"}`,
			Expected:    "blocked",
			Severity:    "High",
			Description: "Stored XSS in comment field",
		},
		{
			Name:        "DOM-based XSS",
			Method:      "GET",
			Path:        "/profile?name=<svg/onload=alert('XSS')>",
			Expected:    "blocked",
			Severity:    "Medium",
			Description: "DOM-based XSS in profile name",
		},
	}
	
	for _, vector := range vectors {
		rt.executeTest(vector)
	}
}

func (rt *RedTeamTest) runAuthBypassTests() {
	log.Println("Running Authentication Bypass Tests...")
	
	vectors := []AttackVector{
		{
			Name:        "JWT Token Manipulation",
			Method:      "GET",
			Path:        "/api/admin/users",
			Headers:     map[string]string{"Authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJub25lIn0.eyJ1c2VyIjoiYWRtaW4ifQ."},
			Expected:    "blocked",
			Severity:    "Critical",
			Description: "JWT token with 'none' algorithm",
		},
		{
			Name:        "Session Fixation",
			Method:      "POST",
			Path:        "/api/login",
			Headers:     map[string]string{"Cookie": "SESSIONID=attacker_controlled_session"},
			Body:        `{"username":"user","password":"pass"}`,
			Expected:    "blocked",
			Severity:    "High",
			Description: "Session fixation attack",
		},
		{
			Name:        "Privilege Escalation",
			Method:      "PUT",
			Path:        "/api/users/1",
			Body:        `{"role":"admin","user_id":1}`,
			Expected:    "blocked",
			Severity:    "Critical",
			Description: "Privilege escalation attempt",
		},
	}
	
	for _, vector := range vectors {
		rt.executeTest(vector)
	}
}

func (rt *RedTeamTest) runRateLimitTests() {
	log.Println("Running Rate Limiting Tests...")
	
	// Concurrent requests to test rate limiting
	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			
			vector := AttackVector{
				Name:        fmt.Sprintf("Rate Limit Test %d", id),
				Method:      "POST",
				Path:        "/api/login",
				Body:        `{"username":"test","password":"test"}`,
				Expected:    "rate_limited",
				Severity:    "Medium",
				Description: "Brute force login attempt",
			}
			
			rt.executeTest(vector)
		}(i)
	}
	wg.Wait()
}

func (rt *RedTeamTest) runDirectoryTraversalTests() {
	log.Println("Running Directory Traversal Tests...")
	
	vectors := []AttackVector{
		{
			Name:        "Path Traversal - etc/passwd",
			Method:      "GET",
			Path:        "/api/files?path=../../../etc/passwd",
			Expected:    "blocked",
			Severity:    "High",
			Description: "Directory traversal to access /etc/passwd",
		},
		{
			Name:        "Path Traversal - Windows",
			Method:      "GET",
			Path:        "/api/files?path=..\\..\\..\\windows\\system32\\config\\sam",
			Expected:    "blocked",
			Severity:    "High",
			Description: "Directory traversal to access Windows SAM file",
		},
		{
			Name:        "Null Byte Injection",
			Method:      "GET",
			Path:        "/api/download?file=../../../etc/passwd%00.txt",
			Expected:    "blocked",
			Severity:    "High",
			Description: "Null byte injection for path traversal",
		},
	}
	
	for _, vector := range vectors {
		rt.executeTest(vector)
	}
}

func (rt *RedTeamTest) runCommandInjectionTests() {
	log.Println("Running Command Injection Tests...")
	
	vectors := []AttackVector{
		{
			Name:        "OS Command Injection",
			Method:      "POST",
			Path:        "/api/ping",
			Body:        `{"host":"127.0.0.1; cat /etc/passwd"}`,
			Expected:    "blocked",
			Severity:    "Critical",
			Description: "OS command injection in ping utility",
		},
		{
			Name:        "Blind Command Injection",
			Method:      "POST",
			Path:        "/api/system/info",
			Body:        `{"command":"whoami && sleep 5"}`,
			Expected:    "blocked",
			Severity:    "Critical",
			Description: "Blind command injection with time delay",
		},
	}
	
	for _, vector := range vectors {
		rt.executeTest(vector)
	}
}

func (rt *RedTeamTest) runCSRFTests() {
	log.Println("Running CSRF Tests...")
	
	vectors := []AttackVector{
		{
			Name:        "CSRF - Password Change",
			Method:      "POST",
			Path:        "/api/change-password",
			Body:        `{"new_password":"hacked123"}`,
			Expected:    "blocked",
			Severity:    "High",
			Description: "CSRF attack to change user password",
		},
		{
			Name:        "CSRF - Admin Action",
			Method:      "DELETE",
			Path:        "/api/admin/users/1",
			Expected:    "blocked",
			Severity:    "High",
			Description: "CSRF attack to delete user account",
		},
	}
	
	for _, vector := range vectors {
		rt.executeTest(vector)
	}
}

func (rt *RedTeamTest) runContainerEscapeTests() {
	log.Println("Running Container Escape Tests...")
	
	vectors := []AttackVector{
		{
			Name:        "Docker Socket Access",
			Method:      "GET",
			Path:        "/api/system/docker/containers",
			Expected:    "blocked",
			Severity:    "Critical",
			Description: "Attempt to access Docker socket",
		},
		{
			Name:        "Privileged Container Check",
			Method:      "POST",
			Path:        "/api/system/exec",
			Body:        `{"command":"mount"}`,
			Expected:    "blocked",
			Severity:    "High",
			Description: "Check for privileged container capabilities",
		},
		{
			Name:        "Proc Filesystem Access",
			Method:      "GET",
			Path:        "/api/files?path=/proc/1/cgroup",
			Expected:    "blocked",
			Severity:    "Medium",
			Description: "Access to container process information",
		},
	}
	
	for _, vector := range vectors {
		rt.executeTest(vector)
	}
}

func (rt *RedTeamTest) executeTest(vector AttackVector) {
	start := time.Now()
	
	var req *http.Request
	var err error
	
	if vector.Body != "" {
		req, err = http.NewRequest(vector.Method, rt.baseURL+vector.Path, strings.NewReader(vector.Body))
		if err != nil {
			rt.recordResult(vector.Name, "error", err.Error(), time.Since(start).Milliseconds(), vector.Severity, vector.Description)
			return
		}
		req.Header.Set("Content-Type", "application/json")
	} else {
		req, err = http.NewRequest(vector.Method, rt.baseURL+vector.Path, nil)
		if err != nil {
			rt.recordResult(vector.Name, "error", err.Error(), time.Since(start).Milliseconds(), vector.Severity, vector.Description)
			return
		}
	}
	
	// Add custom headers
	for key, value := range vector.Headers {
		req.Header.Set(key, value)
	}
	
	// Add attack signatures to User-Agent
	req.Header.Set("User-Agent", "RedTeam-Scanner/1.0 (Security-Test)")
	
	resp, err := rt.client.Do(req)
	if err != nil {
		rt.recordResult(vector.Name, "error", err.Error(), time.Since(start).Milliseconds(), vector.Severity, vector.Description)
		return
	}
	defer resp.Body.Close()
	
	body, _ := io.ReadAll(resp.Body)
	responseText := string(body)
	
	// Analyze response
	status := rt.analyzeResponse(resp.StatusCode, responseText, vector.Expected)
	
	rt.recordResult(vector.Name, status, fmt.Sprintf("HTTP %d: %s", resp.StatusCode, responseText), 
		time.Since(start).Milliseconds(), vector.Severity, vector.Description)
}

func (rt *RedTeamTest) analyzeResponse(statusCode int, body, expected string) string {
	switch expected {
	case "blocked":
		if statusCode == 403 || statusCode == 406 || statusCode == 429 || 
		   strings.Contains(body, "blocked") || strings.Contains(body, "forbidden") {
			return "PASS"
		}
		return "FAIL"
	case "rate_limited":
		if statusCode == 429 || strings.Contains(body, "rate limit") {
			return "PASS"
		}
		return "FAIL"
	default:
		if statusCode >= 400 {
			return "PASS"
		}
		return "FAIL"
	}
}

func (rt *RedTeamTest) recordResult(name, status, response string, duration int64, severity, description string) {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	
	result := TestResult{
		TestName:    name,
		Status:      status,
		Response:    response,
		Duration:    duration,
		Timestamp:   time.Now(),
		Severity:    severity,
		Description: description,
	}
	
	rt.results = append(rt.results, result)
	
	// Log result
	log.Printf("[%s] %s: %s (%dms)", severity, name, status, duration)
}

func (rt *RedTeamTest) generateReport() {
	log.Println("Generating Red Team Assessment Report...")
	
	report := map[string]interface{}{
		"assessment_date": time.Now().Format(time.RFC3339),
		"target_url":      rt.baseURL,
		"total_tests":     len(rt.results),
		"summary":         rt.generateSummary(),
		"results":         rt.results,
		"recommendations": rt.generateRecommendations(),
	}
	
	// Write to file
	reportJSON, _ := json.MarshalIndent(report, "", "  ")
	filename := fmt.Sprintf("red-team-report-%s.json", time.Now().Format("20060102-150405"))
	
	err := os.WriteFile(filename, reportJSON, 0644)
	if err != nil {
		log.Printf("Failed to write report: %v", err)
		return
	}
	
	log.Printf("Red Team Assessment Report saved to: %s", filename)
	
	// Print summary
	summary := rt.generateSummary()
	log.Printf("Assessment Summary:")
	log.Printf("  Total Tests: %d", summary["total_tests"])
	log.Printf("  Passed: %d", summary["passed"])
	log.Printf("  Failed: %d", summary["failed"])
	log.Printf("  Errors: %d", summary["errors"])
	log.Printf("  Critical Issues: %d", summary["critical"])
	log.Printf("  High Issues: %d", summary["high"])
	log.Printf("  Security Score: %.1f%%", summary["security_score"])
}

func (rt *RedTeamTest) generateSummary() map[string]interface{} {
	var passed, failed, errors, critical, high, medium, low int
	
	for _, result := range rt.results {
		switch result.Status {
		case "PASS":
			passed++
		case "FAIL":
			failed++
		case "error":
			errors++
		}
		
		switch result.Severity {
		case "Critical":
			critical++
		case "High":
			high++
		case "Medium":
			medium++
		case "Low":
			low++
		}
	}
	
	total := len(rt.results)
	securityScore := float64(passed) / float64(total) * 100
	
	return map[string]interface{}{
		"total_tests":    total,
		"passed":         passed,
		"failed":         failed,
		"errors":         errors,
		"critical":       critical,
		"high":           high,
		"medium":         medium,
		"low":            low,
		"security_score": securityScore,
	}
}

func (rt *RedTeamTest) generateRecommendations() []string {
	recommendations := []string{}
	
	summary := rt.generateSummary()
	
	if summary["failed"].(int) > 0 {
		recommendations = append(recommendations, "Implement additional input validation and sanitization")
	}
	
	if summary["critical"].(int) > 0 {
		recommendations = append(recommendations, "Address critical security vulnerabilities immediately")
	}
	
	if summary["high"].(int) > 0 {
		recommendations = append(recommendations, "Implement additional security controls for high-risk areas")
	}
	
	if summary["security_score"].(float64) < 90 {
		recommendations = append(recommendations, "Enhance overall security posture - target 90%+ security score")
	}
	
	recommendations = append(recommendations, "Regular security assessments and penetration testing")
	recommendations = append(recommendations, "Implement Web Application Firewall (WAF) rules")
	recommendations = append(recommendations, "Enable comprehensive security monitoring and alerting")
	
	return recommendations
}

func main() {
	baseURL := os.Getenv("TARGET_URL")
	if baseURL == "" {
		baseURL = "https://localhost:8443"
	}
	
	log.Printf("Starting Red Team Assessment against: %s", baseURL)
	
	redTeam := NewRedTeamTest(baseURL)
	redTeam.RunAllTests()
	
	log.Println("Red Team Assessment completed!")
}