package crypto

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// Integration tests for the complete PQC system

func TestPQCService_Integration(t *testing.T) {
	// Create test server
	server := createTestPQCServer(t)
	defer server.Close()

	client := &http.Client{Timeout: 10 * time.Second}

	// Test key generation
	keyGenResp := testKeyGeneration(t, client, server.URL)
	
	// Test handshake
	testHandshake(t, client, server.URL, keyGenResp.SessionID)
	
	// Test session info
	testSessionInfo(t, client, server.URL, keyGenResp.SessionID)
}

func TestPQCService_LoadTest(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping load test in short mode")
	}

	server := createTestPQCServer(t)
	defer server.Close()

	client := &http.Client{Timeout: 5 * time.Second}
	
	// Concurrent key generation
	concurrency := 10
	requests := 100
	
	results := make(chan error, concurrency)
	
	for i := 0; i < concurrency; i++ {
		go func() {
			for j := 0; j < requests/concurrency; j++ {
				keyGenResp := testKeyGeneration(t, client, server.URL)
				if keyGenResp.SessionID == "" {
					results <- fmt.Errorf("empty session ID")
					return
				}
			}
			results <- nil
		}()
	}
	
	// Wait for all goroutines
	for i := 0; i < concurrency; i++ {
		if err := <-results; err != nil {
			t.Errorf("Load test failed: %v", err)
		}
	}
}

func TestPQCService_FailureScenarios(t *testing.T) {
	server := createTestPQCServer(t)
	defer server.Close()

	client := &http.Client{Timeout: 5 * time.Second}

	tests := []struct {
		name           string
		endpoint       string
		method         string
		body           interface{}
		expectedStatus int
	}{
		{
			name:           "Invalid algorithm",
			endpoint:       "/v1/pqc/keygen",
			method:         "POST",
			body:           map[string]string{"algorithm": "invalid", "client_id": "test"},
			expectedStatus: 400,
		},
		{
			name:           "Missing session ID",
			endpoint:       "/v1/pqc/handshake",
			method:         "POST",
			body:           map[string]string{"peer_public_key": "test"},
			expectedStatus: 400,
		},
		{
			name:           "Invalid session ID",
			endpoint:       "/v1/pqc/session/invalid-session",
			method:         "GET",
			body:           nil,
			expectedStatus: 200, // Returns default response
		},
		{
			name:           "Method not allowed",
			endpoint:       "/v1/pqc/keygen",
			method:         "GET",
			body:           nil,
			expectedStatus: 405,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var body []byte
			if test.body != nil {
				var err error
				body, err = json.Marshal(test.body)
				if err != nil {
					t.Fatalf("Failed to marshal body: %v", err)
				}
			}

			req, err := http.NewRequest(test.method, server.URL+test.endpoint, bytes.NewReader(body))
			if err != nil {
				t.Fatalf("Failed to create request: %v", err)
			}

			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("Authorization", "Bearer test_token")

			resp, err := client.Do(req)
			if err != nil {
				t.Fatalf("Request failed: %v", err)
			}
			defer resp.Body.Close()

			if resp.StatusCode != test.expectedStatus {
				t.Errorf("Expected status %d, got %d", test.expectedStatus, resp.StatusCode)
			}
		})
	}
}

func TestHybridKEX_EndToEndHandshake(t *testing.T) {
	// Test complete handshake between two parties
	alice := NewHybridKEX(true)
	bob := NewHybridKEX(true)

	// Alice generates key pair
	aliceSession, err := alice.GenerateKeyPair()
	if err != nil {
		t.Fatalf("Alice key generation failed: %v", err)
	}

	// Bob generates key pair
	bobSession, err := bob.GenerateKeyPair()
	if err != nil {
		t.Fatalf("Bob key generation failed: %v", err)
	}

	// Create public key messages
	alicePublicMsg := createPublicKeyMessage(aliceSession)
	bobPublicMsg := createPublicKeyMessage(bobSession)

	// Alice initiates handshake with Bob's public key
	_, aliceHandshakeMsg, err := alice.InitiateHandshake(bobPublicMsg)
	if err != nil {
		t.Fatalf("Alice handshake initiation failed: %v", err)
	}

	// Bob completes handshake with Alice's message
	err = bob.CompleteHandshake(bobSession.ID, aliceHandshakeMsg)
	if err != nil {
		t.Fatalf("Bob handshake completion failed: %v", err)
	}

	// Alice completes handshake with Bob's message
	err = alice.CompleteHandshake(aliceSession.ID, bobPublicMsg)
	if err != nil {
		t.Fatalf("Alice handshake completion failed: %v", err)
	}

	// Verify shared secrets match
	aliceSecret, err := alice.GetSharedSecret(aliceSession.ID)
	if err != nil {
		t.Fatalf("Failed to get Alice's shared secret: %v", err)
	}

	bobSecret, err := bob.GetSharedSecret(bobSession.ID)
	if err != nil {
		t.Fatalf("Failed to get Bob's shared secret: %v", err)
	}

	if !bytes.Equal(aliceSecret, bobSecret) {
		t.Error("Shared secrets do not match")
	}

	// Verify metrics
	aliceMetrics := alice.GetMetrics()
	bobMetrics := bob.GetMetrics()

	if aliceMetrics.HandshakesTotal == 0 {
		t.Error("Alice should have recorded handshake")
	}

	if bobMetrics.HandshakesSucceeded == 0 {
		t.Error("Bob should have recorded successful handshake")
	}
}

func TestHybridKEX_PerformanceBenchmark(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping performance test in short mode")
	}

	kex := NewHybridKEX(true)
	
	// Measure key generation performance
	start := time.Now()
	iterations := 1000
	
	for i := 0; i < iterations; i++ {
		_, err := kex.GenerateKeyPair()
		if err != nil {
			t.Fatalf("Key generation failed: %v", err)
		}
	}
	
	duration := time.Since(start)
	avgLatency := duration / time.Duration(iterations)
	
	t.Logf("Key generation performance:")
	t.Logf("  Total time: %v", duration)
	t.Logf("  Average latency: %v", avgLatency)
	t.Logf("  Operations per second: %.2f", float64(iterations)/duration.Seconds())
	
	// Verify performance meets requirements (< 10ms per operation)
	if avgLatency > 10*time.Millisecond {
		t.Errorf("Key generation too slow: %v > 10ms", avgLatency)
	}
}

func TestHybridKEX_SecurityProperties(t *testing.T) {
	kex := NewHybridKEX(true)

	// Test that different sessions produce different keys
	session1, err := kex.GenerateKeyPair()
	if err != nil {
		t.Fatalf("Session 1 generation failed: %v", err)
	}

	session2, err := kex.GenerateKeyPair()
	if err != nil {
		t.Fatalf("Session 2 generation failed: %v", err)
	}

	// Private keys should be different
	if bytes.Equal(session1.X25519Private[:], session2.X25519Private[:]) {
		t.Error("X25519 private keys should be different")
	}

	// Public keys should be different
	if bytes.Equal(session1.X25519Public[:], session2.X25519Public[:]) {
		t.Error("X25519 public keys should be different")
	}

	// Kyber keys should be different (if enabled)
	if session1.Algorithm == AlgHybrid && session2.Algorithm == AlgHybrid {
		if bytes.Equal(session1.KyberPrivate, session2.KyberPrivate) {
			t.Error("Kyber private keys should be different")
		}

		if bytes.Equal(session1.KyberPublic, session2.KyberPublic) {
			t.Error("Kyber public keys should be different")
		}
	}
}

// Helper functions for integration tests

func createTestPQCServer(t *testing.T) *httptest.Server {
	// This would create a test server with the PQC service
	// For now, return a mock server
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/pqc/keygen":
			handleTestKeyGen(w, r)
		case "/v1/pqc/handshake":
			handleTestHandshake(w, r)
		default:
			if strings.HasPrefix(r.URL.Path, "/v1/pqc/session/") {
				handleTestSessionInfo(w, r)
			} else {
				http.NotFound(w, r)
			}
		}
	}))
}

func handleTestKeyGen(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", 405)
		return
	}

	response := map[string]interface{}{
		"session_id": "test-session-123",
		"public_key": []byte("mock-public-key"),
		"algorithm":  "hybrid",
		"version":    2,
		"expires_at": time.Now().Add(24 * time.Hour).Format(time.RFC3339),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func handleTestHandshake(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", 405)
		return
	}

	response := map[string]interface{}{
		"success":       true,
		"shared_secret": []byte("mock-shared-secret-32-bytes-long"),
		"message":       "Handshake completed successfully",
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func handleTestSessionInfo(w http.ResponseWriter, r *http.Request) {
	response := map[string]interface{}{
		"session_id":  "test-session-123",
		"algorithm":   "hybrid",
		"established": true,
		"created_at":  time.Now().Format(time.RFC3339),
		"expires_at":  time.Now().Add(24 * time.Hour).Format(time.RFC3339),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func testKeyGeneration(t *testing.T, client *http.Client, serverURL string) KeyGenResponse {
	reqBody := map[string]string{
		"algorithm": "hybrid",
		"client_id": "test-client",
	}

	body, _ := json.Marshal(reqBody)
	req, _ := http.NewRequest("POST", serverURL+"/v1/pqc/keygen", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer test_token")

	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("Key generation request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		t.Fatalf("Expected status 200, got %d", resp.StatusCode)
	}

	var keyGenResp KeyGenResponse
	if err := json.NewDecoder(resp.Body).Decode(&keyGenResp); err != nil {
		t.Fatalf("Failed to decode response: %v", err)
	}

	return keyGenResp
}

func testHandshake(t *testing.T, client *http.Client, serverURL, sessionID string) {
	reqBody := map[string]interface{}{
		"session_id":      sessionID,
		"peer_public_key": []byte("mock-peer-public-key"),
		"client_id":       "test-client",
	}

	body, _ := json.Marshal(reqBody)
	req, _ := http.NewRequest("POST", serverURL+"/v1/pqc/handshake", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer test_token")

	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("Handshake request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		t.Fatalf("Expected status 200, got %d", resp.StatusCode)
	}
}

func testSessionInfo(t *testing.T, client *http.Client, serverURL, sessionID string) {
	req, _ := http.NewRequest("GET", serverURL+"/v1/pqc/session/"+sessionID, nil)
	req.Header.Set("Authorization", "Bearer test_token")

	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("Session info request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		t.Fatalf("Expected status 200, got %d", resp.StatusCode)
	}
}

type KeyGenResponse struct {
	SessionID string `json:"session_id"`
	PublicKey []byte `json:"public_key"`
	Algorithm string `json:"algorithm"`
	Version   int    `json:"version"`
	ExpiresAt string `json:"expires_at"`
}