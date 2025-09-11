package crypto

import (
	"bytes"
	"testing"
	"time"
)

func TestHybridKEX_GenerateKeyPair(t *testing.T) {
	tests := []struct {
		name         string
		kyberEnabled bool
		expectAlg    byte
	}{
		{
			name:         "X25519 only",
			kyberEnabled: false,
			expectAlg:    AlgX25519Only,
		},
		{
			name:         "Hybrid mode",
			kyberEnabled: true,
			expectAlg:    AlgHybrid,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			kex := NewHybridKEX(tt.kyberEnabled)
			
			session, err := kex.GenerateKeyPair()
			if err != nil {
				t.Fatalf("GenerateKeyPair failed: %v", err)
			}

			// Validate session
			if session.ID == "" {
				t.Error("Session ID should not be empty")
			}

			if session.Version != ProtocolVersion2 {
				t.Errorf("Expected version %d, got %d", ProtocolVersion2, session.Version)
			}

			if session.Algorithm != tt.expectAlg {
				t.Errorf("Expected algorithm %d, got %d", tt.expectAlg, session.Algorithm)
			}

			// Check X25519 keys
			if isZero(session.X25519Private[:]) {
				t.Error("X25519 private key should not be zero")
			}

			if isZero(session.X25519Public[:]) {
				t.Error("X25519 public key should not be zero")
			}

			// Check Kyber keys for hybrid mode
			if tt.kyberEnabled {
				if len(session.KyberPrivate) != KyberPrivateKeySize {
					t.Errorf("Expected Kyber private key size %d, got %d", 
						KyberPrivateKeySize, len(session.KyberPrivate))
				}

				if len(session.KyberPublic) != KyberPublicKeySize {
					t.Errorf("Expected Kyber public key size %d, got %d", 
						KyberPublicKeySize, len(session.KyberPublic))
				}
			}

			// Check session is cached
			cachedSession, exists := kex.sessionCache[session.ID]
			if !exists {
				t.Error("Session should be cached")
			}

			if cachedSession.ID != session.ID {
				t.Error("Cached session ID mismatch")
			}
		})
	}
}

func TestHybridKEX_HandshakeFlow(t *testing.T) {
	tests := []struct {
		name         string
		kyberEnabled bool
	}{
		{
			name:         "X25519 handshake",
			kyberEnabled: false,
		},
		{
			name:         "Hybrid handshake",
			kyberEnabled: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create two KEX instances (Alice and Bob)
			alice := NewHybridKEX(tt.kyberEnabled)
			bob := NewHybridKEX(tt.kyberEnabled)

			// Alice generates key pair
			aliceSession, err := alice.GenerateKeyPair()
			if err != nil {
				t.Fatalf("Alice GenerateKeyPair failed: %v", err)
			}

			// Bob generates key pair
			bobSession, err := bob.GenerateKeyPair()
			if err != nil {
				t.Fatalf("Bob GenerateKeyPair failed: %v", err)
			}

			// Alice initiates handshake with Bob's public key
			bobPublicKey := createPublicKeyMessage(bobSession)
			_, aliceMessage, err := alice.InitiateHandshake(bobPublicKey)
			if err != nil {
				t.Fatalf("Alice InitiateHandshake failed: %v", err)
			}

			// Bob completes handshake with Alice's message
			err = bob.CompleteHandshake(bobSession.ID, aliceMessage)
			if err != nil {
				t.Fatalf("Bob CompleteHandshake failed: %v", err)
			}

			// Alice completes handshake with Bob's message
			alicePublicKey := createPublicKeyMessage(aliceSession)
			err = alice.CompleteHandshake(aliceSession.ID, alicePublicKey)
			if err != nil {
				t.Fatalf("Alice CompleteHandshake failed: %v", err)
			}

			// Verify both parties have the same shared secret
			aliceSecret, err := alice.GetSharedSecret(aliceSession.ID)
			if err != nil {
				t.Fatalf("Alice GetSharedSecret failed: %v", err)
			}

			bobSecret, err := bob.GetSharedSecret(bobSession.ID)
			if err != nil {
				t.Fatalf("Bob GetSharedSecret failed: %v", err)
			}

			if !bytes.Equal(aliceSecret, bobSecret) {
				t.Error("Shared secrets do not match")
			}

			if len(aliceSecret) != SharedSecretSize {
				t.Errorf("Expected shared secret size %d, got %d", 
					SharedSecretSize, len(aliceSecret))
			}

			// Verify sessions are established
			if !aliceSession.Established {
				t.Error("Alice session should be established")
			}

			if !bobSession.Established {
				t.Error("Bob session should be established")
			}
		})
	}
}

func TestHybridKEX_MessageParsing(t *testing.T) {
	kex := NewHybridKEX(true)
	session, err := kex.GenerateKeyPair()
	if err != nil {
		t.Fatalf("GenerateKeyPair failed: %v", err)
	}

	// Create handshake message
	message, err := kex.createHandshakeMessage(session, nil)
	if err != nil {
		t.Fatalf("createHandshakeMessage failed: %v", err)
	}

	// Parse the message
	data, err := kex.parseHandshakeMessage(message)
	if err != nil {
		t.Fatalf("parseHandshakeMessage failed: %v", err)
	}

	// Verify parsed data
	x25519Public, exists := data["x25519_public"]
	if !exists {
		t.Error("X25519 public key not found in parsed data")
	}

	if !bytes.Equal(x25519Public, session.X25519Public[:]) {
		t.Error("Parsed X25519 public key does not match original")
	}

	if session.Algorithm == AlgHybrid {
		kyberPublic, exists := data["kyber_public"]
		if !exists {
			t.Error("Kyber public key not found in parsed data")
		}

		if !bytes.Equal(kyberPublic, session.KyberPublic) {
			t.Error("Parsed Kyber public key does not match original")
		}
	}
}

func TestHybridKEX_ErrorHandling(t *testing.T) {
	kex := NewHybridKEX(true)

	// Test invalid session ID
	_, err := kex.GetSharedSecret("invalid-session-id")
	if err == nil {
		t.Error("Expected error for invalid session ID")
	}

	// Test handshake on non-existent session
	err = kex.CompleteHandshake("invalid-session-id", []byte("dummy"))
	if err == nil {
		t.Error("Expected error for non-existent session")
	}

	// Test invalid message parsing
	_, err = kex.parseHandshakeMessage([]byte{0x01}) // Too short
	if err == nil {
		t.Error("Expected error for invalid message")
	}

	// Test unsupported version
	invalidMessage := []byte{0xFF, 0x01} // Invalid version
	_, err = kex.parseHandshakeMessage(invalidMessage)
	if err != ErrUnsupportedVersion {
		t.Errorf("Expected ErrUnsupportedVersion, got %v", err)
	}
}

func TestHybridKEX_SessionCleanup(t *testing.T) {
	kex := NewHybridKEX(false)

	// Create session with short expiry
	session, err := kex.GenerateKeyPair()
	if err != nil {
		t.Fatalf("GenerateKeyPair failed: %v", err)
	}

	// Manually set expiry to past
	session.ExpiresAt = time.Now().Add(-1 * time.Hour)
	kex.sessionCache[session.ID] = session

	// Verify session exists
	if len(kex.sessionCache) != 1 {
		t.Error("Session should exist before cleanup")
	}

	// Run cleanup
	kex.CleanupExpiredSessions()

	// Verify session is removed
	if len(kex.sessionCache) != 0 {
		t.Error("Expired session should be removed")
	}
}

func TestHybridKEX_Metrics(t *testing.T) {
	kex := NewHybridKEX(true)

	// Initial metrics should be zero
	metrics := kex.GetMetrics()
	if metrics.HandshakesTotal != 0 {
		t.Error("Initial handshakes total should be 0")
	}

	// Perform successful handshake
	alice := NewHybridKEX(true)
	bob := NewHybridKEX(true)

	aliceSession, _ := alice.GenerateKeyPair()
	bobSession, _ := bob.GenerateKeyPair()

	bobPublicKey := createPublicKeyMessage(bobSession)
	_, aliceMessage, _ := alice.InitiateHandshake(bobPublicKey)
	bob.CompleteHandshake(bobSession.ID, aliceMessage)

	// Check metrics updated
	aliceMetrics := alice.GetMetrics()
	if aliceMetrics.HandshakesTotal != 1 {
		t.Errorf("Expected 1 total handshake, got %d", aliceMetrics.HandshakesTotal)
	}

	if aliceMetrics.AverageLatencyMs <= 0 {
		t.Error("Average latency should be greater than 0")
	}
}

func TestMockKyberProvider(t *testing.T) {
	// Test available provider
	provider := &MockKyberProvider{available: true}

	if !provider.IsAvailable() {
		t.Error("Provider should be available")
	}

	private, public, err := provider.GenerateKeyPair()
	if err != nil {
		t.Fatalf("GenerateKeyPair failed: %v", err)
	}

	if len(private) != KyberPrivateKeySize {
		t.Errorf("Expected private key size %d, got %d", KyberPrivateKeySize, len(private))
	}

	if len(public) != KyberPublicKeySize {
		t.Errorf("Expected public key size %d, got %d", KyberPublicKeySize, len(public))
	}

	// Test encapsulation
	ciphertext, sharedSecret, err := provider.Encapsulate(public)
	if err != nil {
		t.Fatalf("Encapsulate failed: %v", err)
	}

	if len(ciphertext) != KyberCiphertextSize {
		t.Errorf("Expected ciphertext size %d, got %d", KyberCiphertextSize, len(ciphertext))
	}

	if len(sharedSecret) != 32 {
		t.Errorf("Expected shared secret size 32, got %d", len(sharedSecret))
	}

	// Test decapsulation
	decapsulatedSecret, err := provider.Decapsulate(private, ciphertext)
	if err != nil {
		t.Fatalf("Decapsulate failed: %v", err)
	}

	if len(decapsulatedSecret) != 32 {
		t.Errorf("Expected decapsulated secret size 32, got %d", len(decapsulatedSecret))
	}

	// Test unavailable provider
	unavailableProvider := &MockKyberProvider{available: false}

	if unavailableProvider.IsAvailable() {
		t.Error("Provider should not be available")
	}

	_, _, err = unavailableProvider.GenerateKeyPair()
	if err == nil {
		t.Error("Expected error for unavailable provider")
	}
}

func BenchmarkHybridKEX_GenerateKeyPair(b *testing.B) {
	kex := NewHybridKEX(true)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := kex.GenerateKeyPair()
		if err != nil {
			b.Fatalf("GenerateKeyPair failed: %v", err)
		}
	}
}

func BenchmarkHybridKEX_Handshake(b *testing.B) {
	alice := NewHybridKEX(true)
	bob := NewHybridKEX(true)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		aliceSession, _ := alice.GenerateKeyPair()
		bobSession, _ := bob.GenerateKeyPair()

		bobPublicKey := createPublicKeyMessage(bobSession)
		_, aliceMessage, _ := alice.InitiateHandshake(bobPublicKey)
		bob.CompleteHandshake(bobSession.ID, aliceMessage)
	}
}

func BenchmarkX25519Only_Handshake(b *testing.B) {
	alice := NewHybridKEX(false)
	bob := NewHybridKEX(false)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		aliceSession, _ := alice.GenerateKeyPair()
		bobSession, _ := bob.GenerateKeyPair()

		bobPublicKey := createPublicKeyMessage(bobSession)
		_, aliceMessage, _ := alice.InitiateHandshake(bobPublicKey)
		bob.CompleteHandshake(bobSession.ID, aliceMessage)
	}
}

// Helper functions for tests

func isZero(data []byte) bool {
	for _, b := range data {
		if b != 0 {
			return false
		}
	}
	return true
}

func createPublicKeyMessage(session *Session) []byte {
	message := make([]byte, 0, 2048)
	message = append(message, session.Version)
	message = append(message, session.Algorithm)
	message = append(message, session.X25519Public[:]...)
	
	if session.Algorithm == AlgHybrid {
		message = append(message, session.KyberPublic...)
	}
	
	return message
}