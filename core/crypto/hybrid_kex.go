package crypto

import (
	"crypto/rand"
	"crypto/subtle"
	"errors"
	"fmt"
	"sync"
	"time"

	"golang.org/x/crypto/curve25519"
	"golang.org/x/crypto/hkdf"
	"golang.org/x/crypto/sha3"
)

const (
	// Key sizes
	X25519PrivateKeySize = 32
	X25519PublicKeySize  = 32
	KyberPrivateKeySize  = 2400
	KyberPublicKeySize   = 1184
	KyberCiphertextSize  = 1568
	SharedSecretSize     = 32
	
	// Protocol versions
	ProtocolVersion1 = 0x01
	ProtocolVersion2 = 0x02
	
	// Algorithm identifiers
	AlgX25519Only = 0x01
	AlgHybrid     = 0x02
)

var (
	ErrInvalidKeySize     = errors.New("invalid key size")
	ErrInvalidCiphertext  = errors.New("invalid ciphertext")
	ErrUnsupportedVersion = errors.New("unsupported protocol version")
	ErrHandshakeFailed    = errors.New("handshake failed")
)

// HybridKEX implements post-quantum hybrid key exchange
type HybridKEX struct {
	mu              sync.RWMutex
	kyberEnabled    bool
	fallbackMode    bool
	sessionCache    map[string]*Session
	metrics         *KEXMetrics
	pqProvider      PQProvider
}

// Session represents an active key exchange session
type Session struct {
	ID              string
	Version         byte
	Algorithm       byte
	X25519Private   [32]byte
	X25519Public    [32]byte
	KyberPrivate    []byte
	KyberPublic     []byte
	SharedSecret    [32]byte
	CreatedAt       time.Time
	ExpiresAt       time.Time
	PeerPublicKey   []byte
	Established     bool
}

// KEXMetrics tracks key exchange performance
type KEXMetrics struct {
	HandshakesTotal     int64
	HandshakesSucceeded int64
	HandshakesFailed    int64
	QuantumHandshakes   int64
	ClassicalHandshakes int64
	AverageLatencyMs    float64
}

// PQProvider interface for post-quantum operations
type PQProvider interface {
	GenerateKeyPair() (privateKey, publicKey []byte, err error)
	Encapsulate(publicKey []byte) (ciphertext, sharedSecret []byte, err error)
	Decapsulate(privateKey, ciphertext []byte) (sharedSecret []byte, err error)
	IsAvailable() bool
}

// MockKyberProvider implements PQProvider for testing
type MockKyberProvider struct {
	available bool
}

func (m *MockKyberProvider) GenerateKeyPair() ([]byte, []byte, error) {
	if !m.available {
		return nil, nil, errors.New("kyber not available")
	}
	
	private := make([]byte, KyberPrivateKeySize)
	public := make([]byte, KyberPublicKeySize)
	
	rand.Read(private)
	rand.Read(public)
	
	return private, public, nil
}

func (m *MockKyberProvider) Encapsulate(publicKey []byte) ([]byte, []byte, error) {
	if !m.available {
		return nil, nil, errors.New("kyber not available")
	}
	
	if len(publicKey) != KyberPublicKeySize {
		return nil, nil, ErrInvalidKeySize
	}
	
	ciphertext := make([]byte, KyberCiphertextSize)
	sharedSecret := make([]byte, 32)
	
	rand.Read(ciphertext)
	rand.Read(sharedSecret)
	
	return ciphertext, sharedSecret, nil
}

func (m *MockKyberProvider) Decapsulate(privateKey, ciphertext []byte) ([]byte, error) {
	if !m.available {
		return nil, errors.New("kyber not available")
	}
	
	if len(privateKey) != KyberPrivateKeySize {
		return nil, ErrInvalidKeySize
	}
	
	if len(ciphertext) != KyberCiphertextSize {
		return nil, ErrInvalidCiphertext
	}
	
	sharedSecret := make([]byte, 32)
	rand.Read(sharedSecret)
	
	return sharedSecret, nil
}

func (m *MockKyberProvider) IsAvailable() bool {
	return m.available
}

// NewHybridKEX creates a new hybrid key exchange instance
func NewHybridKEX(kyberEnabled bool) *HybridKEX {
	return &HybridKEX{
		kyberEnabled: kyberEnabled,
		fallbackMode: false,
		sessionCache: make(map[string]*Session),
		metrics:      &KEXMetrics{},
		pqProvider:   &MockKyberProvider{available: kyberEnabled},
	}
}

// GenerateKeyPair generates a hybrid key pair
func (h *HybridKEX) GenerateKeyPair() (*Session, error) {
	session := &Session{
		ID:        generateSessionID(),
		Version:   ProtocolVersion2,
		CreatedAt: time.Now(),
		ExpiresAt: time.Now().Add(24 * time.Hour),
	}
	
	// Always generate X25519 keys
	var err error
	session.X25519Private, err = generateX25519Private()
	if err != nil {
		return nil, fmt.Errorf("failed to generate X25519 private key: %w", err)
	}
	
	session.X25519Public, err = generateX25519Public(session.X25519Private)
	if err != nil {
		return nil, fmt.Errorf("failed to generate X25519 public key: %w", err)
	}
	
	// Generate Kyber keys if enabled and available
	if h.kyberEnabled && h.pqProvider.IsAvailable() {
		session.KyberPrivate, session.KyberPublic, err = h.pqProvider.GenerateKeyPair()
		if err != nil {
			// Fallback to X25519 only
			session.Algorithm = AlgX25519Only
		} else {
			session.Algorithm = AlgHybrid
		}
	} else {
		session.Algorithm = AlgX25519Only
	}
	
	h.mu.Lock()
	h.sessionCache[session.ID] = session
	h.mu.Unlock()
	
	return session, nil
}

// InitiateHandshake starts a key exchange handshake
func (h *HybridKEX) InitiateHandshake(peerPublicKey []byte) (*Session, []byte, error) {
	start := time.Now()
	defer func() {
		latency := time.Since(start).Milliseconds()
		h.updateMetrics(latency)
	}()
	
	session, err := h.GenerateKeyPair()
	if err != nil {
		h.metrics.HandshakesFailed++
		return nil, nil, err
	}
	
	// Create handshake message
	message, err := h.createHandshakeMessage(session, peerPublicKey)
	if err != nil {
		h.metrics.HandshakesFailed++
		return nil, nil, err
	}
	
	h.metrics.HandshakesTotal++
	return session, message, nil
}

// CompleteHandshake completes the key exchange
func (h *HybridKEX) CompleteHandshake(sessionID string, peerMessage []byte) error {
	h.mu.RLock()
	session, exists := h.sessionCache[sessionID]
	h.mu.RUnlock()
	
	if !exists {
		return errors.New("session not found")
	}
	
	if session.Established {
		return errors.New("handshake already completed")
	}
	
	// Parse peer message
	peerData, err := h.parseHandshakeMessage(peerMessage)
	if err != nil {
		h.metrics.HandshakesFailed++
		return err
	}
	
	// Derive shared secret
	err = h.deriveSharedSecret(session, peerData)
	if err != nil {
		h.metrics.HandshakesFailed++
		return err
	}
	
	session.Established = true
	h.metrics.HandshakesSucceeded++
	
	if session.Algorithm == AlgHybrid {
		h.metrics.QuantumHandshakes++
	} else {
		h.metrics.ClassicalHandshakes++
	}
	
	return nil
}

// GetSharedSecret returns the derived shared secret
func (h *HybridKEX) GetSharedSecret(sessionID string) ([]byte, error) {
	h.mu.RLock()
	session, exists := h.sessionCache[sessionID]
	h.mu.RUnlock()
	
	if !exists {
		return nil, errors.New("session not found")
	}
	
	if !session.Established {
		return nil, errors.New("handshake not completed")
	}
	
	return session.SharedSecret[:], nil
}

// CleanupExpiredSessions removes expired sessions
func (h *HybridKEX) CleanupExpiredSessions() {
	h.mu.Lock()
	defer h.mu.Unlock()
	
	now := time.Now()
	for id, session := range h.sessionCache {
		if now.After(session.ExpiresAt) {
			delete(h.sessionCache, id)
		}
	}
}

// GetMetrics returns current metrics
func (h *HybridKEX) GetMetrics() KEXMetrics {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return *h.metrics
}

// createHandshakeMessage creates a handshake message
func (h *HybridKEX) createHandshakeMessage(session *Session, peerPublicKey []byte) ([]byte, error) {
	message := make([]byte, 0, 4096)
	
	// Protocol version
	message = append(message, session.Version)
	
	// Algorithm identifier
	message = append(message, session.Algorithm)
	
	// X25519 public key
	message = append(message, session.X25519Public[:]...)
	
	// Kyber public key (if hybrid)
	if session.Algorithm == AlgHybrid {
		message = append(message, session.KyberPublic...)
	}
	
	return message, nil
}

// parseHandshakeMessage parses a handshake message
func (h *HybridKEX) parseHandshakeMessage(message []byte) (map[string][]byte, error) {
	if len(message) < 2 {
		return nil, errors.New("message too short")
	}
	
	data := make(map[string][]byte)
	offset := 0
	
	// Protocol version
	version := message[offset]
	offset++
	
	if version != ProtocolVersion1 && version != ProtocolVersion2 {
		return nil, ErrUnsupportedVersion
	}
	
	// Algorithm
	algorithm := message[offset]
	offset++
	
	// X25519 public key
	if len(message) < offset+X25519PublicKeySize {
		return nil, errors.New("invalid X25519 key")
	}
	
	data["x25519_public"] = message[offset : offset+X25519PublicKeySize]
	offset += X25519PublicKeySize
	
	// Kyber public key (if hybrid)
	if algorithm == AlgHybrid {
		if len(message) < offset+KyberPublicKeySize {
			return nil, errors.New("invalid Kyber key")
		}
		
		data["kyber_public"] = message[offset : offset+KyberPublicKeySize]
		offset += KyberPublicKeySize
	}
	
	return data, nil
}

// deriveSharedSecret derives the final shared secret
func (h *HybridKEX) deriveSharedSecret(session *Session, peerData map[string][]byte) error {
	var secrets [][]byte
	
	// X25519 ECDH
	peerX25519Public := peerData["x25519_public"]
	if len(peerX25519Public) != X25519PublicKeySize {
		return ErrInvalidKeySize
	}
	
	var peerPublic, sharedPoint [32]byte
	copy(peerPublic[:], peerX25519Public)
	
	curve25519.ScalarMult(&sharedPoint, &session.X25519Private, &peerPublic)
	secrets = append(secrets, sharedPoint[:])
	
	// Kyber KEM (if hybrid)
	if session.Algorithm == AlgHybrid {
		peerKyberPublic := peerData["kyber_public"]
		if len(peerKyberPublic) != KyberPublicKeySize {
			return ErrInvalidKeySize
		}
		
		_, kyberSecret, err := h.pqProvider.Encapsulate(peerKyberPublic)
		if err != nil {
			return fmt.Errorf("Kyber encapsulation failed: %w", err)
		}
		
		secrets = append(secrets, kyberSecret)
	}
	
	// Combine secrets using HKDF
	combinedSecret := combineSecrets(secrets)
	copy(session.SharedSecret[:], combinedSecret)
	
	return nil
}

// updateMetrics updates performance metrics
func (h *HybridKEX) updateMetrics(latencyMs int64) {
	h.mu.Lock()
	defer h.mu.Unlock()
	
	// Update average latency using exponential moving average
	alpha := 0.1
	h.metrics.AverageLatencyMs = (1-alpha)*h.metrics.AverageLatencyMs + alpha*float64(latencyMs)
}

// Helper functions

func generateSessionID() string {
	bytes := make([]byte, 16)
	rand.Read(bytes)
	return fmt.Sprintf("%x", bytes)
}

func generateX25519Private() ([32]byte, error) {
	var private [32]byte
	_, err := rand.Read(private[:])
	return private, err
}

func generateX25519Public(private [32]byte) ([32]byte, error) {
	var public [32]byte
	curve25519.ScalarBaseMult(&public, &private)
	return public, nil
}

func combineSecrets(secrets [][]byte) []byte {
	// Concatenate all secrets
	var combined []byte
	for _, secret := range secrets {
		combined = append(combined, secret...)
	}
	
	// Derive final secret using HKDF-SHA3
	hkdf := hkdf.New(sha3.New256, combined, nil, []byte("ShieldX-HybridKEX-v1"))
	
	finalSecret := make([]byte, SharedSecretSize)
	hkdf.Read(finalSecret)
	
	return finalSecret
}

// Constant-time comparison for security
func constantTimeCompare(a, b []byte) bool {
	return subtle.ConstantTimeCompare(a, b) == 1
}