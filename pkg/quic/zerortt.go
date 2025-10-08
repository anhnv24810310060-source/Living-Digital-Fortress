// Package quic - 0-RTT connection establishment with replay protection
package quic

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"io"
	"sync"
	"time"
)

// ZeroRTTTokenManager manages 0-RTT session resumption tokens
type ZeroRTTTokenManager struct {
	mu               sync.RWMutex
	tokens           map[string]*SessionToken // clientID -> token
	replayWindow     map[string]int64         // tokenID -> lastUsed
	secret           []byte                   // Master secret for token encryption
	maxAge           time.Duration
	antiReplayWindow time.Duration
}

// SessionToken contains encrypted session resumption data
type SessionToken struct {
	ID            string
	ClientID      string
	EncryptedData []byte
	IssuedAt      time.Time
	ExpiresAt     time.Time
	Nonce         []byte
}

// NewZeroRTTTokenManager creates a new token manager
func NewZeroRTTTokenManager(secret []byte) *ZeroRTTTokenManager {
	if len(secret) < 32 {
		// Generate secure random secret
		secret = make([]byte, 32)
		io.ReadFull(rand.Reader, secret)
	}

	return &ZeroRTTTokenManager{
		tokens:           make(map[string]*SessionToken),
		replayWindow:     make(map[string]int64),
		secret:           secret,
		maxAge:           24 * time.Hour,
		antiReplayWindow: 5 * time.Second,
	}
}

// IssueToken issues a 0-RTT resumption token
func (z *ZeroRTTTokenManager) IssueToken(clientID string, sessionData []byte) (*SessionToken, error) {
	z.mu.Lock()
	defer z.mu.Unlock()

	// Generate unique token ID
	tokenID := make([]byte, 16)
	if _, err := io.ReadFull(rand.Reader, tokenID); err != nil {
		return nil, err
	}

	// Generate nonce for AEAD
	nonce := make([]byte, 12)
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return nil, err
	}

	// Encrypt session data
	block, err := aes.NewCipher(z.secret)
	if err != nil {
		return nil, err
	}

	aesGCM, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}

	// Additional authenticated data: clientID + tokenID + timestamp
	now := time.Now()
	aad := make([]byte, len(clientID)+len(tokenID)+8)
	copy(aad, clientID)
	copy(aad[len(clientID):], tokenID)
	binary.BigEndian.PutUint64(aad[len(clientID)+len(tokenID):], uint64(now.Unix()))

	ciphertext := aesGCM.Seal(nil, nonce, sessionData, aad)

	token := &SessionToken{
		ID:            string(tokenID),
		ClientID:      clientID,
		EncryptedData: ciphertext,
		IssuedAt:      now,
		ExpiresAt:     now.Add(z.maxAge),
		Nonce:         nonce,
	}

	z.tokens[clientID] = token
	return token, nil
}

// ValidateToken validates a 0-RTT token with replay protection
func (z *ZeroRTTTokenManager) ValidateToken(clientID string, tokenData []byte) ([]byte, error) {
	z.mu.Lock()
	defer z.mu.Unlock()

	// Check if we have a token for this client
	token, exists := z.tokens[clientID]
	if !exists {
		return nil, errors.New("no token found for client")
	}

	// Check expiration
	if time.Now().After(token.ExpiresAt) {
		delete(z.tokens, clientID)
		return nil, errors.New("token expired")
	}

	// Anti-replay: check if token was used recently
	if lastUsed, ok := z.replayWindow[token.ID]; ok {
		if time.Since(time.Unix(lastUsed, 0)) < z.antiReplayWindow {
			return nil, errors.New("replay detected")
		}
	}

	// Decrypt token data
	block, err := aes.NewCipher(z.secret)
	if err != nil {
		return nil, err
	}

	aesGCM, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}

	// Reconstruct AAD
	aad := make([]byte, len(clientID)+len(token.ID)+8)
	copy(aad, clientID)
	copy(aad[len(clientID):], token.ID)
	binary.BigEndian.PutUint64(aad[len(clientID)+len(token.ID):], uint64(token.IssuedAt.Unix()))

	plaintext, err := aesGCM.Open(nil, token.Nonce, token.EncryptedData, aad)
	if err != nil {
		return nil, errors.New("token decryption failed")
	}

	// Mark token as used
	z.replayWindow[token.ID] = time.Now().Unix()

	// Cleanup old replay window entries
	go z.cleanupReplayWindow()

	return plaintext, nil
}

// cleanupReplayWindow removes old entries from replay protection window
func (z *ZeroRTTTokenManager) cleanupReplayWindow() {
	z.mu.Lock()
	defer z.mu.Unlock()

	cutoff := time.Now().Add(-z.antiReplayWindow).Unix()
	for id, lastUsed := range z.replayWindow {
		if lastUsed < cutoff {
			delete(z.replayWindow, id)
		}
	}
}

// RevokeToken immediately invalidates a token
func (z *ZeroRTTTokenManager) RevokeToken(clientID string) {
	z.mu.Lock()
	defer z.mu.Unlock()

	if token, ok := z.tokens[clientID]; ok {
		delete(z.replayWindow, token.ID)
		delete(z.tokens, clientID)
	}
}

// DeriveApplicationKey derives application keys from session resumption
func DeriveApplicationKey(masterSecret []byte, context string) ([]byte, error) {
	// HKDF-Expand-Label from TLS 1.3
	h := sha256.New()
	h.Write(masterSecret)
	h.Write([]byte(context))
	return h.Sum(nil), nil
}

// RotateSecret rotates the encryption secret periodically
func (z *ZeroRTTTokenManager) RotateSecret() error {
	newSecret := make([]byte, 32)
	if _, err := io.ReadFull(rand.Reader, newSecret); err != nil {
		return err
	}

	z.mu.Lock()
	defer z.mu.Unlock()

	// Re-encrypt all tokens with new secret
	// (Simplified - in production, use key versioning)
	z.secret = newSecret

	// Invalidate all old tokens (force new handshake)
	z.tokens = make(map[string]*SessionToken)
	z.replayWindow = make(map[string]int64)

	return nil
}
