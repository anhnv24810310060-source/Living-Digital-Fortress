package main

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"sync"
	"time"

	"golang.org/x/crypto/argon2"
	"golang.org/x/crypto/chacha20poly1305"
)

// EncryptionManager handles at-rest encryption for sensitive telemetry data
// Uses AES-256-GCM for general encryption and ChaCha20-Poly1305 for high-performance scenarios
type EncryptionManager struct {
	masterKey     []byte
	keyRotation   *KeyRotationPolicy
	activeKeyID   string
	keys          map[string][]byte // keyID -> derived key
	mu            sync.RWMutex
}

// KeyRotationPolicy defines automatic key rotation
type KeyRotationPolicy struct {
	RotationInterval time.Duration
	MaxKeyAge        time.Duration
	GracePeriod      time.Duration // Allow old keys during transition
	lastRotation     time.Time
	mu               sync.Mutex
}

// EncryptedData represents encrypted payload with metadata
type EncryptedData struct {
	KeyID       string    `json:"key_id"`
	Algorithm   string    `json:"algorithm"` // aes-gcm or chacha20poly1305
	Nonce       string    `json:"nonce"`
	Ciphertext  string    `json:"ciphertext"`
	Tag         string    `json:"tag,omitempty"`
	EncryptedAt time.Time `json:"encrypted_at"`
	Version     int       `json:"version"` // Encryption scheme version
}

// TelemetryEncryptor wraps telemetry data with encryption
type TelemetryEncryptor struct {
	em *EncryptionManager
}

// FieldEncryption provides field-level encryption for selective encryption
type FieldEncryption struct {
	sensitiveFields []string
	em              *EncryptionManager
}

// NewEncryptionManager initializes the encryption system
func NewEncryptionManager(masterPassword string) (*EncryptionManager, error) {
	// Derive master key using Argon2id (memory-hard KDF)
	salt := []byte("shieldx-contauth-salt-v1") // In production, use random salt stored securely
	masterKey := argon2.IDKey([]byte(masterPassword), salt, 1, 64*1024, 4, 32)

	em := &EncryptionManager{
		masterKey: masterKey,
		keys:      make(map[string][]byte),
		keyRotation: &KeyRotationPolicy{
			RotationInterval: 30 * 24 * time.Hour, // 30 days
			MaxKeyAge:        60 * 24 * time.Hour, // 60 days
			GracePeriod:      7 * 24 * time.Hour,  // 7 days
			lastRotation:     time.Now(),
		},
	}

	// Generate initial key
	if err := em.rotateKey(); err != nil {
		return nil, err
	}

	// Start automatic key rotation
	go em.autoRotateKeys()

	return em, nil
}

// EncryptTelemetry encrypts complete telemetry payload
func (em *EncryptionManager) EncryptTelemetry(data interface{}) (*EncryptedData, error) {
	// Serialize data to JSON
	plaintext, err := json.Marshal(data)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal data: %w", err)
	}

	// Use ChaCha20-Poly1305 for better performance
	return em.encryptWithChaCha20(plaintext)
}

// DecryptTelemetry decrypts encrypted telemetry
func (em *EncryptionManager) DecryptTelemetry(encrypted *EncryptedData, target interface{}) error {
	var plaintext []byte
	var err error

	switch encrypted.Algorithm {
	case "chacha20poly1305":
		plaintext, err = em.decryptWithChaCha20(encrypted)
	case "aes-gcm":
		plaintext, err = em.decryptWithAESGCM(encrypted)
	default:
		return fmt.Errorf("unsupported algorithm: %s", encrypted.Algorithm)
	}

	if err != nil {
		return err
	}

	return json.Unmarshal(plaintext, target)
}

// encryptWithChaCha20 uses ChaCha20-Poly1305 AEAD
func (em *EncryptionManager) encryptWithChaCha20(plaintext []byte) (*EncryptedData, error) {
	em.mu.RLock()
	key, exists := em.keys[em.activeKeyID]
	keyID := em.activeKeyID
	em.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("active key not found")
	}

	aead, err := chacha20poly1305.NewX(key)
	if err != nil {
		return nil, fmt.Errorf("failed to create cipher: %w", err)
	}

	// Generate nonce
	nonce := make([]byte, aead.NonceSize())
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return nil, fmt.Errorf("failed to generate nonce: %w", err)
	}

	// Encrypt and authenticate
	ciphertext := aead.Seal(nil, nonce, plaintext, nil)

	return &EncryptedData{
		KeyID:       keyID,
		Algorithm:   "chacha20poly1305",
		Nonce:       base64.StdEncoding.EncodeToString(nonce),
		Ciphertext:  base64.StdEncoding.EncodeToString(ciphertext),
		EncryptedAt: time.Now(),
		Version:     1,
	}, nil
}

// decryptWithChaCha20 decrypts ChaCha20-Poly1305 AEAD
func (em *EncryptionManager) decryptWithChaCha20(encrypted *EncryptedData) ([]byte, error) {
	em.mu.RLock()
	key, exists := em.keys[encrypted.KeyID]
	em.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("decryption key not found: %s", encrypted.KeyID)
	}

	aead, err := chacha20poly1305.NewX(key)
	if err != nil {
		return nil, fmt.Errorf("failed to create cipher: %w", err)
	}

	nonce, err := base64.StdEncoding.DecodeString(encrypted.Nonce)
	if err != nil {
		return nil, fmt.Errorf("invalid nonce: %w", err)
	}

	ciphertext, err := base64.StdEncoding.DecodeString(encrypted.Ciphertext)
	if err != nil {
		return nil, fmt.Errorf("invalid ciphertext: %w", err)
	}

	plaintext, err := aead.Open(nil, nonce, ciphertext, nil)
	if err != nil {
		return nil, fmt.Errorf("decryption failed: %w", err)
	}

	return plaintext, nil
}

// encryptWithAESGCM uses AES-256-GCM AEAD
func (em *EncryptionManager) encryptWithAESGCM(plaintext []byte) (*EncryptedData, error) {
	em.mu.RLock()
	key, exists := em.keys[em.activeKeyID]
	keyID := em.activeKeyID
	em.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("active key not found")
	}

	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, fmt.Errorf("failed to create cipher: %w", err)
	}

	aead, err := cipher.NewGCM(block)
	if err != nil {
		return nil, fmt.Errorf("failed to create GCM: %w", err)
	}

	nonce := make([]byte, aead.NonceSize())
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return nil, fmt.Errorf("failed to generate nonce: %w", err)
	}

	ciphertext := aead.Seal(nil, nonce, plaintext, nil)

	return &EncryptedData{
		KeyID:       keyID,
		Algorithm:   "aes-gcm",
		Nonce:       base64.StdEncoding.EncodeToString(nonce),
		Ciphertext:  base64.StdEncoding.EncodeToString(ciphertext),
		EncryptedAt: time.Now(),
		Version:     1,
	}, nil
}

// decryptWithAESGCM decrypts AES-256-GCM
func (em *EncryptionManager) decryptWithAESGCM(encrypted *EncryptedData) ([]byte, error) {
	em.mu.RLock()
	key, exists := em.keys[encrypted.KeyID]
	em.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("decryption key not found: %s", encrypted.KeyID)
	}

	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, fmt.Errorf("failed to create cipher: %w", err)
	}

	aead, err := cipher.NewGCM(block)
	if err != nil {
		return nil, fmt.Errorf("failed to create GCM: %w", err)
	}

	nonce, err := base64.StdEncoding.DecodeString(encrypted.Nonce)
	if err != nil {
		return nil, fmt.Errorf("invalid nonce: %w", err)
	}

	ciphertext, err := base64.StdEncoding.DecodeString(encrypted.Ciphertext)
	if err != nil {
		return nil, fmt.Errorf("invalid ciphertext: %w", err)
	}

	plaintext, err := aead.Open(nil, nonce, ciphertext, nil)
	if err != nil {
		return nil, fmt.Errorf("decryption failed: %w", err)
	}

	return plaintext, nil
}

// rotateKey generates a new encryption key
func (em *EncryptionManager) rotateKey() error {
	em.mu.Lock()
	defer em.mu.Unlock()

	// Generate key ID
	keyID := fmt.Sprintf("key-%d", time.Now().Unix())

	// Derive new key from master key using HKDF-like approach
	info := []byte(keyID)
	h := sha256.New()
	h.Write(em.masterKey)
	h.Write(info)
	derivedKey := h.Sum(nil)

	em.keys[keyID] = derivedKey
	em.activeKeyID = keyID
	em.keyRotation.lastRotation = time.Now()

	// Clean up old keys outside grace period
	em.cleanupOldKeys()

	return nil
}

// cleanupOldKeys removes keys older than MaxKeyAge
func (em *EncryptionManager) cleanupOldKeys() {
	cutoff := time.Now().Add(-em.keyRotation.MaxKeyAge)

	for keyID := range em.keys {
		// Extract timestamp from keyID
		var timestamp int64
		fmt.Sscanf(keyID, "key-%d", &timestamp)
		keyTime := time.Unix(timestamp, 0)

		if keyTime.Before(cutoff) && keyID != em.activeKeyID {
			delete(em.keys, keyID)
		}
	}
}

// autoRotateKeys periodically rotates encryption keys
func (em *EncryptionManager) autoRotateKeys() {
	ticker := time.NewTicker(24 * time.Hour) // Check daily
	defer ticker.Stop()

	for range ticker.C {
		em.keyRotation.mu.Lock()
		shouldRotate := time.Since(em.keyRotation.lastRotation) >= em.keyRotation.RotationInterval
		em.keyRotation.mu.Unlock()

		if shouldRotate {
			if err := em.rotateKey(); err != nil {
				// Log error but continue
				fmt.Printf("Key rotation failed: %v\n", err)
			}
		}
	}
}

// === Field-Level Encryption ===

// NewFieldEncryption creates field-level encryptor
func NewFieldEncryption(em *EncryptionManager, sensitiveFields []string) *FieldEncryption {
	return &FieldEncryption{
		sensitiveFields: sensitiveFields,
		em:              em,
	}
}

// EncryptFields selectively encrypts sensitive fields in a map
func (fe *FieldEncryption) EncryptFields(data map[string]interface{}) (map[string]interface{}, error) {
	result := make(map[string]interface{})

	for key, value := range data {
		if fe.isSensitive(key) {
			// Encrypt this field
			valueBytes, err := json.Marshal(value)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal field %s: %w", key, err)
			}

			encrypted, err := fe.em.encryptWithChaCha20(valueBytes)
			if err != nil {
				return nil, fmt.Errorf("failed to encrypt field %s: %w", key, err)
			}

			result[key] = encrypted
		} else {
			// Keep as-is
			result[key] = value
		}
	}

	return result, nil
}

// DecryptFields decrypts previously encrypted fields
func (fe *FieldEncryption) DecryptFields(data map[string]interface{}) (map[string]interface{}, error) {
	result := make(map[string]interface{})

	for key, value := range data {
		if encMap, ok := value.(map[string]interface{}); ok {
			// Check if this looks like encrypted data
			if _, hasKeyID := encMap["key_id"]; hasKeyID {
				// Try to decrypt
				encData := &EncryptedData{
					KeyID:      encMap["key_id"].(string),
					Algorithm:  encMap["algorithm"].(string),
					Nonce:      encMap["nonce"].(string),
					Ciphertext: encMap["ciphertext"].(string),
				}

				var decrypted interface{}
				if err := fe.em.DecryptTelemetry(encData, &decrypted); err != nil {
					return nil, fmt.Errorf("failed to decrypt field %s: %w", key, err)
				}

				result[key] = decrypted
				continue
			}
		}

		result[key] = value
	}

	return result, nil
}

func (fe *FieldEncryption) isSensitive(field string) bool {
	for _, sensitive := range fe.sensitiveFields {
		if field == sensitive {
			return true
		}
	}
	return false
}

// === Utility Functions ===

// HashForLogging creates one-way hash for safe logging
func HashForLogging(data string) string {
	hash := sha256.Sum256([]byte(data))
	return base64.URLEncoding.EncodeToString(hash[:])[:16] // First 16 chars
}

// MaskPII masks personally identifiable information
func MaskPII(data string) string {
	if len(data) <= 4 {
		return "****"
	}
	return data[:2] + "***" + data[len(data)-2:]
}

// SecureEraseBytes securely erases sensitive data from memory
func SecureEraseBytes(data []byte) {
	for i := range data {
		data[i] = 0
	}
}

// ValidateEncryptedData checks integrity of encrypted data structure
func ValidateEncryptedData(encrypted *EncryptedData) error {
	if encrypted.KeyID == "" {
		return fmt.Errorf("missing key_id")
	}
	if encrypted.Algorithm != "chacha20poly1305" && encrypted.Algorithm != "aes-gcm" {
		return fmt.Errorf("unsupported algorithm: %s", encrypted.Algorithm)
	}
	if encrypted.Nonce == "" {
		return fmt.Errorf("missing nonce")
	}
	if encrypted.Ciphertext == "" {
		return fmt.Errorf("missing ciphertext")
	}
	if encrypted.Version != 1 {
		return fmt.Errorf("unsupported version: %d", encrypted.Version)
	}
	return nil
}

// GetKeyMetadata returns information about active keys (without exposing key material)
func (em *EncryptionManager) GetKeyMetadata() map[string]interface{} {
	em.mu.RLock()
	defer em.mu.RUnlock()

	metadata := map[string]interface{}{
		"active_key_id": em.activeKeyID,
		"total_keys":    len(em.keys),
		"last_rotation": em.keyRotation.lastRotation,
		"next_rotation": em.keyRotation.lastRotation.Add(em.keyRotation.RotationInterval),
		"key_ids":       make([]string, 0, len(em.keys)),
	}

	for keyID := range em.keys {
		metadata["key_ids"] = append(metadata["key_ids"].([]string), keyID)
	}

	return metadata
}
