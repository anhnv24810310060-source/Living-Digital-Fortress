package cryptoatrest

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
)

// Encryptor provides simple AES-256-GCM encryption for data-at-rest.
type Encryptor struct {
	aead cipher.AEAD
}

// New creates an Encryptor from a 32-byte key.
func New(key []byte) (*Encryptor, error) {
	if len(key) != 32 {
		return nil, fmt.Errorf("key must be 32 bytes, got %d", len(key))
	}
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}
	aead, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}
	return &Encryptor{aead: aead}, nil
}

// NewFromEnv creates an Encryptor using key from env var (raw/base64/hex).
func NewFromEnv(envKey string) (*Encryptor, error) {
	v := os.Getenv(envKey)
	if v == "" {
		return nil, errors.New("encryption key env not set: " + envKey)
	}
	// try raw 32 bytes
	if len(v) == 32 {
		if e, err := New([]byte(v)); err == nil {
			return e, nil
		}
	}
	if b, err := base64.StdEncoding.DecodeString(v); err == nil && len(b) == 32 {
		return New(b)
	}
	if b, err := hex.DecodeString(v); err == nil && len(b) == 32 {
		return New(b)
	}
	return nil, errors.New("invalid key format for " + envKey + ": need 32B raw/base64/hex")
}

// Encrypt returns base64(nonce||ciphertext) for input plaintext.
func (e *Encryptor) Encrypt(plain []byte) (string, error) {
	nonce := make([]byte, e.aead.NonceSize())
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return "", err
	}
	ct := e.aead.Seal(nil, nonce, plain, nil)
	out := append(nonce, ct...)
	return base64.StdEncoding.EncodeToString(out), nil
}

// Decrypt accepts base64(nonce||ciphertext) and returns plaintext.
func (e *Encryptor) Decrypt(b64 string) ([]byte, error) {
	raw, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		return nil, err
	}
	if len(raw) < e.aead.NonceSize() {
		return nil, errors.New("ciphertext too short")
	}
	nonce := raw[:e.aead.NonceSize()]
	ct := raw[e.aead.NonceSize():]
	return e.aead.Open(nil, nonce, ct, nil)
}
