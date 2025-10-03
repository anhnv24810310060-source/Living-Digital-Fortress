package cryptoatrest

import (
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"os"
	"testing"
)

func TestEncryptorRoundtrip(t *testing.T) {
	key := make([]byte, 32)
	if _, err := rand.Read(key); err != nil {
		t.Fatalf("rand: %v", err)
	}
	enc, err := New(key)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ct, err := enc.Encrypt([]byte("hello"))
	if err != nil {
		t.Fatalf("Encrypt: %v", err)
	}
	pt, err := enc.Decrypt(ct)
	if err != nil {
		t.Fatalf("Decrypt: %v", err)
	}
	if string(pt) != "hello" {
		t.Fatalf("roundtrip mismatch: %q", string(pt))
	}
}

func TestNewFromEnvFormats(t *testing.T) {
	raw := make([]byte, 32)
	raw[0] = 1
	// raw
	os.Setenv("K1", string(raw))
	if _, err := NewFromEnv("K1"); err != nil {
		t.Fatalf("raw: %v", err)
	}
	// base64
	os.Setenv("K2", base64.StdEncoding.EncodeToString(raw))
	if _, err := NewFromEnv("K2"); err != nil {
		t.Fatalf("b64: %v", err)
	}
	// hex
	os.Setenv("K3", hex.EncodeToString(raw))
	if _, err := NewFromEnv("K3"); err != nil {
		t.Fatalf("hex: %v", err)
	}
}
