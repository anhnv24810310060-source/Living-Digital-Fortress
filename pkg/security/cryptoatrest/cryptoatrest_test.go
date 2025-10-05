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
	// fallback to os.Setenv for environment (Go Setenv failing in this environment)
	if err := os.Setenv("K1", string(raw)); err != nil {
		t.Fatalf("setenv K1: %v", err)
	}
	if _, err := NewFromEnv("K1"); err != nil {
		t.Fatalf("raw: %v", err)
	}
	// base64
	if err := os.Setenv("K2", base64.StdEncoding.EncodeToString(raw)); err != nil {
		t.Fatalf("setenv K2: %v", err)
	}
	if _, err := NewFromEnv("K2"); err != nil {
		t.Fatalf("b64: %v", err)
	}
	// hex via _FILE fallback if normal setenv fails
	if err := os.Setenv("K3", hex.EncodeToString(raw)); err != nil {
		// try file fallback
		path := createTempKeyFile(t, hex.EncodeToString(raw))
		if err2 := os.Setenv("K3_FILE", path); err2 != nil {
			t.Fatalf("setenv K3_FILE: %v", err2)
		}
	}
	if _, err := NewFromEnv("K3"); err != nil {
		t.Fatalf("hex: %v", err)
	}
}

func createTempKeyFile(t *testing.T, data string) string {
	f, err := os.CreateTemp("", "key-*.txt")
	if err != nil {
		t.Fatalf("tempfile: %v", err)
	}
	if _, err := f.Write([]byte(data)); err != nil {
		t.Fatalf("write: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	return f.Name()
}
