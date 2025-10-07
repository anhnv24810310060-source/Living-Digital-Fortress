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
	// raw via _FILE fallback
	rawPath := createTempKeyFile(t, raw)
	t.Setenv("K1_FILE", rawPath)
	if _, err := NewFromEnv("K1"); err != nil {
		t.Fatalf("raw file: %v", err)
	}

	// base64
	b64 := base64.StdEncoding.EncodeToString(raw)
	t.Setenv("K2", b64)
	if _, err := NewFromEnv("K2"); err != nil {
		t.Fatalf("b64: %v", err)
	}

	// hex string
	hexVal := hex.EncodeToString(raw)
	t.Setenv("K3", hexVal)
	if _, err := NewFromEnv("K3"); err != nil {
		t.Fatalf("hex: %v", err)
	}

	// hex via _FILE fallback
	hexPath := createTempKeyFile(t, []byte(hexVal))
	t.Setenv("K4_FILE", hexPath)
	if _, err := NewFromEnv("K4"); err != nil {
		t.Fatalf("hex file fallback: %v", err)
	}
}

func createTempKeyFile(t *testing.T, data []byte) string {
	f, err := os.CreateTemp("", "key-*.txt")
	if err != nil {
		t.Fatalf("tempfile: %v", err)
	}
	if _, err := f.Write(data); err != nil {
		t.Fatalf("write: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	path := f.Name()
	t.Cleanup(func() {
		_ = os.Remove(path)
	})
	return path
}
