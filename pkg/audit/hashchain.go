package audit

import (
	"crypto/sha256"
	"encoding/hex"
	"io"
	"os"
)

// HashChain computes a simple rolling SHA-256 hash over a file to produce an anchor value.
func HashChain(filePath string) (string, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
