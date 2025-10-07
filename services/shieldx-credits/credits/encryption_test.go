package main

import (
	"encoding/base64"
	"os"
	"strings"
	"testing"
)

func TestEncryptAtRest_WithKey(t *testing.T) {
    // Build a deterministic 32-byte key and encode to base64
    key := make([]byte, 32)
    for i := range key { key[i] = byte(i) }
    os.Setenv("PAYMENT_ENC_KEY", "base64:"+base64.StdEncoding.EncodeToString(key))
    cl := &CreditLedger{}
    out, err := cl.encryptAtRest("card_ref_4111111111111111")
    if err != nil {
        t.Fatalf("encryptAtRest error: %v", err)
    }
    if !strings.HasPrefix(out, "enc:v1:") {
        t.Fatalf("expected prefix enc:v1:, got %s", out)
    }
}

func TestEncryptAtRest_NoKey(t *testing.T) {
    os.Unsetenv("PAYMENT_ENC_KEY")
    cl := &CreditLedger{}
    in := "plain_ref"
    out, err := cl.encryptAtRest(in)
    if err != nil {
        t.Fatalf("unexpected err: %v", err)
    }
    if out != in {
        t.Fatalf("expected passthrough without key, got %s", out)
    }
}
