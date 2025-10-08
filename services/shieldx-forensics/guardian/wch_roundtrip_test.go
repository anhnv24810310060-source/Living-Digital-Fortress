package main

import (
	"crypto/ecdh"
	"crypto/rand"
	w "shieldx/shared/shieldx-common/pkg/wch"
	"testing"
)

// helper to derive shared secret between two X25519 keypairs
func x25519Shared(aPriv *ecdh.PrivateKey, bPubBytes []byte) []byte {
	curve := ecdh.X25519()
	bPub, _ := curve.NewPublicKey(bPubBytes)
	shared, _ := aPriv.ECDH(bPub)
	return shared
}

// TestWCHKeyDerivationRoundTrip verifies shared secret + HKDF derivation matches between sides.
func TestWCHKeyDerivationRoundTrip(t *testing.T) {
	curve := ecdh.X25519()
	gPriv, _ := curve.GenerateKey(rand.Reader) // guardian static
	cPriv, cPub, _ := w.GenerateClientEphemeral()
	sharedClient := x25519Shared(cPriv, gPriv.PublicKey().Bytes())
	sharedGuardian := x25519Shared(gPriv, cPub)
	if string(sharedClient) != string(sharedGuardian) {
		t.Fatal("shared secret mismatch")
	}
	key1, err := w.DeriveKey(sharedClient, "chan-1")
	if err != nil {
		t.Fatal(err)
	}
	key2, err := w.DeriveKey(sharedGuardian, "chan-1")
	if err != nil {
		t.Fatal(err)
	}
	if string(key1) != string(key2) {
		t.Fatal("HKDF key mismatch")
	}
}

// TestWCHSealOpen ensures plaintext survives encrypt/decrypt cycle.
func TestWCHSealOpen(t *testing.T) {
	curve := ecdh.X25519()
	gPriv, _ := curve.GenerateKey(rand.Reader)
	cPriv, cPub, _ := w.GenerateClientEphemeral()
	shared := x25519Shared(cPriv, gPriv.PublicKey().Bytes())
	key, _ := w.DeriveKey(shared, "channelA")
	pt := []byte("hello whisper channel")
	nonce, ct, err := w.Seal(key, pt)
	if err != nil {
		t.Fatalf("seal err: %v", err)
	}
	// guardian reconstructs shared from its priv + client pub
	shared2 := x25519Shared(gPriv, cPub)
	key2, _ := w.DeriveKey(shared2, "channelA")
	out, err := w.Open(key2, nonce, ct)
	if err != nil {
		t.Fatalf("open err: %v", err)
	}
	if string(out) != string(pt) {
		t.Fatalf("plaintext mismatch got %s", out)
	}
}

// TestWCHRekeyCounter verifies DeriveKeyWithCounter changes key material.
func TestWCHRekeyCounter(t *testing.T) {
	curve := ecdh.X25519()
	a, _ := curve.GenerateKey(rand.Reader)
	b, _ := curve.GenerateKey(rand.Reader)
	shared := x25519Shared(a, b.PublicKey().Bytes())
	base, _ := w.DeriveKey(shared, "chanX")
	k1, _ := w.DeriveKeyWithCounter(shared, "chanX", 1)
	k2, _ := w.DeriveKeyWithCounter(shared, "chanX", 2)
	if string(base) == string(k1) {
		t.Fatalf("expected counter key differ from base")
	}
	if string(k1) == string(k2) {
		t.Fatalf("expected counter 1 and 2 keys differ")
	}
	// Counter <=0 fallback equals base
	k0, _ := w.DeriveKeyWithCounter(shared, "chanX", 0)
	if string(k0) != string(base) {
		t.Fatalf("expected counter 0 same as base")
	}
}
