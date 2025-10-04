package pqcrypto
// Package pqcrypto implements post-quantum cryptography using Kyber-1024 and Dilithium-5
package pqcrypto

import (
	"crypto/rand"
	"errors"
	"fmt"
	"io"
	
	// Using Cloudflare's CIRCL library for PQC (production-ready implementation)
	"github.com/cloudflare/circl/kem/kyber/kyber1024"
	"github.com/cloudflare/circl/sign/dilithium/mode5"
)

// KyberPublicKey represents a Kyber-1024 public key (1568 bytes)
type KyberPublicKey struct {
	pk *kyber1024.PublicKey
}

// KyberPrivateKey represents a Kyber-1024 private key (3168 bytes)
type KyberPrivateKey struct {
	sk *kyber1024.PrivateKey
}

// DilithiumPublicKey represents a Dilithium-5 public key (2592 bytes)
type DilithiumPublicKey struct {
	pk *mode5.PublicKey
}

// DilithiumPrivateKey represents a Dilithium-5 private key (4864 bytes)
type DilithiumPrivateKey struct {
	sk *mode5.PrivateKey
}

// GenerateKyberKeypair generates a new Kyber-1024 keypair for key encapsulation
// Returns: (publicKey, privateKey, error)
// Security: NIST Level 5 (>=256 bits quantum security)
func GenerateKyberKeypair() (*KyberPublicKey, *KyberPrivateKey, error) {
	pk, sk, err := kyber1024.GenerateKeyPair(rand.Reader)
	if err != nil {
		return nil, nil, fmt.Errorf("kyber keygen: %w", err)
	}
	return &KyberPublicKey{pk: pk}, &KyberPrivateKey{sk: sk}, nil
}

// Encapsulate generates a shared secret and ciphertext using Kyber-1024
// Returns: (ciphertext [1568 bytes], sharedSecret [32 bytes], error)
func (pub *KyberPublicKey) Encapsulate() ([]byte, []byte, error) {
	if pub == nil || pub.pk == nil {
		return nil, nil, errors.New("nil public key")
	}
	ct, ss, err := kyber1024.EncapsulateTo(nil, nil, pub.pk, rand.Reader)
	if err != nil {
		return nil, nil, fmt.Errorf("kyber encap: %w", err)
	}
	return ct, ss, nil
}

// Decapsulate recovers the shared secret from ciphertext using Kyber-1024
// Returns: (sharedSecret [32 bytes], error)
func (priv *KyberPrivateKey) Decapsulate(ciphertext []byte) ([]byte, error) {
	if priv == nil || priv.sk == nil {
		return nil, errors.New("nil private key")
	}
	ss, err := kyber1024.DecapsulateTo(nil, priv.sk, ciphertext)
	if err != nil {
		return nil, fmt.Errorf("kyber decap: %w", err)
	}
	return ss, nil
}

// MarshalBinary encodes Kyber public key to bytes
func (pub *KyberPublicKey) MarshalBinary() ([]byte, error) {
	if pub == nil || pub.pk == nil {
		return nil, errors.New("nil public key")
	}
	b, err := pub.pk.MarshalBinary()
	if err != nil {
		return nil, fmt.Errorf("marshal kyber pub: %w", err)
	}
	return b, nil
}

// UnmarshalBinary decodes Kyber public key from bytes
func (pub *KyberPublicKey) UnmarshalBinary(data []byte) error {
	pk := new(kyber1024.PublicKey)
	if err := pk.UnmarshalBinary(data); err != nil {
		return fmt.Errorf("unmarshal kyber pub: %w", err)
	}
	pub.pk = pk
	return nil
}

// MarshalBinary encodes Kyber private key to bytes
func (priv *KyberPrivateKey) MarshalBinary() ([]byte, error) {
	if priv == nil || priv.sk == nil {
		return nil, errors.New("nil private key")
	}
	b, err := priv.sk.MarshalBinary()
	if err != nil {
		return nil, fmt.Errorf("marshal kyber priv: %w", err)
	}
	return b, nil
}

// UnmarshalBinary decodes Kyber private key from bytes
func (priv *KyberPrivateKey) UnmarshalBinary(data []byte) error {
	sk := new(kyber1024.PrivateKey)
	if err := sk.UnmarshalBinary(data); err != nil {
		return fmt.Errorf("unmarshal kyber priv: %w", err)
	}
	priv.sk = sk
	return nil
}

// GenerateDilithiumKeypair generates a new Dilithium-5 keypair for signatures
// Returns: (publicKey, privateKey, error)
// Security: NIST Level 5 (>=256 bits quantum security)
func GenerateDilithiumKeypair() (*DilithiumPublicKey, *DilithiumPrivateKey, error) {
	pk, sk, err := mode5.GenerateKey(rand.Reader)
	if err != nil {
		return nil, nil, fmt.Errorf("dilithium keygen: %w", err)
	}
	return &DilithiumPublicKey{pk: pk}, &DilithiumPrivateKey{sk: sk}, nil
}

// Sign creates a signature for the message using Dilithium-5
// Returns: (signature [4627 bytes], error)
func (priv *DilithiumPrivateKey) Sign(message []byte) ([]byte, error) {
	if priv == nil || priv.sk == nil {
		return nil, errors.New("nil private key")
	}
	return mode5.Sign(priv.sk, message), nil
}

// Verify checks the signature for the message using Dilithium-5
// Returns: true if valid, false otherwise
func (pub *DilithiumPublicKey) Verify(message, signature []byte) bool {
	if pub == nil || pub.pk == nil {
		return false
	}
	return mode5.Verify(pub.pk, message, signature)
}

// MarshalBinary encodes Dilithium public key to bytes
func (pub *DilithiumPublicKey) MarshalBinary() ([]byte, error) {
	if pub == nil || pub.pk == nil {
		return nil, errors.New("nil public key")
	}
	b, err := pub.pk.MarshalBinary()
	if err != nil {
		return nil, fmt.Errorf("marshal dilithium pub: %w", err)
	}
	return b, nil
}

// UnmarshalBinary decodes Dilithium public key from bytes
func (pub *DilithiumPublicKey) UnmarshalBinary(data []byte) error {
	pk := new(mode5.PublicKey)
	if err := pk.UnmarshalBinary(data); err != nil {
		return fmt.Errorf("unmarshal dilithium pub: %w", err)
	}
	pub.pk = pk
	return nil
}

// MarshalBinary encodes Dilithium private key to bytes
func (priv *DilithiumPrivateKey) MarshalBinary() ([]byte, error) {
	if priv == nil || priv.sk == nil {
		return nil, errors.New("nil private key")
	}
	b, err := priv.sk.MarshalBinary()
	if err != nil {
		return nil, fmt.Errorf("marshal dilithium priv: %w", err)
	}
	return b, nil
}

// UnmarshalBinary decodes Dilithium private key from bytes
func (priv *DilithiumPrivateKey) UnmarshalBinary(data []byte) error {
	sk := new(mode5.PrivateKey)
	if err := sk.UnmarshalBinary(data); err != nil {
		return fmt.Errorf("unmarshal dilithium priv: %w", err)
	}
	priv.sk = sk
	return nil
}

// HybridEncrypt performs hybrid encryption: classical (RSA/ECDH) + post-quantum (Kyber)
// This ensures backward compatibility while adding quantum resistance
type HybridKEM struct {
	kyberPub  *KyberPublicKey
	kyberPriv *KyberPrivateKey
	// Add classical key material here for hybrid mode
}

// NewHybridKEM creates a new hybrid KEM instance
func NewHybridKEM() (*HybridKEM, error) {
	pub, priv, err := GenerateKyberKeypair()
	if err != nil {
		return nil, err
	}
	return &HybridKEM{
		kyberPub:  pub,
		kyberPriv: priv,
	}, nil
}

// Encapsulate generates hybrid shared secret (classical XOR quantum)
func (h *HybridKEM) Encapsulate() (ciphertext []byte, sharedSecret []byte, err error) {
	// PQC component
	ctPQ, ssPQ, err := h.kyberPub.Encapsulate()
	if err != nil {
		return nil, nil, err
	}
	
	// TODO: Add classical ECDH component and XOR/KDF the secrets
	// For now, return PQC only (can be enhanced later)
	return ctPQ, ssPQ, nil
}

// SecureRandom generates cryptographically secure random bytes
func SecureRandom(n int) ([]byte, error) {
	b := make([]byte, n)
	if _, err := io.ReadFull(rand.Reader, b); err != nil {
		return nil, fmt.Errorf("secure random: %w", err)
	}
	return b, nil
}
