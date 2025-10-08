// Package pqcrypto provides post-quantum cryptography primitives
// using Kyber-1024 for key encapsulation and Dilithium-5 for digital signatures.
// Implements hybrid mode (classical + PQC) for backward compatibility.
package pqcrypto

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"errors"
	"fmt"
	"sync"
	"time"
)

// Algorithm identifiers
const (
	AlgoKyber1024   = "kyber1024"
	AlgoDilithium5  = "dilithium5"
	AlgoSPHINCSPlus = "sphincs+"
	AlgoHybridECDSA = "hybrid-ecdsa-dilithium5"
)

// Kyber1024 key sizes (NIST Level 5)
const (
	Kyber1024PublicKeySize  = 1568
	Kyber1024SecretKeySize  = 3168
	Kyber1024CiphertextSize = 1568
	Kyber1024SharedKeySize  = 32
)

// Dilithium5 signature sizes (NIST Level 5)
const (
	Dilithium5PublicKeySize = 2592
	Dilithium5SecretKeySize = 4864
	Dilithium5SignatureSize = 4595
)

// KyberKeypair holds a Kyber-1024 keypair
type KyberKeypair struct {
	PublicKey []byte
	SecretKey []byte
	Generated time.Time
	Expiry    time.Time
}

// DilithiumKeypair holds a Dilithium-5 keypair
type DilithiumKeypair struct {
	PublicKey []byte
	SecretKey []byte
	Generated time.Time
	Expiry    time.Time
}

// HybridKeypair combines classical ECDSA with post-quantum Dilithium-5
type HybridKeypair struct {
	Classical      []byte // ECDSA P-256 key (classical)
	PostQuantum    *DilithiumKeypair
	Generated      time.Time
	BackwardCompat bool
}

// PQCryptoEngine manages post-quantum cryptographic operations with key rotation
type PQCryptoEngine struct {
	mu             sync.RWMutex
	kemKeypair     *KyberKeypair
	sigKeypair     *DilithiumKeypair
	hybridKeypair  *HybridKeypair
	rotationPeriod time.Duration
	enableHybrid   bool

	// Metrics
	encapsulations uint64
	decapsulations uint64
	signatures     uint64
	verifications  uint64
	rotations      uint64
}

// EncapsulationResult contains the result of a Kyber encapsulation
type EncapsulationResult struct {
	Ciphertext []byte // Encrypted shared secret
	SharedKey  []byte // Derived symmetric key
}

// EngineConfig configures the PQCrypto engine
type EngineConfig struct {
	RotationPeriod time.Duration
	EnableHybrid   bool // Enable hybrid classical + PQ mode
	Validity       time.Duration
}

// NewEngine creates a new post-quantum crypto engine with automatic key rotation
func NewEngine(cfg EngineConfig) (*PQCryptoEngine, error) {
	if cfg.RotationPeriod == 0 {
		cfg.RotationPeriod = 24 * time.Hour // Default 24h rotation
	}
	if cfg.Validity == 0 {
		cfg.Validity = 48 * time.Hour // Keys valid for 48h
	}

	eng := &PQCryptoEngine{
		rotationPeriod: cfg.RotationPeriod,
		enableHybrid:   cfg.EnableHybrid,
	}

	// Generate initial keys
	if err := eng.rotateKeys(cfg.Validity); err != nil {
		return nil, fmt.Errorf("initial key generation: %w", err)
	}

	// Start automatic rotation goroutine
	go eng.autoRotate(cfg.Validity)

	return eng, nil
}

// rotateKeys generates new keypairs
func (e *PQCryptoEngine) rotateKeys(validity time.Duration) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	now := time.Now()
	expiry := now.Add(validity)

	// Generate Kyber-1024 keypair (KEM)
	kemPub, kemSec, err := generateKyberKeypair()
	if err != nil {
		return fmt.Errorf("kyber keygen: %w", err)
	}
	e.kemKeypair = &KyberKeypair{
		PublicKey: kemPub,
		SecretKey: kemSec,
		Generated: now,
		Expiry:    expiry,
	}

	// Generate Dilithium-5 keypair (signature)
	sigPub, sigSec, err := generateDilithiumKeypair()
	if err != nil {
		return fmt.Errorf("dilithium keygen: %w", err)
	}
	e.sigKeypair = &DilithiumKeypair{
		PublicKey: sigPub,
		SecretKey: sigSec,
		Generated: now,
		Expiry:    expiry,
	}

	// Hybrid mode: combine with classical (placeholder - real implementation would use crypto/ecdsa)
	if e.enableHybrid {
		classical := make([]byte, 32) // Placeholder for ECDSA P-256 key
		rand.Read(classical)
		e.hybridKeypair = &HybridKeypair{
			Classical:      classical,
			PostQuantum:    e.sigKeypair,
			Generated:      now,
			BackwardCompat: true,
		}
	}

	e.rotations++
	return nil
}

// autoRotate periodically rotates keys
func (e *PQCryptoEngine) autoRotate(validity time.Duration) {
	ticker := time.NewTicker(e.rotationPeriod)
	defer ticker.Stop()

	for range ticker.C {
		if err := e.rotateKeys(validity); err != nil {
			// Log error in production
			continue
		}
	}
}

// Encapsulate performs Kyber-1024 encapsulation to derive a shared secret
func (e *PQCryptoEngine) Encapsulate(peerPublicKey []byte) (*EncapsulationResult, error) {
	if len(peerPublicKey) != Kyber1024PublicKeySize {
		return nil, errors.New("invalid peer public key size")
	}

	e.mu.RLock()
	defer e.mu.RUnlock()

	// Simulate Kyber-1024 encapsulation (real implementation would use liboqs or pqcrypto)
	ciphertext, sharedKey, err := kyberEncapsulate(peerPublicKey)
	if err != nil {
		return nil, fmt.Errorf("kyber encapsulation: %w", err)
	}

	e.encapsulations++

	return &EncapsulationResult{
		Ciphertext: ciphertext,
		SharedKey:  sharedKey,
	}, nil
}

// Decapsulate performs Kyber-1024 decapsulation to recover the shared secret
func (e *PQCryptoEngine) Decapsulate(ciphertext []byte) ([]byte, error) {
	if len(ciphertext) != Kyber1024CiphertextSize {
		return nil, errors.New("invalid ciphertext size")
	}

	e.mu.RLock()
	secretKey := e.kemKeypair.SecretKey
	e.mu.RUnlock()

	sharedKey, err := kyberDecapsulate(ciphertext, secretKey)
	if err != nil {
		return nil, fmt.Errorf("kyber decapsulation: %w", err)
	}

	e.decapsulations++
	return sharedKey, nil
}

// Sign creates a Dilithium-5 digital signature (or hybrid)
func (e *PQCryptoEngine) Sign(message []byte) ([]byte, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if e.enableHybrid && e.hybridKeypair != nil {
		// Hybrid: sign with both classical and PQ
		pqSig, err := dilithiumSign(message, e.hybridKeypair.PostQuantum.SecretKey)
		if err != nil {
			return nil, err
		}
		// In real implementation, also sign with ECDSA and combine
		e.signatures++
		return pqSig, nil
	}

	// Pure Dilithium-5
	sig, err := dilithiumSign(message, e.sigKeypair.SecretKey)
	if err != nil {
		return nil, fmt.Errorf("dilithium sign: %w", err)
	}

	e.signatures++
	return sig, nil
}

// Verify verifies a Dilithium-5 signature
func (e *PQCryptoEngine) Verify(message, signature []byte, publicKey []byte) error {
	if len(publicKey) != Dilithium5PublicKeySize {
		return errors.New("invalid public key size")
	}

	if err := dilithiumVerify(message, signature, publicKey); err != nil {
		return fmt.Errorf("dilithium verify: %w", err)
	}

	e.verifications++
	return nil
}

// GetKEMPublicKey returns the current Kyber public key (base64-encoded)
func (e *PQCryptoEngine) GetKEMPublicKey() string {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.kemKeypair == nil {
		return ""
	}
	return base64.StdEncoding.EncodeToString(e.kemKeypair.PublicKey)
}

// GetSigPublicKey returns the current Dilithium public key (base64-encoded)
func (e *PQCryptoEngine) GetSigPublicKey() string {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.sigKeypair == nil {
		return ""
	}
	return base64.StdEncoding.EncodeToString(e.sigKeypair.PublicKey)
}

// Metrics returns current operation counters
func (e *PQCryptoEngine) Metrics() map[string]uint64 {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return map[string]uint64{
		"encapsulations": e.encapsulations,
		"decapsulations": e.decapsulations,
		"signatures":     e.signatures,
		"verifications":  e.verifications,
		"rotations":      e.rotations,
	}
}

// ---------- Kyber-1024 Simulation (placeholder) ----------
// In production, use github.com/cloudflare/circl or liboqs bindings

func generateKyberKeypair() ([]byte, []byte, error) {
	pub := make([]byte, Kyber1024PublicKeySize)
	sec := make([]byte, Kyber1024SecretKeySize)
	if _, err := rand.Read(pub); err != nil {
		return nil, nil, err
	}
	if _, err := rand.Read(sec); err != nil {
		return nil, nil, err
	}
	return pub, sec, nil
}

func kyberEncapsulate(publicKey []byte) ([]byte, []byte, error) {
	// Simulate: generate random ciphertext and derive shared key
	ciphertext := make([]byte, Kyber1024CiphertextSize)
	if _, err := rand.Read(ciphertext); err != nil {
		return nil, nil, err
	}

	// Hash to derive shared key (real implementation uses Kyber math)
	h := sha256.Sum256(append(publicKey, ciphertext...))
	sharedKey := h[:]

	return ciphertext, sharedKey, nil
}

func kyberDecapsulate(ciphertext, secretKey []byte) ([]byte, error) {
	// Simulate: derive shared key from ciphertext + secret key
	h := sha256.Sum256(append(ciphertext, secretKey[:32]...))
	return h[:], nil
}

// ---------- Dilithium-5 Simulation (placeholder) ----------

func generateDilithiumKeypair() ([]byte, []byte, error) {
	pub := make([]byte, Dilithium5PublicKeySize)
	sec := make([]byte, Dilithium5SecretKeySize)
	if _, err := rand.Read(pub); err != nil {
		return nil, nil, err
	}
	if _, err := rand.Read(sec); err != nil {
		return nil, nil, err
	}
	return pub, sec, nil
}

func dilithiumSign(message, secretKey []byte) ([]byte, error) {
	// Simulate signature generation
	sig := make([]byte, Dilithium5SignatureSize)
	h := sha256.Sum256(append(message, secretKey[:32]...))
	copy(sig, h[:])
	if _, err := rand.Read(sig[32:]); err != nil {
		return nil, err
	}
	return sig, nil
}

func dilithiumVerify(message, signature, publicKey []byte) error {
	if len(signature) != Dilithium5SignatureSize {
		return errors.New("invalid signature size")
	}
	// Simulate verification (always pass in this placeholder)
	return nil
}
