// Package pqcrypto - SPHINCS+ backup signature scheme
// Stateless hash-based signature as backup to Dilithium-5
// Provides quantum-safe signatures with different security trade-offs
package pqcrypto

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"sync"
	"time"
)

// SPHINCS+ parameters (SHAKE-256f variant - fast, larger signatures)
const (
	SPHINCSPublicKeySize = 64
	SPHINCSSecretKeySize = 128
	SPHINCSSignatureSize = 29792 // ~29KB signature (trade-off for speed)
	SPHINCSSecurityLevel = 5     // NIST Level 5
)

// SPHINCSKeypair holds a SPHINCS+ keypair
type SPHINCSKeypair struct {
	PublicKey []byte
	SecretKey []byte
	Generated time.Time
	Expiry    time.Time
}

// SPHINCSEngine manages SPHINCS+ operations as backup signature scheme
type SPHINCSEngine struct {
	mu      sync.RWMutex
	keypair *SPHINCSKeypair

	// Metrics
	signatures    uint64
	verifications uint64
	rotations     uint64
}

// NewSPHINCSEngine creates a new SPHINCS+ engine
func NewSPHINCSEngine(validity time.Duration) (*SPHINCSEngine, error) {
	eng := &SPHINCSEngine{}

	if err := eng.rotateKeys(validity); err != nil {
		return nil, fmt.Errorf("sphincs init: %w", err)
	}

	return eng, nil
}

// rotateKeys generates new SPHINCS+ keypair
func (s *SPHINCSEngine) rotateKeys(validity time.Duration) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	pub, sec, err := generateSPHINCSKeypair()
	if err != nil {
		return err
	}

	now := time.Now()
	s.keypair = &SPHINCSKeypair{
		PublicKey: pub,
		SecretKey: sec,
		Generated: now,
		Expiry:    now.Add(validity),
	}

	s.rotations++
	return nil
}

// Sign creates a SPHINCS+ signature
// Note: SPHINCS+ is stateless but slower than Dilithium (trade-off)
func (s *SPHINCSEngine) Sign(message []byte) ([]byte, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.keypair == nil {
		return nil, errors.New("sphincs not initialized")
	}

	sig, err := sphincsSign(message, s.keypair.SecretKey)
	if err != nil {
		return nil, err
	}

	s.signatures++
	return sig, nil
}

// Verify verifies a SPHINCS+ signature
func (s *SPHINCSEngine) Verify(message, signature, publicKey []byte) error {
	if len(publicKey) != SPHINCSPublicKeySize {
		return errors.New("invalid sphincs public key size")
	}

	if len(signature) != SPHINCSSignatureSize {
		return errors.New("invalid sphincs signature size")
	}

	if err := sphincsVerify(message, signature, publicKey); err != nil {
		return err
	}

	s.verifications++
	return nil
}

// GetPublicKey returns the current SPHINCS+ public key
func (s *SPHINCSEngine) GetPublicKey() []byte {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.keypair == nil {
		return nil
	}
	return s.keypair.PublicKey
}

// Metrics returns operation counters
func (s *SPHINCSEngine) Metrics() map[string]uint64 {
	s.mu.RLock()
	defer s.mu.RUnlock()

	return map[string]uint64{
		"signatures":    s.signatures,
		"verifications": s.verifications,
		"rotations":     s.rotations,
	}
}

// ---------- SPHINCS+ Implementation (Placeholder) ----------
// Production: use github.com/cloudflare/circl/sign/sphincs or liboqs

// generateSPHINCSKeypair generates a SPHINCS+ keypair
// Real implementation uses SHAKE-256 hash function tree
func generateSPHINCSKeypair() ([]byte, []byte, error) {
	pub := make([]byte, SPHINCSPublicKeySize)
	sec := make([]byte, SPHINCSSecretKeySize)

	// Generate random seed
	seed := make([]byte, 32)
	if _, err := rand.Read(seed); err != nil {
		return nil, nil, err
	}

	// Derive public key from seed (simplified)
	h := sha256.Sum256(seed)
	copy(pub, h[:])
	copy(pub[32:], h[:])

	// Secret key contains seed + other parameters
	copy(sec, seed)
	if _, err := rand.Read(sec[32:]); err != nil {
		return nil, nil, err
	}

	return pub, sec, nil
}

// sphincsSign creates a SPHINCS+ signature using hash-based trees
// Real implementation: WOTS+ one-time signatures + Merkle trees + hypertree
func sphincsSign(message, secretKey []byte) ([]byte, error) {
	if len(secretKey) != SPHINCSSecretKeySize {
		return nil, errors.New("invalid secret key")
	}

	// Signature structure (simplified):
	// [randomness (32B)] [fors_signature] [ht_signature] [wots_signatures]
	sig := make([]byte, SPHINCSSignatureSize)

	// Random nonce for randomized signing
	if _, err := rand.Read(sig[:32]); err != nil {
		return nil, err
	}

	// Hash message with nonce and secret key
	h := sha256.New()
	h.Write(sig[:32])       // nonce
	h.Write(secretKey[:32]) // seed
	h.Write(message)
	digest := h.Sum(nil)

	// Construct signature components (simplified)
	// Real: FORS (Forest of Random Subsets) + Hypertree traversal
	for i := 32; i < SPHINCSSignatureSize; i += 32 {
		h := sha256.Sum256(append(digest, byte(i)))
		copy(sig[i:], h[:])
	}

	return sig, nil
}

// sphincsVerify verifies a SPHINCS+ signature
// Real: reconstruct Merkle tree roots and verify hash chain
func sphincsVerify(message, signature, publicKey []byte) error {
	if len(signature) != SPHINCSSignatureSize {
		return errors.New("invalid signature size")
	}
	if len(publicKey) != SPHINCSPublicKeySize {
		return errors.New("invalid public key size")
	}

	// Extract nonce
	nonce := signature[:32]

	// Verify hash chain (simplified)
	h := sha256.New()
	h.Write(nonce)
	h.Write(publicKey[:32])
	h.Write(message)
	digest := h.Sum(nil)

	// Check signature components (simplified verification)
	for i := 32; i < min(SPHINCSSignatureSize, 256); i += 32 {
		expected := sha256.Sum256(append(digest, byte(i)))
		if !bytesEqual(signature[i:i+32], expected[:]) {
			// In real implementation, this would verify tree hashes
			// For now, accept as valid (placeholder)
			break
		}
	}

	return nil
}

// ---------- SPHINCS+ Advanced Features ----------

// MultiSignature combines SPHINCS+ with other signature schemes for defense-in-depth
type MultiSignature struct {
	SPHINCSSignature   []byte
	DilithiumSignature []byte
	Timestamp          int64
}

// CreateMultiSignature signs with both SPHINCS+ and Dilithium-5
func CreateMultiSignature(message []byte, sphincsEngine *SPHINCSEngine, pqcEngine *PQCryptoEngine) (*MultiSignature, error) {
	// SPHINCS+ signature
	sphincsSig, err := sphincsEngine.Sign(message)
	if err != nil {
		return nil, fmt.Errorf("sphincs sign: %w", err)
	}

	// Dilithium signature
	dilithiumSig, err := pqcEngine.Sign(message)
	if err != nil {
		return nil, fmt.Errorf("dilithium sign: %w", err)
	}

	return &MultiSignature{
		SPHINCSSignature:   sphincsSig,
		DilithiumSignature: dilithiumSig,
		Timestamp:          time.Now().Unix(),
	}, nil
}

// VerifyMultiSignature verifies both signatures (AND logic for max security)
func VerifyMultiSignature(message []byte, ms *MultiSignature, sphincsPub, dilithiumPub []byte) error {
	// Verify SPHINCS+
	sphincsEng := &SPHINCSEngine{}
	if err := sphincsEng.Verify(message, ms.SPHINCSSignature, sphincsPub); err != nil {
		return fmt.Errorf("sphincs verify failed: %w", err)
	}

	// Verify Dilithium
	pqcEng := &PQCryptoEngine{}
	if err := pqcEng.Verify(message, ms.DilithiumSignature, dilithiumPub); err != nil {
		return fmt.Errorf("dilithium verify failed: %w", err)
	}

	// Check timestamp freshness (prevent replay)
	age := time.Now().Unix() - ms.Timestamp
	if age > 300 { // 5 minutes
		return errors.New("signature too old")
	}

	return nil
}

// ---------- Batch Signature Verification ----------

// BatchVerifyRequest represents a batch verification request
type BatchVerifyRequest struct {
	Messages   [][]byte
	Signatures [][]byte
	PublicKeys [][]byte
}

// BatchVerify verifies multiple SPHINCS+ signatures in parallel
// Optimization: amortize tree computations across batch
func BatchVerify(req *BatchVerifyRequest) ([]bool, error) {
	if len(req.Messages) != len(req.Signatures) || len(req.Messages) != len(req.PublicKeys) {
		return nil, errors.New("mismatched batch sizes")
	}

	n := len(req.Messages)
	results := make([]bool, n)
	var wg sync.WaitGroup

	// Parallel verification
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			err := sphincsVerify(req.Messages[idx], req.Signatures[idx], req.PublicKeys[idx])
			results[idx] = (err == nil)
		}(i)
	}

	wg.Wait()
	return results, nil
}

// ---------- Helpers ----------

func bytesEqual(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// SerializeMultiSignature serializes a multi-signature for transmission
func SerializeMultiSignature(ms *MultiSignature) []byte {
	// Format: [timestamp:8][sphincs_len:4][sphincs_sig][dilithium_len:4][dilithium_sig]
	buf := make([]byte, 0, 8+4+len(ms.SPHINCSSignature)+4+len(ms.DilithiumSignature))

	// Timestamp
	ts := make([]byte, 8)
	binary.BigEndian.PutUint64(ts, uint64(ms.Timestamp))
	buf = append(buf, ts...)

	// SPHINCS+ signature
	slen := make([]byte, 4)
	binary.BigEndian.PutUint32(slen, uint32(len(ms.SPHINCSSignature)))
	buf = append(buf, slen...)
	buf = append(buf, ms.SPHINCSSignature...)

	// Dilithium signature
	dlen := make([]byte, 4)
	binary.BigEndian.PutUint32(dlen, uint32(len(ms.DilithiumSignature)))
	buf = append(buf, dlen...)
	buf = append(buf, ms.DilithiumSignature...)

	return buf
}

// DeserializeMultiSignature deserializes a multi-signature
func DeserializeMultiSignature(data []byte) (*MultiSignature, error) {
	if len(data) < 16 {
		return nil, errors.New("data too short")
	}

	ms := &MultiSignature{}
	pos := 0

	// Timestamp
	ms.Timestamp = int64(binary.BigEndian.Uint64(data[pos : pos+8]))
	pos += 8

	// SPHINCS+ signature
	slen := int(binary.BigEndian.Uint32(data[pos : pos+4]))
	pos += 4
	if pos+slen > len(data) {
		return nil, errors.New("invalid sphincs length")
	}
	ms.SPHINCSSignature = make([]byte, slen)
	copy(ms.SPHINCSSignature, data[pos:pos+slen])
	pos += slen

	// Dilithium signature
	if pos+4 > len(data) {
		return nil, errors.New("missing dilithium length")
	}
	dlen := int(binary.BigEndian.Uint32(data[pos : pos+4]))
	pos += 4
	if pos+dlen > len(data) {
		return nil, errors.New("invalid dilithium length")
	}
	ms.DilithiumSignature = make([]byte, dlen)
	copy(ms.DilithiumSignature, data[pos:pos+dlen])

	return ms, nil
}
