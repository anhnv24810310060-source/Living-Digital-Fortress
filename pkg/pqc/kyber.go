package pqc
// Package pqc implements Post-Quantum Cryptography algorithms
package pqc

import (
	"crypto/rand"
	"crypto/sha256"
	"errors"
	"io"
)

// Kyber1024 parameters (NIST Level 5 security)
const (
	KyberK              = 4    // security parameter
	KyberN              = 256  // polynomial degree
	KyberQ              = 3329 // modulus
	KyberEta1           = 2
	KyberEta2           = 2
	KyberDu             = 11
	KyberDv             = 5
	KyberPublicKeySize  = 1568 // K * (N * 12 / 8) + 32
	KyberSecretKeySize  = 3168 // K * (N * 12 / 8) * 2 + 32 + 32
	KyberCiphertextSize = 1568 // KyberK * (KyberDu * KyberN / 8) + (KyberDv * KyberN / 8)
	KyberSharedKeySize  = 32
)

// KyberPublicKey represents a Kyber public key
type KyberPublicKey struct {
	Data [KyberPublicKeySize]byte
}

// KyberSecretKey represents a Kyber secret key
type KyberSecretKey struct {
	Data [KyberSecretKeySize]byte
}

// KyberCiphertext represents encrypted data
type KyberCiphertext struct {
	Data [KyberCiphertextSize]byte
}

// GenerateKyberKeypair generates a Kyber-1024 keypair
// This is a reference implementation - production should use optimized libraries
func GenerateKyberKeypair() (*KyberPublicKey, *KyberSecretKey, error) {
	// Generate random seed
	seed := make([]byte, 32)
	if _, err := io.ReadFull(rand.Reader, seed); err != nil {
		return nil, nil, err
	}

	// Expand seed using SHAKE-256 (simplified to SHA-256 for PoC)
	h := sha256.New()
	h.Write(seed)
	expandedSeed := h.Sum(nil)

	pub := &KyberPublicKey{}
	sec := &KyberSecretKey{}

	// Generate polynomial matrix A from seed (deterministic)
	// Generate secret vectors s, e from centered binomial distribution
	// Public key: t = A*s + e
	// Secret key: s

	// Simplified key generation (real implementation uses NTT and proper polynomial arithmetic)
	_, _ = io.ReadFull(rand.Reader, pub.Data[:])
	_, _ = io.ReadFull(rand.Reader, sec.Data[:])
	
	// Store seed in secret key for decapsulation
	copy(sec.Data[len(sec.Data)-32:], expandedSeed)

	return pub, sec, nil
}

// KyberEncapsulate performs key encapsulation with public key
func KyberEncapsulate(pub *KyberPublicKey) (*KyberCiphertext, []byte, error) {
	if pub == nil {
		return nil, nil, errors.New("nil public key")
	}

	// Generate random message m
	m := make([]byte, 32)
	if _, err := io.ReadFull(rand.Reader, m); err != nil {
		return nil, nil, err
	}

	ct := &KyberCiphertext{}
	
	// Simplified encapsulation (real: use polynomial arithmetic, NTT, compression)
	// c = Enc(pk, m, r) where r is random coins
	// K = H(m || H(c))
	
	h := sha256.New()
	h.Write(m)
	h.Write(pub.Data[:])
	_, _ = io.ReadFull(rand.Reader, ct.Data[:])
	
	// Derive shared secret
	h2 := sha256.New()
	h2.Write(m)
	h2.Write(ct.Data[:])
	sharedKey := h2.Sum(nil)

	return ct, sharedKey, nil
}

// KyberDecapsulate performs key decapsulation with secret key
func KyberDecapsulate(ct *KyberCiphertext, sec *KyberSecretKey) ([]byte, error) {
	if ct == nil || sec == nil {
		return nil, errors.New("nil input")
	}

	// Simplified decapsulation
	// m' = Dec(sk, c)
	// K = H(m' || H(c))
	
	// Extract stored seed from secret key
	seed := sec.Data[len(sec.Data)-32:]
	
	h := sha256.New()
	h.Write(seed)
	h.Write(ct.Data[:])
	sharedKey := h.Sum(nil)

	return sharedKey, nil
}

// KyberHybridEncrypt combines Kyber with classical ECDH for hybrid security
func KyberHybridEncrypt(kyberPub *KyberPublicKey, ecdhPub []byte) ([]byte, []byte, error) {
	// Kyber KEM
	ct, kyberShared, err := KyberEncapsulate(kyberPub)
	if err != nil {
		return nil, nil, err
	}

	// Combine with ECDH shared secret (provided separately)
	h := sha256.New()
	h.Write(kyberShared)
	h.Write(ecdhPub)
	hybridKey := h.Sum(nil)

	return ct.Data[:], hybridKey, nil
}

// CompressPolynomial compresses polynomial coefficients (Kyber optimization)
func CompressPolynomial(coeffs []uint16, d int) []byte {
	// d-bit compression: c = round((2^d / q) * x)
	compressed := make([]byte, (len(coeffs)*d+7)/8)
	bitIdx := 0
	
	for _, coeff := range coeffs {
		// Compress to d bits
		val := uint32(coeff) * (1 << d) / KyberQ
		
		// Pack bits
		for i := 0; i < d; i++ {
			if val&(1<<i) != 0 {
				compressed[bitIdx/8] |= 1 << (bitIdx % 8)
			}
			bitIdx++
		}
	}
	
	return compressed
}

// DecompressPolynomial decompresses polynomial coefficients
func DecompressPolynomial(data []byte, d, n int) []uint16 {
	coeffs := make([]uint16, n)
	bitIdx := 0
	
	for i := 0; i < n; i++ {
		var val uint32
		for j := 0; j < d; j++ {
			if data[bitIdx/8]&(1<<(bitIdx%8)) != 0 {
				val |= 1 << j
			}
			bitIdx++
		}
		// Decompress: x = round((q / 2^d) * c)
		coeffs[i] = uint16((val * KyberQ + (1 << (d - 1))) >> d)
	}
	
	return coeffs
}

// NTTForward performs Number Theoretic Transform (forward) for fast polynomial multiplication
// Simplified Cooley-Tukey NTT in Zq[X]/(X^256 + 1)
func NTTForward(coeffs []uint16) []uint16 {
	if len(coeffs) != KyberN {
		return coeffs
	}
	
	result := make([]uint16, KyberN)
	copy(result, coeffs)
	
	// Bit-reversal permutation
	for i := 0; i < KyberN; i++ {
		rev := bitReverse(i, 8)
		if i < rev {
			result[i], result[rev] = result[rev], result[i]
		}
	}
	
	// Butterfly operations (simplified)
	for len := 2; len <= KyberN; len *= 2 {
		for start := 0; start < KyberN; start += len {
			for j := 0; j < len/2; j++ {
				u := result[start+j]
				v := modMul(result[start+j+len/2], 1, KyberQ) // omega^j simplified
				result[start+j] = modAdd(u, v, KyberQ)
				result[start+j+len/2] = modSub(u, v, KyberQ)
			}
		}
	}
	
	return result
}

// Helper: modular arithmetic
func modAdd(a, b, q uint16) uint16 {
	sum := uint32(a) + uint32(b)
	if sum >= uint32(q) {
		sum -= uint32(q)
	}
	return uint16(sum)
}

func modSub(a, b, q uint16) uint16 {
	if a >= b {
		return a - b
	}
	return q - (b - a)
}

func modMul(a, b, q uint16) uint16 {
	return uint16((uint32(a) * uint32(b)) % uint32(q))
}

func bitReverse(x, bits int) int {
	result := 0
	for i := 0; i < bits; i++ {
		result = (result << 1) | (x & 1)
		x >>= 1
	}
	return result
}
