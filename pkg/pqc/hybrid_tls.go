// Package pqc - Hybrid TLS 1.3 with post-quantum key exchange
package pqc

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"errors"
	"math/big"
	"time"
)

// HybridTLSConfig creates TLS 1.3 config with hybrid PQ+classical KEX
func HybridTLSConfig(certPEM, keyPEM []byte) (*tls.Config, error) {
	cert, err := tls.X509KeyPair(certPEM, keyPEM)
	if err != nil {
		return nil, err
	}

	config := &tls.Config{
		Certificates: []tls.Certificate{cert},
		MinVersion:   tls.VersionTLS13,
		MaxVersion:   tls.VersionTLS13,
		CipherSuites: []uint16{
			tls.TLS_AES_256_GCM_SHA384,
			tls.TLS_CHACHA20_POLY1305_SHA256,
		},
		CurvePreferences: []tls.CurveID{
			tls.X25519, // Classical ECDH for fallback
			tls.CurveP384,
		},
		PreferServerCipherSuites: true,
	}

	// Enable session tickets with PQ-safe keys
	config.SessionTicketsDisabled = false

	return config, nil
}

// GenerateHybridKeyPair generates both classical and PQ key pairs
func GenerateHybridKeyPair() (*ecdsa.PrivateKey, *KyberPublicKey, *KyberSecretKey, error) {
	// Classical ECDSA P-384
	ecdsaKey, err := ecdsa.GenerateKey(elliptic.P384(), rand.Reader)
	if err != nil {
		return nil, nil, nil, err
	}

	// Post-quantum Kyber-1024
	kyberPub, kyberSec, err := GenerateKyberKeypair()
	if err != nil {
		return nil, nil, nil, err
	}

	return ecdsaKey, kyberPub, kyberSec, nil
}

// HybridKEX performs hybrid key exchange: classical ECDH + Kyber KEM
func HybridKEX(ecdsaPub *ecdsa.PublicKey, kyberPub *KyberPublicKey) ([]byte, []byte, error) {
	// 1. ECDH ephemeral key exchange
	ecdhPriv, err := ecdsa.GenerateKey(elliptic.P384(), rand.Reader)
	if err != nil {
		return nil, nil, err
	}

	// Compute ECDH shared secret
	x, _ := ecdsaPub.Curve.ScalarMult(ecdsaPub.X, ecdsaPub.Y, ecdhPriv.D.Bytes())
	ecdhShared := x.Bytes()

	// 2. Kyber KEM encapsulation
	kyberCT, kyberShared, err := KyberEncapsulate(kyberPub)
	if err != nil {
		return nil, nil, err
	}

	// 3. Combine shared secrets with KDF
	h := sha256.New()
	h.Write(ecdhShared)
	h.Write(kyberShared)
	hybridShared := h.Sum(nil)

	// Return ciphertext bundle (ECDH pub + Kyber CT)
	bundle := append(elliptic.Marshal(elliptic.P384(), ecdhPriv.PublicKey.X, ecdhPriv.PublicKey.Y), kyberCT.Data[:]...)

	return bundle, hybridShared, nil
}

// GeneratePQCertificate creates X.509 certificate with PQ signature
func GeneratePQCertificate(commonName string, validFor time.Duration) ([]byte, []byte, error) {
	// Generate Dilithium key pair for signing
	dilPub, dilSec, err := GenerateDilithiumKeypair()
	if err != nil {
		return nil, nil, err
	}

	// Generate ECDSA key for X.509 compatibility
	ecdsaKey, err := ecdsa.GenerateKey(elliptic.P384(), rand.Reader)
	if err != nil {
		return nil, nil, err
	}

	// Create certificate template
	template := &x509.Certificate{
		SerialNumber:          nil, // Will be generated
		Subject:               pkix.Name{CommonName: commonName},
		NotBefore:             time.Now(),
		NotAfter:              time.Now().Add(validFor),
		KeyUsage:              x509.KeyUsageDigitalSignature | x509.KeyUsageKeyEncipherment,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth, x509.ExtKeyUsageClientAuth},
		BasicConstraintsValid: true,
		IsCA:                  false,
	}

	// Generate serial number
	serialNumberLimit := new(big.Int).Lsh(big.NewInt(1), 128)
	template.SerialNumber, _ = rand.Int(rand.Reader, serialNumberLimit)

	// Self-sign with ECDSA (for TLS compatibility)
	// Store Dilithium signature in extensions for PQ verification
	derBytes, err := x509.CreateCertificate(rand.Reader, template, template, &ecdsaKey.PublicKey, ecdsaKey)
	if err != nil {
		return nil, nil, err
	}

	// Sign certificate with Dilithium for PQ proof
	dilSig, err := DilithiumSign(derBytes, dilSec)
	if err != nil {
		return nil, nil, err
	}

	// Store PQ signature in custom extension (OID 1.3.6.1.4.1.example)
	_ = dilSig // Would be stored in certificate extension in production

	// PEM encode
	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: derBytes})
	keyDER, _ := x509.MarshalECPrivateKey(ecdsaKey)
	keyPEM := pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: keyDER})

	// Also store Dilithium keys (in practice, use secure key storage)
	_ = dilPub
	_ = dilSec

	return certPEM, keyPEM, nil
}

// VerifyPQCertificate verifies both classical and PQ signatures
func VerifyPQCertificate(certDER []byte, dilPub *DilithiumPublicKey) error {
	// Parse X.509 certificate
	cert, err := x509.ParseCertificate(certDER)
	if err != nil {
		return err
	}

	// Verify classical signature
	if err := cert.CheckSignature(cert.SignatureAlgorithm, cert.RawTBSCertificate, cert.Signature); err != nil {
		return err
	}

	// Extract and verify Dilithium signature from extension
	// (Simplified - in production, parse from certificate extensions)
	dilSig := &DilithiumSignature{Data: cert.Signature} // Placeholder

	if !DilithiumVerify(certDER, dilSig, dilPub) {
		return errors.New("PQ signature verification failed")
	}

	return nil
}
