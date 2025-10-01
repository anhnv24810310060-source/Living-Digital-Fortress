package auth
package auth

import (
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/pem"
	"os"
	"testing"
)

// GenerateTestKeyPair generates RSA key pair for testing
func GenerateTestKeyPair() (privateKeyPEM, publicKeyPEM string, err error) {
	privateKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		return "", "", err
	}

	privateKeyBytes := x509.MarshalPKCS1PrivateKey(privateKey)
	privateKeyPEM = string(pem.EncodeToMemory(&pem.Block{
		Type:  "RSA PRIVATE KEY",
		Bytes: privateKeyBytes,
	}))

	publicKeyBytes, err := x509.MarshalPKIXPublicKey(&privateKey.PublicKey)
	if err != nil {
		return "", "", err
	}

	publicKeyPEM = string(pem.EncodeToMemory(&pem.Block{
		Type:  "PUBLIC KEY",
		Bytes: publicKeyBytes,
	}))

	return privateKeyPEM, publicKeyPEM, nil
}

// LoadKeysFromEnv loads RSA keys from environment variables
func LoadKeysFromEnv() (privateKeyPEM, publicKeyPEM string, err error) {
	privateKeyPEM = os.Getenv("JWT_PRIVATE_KEY")
	publicKeyPEM = os.Getenv("JWT_PUBLIC_KEY")

	if privateKeyPEM == "" || publicKeyPEM == "" {
		// Generate new keys if not provided
		return GenerateTestKeyPair()
	}

	return privateKeyPEM, publicKeyPEM, nil
}

// TestHelper provides helper functions for testing auth components
type TestHelper struct {
	t              *testing.T
	JWTManager     *JWTManager
	SessionManager *SessionManager
	RBACEngine     *RBACEngine
}

// NewTestHelper creates a test helper with initialized components
func NewTestHelper(t *testing.T) (*TestHelper, error) {
	privateKey, publicKey, err := GenerateTestKeyPair()
	if err != nil {
		return nil, err
	}

	jwtManager, err := NewJWTManager(JWTConfig{
		PrivateKeyPEM:    privateKey,
		PublicKeyPEM:     publicKey,
		Issuer:           "test-issuer",
		RevokedTokenStore: NewInMemoryRevokedStore(),
	})
	if err != nil {
		return nil, err
	}

	rbacEngine, err := NewRBACEngine(RBACConfig{
		Roles: GetDefaultRoles(),
	})
	if err != nil {
		return nil, err
	}

	return &TestHelper{
		t:          t,
		JWTManager: jwtManager,
		RBACEngine: rbacEngine,
	}, nil
}
