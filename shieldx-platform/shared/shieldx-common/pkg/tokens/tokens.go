package tokens

import (
    "crypto/ed25519"
    "crypto/rand"
    "encoding/base64"
    "encoding/json"
    "errors"
    "fmt"
    "strings"
    "time"
)

const (
    // TokenPrefix identifies the token scheme/version used for locator tokens
    TokenPrefix = "LKT1"
)

var (
    b64 = base64.URLEncoding.WithPadding(base64.NoPadding)
)

// LocatorClaims represents the minimal claims for a locator token.
// All time fields are expressed as Unix seconds.
type LocatorClaims struct {
    Tenant   string `json:"tenant"`
    Scope    string `json:"scope"`
    Audience string `json:"audience,omitempty"`
    IssuedAt int64  `json:"iat"`
    Expires  int64  `json:"exp"`
    Nonce    string `json:"nonce"`
    DPoPJKT  string `json:"dpop_jkt,omitempty"`
}

// GenerateEd25519Keypair returns (publicKey, privateKey).
func GenerateEd25519Keypair() (ed25519.PublicKey, ed25519.PrivateKey, error) {
    pub, priv, err := ed25519.GenerateKey(rand.Reader)
    if err != nil {
        return nil, nil, err
    }
    return pub, priv, nil
}

// RandomNonce returns a URL-safe base64 string of n random bytes.
func RandomNonce(n int) (string, error) {
    if n <= 0 {
        n = 16
    }
    buf := make([]byte, n)
    if _, err := rand.Read(buf); err != nil {
        return "", err
    }
    return b64.EncodeToString(buf), nil
}

// EncodeToken signs the claims and returns a compact token string.
// Format: LKT1.<base64url(payloadJSON)>.{base64url(signature)}
func EncodeToken(claims LocatorClaims, priv ed25519.PrivateKey) (string, error) {
    if priv == nil || len(priv) == 0 {
        return "", errors.New("empty private key")
    }
    payload, err := json.Marshal(claims)
    if err != nil {
        return "", err
    }
    payloadEnc := b64.EncodeToString(payload)
    sig := ed25519.Sign(priv, payload)
    sigEnc := b64.EncodeToString(sig)
    return fmt.Sprintf("%s.%s.%s", TokenPrefix, payloadEnc, sigEnc), nil
}

// DecodeAndVerify parses and verifies a token using the given public key.
// It returns the claims if valid and not expired at the time of call.
func DecodeAndVerify(token string, pub ed25519.PublicKey) (LocatorClaims, error) {
    var empty LocatorClaims
    if token == "" {
        return empty, errors.New("empty token")
    }
    parts := strings.Split(token, ".")
    if len(parts) != 3 {
        return empty, errors.New("invalid token format")
    }
    if parts[0] != TokenPrefix {
        return empty, errors.New("unsupported token version")
    }
    payloadBytes, err := b64.DecodeString(parts[1])
    if err != nil {
        return empty, fmt.Errorf("payload decode: %w", err)
    }
    sigBytes, err := b64.DecodeString(parts[2])
    if err != nil {
        return empty, fmt.Errorf("signature decode: %w", err)
    }
    if !ed25519.Verify(pub, payloadBytes, sigBytes) {
        return empty, errors.New("signature verification failed")
    }
    var claims LocatorClaims
    if err := json.Unmarshal(payloadBytes, &claims); err != nil {
        return empty, fmt.Errorf("claims unmarshal: %w", err)
    }
    now := time.Now().Unix()
    if claims.Expires <= now {
        return empty, errors.New("token expired")
    }
    return claims, nil
}

// PublicKeyB64 encodes an Ed25519 public key as standard base64.
func PublicKeyB64(pub ed25519.PublicKey) string {
    return base64.StdEncoding.EncodeToString(pub)
}

// PrivateKeyFromB64 decodes a base64-encoded Ed25519 private key.
func PrivateKeyFromB64(b64str string) (ed25519.PrivateKey, error) {
    raw, err := base64.StdEncoding.DecodeString(strings.TrimSpace(b64str))
    if err != nil {
        return nil, err
    }
    if l := len(raw); l != ed25519.PrivateKeySize {
        return nil, fmt.Errorf("invalid ed25519 private key length: %d", l)
    }
    return ed25519.PrivateKey(raw), nil
}



