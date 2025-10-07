package wch

import (
    "crypto/aes"
    "crypto/cipher"
    "crypto/ecdh"
    "crypto/rand"
    "crypto/sha256"
    "encoding/base64"
    "encoding/json"
    "errors"
    "io"
    "fmt"
    xhkdf "golang.org/x/crypto/hkdf"
)

const (
    Protocol = "WCHv1"
)

var b64 = base64.StdEncoding

// Envelope is a sealed message carried via Ingress without decryption.
type Envelope struct {
    ChannelID        string `json:"channelId"`
    EphemeralPubB64  string `json:"ephemeralPubKey"`
    NonceB64         string `json:"nonce"`
    CiphertextB64    string `json:"ciphertext"`
    RekeyCounter     int    `json:"rekeyCounter,omitempty"`
}

// ConnectResponse contains channel info sent to the client by Ingress.
type ConnectResponse struct {
    ChannelID      string `json:"channelId"`
    GuardianPubB64 string `json:"guardianPubKey"`
    Protocol       string `json:"protocol"`
    ExpiresAt      int64  `json:"expiresAt"`
    RebindHintMs   int    `json:"rebindHintMs,omitempty"`
}

// InnerRequest is the plaintext structure the client seals.
type InnerRequest struct {
    Method  string            `json:"method"`
    Path    string            `json:"path"`
    Headers map[string]string `json:"headers,omitempty"`
    Body    []byte            `json:"body,omitempty"`
}

// InnerResponse is the plaintext response the guardian returns.
type InnerResponse struct {
    Status  int               `json:"status"`
    Headers map[string]string `json:"headers,omitempty"`
    Body    []byte            `json:"body,omitempty"`
}

// InnerUDPRequest represents a single UDP exchange via MASQUE/QUIC relay.
type InnerUDPRequest struct {
    Target   string `json:"target"` // host:port
    Data     []byte `json:"data"`
    TimeoutMs int   `json:"timeoutMs,omitempty"`
}

// UDPResponse is a single UDP response payload
type UDPResponse struct {
    Data []byte `json:"data"`
}

// GenerateClientEphemeral creates an ephemeral X25519 keypair for client use.
func GenerateClientEphemeral() (*ecdh.PrivateKey, []byte, error) {
    curve := ecdh.X25519()
    priv, err := curve.GenerateKey(rand.Reader)
    if err != nil {
        return nil, nil, err
    }
    return priv, priv.PublicKey().Bytes(), nil
}

// DeriveKey derives an AES-256 key via HKDF-SHA256 from shared secret and channelId as salt/info.
func DeriveKey(sharedSecret []byte, channelID string) ([]byte, error) {
    salt := []byte("wch:" + channelID)
    hk := xhkdf.New(sha256.New, sharedSecret, salt, []byte(Protocol))
    key := make([]byte, 32)
    if _, err := io.ReadFull(hk, key); err != nil {
        return nil, err
    }
    return key, nil
}

// DeriveKeyWithCounter derives an AES-256 key with an additional counter for mid-session rekey.
func DeriveKeyWithCounter(sharedSecret []byte, channelID string, counter int) ([]byte, error) {
    if counter <= 0 { return DeriveKey(sharedSecret, channelID) }
    salt := []byte(fmt.Sprintf("wch:%s:%d", channelID, counter))
    hk := xhkdf.New(sha256.New, sharedSecret, salt, []byte(Protocol))
    key := make([]byte, 32)
    if _, err := io.ReadFull(hk, key); err != nil { return nil, err }
    return key, nil
}

// Seal encrypts plaintext using AES-GCM with a random 12-byte nonce.
func Seal(key []byte, plaintext []byte) (nonce []byte, ciphertext []byte, err error) {
    block, err := aes.NewCipher(key)
    if err != nil { return nil, nil, err }
    aead, err := cipher.NewGCM(block)
    if err != nil { return nil, nil, err }
    nonce = make([]byte, aead.NonceSize())
    if _, err := rand.Read(nonce); err != nil { return nil, nil, err }
    ciphertext = aead.Seal(nil, nonce, plaintext, nil)
    return nonce, ciphertext, nil
}

// Open decrypts ciphertext using AES-GCM.
func Open(key []byte, nonce []byte, ciphertext []byte) ([]byte, error) {
    block, err := aes.NewCipher(key)
    if err != nil { return nil, err }
    aead, err := cipher.NewGCM(block)
    if err != nil { return nil, err }
    if len(nonce) != aead.NonceSize() { return nil, errors.New("invalid nonce length") }
    return aead.Open(nil, nonce, ciphertext, nil)
}

// MarshalB64 encodes bytes to std base64.
func MarshalB64(b []byte) string { return b64.EncodeToString(b) }

// UnmarshalB64 decodes std base64.
func UnmarshalB64(s string) ([]byte, error) { return b64.DecodeString(s) }

// JSON helpers
func ToJSON(v any) []byte {
    b, _ := json.Marshal(v)
    return b
}



