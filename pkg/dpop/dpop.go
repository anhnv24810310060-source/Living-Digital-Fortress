package dpop

import (
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"
)

var b64url = base64.URLEncoding.WithPadding(base64.NoPadding)

// VerifyEdDSA verifies a DPoP JWT (EdDSA/Ed25519) and returns (jkt, jti, iat) if valid.
// Minimal fields checked: header.jwk (OKP/Ed25519), payload.htu, payload.htm, payload.jti, payload.iat
func VerifyEdDSA(dpopJWT string, method string, htu string, now time.Time, skewSec int64) (string, string, int64, error) {
	parts := strings.Split(dpopJWT, ".")
	if len(parts) != 3 {
		return "", "", 0, errors.New("invalid dpop format")
	}
	headerJSON, err := b64url.DecodeString(parts[0])
	if err != nil {
		return "", "", 0, err
	}
	payloadJSON, err := b64url.DecodeString(parts[1])
	if err != nil {
		return "", "", 0, err
	}
	sig, err := b64url.DecodeString(parts[2])
	if err != nil {
		return "", "", 0, err
	}
	var hdr map[string]any
	if err := json.Unmarshal(headerJSON, &hdr); err != nil {
		return "", "", 0, err
	}
	if alg, _ := hdr["alg"].(string); alg != "EdDSA" {
		return "", "", 0, errors.New("unsupported alg")
	}
	jwk, ok := hdr["jwk"].(map[string]any)
	if !ok {
		return "", "", 0, errors.New("missing jwk")
	}
	if kty, _ := jwk["kty"].(string); kty != "OKP" {
		return "", "", 0, errors.New("unsupported kty")
	}
	if crv, _ := jwk["crv"].(string); crv != "Ed25519" {
		return "", "", 0, errors.New("unsupported crv")
	}
	xstr, _ := jwk["x"].(string)
	x, err := b64url.DecodeString(xstr)
	if err != nil {
		return "", "", 0, fmt.Errorf("jwk x: %w", err)
	}
	if len(x) != ed25519.PublicKeySize {
		return "", "", 0, errors.New("bad jwk x size")
	}
	pub := ed25519.PublicKey(x)
	// Compute JKT per RFC7638 canonical members for OKP
	can := fmt.Sprintf("{\"crv\":\"Ed25519\",\"kty\":\"OKP\",\"x\":\"%s\"}", xstr)
	sum := sha256.Sum256([]byte(can))
	jkt := b64url.EncodeToString(sum[:])
	var pl map[string]any
	if err := json.Unmarshal(payloadJSON, &pl); err != nil {
		return "", "", 0, err
	}
	htm, _ := pl["htm"].(string)
	htuPl, _ := pl["htu"].(string)
	jti, _ := pl["jti"].(string)
	iatFloat, _ := pl["iat"].(float64)
	iat := int64(iatFloat)
	if strings.ToUpper(htm) != strings.ToUpper(method) {
		return "", "", 0, errors.New("htm mismatch")
	}
	if htuPl != htu {
		return "", "", 0, errors.New("htu mismatch")
	}
	if jti == "" {
		return "", "", 0, errors.New("missing jti")
	}
	// iat skew check
	nowSec := now.Unix()
	if iat > nowSec+skewSec || iat < nowSec-skewSec {
		return "", "", 0, errors.New("iat out of range")
	}
	// Verify signature on header.payload
	signing := []byte(parts[0] + "." + parts[1])
	if !ed25519.Verify(pub, signing, sig) {
		return "", "", 0, errors.New("dpop signature invalid")
	}
	return jkt, jti, iat, nil
}
