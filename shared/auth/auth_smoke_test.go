package auth

import "testing"

func TestJWTManagerMinimal(t *testing.T) {
    priv, pub, err := GenerateTestKeyPair()
    if err != nil { t.Fatalf("gen keys: %v", err) }
    jm, err := NewJWTManager(JWTConfig{PrivateKeyPEM: priv, PublicKeyPEM: pub})
    if err != nil { t.Fatalf("new jwt manager: %v", err) }
    pair, err := jm.GenerateTokenPair(t.Context(), "u1", "tenant1", "u1@example", []string{"user"}, nil)
    if err != nil { t.Fatalf("gen pair: %v", err) }
    if pair.AccessToken == "" { t.Fatalf("empty access token") }
}
