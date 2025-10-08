package wgcfg

import (
	"crypto/rand"
	"encoding/base64"
	"fmt"
	"golang.org/x/crypto/curve25519"
)

// GenerateKeypair returns base64-encoded private and public keys using X25519 (WireGuard-compatible).
func GenerateKeypair() (privB64, pubB64 string, err error) {
	var priv [32]byte
	if _, err = rand.Read(priv[:]); err != nil {
		return "", "", err
	}
	clampPrivateKey(&priv)
	var pub [32]byte
	curve25519.ScalarBaseMult(&pub, &priv)
	return base64.StdEncoding.EncodeToString(priv[:]), base64.StdEncoding.EncodeToString(pub[:]), nil
}

// clampPrivateKey clamps a Curve25519 private key as per spec.
func clampPrivateKey(k *[32]byte) {
	k[0] &= 248
	k[31] &= 127
	k[31] |= 64
}

// ClientConfig returns a minimal client config using given keys and endpoints.
func ClientConfig(privKeyB64, clientAddressCIDR, serverPubKeyB64, serverEndpoint string, allowedIPs string, listenPort int) string {
	return fmt.Sprintf(`[Interface]
PrivateKey = %s
Address = %s
ListenPort = %d

[Peer]
PublicKey = %s
AllowedIPs = %s
Endpoint = %s
PersistentKeepalive = 25
`, privKeyB64, clientAddressCIDR, listenPort, serverPubKeyB64, allowedIPs, serverEndpoint)
}

// ServerPeerSnippet returns a peer snippet for the server config.
func ServerPeerSnippet(clientPubKeyB64 string) string {
	return fmt.Sprintf(`[Peer]
PublicKey = %s
AllowedIPs = 10.10.0.2/32
PersistentKeepalive = 25
`, clientPubKeyB64)
}

// end of file
