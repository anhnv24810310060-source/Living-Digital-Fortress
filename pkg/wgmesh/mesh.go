package wgmesh


import (
	"fmt"
	"os/exec"
	"strings"
)

// MeshConfig represents WireGuard mesh configuration
type MeshConfig struct {
	InterfaceName string
	PrivateKey    string
	Address       string
	ListenPort    int
	Peers         []PeerConfig
}

// PeerConfig represents a peer in the mesh
type PeerConfig struct {
	PublicKey           string
	Endpoint            string
	AllowedIPs          []string
	PersistentKeepalive int
}

// SetupMesh configures WireGuard mesh interface
func SetupMesh(config MeshConfig) error {
	// Check if WireGuard is available
	if _, err := exec.LookPath("wg"); err != nil {
		return fmt.Errorf("wireguard tools not installed: %w", err)
	}

	// Remove existing interface if present
	exec.Command("ip", "link", "del", "dev", config.InterfaceName).Run()

	// Create WireGuard interface
	if err := exec.Command("ip", "link", "add", "dev", config.InterfaceName, "type", "wireguard").Run(); err != nil {
		return fmt.Errorf("failed to create interface: %w", err)
	}

	// Set private key
	cmd := exec.Command("wg", "set", config.InterfaceName, 
		"private-key", "/dev/stdin", 
		"listen-port", fmt.Sprintf("%d", config.ListenPort))
	cmd.Stdin = strings.NewReader(config.PrivateKey)
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to set private key: %w", err)
	}

	// Add peers
	for _, peer := range config.Peers {
		peerArgs := []string{"set", config.InterfaceName, "peer", peer.PublicKey}
		
		if peer.Endpoint != "" {
			peerArgs = append(peerArgs, "endpoint", peer.Endpoint)
		}
		
		if len(peer.AllowedIPs) > 0 {
			peerArgs = append(peerArgs, "allowed-ips", strings.Join(peer.AllowedIPs, ","))
		}
		
		if peer.PersistentKeepalive > 0 {
			peerArgs = append(peerArgs, "persistent-keepalive", fmt.Sprintf("%d", peer.PersistentKeepalive))
		}
		
		if err := exec.Command("wg", peerArgs...).Run(); err != nil {
			return fmt.Errorf("failed to add peer %s: %w", peer.PublicKey[:16], err)
		}
	}

	// Set IP address
	if err := exec.Command("ip", "addr", "add", config.Address, "dev", config.InterfaceName).Run(); err != nil {
		return fmt.Errorf("failed to set address: %w", err)
	}

	// Bring interface up
	if err := exec.Command("ip", "link", "set", config.InterfaceName, "up").Run(); err != nil {
		return fmt.Errorf("failed to bring interface up: %w", err)
	}

	return nil
}

// TeardownMesh removes WireGuard mesh interface
func TeardownMesh(interfaceName string) error {
	if err := exec.Command("ip", "link", "del", "dev", interfaceName).Run(); err != nil {
		return fmt.Errorf("failed to delete interface: %w", err)
	}
	return nil
}

// GetMeshStatus returns current mesh status
func GetMeshStatus(interfaceName string) (string, error) {
	cmd := exec.Command("wg", "show", interfaceName)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("failed to get status: %w", err)
	}
	return string(output), nil
}

// GenerateKeyPair generates a WireGuard key pair
func GenerateKeyPair() (privateKey, publicKey string, err error) {
	// Generate private key
	privCmd := exec.Command("wg", "genkey")
	privOut, err := privCmd.Output()
	if err != nil {
		return "", "", fmt.Errorf("failed to generate private key: %w", err)
	}
	privateKey = strings.TrimSpace(string(privOut))

	// Generate public key from private key
	pubCmd := exec.Command("wg", "pubkey")
	pubCmd.Stdin = strings.NewReader(privateKey)
	pubOut, err := pubCmd.Output()
	if err != nil {
		return "", "", fmt.Errorf("failed to generate public key: %w", err)
	}
	publicKey = strings.TrimSpace(string(pubOut))

	return privateKey, publicKey, nil
}
