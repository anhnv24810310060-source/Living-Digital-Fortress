//go:build !linux

package wgctrlmgr

import (
	"time"
)

type Manager interface {
	EnsureDevice(iface string, privKeyB64 string, listenPort int) error
	AddPeer(iface string, clientPubKeyB64 string, allowedCIDR string, endpoint string, persistentKeepalive int, ttl time.Duration) error
	RemovePeer(iface string, clientPubKeyB64 string) error
	DevicePeersCount(iface string) (int, error)
}

type manager struct{}

func New() (Manager, error) { return &manager{}, nil }

func (m *manager) EnsureDevice(iface string, privKeyB64 string, listenPort int) error { return nil }
func (m *manager) AddPeer(iface string, clientPubKeyB64 string, allowedCIDR string, endpoint string, persistentKeepalive int, ttl time.Duration) error {
	return nil
}
func (m *manager) RemovePeer(iface string, clientPubKeyB64 string) error { return nil }
func (m *manager) DevicePeersCount(iface string) (int, error)            { return 0, nil }
