//go:build linux

package wgctrlmgr

import (
    "encoding/base64"
    "errors"
    "fmt"
    "net"
    "sync"
    "time"

    "golang.zx2c4.com/wireguard/wgctrl"
    "golang.zx2c4.com/wireguard/wgctrl/wgtypes"
)

type Manager interface {
    EnsureDevice(iface string, privKeyB64 string, listenPort int) error
    AddPeer(iface string, clientPubKeyB64 string, allowedCIDR string, endpoint string, persistentKeepalive int, ttl time.Duration) error
    RemovePeer(iface string, clientPubKeyB64 string) error
    DevicePeersCount(iface string) (int, error)
    ListPeers(iface string) ([]PeerInfo, error)
}

type manager struct {
    client *wgctrl.Client
    mu     sync.Mutex
    timers map[string]*time.Timer
}

type PeerInfo struct {
    PubKeyB64      string
    Endpoint       string
    LastHandshake  time.Time
    AllowedIPs     []string
    RxBytes        int64
    TxBytes        int64
}

func New() (Manager, error) {
    c, err := wgctrl.New()
    if err != nil { return nil, err }
    return &manager{client: c, timers: map[string]*time.Timer{}}, nil
}

func (m *manager) EnsureDevice(iface string, privKeyB64 string, listenPort int) error {
    if iface == "" { iface = "wg-shieldx" }
    var cfg wgtypes.Config
    if privKeyB64 != "" {
        pkRaw, err := base64.StdEncoding.DecodeString(privKeyB64)
        if err != nil { return err }
        if len(pkRaw) != 32 { return errors.New("invalid wg private key length") }
        pk, err := wgtypes.NewKey(pkRaw)
        if err != nil { return err }
        cfg.PrivateKey = &pk
    }
    if listenPort > 0 { cfg.ListenPort = &listenPort }
    return m.client.ConfigureDevice(iface, cfg)
}

func (m *manager) AddPeer(iface string, clientPubKeyB64 string, allowedCIDR string, endpoint string, persistentKeepalive int, ttl time.Duration) error {
    pubRaw, err := base64.StdEncoding.DecodeString(clientPubKeyB64)
    if err != nil { return err }
    if len(pubRaw) != 32 { return errors.New("invalid wg public key length") }
    pub, err := wgtypes.NewKey(pubRaw)
    if err != nil { return err }
    // Parse allowed IPs
    _, ipnet, err := net.ParseCIDR(allowedCIDR)
    if err != nil { return err }
    allowed := []net.IPNet{*ipnet}
    // Endpoint
    var ep *net.UDPAddr
    if endpoint != "" {
        ep, err = net.ResolveUDPAddr("udp", endpoint)
        if err != nil { return err }
    }
    ka := persistentKeepalive
    peer := wgtypes.PeerConfig{
        PublicKey:                   pub,
        ReplaceAllowedIPs:          true,
        AllowedIPs:                 allowed,
        PersistentKeepaliveInterval: func() *time.Duration { if ka<=0 { return nil }; d:=time.Duration(ka)*time.Second; return &d }(),
        Endpoint:                   ep,
    }
    cfg := wgtypes.Config{Peers: []wgtypes.PeerConfig{peer}}
    if err := m.client.ConfigureDevice(iface, cfg); err != nil { return fmt.Errorf("configure device: %w", err) }
    if ttl > 0 {
        key := iface+"|"+clientPubKeyB64
        m.mu.Lock()
        if t, ok := m.timers[key]; ok { t.Stop() }
        m.timers[key] = time.AfterFunc(ttl, func() {
            _ = m.client.ConfigureDevice(iface, wgtypes.Config{Peers: []wgtypes.PeerConfig{{PublicKey: pub, Remove: true}}})
            m.mu.Lock(); delete(m.timers, key); m.mu.Unlock()
        })
        m.mu.Unlock()
    }
    return nil
}

func (m *manager) RemovePeer(iface string, clientPubKeyB64 string) error {
    pubRaw, err := base64.StdEncoding.DecodeString(clientPubKeyB64)
    if err != nil { return err }
    if len(pubRaw) != 32 { return errors.New("invalid wg public key length") }
    pub, err := wgtypes.NewKey(pubRaw)
    if err != nil { return err }
    key := iface+"|"+clientPubKeyB64
    m.mu.Lock(); if t, ok := m.timers[key]; ok { t.Stop(); delete(m.timers, key) }; m.mu.Unlock()
    return m.client.ConfigureDevice(iface, wgtypes.Config{Peers: []wgtypes.PeerConfig{{PublicKey: pub, Remove: true}}})
}

func (m *manager) DevicePeersCount(iface string) (int, error) {
    dev, err := m.client.Device(iface)
    if err != nil { return 0, err }
    return len(dev.Peers), nil
}

func (m *manager) ListPeers(iface string) ([]PeerInfo, error) {
    dev, err := m.client.Device(iface)
    if err != nil { return nil, err }
    out := make([]PeerInfo, 0, len(dev.Peers))
    for _, p := range dev.Peers {
        pi := PeerInfo{PubKeyB64: base64.StdEncoding.EncodeToString(p.PublicKey[:]), LastHandshake: p.LastHandshakeTime}
        if p.Endpoint != nil { pi.Endpoint = p.Endpoint.String() }
        for _, ipn := range p.AllowedIPs { pi.AllowedIPs = append(pi.AllowedIPs, ipn.String()) }
        pi.RxBytes = int64(p.ReceiveBytes)
        pi.TxBytes = int64(p.TransmitBytes)
        out = append(out, pi)
    }
    return out, nil
}


