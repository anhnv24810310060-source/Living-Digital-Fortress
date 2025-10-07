package wgstate

import (
    "encoding/json"
    "errors"
    "os"
    "path/filepath"
    "sync"
)

type Peer struct {
    Tenant     string `json:"tenant"`
    Scope      string `json:"scope"`
    ClientPub  string `json:"client_pub"`
    AssignedIP string `json:"assigned_ip"`
}

type State struct {
    ServerPrivB64 string         `json:"server_priv_b64"`
    ServerPubB64  string         `json:"server_pub_b64"`
    PublicEndpoint string        `json:"public_endpoint,omitempty"`
    PublicHealthURL string       `json:"public_health_url,omitempty"`
    AltServerPrivB64 string      `json:"alt_server_priv_b64,omitempty"`
    AltServerPubB64  string      `json:"alt_server_pub_b64,omitempty"`
    AltListenPort    int         `json:"alt_listen_port,omitempty"`
    Peers         map[string]Peer `json:"peers"` // key: client pub
    LastIPOctet   int            `json:"last_ip_octet"`
}

type Store struct {
    mu   sync.Mutex
    path string
    s    State
}

func Open(path string) (*Store, error) {
    st := &Store{path: path}
    if path == "" { return nil, errors.New("empty path") }
    if b, err := os.ReadFile(filepath.Clean(path)); err == nil {
        _ = json.Unmarshal(b, &st.s)
    }
    if st.s.Peers == nil { st.s.Peers = map[string]Peer{} }
    return st, nil
}

func (st *Store) Save() error {
    st.mu.Lock(); defer st.mu.Unlock()
    _ = os.MkdirAll(filepath.Dir(st.path), 0o755)
    b, _ := json.MarshalIndent(st.s, "", "  ")
    tmp := st.path + ".tmp"
    if err := os.WriteFile(tmp, b, 0o600); err != nil { return err }
    return os.Rename(tmp, st.path)
}

func (st *Store) Get() State { st.mu.Lock(); defer st.mu.Unlock(); return st.s }

func (st *Store) SetServer(priv, pub string) { st.mu.Lock(); st.s.ServerPrivB64, st.s.ServerPubB64 = priv, pub; st.mu.Unlock() }
func (st *Store) SetEndpoint(endpoint string) { st.mu.Lock(); st.s.PublicEndpoint = endpoint; st.mu.Unlock() }
func (st *Store) SetHealthURL(url string) { st.mu.Lock(); st.s.PublicHealthURL = url; st.mu.Unlock() }
func (st *Store) SetAltServer(priv, pub string, port int) { st.mu.Lock(); st.s.AltServerPrivB64, st.s.AltServerPubB64, st.s.AltListenPort = priv, pub, port; st.mu.Unlock() }
func (st *Store) ClearAltServer() { st.mu.Lock(); st.s.AltServerPrivB64, st.s.AltServerPubB64, st.s.AltListenPort = "", "", 0; st.mu.Unlock() }

func (st *Store) AddPeer(p Peer) { st.mu.Lock(); st.s.Peers[p.ClientPub] = p; st.mu.Unlock() }

func (st *Store) RemovePeer(clientPub string) { st.mu.Lock(); delete(st.s.Peers, clientPub); st.mu.Unlock() }

// AllocateIP assigns the next IP in 10.10.0.X/32 for a peer if not present. Returns assigned IP.
func (st *Store) AllocateIP(clientPub string) string {
    st.mu.Lock(); defer st.mu.Unlock()
    if p, ok := st.s.Peers[clientPub]; ok && p.AssignedIP != "" { return p.AssignedIP }
    st.s.LastIPOctet++
    if st.s.LastIPOctet < 2 { st.s.LastIPOctet = 2 }
    if st.s.LastIPOctet > 254 { st.s.LastIPOctet = 2 }
    ip := "10.10.0." + itoa(st.s.LastIPOctet) + "/32"
    if p, ok := st.s.Peers[clientPub]; ok { p.AssignedIP = ip; st.s.Peers[clientPub] = p }
    return ip
}

func itoa(n int) string { return fmtInt(n) }

func fmtInt(n int) string {
    // avoid importing strconv for this tiny use
    if n == 0 { return "0" }
    neg := false
    if n < 0 { neg = true; n = -n }
    buf := make([]byte, 0, 3)
    for n > 0 { buf = append([]byte{'0' + byte(n%10)}, buf...); n/=10 }
    if neg { buf = append([]byte{'-'}, buf...) }
    return string(buf)
}


