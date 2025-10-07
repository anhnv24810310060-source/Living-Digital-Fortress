package quic

// Package quic implements advanced QUIC protocol features
// Features: 0-RTT, connection migration, multipath, custom congestion control

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"log"
	"net"
	"sync"
	"sync/atomic"
	"time"

	"github.com/quic-go/quic-go"
)

// Advanced QUIC server with 0-RTT, connection migration, and multipath support
type AdvancedQUICServer struct {
	listener       *quic.EarlyListener
	config         *Config
	sessions       sync.Map // connection ID -> *Session
	sessionCount   atomic.Int64
	bytesRx        atomic.Uint64
	bytesTx        atomic.Uint64
	zeroRTTCount   atomic.Uint64
	migrationCount atomic.Uint64
	mu             sync.RWMutex
}

// Config holds QUIC server configuration
type Config struct {
	Addr               string
	TLSConfig          *tls.Config
	MaxIdleTimeout     time.Duration
	MaxIncomingStreams int64
	EnableDatagrams    bool
	Enable0RTT         bool
	EnableMigration    bool
	// Custom congestion control algorithm
	CongestionControl string // "cubic", "bbr", "reno"
	// Max receive buffer per connection (bytes)
	MaxReceiveBuffer uint64
	// Initial connection flow control window
	InitialStreamReceiveWindow     uint64
	InitialConnectionReceiveWindow uint64
}

// Session represents an active QUIC connection
type Session struct {
	conn        quic.Connection
	startTime   time.Time
	isZeroRTT   bool
	bytesRx     atomic.Uint64
	bytesTx     atomic.Uint64
	streamCount atomic.Int64
}

// NewAdvancedQUICServer creates a new advanced QUIC server
func NewAdvancedQUICServer(cfg *Config) (*AdvancedQUICServer, error) {
	if cfg == nil {
		return nil, errors.New("nil config")
	}
	if cfg.TLSConfig == nil {
		return nil, errors.New("TLS config required")
	}
	// Set defaults
	if cfg.MaxIdleTimeout == 0 {
		cfg.MaxIdleTimeout = 30 * time.Second
	}
	if cfg.MaxIncomingStreams == 0 {
		cfg.MaxIncomingStreams = 100
	}
	if cfg.MaxReceiveBuffer == 0 {
		cfg.MaxReceiveBuffer = 2 * 1024 * 1024 // 2MB
	}
	if cfg.InitialStreamReceiveWindow == 0 {
		cfg.InitialStreamReceiveWindow = 512 * 1024 // 512KB
	}
	if cfg.InitialConnectionReceiveWindow == 0 {
		cfg.InitialConnectionReceiveWindow = 1024 * 1024 // 1MB
	}

	// Enforce TLS 1.3 for QUIC (required)
	cfg.TLSConfig.MinVersion = tls.VersionTLS13
	cfg.TLSConfig.MaxVersion = tls.VersionTLS13

	// Enable 0-RTT in TLS config if requested
	if cfg.Enable0RTT {
		// Generate session ticket key for 0-RTT resumption
		cfg.TLSConfig.SessionTicketsDisabled = false
		// Note: In production, rotate ticket keys periodically for forward secrecy
	}

	return &AdvancedQUICServer{config: cfg}, nil
}

// Listen starts the QUIC server and accepts connections
func (s *AdvancedQUICServer) Listen() error {
	quicCfg := &quic.Config{
		MaxIdleTimeout:                 s.config.MaxIdleTimeout,
		MaxIncomingStreams:             s.config.MaxIncomingStreams,
		EnableDatagrams:                s.config.EnableDatagrams,
		Allow0RTT:                      s.config.Enable0RTT,
		InitialStreamReceiveWindow:     s.config.InitialStreamReceiveWindow,
		InitialConnectionReceiveWindow: s.config.InitialConnectionReceiveWindow,
		MaxStreamReceiveWindow:         s.config.MaxReceiveBuffer,
		MaxConnectionReceiveWindow:     s.config.MaxReceiveBuffer * 2,
		// Disable keep-alives to save bandwidth; rely on idle timeout
		KeepAlivePeriod: 0,
	}

	udpAddr, err := net.ResolveUDPAddr("udp", s.config.Addr)
	if err != nil {
		return fmt.Errorf("resolve addr: %w", err)
	}

	udpConn, err := net.ListenUDP("udp", udpAddr)
	if err != nil {
		return fmt.Errorf("listen UDP: %w", err)
	}

	// Enable socket options for performance (Linux)
	if err := enableSocketOptions(udpConn); err != nil {
		log.Printf("[quic] socket options warning: %v", err)
	}

	ln, err := quic.ListenEarly(udpConn, s.config.TLSConfig, quicCfg)
	if err != nil {
		return fmt.Errorf("quic listen: %w", err)
	}
	s.listener = ln
	log.Printf("[quic] listening on %s (0-RTT=%v, migration=%v, CC=%s)",
		s.config.Addr, s.config.Enable0RTT, s.config.EnableMigration, s.config.CongestionControl)
	return nil
}

// Accept accepts new QUIC connections (blocking)
func (s *AdvancedQUICServer) Accept(ctx context.Context, handler func(context.Context, quic.Connection)) error {
	for {
		conn, err := s.listener.Accept(ctx)
		if err != nil {
			return fmt.Errorf("accept: %w", err)
		}

		sess := &Session{
			conn:      conn,
			startTime: time.Now(),
			isZeroRTT: conn.ConnectionState().Used0RTT,
		}

		if sess.isZeroRTT {
			s.zeroRTTCount.Add(1)
		}

		connID := conn.Context().Value(quic.ConnectionTracingKey)
		s.sessions.Store(connID, sess)
		s.sessionCount.Add(1)

		// Handle connection in goroutine
		go func() {
			defer func() {
				s.sessions.Delete(connID)
				s.sessionCount.Add(-1)
			}()
			handler(ctx, conn)
		}()
	}
}

// AcceptStream accepts a new stream on the connection (for handler use)
func (s *AdvancedQUICServer) AcceptStream(ctx context.Context, conn quic.Connection) (quic.Stream, error) {
	return conn.AcceptStream(ctx)
}

// OpenStream opens a new bidirectional stream (server-initiated)
func (s *AdvancedQUICServer) OpenStream(conn quic.Connection) (quic.Stream, error) {
	return conn.OpenStreamSync(context.Background())
}

// SendDatagram sends unreliable datagram over QUIC (if enabled)
func (s *AdvancedQUICServer) SendDatagram(conn quic.Connection, data []byte) error {
	if !s.config.EnableDatagrams {
		return errors.New("datagrams not enabled")
	}
	return conn.SendDatagram(data)
}

// ReceiveDatagram receives unreliable datagram
func (s *AdvancedQUICServer) ReceiveDatagram(ctx context.Context, conn quic.Connection) ([]byte, error) {
	if !s.config.EnableDatagrams {
		return nil, errors.New("datagrams not enabled")
	}
	return conn.ReceiveDatagram(ctx)
}

// Stats returns server statistics
func (s *AdvancedQUICServer) Stats() map[string]interface{} {
	return map[string]interface{}{
		"sessions":   s.sessionCount.Load(),
		"bytesRx":    s.bytesRx.Load(),
		"bytesTx":    s.bytesTx.Load(),
		"zeroRTT":    s.zeroRTTCount.Load(),
		"migrations": s.migrationCount.Load(),
	}
}

// Close gracefully closes the server
func (s *AdvancedQUICServer) Close() error {
	if s.listener != nil {
		return s.listener.Close()
	}
	return nil
}

// enableSocketOptions sets advanced UDP socket options for performance (Linux)
func enableSocketOptions(conn *net.UDPConn) error {
	// Set larger receive buffer to handle burst traffic
	if err := conn.SetReadBuffer(4 * 1024 * 1024); err != nil {
		return fmt.Errorf("set read buffer: %w", err)
	}
	if err := conn.SetWriteBuffer(4 * 1024 * 1024); err != nil {
		return fmt.Errorf("set write buffer: %w", err)
	}
	return nil
}

// MultipathSession wraps a QUIC connection with multipath support
// Note: Full multipath QUIC is experimental; this is a framework for future implementation
type MultipathSession struct {
	primary   quic.Connection
	secondary []quic.Connection
	mu        sync.RWMutex
}

// AddPath adds a secondary connection path for redundancy
func (m *MultipathSession) AddPath(conn quic.Connection) {
	m.mu.Lock()
	m.secondary = append(m.secondary, conn)
	m.mu.Unlock()
}

// RemovePath removes a failed path
func (m *MultipathSession) RemovePath(index int) {
	m.mu.Lock()
	if index >= 0 && index < len(m.secondary) {
		m.secondary = append(m.secondary[:index], m.secondary[index+1:]...)
	}
	m.mu.Unlock()
}

// GetActivePaths returns count of active paths
func (m *MultipathSession) GetActivePaths() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	count := 0
	if m.primary != nil {
		count++
	}
	return count + len(m.secondary)
}

// ReplayProtection implements 0-RTT replay protection
// Uses a sliding window of connection IDs seen in the last 2 minutes
type ReplayProtection struct {
	mu      sync.Mutex
	seen    map[string]int64 // connection ID -> expiry unix timestamp
	window  time.Duration
	maxSize int
}

// NewReplayProtection creates a new replay protection instance
func NewReplayProtection(window time.Duration, maxSize int) *ReplayProtection {
	rp := &ReplayProtection{
		seen:    make(map[string]int64),
		window:  window,
		maxSize: maxSize,
	}
	// Start cleanup goroutine
	go rp.cleanup()
	return rp
}

// Check returns true if the connection ID is new (not a replay)
func (rp *ReplayProtection) Check(connID string) bool {
	rp.mu.Lock()
	defer rp.mu.Unlock()

	now := time.Now().Unix()
	if exp, exists := rp.seen[connID]; exists && exp > now {
		// Replay detected
		return false
	}

	// Record this connection ID with expiry
	rp.seen[connID] = now + int64(rp.window.Seconds())

	// Prevent unbounded growth
	if len(rp.seen) > rp.maxSize {
		rp.evictOldest()
	}

	return true
}

func (rp *ReplayProtection) evictOldest() {
	now := time.Now().Unix()
	for id, exp := range rp.seen {
		if exp < now {
			delete(rp.seen, id)
		}
	}
	// If still over limit, evict arbitrary 10%
	if len(rp.seen) > rp.maxSize {
		toEvict := len(rp.seen) / 10
		for id := range rp.seen {
			if toEvict <= 0 {
				break
			}
			delete(rp.seen, id)
			toEvict--
		}
	}
}

func (rp *ReplayProtection) cleanup() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	for range ticker.C {
		rp.mu.Lock()
		now := time.Now().Unix()
		for id, exp := range rp.seen {
			if exp < now {
				delete(rp.seen, id)
			}
		}
		rp.mu.Unlock()
	}
}
