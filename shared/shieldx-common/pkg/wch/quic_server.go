package wch

import (
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/quic-go/quic-go"
	"github.com/quic-go/quic-go/http3"
)

// QUICServer represents a production-grade QUIC server for WCH
type QUICServer struct {
	addr        string
	tlsConfig   *tls.Config
	quicConfig  *quic.Config
	server      *http3.Server
	handler     http.Handler
	sessions    *SessionManager
	rateLimiter RateLimiter
	camouflage  *CamouflageEngine
	metrics     *QUICMetrics
	mu          sync.RWMutex
	shutdown    chan struct{}
}

// QUICConfig configuration for QUIC server
type QUICConfig struct {
	Addr                       string
	TLSConfig                  *tls.Config
	MaxIdleTimeout             time.Duration
	MaxIncomingStreams         int64
	MaxIncomingUniStreams      int64
	EnableDatagrams            bool
	MaxStreamReceiveWindow     uint64
	MaxConnectionReceiveWindow uint64
	KeepAlivePeriod            time.Duration
	HandshakeIdleTimeout       time.Duration
	RateLimiter                RateLimiter
	SessionManager             *SessionManager
	CamouflageEngine           *CamouflageEngine
}

// QUICMetrics tracks QUIC server metrics
type QUICMetrics struct {
	mu                 sync.RWMutex
	ConnectionsTotal   int64
	ConnectionsActive  int64
	StreamsTotal       int64
	StreamsActive      int64
	BytesSent          int64
	BytesReceived      int64
	PacketsLost        int64
	RTTHistogram       map[string]int64
	HandshakeFailures  int64
	HandshakeSuccesses int64
	DatagramsReceived  int64
	DatagramsSent      int64
}

// NewQUICServer creates a new production-grade QUIC server
func NewQUICServer(config QUICConfig) (*QUICServer, error) {
	if config.Addr == "" {
		config.Addr = ":443"
	}

	// Configure QUIC parameters for production
	quicConfig := &quic.Config{
		MaxIdleTimeout:        config.MaxIdleTimeout,
		MaxIncomingStreams:    config.MaxIncomingStreams,
		MaxIncomingUniStreams: config.MaxIncomingUniStreams,
		EnableDatagrams:       config.EnableDatagrams,
		KeepAlivePeriod:       config.KeepAlivePeriod,
		HandshakeIdleTimeout:  config.HandshakeIdleTimeout,
	}

	// Set defaults
	if quicConfig.MaxIdleTimeout == 0 {
		quicConfig.MaxIdleTimeout = 30 * time.Second
	}
	if quicConfig.MaxIncomingStreams == 0 {
		quicConfig.MaxIncomingStreams = 100
	}
	if quicConfig.MaxIncomingUniStreams == 0 {
		quicConfig.MaxIncomingUniStreams = 100
	}
	if quicConfig.KeepAlivePeriod == 0 {
		quicConfig.KeepAlivePeriod = 10 * time.Second
	}
	if quicConfig.HandshakeIdleTimeout == 0 {
		quicConfig.HandshakeIdleTimeout = 10 * time.Second
	}
	// no MaxTokenAge in current quic-go; rely on defaults

	// Configure TLS for QUIC
	if config.TLSConfig == nil {
		return nil, fmt.Errorf("TLS config is required")
	}

	// Ensure ALPN includes HTTP/3
	if config.TLSConfig.NextProtos == nil {
		config.TLSConfig.NextProtos = []string{"h3", "h3-29", "shieldx-wch"}
	}

	metrics := &QUICMetrics{
		RTTHistogram: make(map[string]int64),
	}

	server := &QUICServer{
		addr:        config.Addr,
		tlsConfig:   config.TLSConfig,
		quicConfig:  quicConfig,
		sessions:    config.SessionManager,
		rateLimiter: config.RateLimiter,
		camouflage:  config.CamouflageEngine,
		metrics:     metrics,
		shutdown:    make(chan struct{}),
	}

	return server, nil
}

// SetHandler sets the HTTP handler for the QUIC server
func (s *QUICServer) SetHandler(handler http.Handler) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.handler = handler
}

// Start starts the QUIC server
func (s *QUICServer) Start(ctx context.Context) error {
	s.mu.RLock()
	handler := s.handler
	s.mu.RUnlock()

	if handler == nil {
		return fmt.Errorf("no handler set")
	}

	// Wrap handler with middleware
	wrappedHandler := s.wrapHandler(handler)

	// Create HTTP/3 server
	s.server = &http3.Server{
		Addr:       s.addr,
		Handler:    wrappedHandler,
		TLSConfig:  s.tlsConfig,
		QUICConfig: s.quicConfig,
	}

	log.Printf("🚀 QUIC/HTTP3 server starting on %s", s.addr)
	log.Printf("📊 Max streams: %d | Idle timeout: %v",
		s.quicConfig.MaxIncomingStreams,
		s.quicConfig.MaxIdleTimeout,
	)

	// Start metrics collector
	go s.collectMetrics(ctx)

	// Listen and serve
	errCh := make(chan error, 1)
	go func() {
		errCh <- s.server.ListenAndServe()
	}()

	// Wait for shutdown or error
	select {
	case <-ctx.Done():
		log.Println("🛑 QUIC server shutting down...")
		return s.server.Close()
	case err := <-errCh:
		if err != nil && err != http.ErrServerClosed {
			return fmt.Errorf("QUIC server error: %w", err)
		}
		return nil
	case <-s.shutdown:
		log.Println("🛑 QUIC server shutting down...")
		return s.server.Close()
	}
}

// Shutdown gracefully shuts down the server
func (s *QUICServer) Shutdown(ctx context.Context) error {
	close(s.shutdown)
	if s.server != nil {
		return s.server.Close()
	}
	return nil
}

// wrapHandler wraps the handler with middleware
func (s *QUICServer) wrapHandler(handler http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Increment connection counter
		s.metrics.mu.Lock()
		s.metrics.ConnectionsTotal++
		s.metrics.ConnectionsActive++
		s.metrics.mu.Unlock()

		defer func() {
			s.metrics.mu.Lock()
			s.metrics.ConnectionsActive--
			s.metrics.mu.Unlock()
		}()

		// Rate limiting
		if s.rateLimiter != nil {
			allowed, err := s.rateLimiter.Allow(r.Context(), r.RemoteAddr)
			if err != nil {
				http.Error(w, "Rate limiter error", http.StatusInternalServerError)
				return
			}
			if !allowed {
				http.Error(w, "Rate limit exceeded", http.StatusTooManyRequests)
				return
			}
		}

		// Camouflage/fingerprint rotation
		if s.camouflage != nil {
			s.camouflage.ApplyFingerprint(w, r)
		}

		// Track bytes
		wrappedWriter := &metricsResponseWriter{
			ResponseWriter: w,
			metrics:        s.metrics,
		}

		// Call original handler
		handler.ServeHTTP(wrappedWriter, r)
	})
}

// collectMetrics periodically collects and logs metrics
func (s *QUICServer) collectMetrics(ctx context.Context) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			s.logMetrics()
		}
	}
}

// logMetrics logs current metrics
func (s *QUICServer) logMetrics() {
	s.metrics.mu.RLock()
	defer s.metrics.mu.RUnlock()

	log.Printf("📊 QUIC Metrics: Connections=%d/%d | Streams=%d/%d | TX=%d MB | RX=%d MB",
		s.metrics.ConnectionsActive,
		s.metrics.ConnectionsTotal,
		s.metrics.StreamsActive,
		s.metrics.StreamsTotal,
		s.metrics.BytesSent/(1024*1024),
		s.metrics.BytesReceived/(1024*1024),
	)
}

// GetMetrics returns current metrics
func (s *QUICServer) GetMetrics() *QUICMetrics {
	s.metrics.mu.RLock()
	defer s.metrics.mu.RUnlock()

	// Return a copy
	return &QUICMetrics{
		ConnectionsTotal:   s.metrics.ConnectionsTotal,
		ConnectionsActive:  s.metrics.ConnectionsActive,
		StreamsTotal:       s.metrics.StreamsTotal,
		StreamsActive:      s.metrics.StreamsActive,
		BytesSent:          s.metrics.BytesSent,
		BytesReceived:      s.metrics.BytesReceived,
		PacketsLost:        s.metrics.PacketsLost,
		HandshakeFailures:  s.metrics.HandshakeFailures,
		HandshakeSuccesses: s.metrics.HandshakeSuccesses,
		DatagramsReceived:  s.metrics.DatagramsReceived,
		DatagramsSent:      s.metrics.DatagramsSent,
	}
}

// metricsResponseWriter wraps http.ResponseWriter to track bytes
type metricsResponseWriter struct {
	http.ResponseWriter
	metrics *QUICMetrics
	written int64
}

func (w *metricsResponseWriter) Write(b []byte) (int, error) {
	n, err := w.ResponseWriter.Write(b)
	w.written += int64(n)
	w.metrics.mu.Lock()
	w.metrics.BytesSent += int64(n)
	w.metrics.mu.Unlock()
	return n, err
}

// Helper function to create a basic TLS config
func CreateTLSConfig(certFile, keyFile string) (*tls.Config, error) {
	cert, err := tls.LoadX509KeyPair(certFile, keyFile)
	if err != nil {
		return nil, fmt.Errorf("failed to load certificate: %w", err)
	}

	return &tls.Config{
		Certificates: []tls.Certificate{cert},
		NextProtos:   []string{"h3", "h3-29", "shieldx-wch"},
		MinVersion:   tls.VersionTLS13,
		CipherSuites: []uint16{
			tls.TLS_AES_128_GCM_SHA256,
			tls.TLS_AES_256_GCM_SHA384,
			tls.TLS_CHACHA20_POLY1305_SHA256,
		},
	}, nil
}

// QUICClient represents a QUIC client for WCH
type QUICClient struct {
	tlsConfig    *tls.Config
	quicConfig   *quic.Config
	roundTripper *http3.RoundTripper
}

// NewQUICClient creates a new QUIC client
func NewQUICClient(tlsConfig *tls.Config) *QUICClient {
	if tlsConfig == nil {
		tlsConfig = &tls.Config{
			NextProtos: []string{"h3", "h3-29", "shieldx-wch"},
			MinVersion: tls.VersionTLS13,
		}
	}

	quicConfig := &quic.Config{
		MaxIdleTimeout:  30 * time.Second,
		KeepAlivePeriod: 10 * time.Second,
	}

	return &QUICClient{
		tlsConfig:  tlsConfig,
		quicConfig: quicConfig,
		roundTripper: &http3.RoundTripper{
			TLSClientConfig: tlsConfig,
			QUICConfig:      quicConfig,
		},
	}
}

// Do performs an HTTP request over QUIC
func (c *QUICClient) Do(req *http.Request) (*http.Response, error) {
	return c.roundTripper.RoundTrip(req)
}

// Close closes the client
func (c *QUICClient) Close() error {
	return c.roundTripper.Close()
}

// SendEnvelope sends a WCH envelope over QUIC
func (c *QUICClient) SendEnvelope(ctx context.Context, url string, envelope *Envelope) (*Envelope, error) {
	data, err := json.Marshal(envelope)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal envelope: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	// body already set by NewRequestWithContext

	resp, err := c.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	var responseEnvelope Envelope
	if err := json.NewDecoder(resp.Body).Decode(&responseEnvelope); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &responseEnvelope, nil
}
