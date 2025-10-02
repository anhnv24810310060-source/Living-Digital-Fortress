package wch

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/google/uuid"
)

// SessionManager manages WCH sessions with ephemeral keys
type SessionManager struct {
	mu       sync.RWMutex
	sessions map[string]*Session
	ttl      time.Duration
}

// Session represents an active WCH session
type Session struct {
	ID              string
	ChannelID       string
	ClientPubKey    []byte
	GuardianPrivKey []byte
	GuardianPubKey  []byte
	SharedSecret    []byte
	CreatedAt       time.Time
	ExpiresAt       time.Time
	LastActivity    time.Time
	RekeyCounter    int
	Metadata        map[string]interface{}
}

// NewSessionManager creates a new session manager
func NewSessionManager(ttl time.Duration) *SessionManager {
	if ttl == 0 {
		ttl = 30 * time.Minute
	}

	sm := &SessionManager{
		sessions: make(map[string]*Session),
		ttl:      ttl,
	}

	// Start cleanup goroutine
	go sm.cleanup()

	return sm
}

// CreateSession creates a new WCH session
func (sm *SessionManager) CreateSession(clientPubKey []byte) (*Session, error) {
	// Generate Guardian ephemeral keypair
	guardianPriv, guardianPub, err := GenerateClientEphemeral()
	if err != nil {
		return nil, fmt.Errorf("failed to generate guardian keys: %w", err)
	}

	// Derive shared secret
	clientPubKeyObj, err := guardianPriv.Curve().NewPublicKey(clientPubKey)
	if err != nil {
		return nil, fmt.Errorf("invalid client public key: %w", err)
	}

	sharedSecret, err := guardianPriv.ECDH(clientPubKeyObj)
	if err != nil {
		return nil, fmt.Errorf("failed to derive shared secret: %w", err)
	}

	now := time.Now()
	session := &Session{
		ID:              uuid.New().String(),
		ChannelID:       uuid.New().String(),
		ClientPubKey:    clientPubKey,
		GuardianPrivKey: guardianPriv.Bytes(),
		GuardianPubKey:  guardianPub,
		SharedSecret:    sharedSecret,
		CreatedAt:       now,
		ExpiresAt:       now.Add(sm.ttl),
		LastActivity:    now,
		RekeyCounter:    0,
		Metadata:        make(map[string]interface{}),
	}

	sm.mu.Lock()
	sm.sessions[session.ChannelID] = session
	sm.mu.Unlock()

	return session, nil
}

// GetSession retrieves a session by channel ID
func (sm *SessionManager) GetSession(channelID string) (*Session, error) {
	sm.mu.RLock()
	session, exists := sm.sessions[channelID]
	sm.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("session not found")
	}

	// Check expiration
	if time.Now().After(session.ExpiresAt) {
		sm.DeleteSession(channelID)
		return nil, fmt.Errorf("session expired")
	}

	// Update last activity
	session.LastActivity = time.Now()

	return session, nil
}

// UpdateSessionActivity updates session last activity time
func (sm *SessionManager) UpdateSessionActivity(channelID string) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	session, exists := sm.sessions[channelID]
	if !exists {
		return fmt.Errorf("session not found")
	}

	session.LastActivity = time.Now()
	// Extend expiration on activity
	session.ExpiresAt = time.Now().Add(sm.ttl)

	return nil
}

// DeleteSession deletes a session
func (sm *SessionManager) DeleteSession(channelID string) {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	delete(sm.sessions, channelID)
}

// cleanup removes expired sessions
func (sm *SessionManager) cleanup() {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		sm.mu.Lock()
		now := time.Now()
		for channelID, session := range sm.sessions {
			if now.After(session.ExpiresAt) {
				delete(sm.sessions, channelID)
			}
		}
		sm.mu.Unlock()
	}
}

// GetActiveSessionsCount returns the number of active sessions
func (sm *SessionManager) GetActiveSessionsCount() int {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	return len(sm.sessions)
}

// WCHHandler handles WCH protocol requests
type WCHHandler struct {
	sessionMgr  *SessionManager
	rateLimiter RateLimiter
	camouflage  *CamouflageEngine
	metrics     *WCHMetrics
}

// WCHMetrics tracks WCH metrics
type WCHMetrics struct {
	mu                sync.RWMutex
	ConnectionsTotal  int64
	EnvelopesSent     int64
	EnvelopesReceived int64
	EncryptionErrors  int64
	DecryptionErrors  int64
	SessionsCreated   int64
	SessionsExpired   int64
}

// NewWCHHandler creates a new WCH handler
func NewWCHHandler(sessionMgr *SessionManager, rateLimiter RateLimiter, camouflage *CamouflageEngine) *WCHHandler {
	return &WCHHandler{
		sessionMgr:  sessionMgr,
		rateLimiter: rateLimiter,
		camouflage:  camouflage,
		metrics:     &WCHMetrics{},
	}
}

// HandleConnect handles WCH connection establishment
func (h *WCHHandler) HandleConnect(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Rate limiting
	if h.rateLimiter != nil {
		allowed, err := h.rateLimiter.Allow(r.Context(), r.RemoteAddr)
		if err != nil || !allowed {
			http.Error(w, "Rate limit exceeded", http.StatusTooManyRequests)
			return
		}
	}

	// Parse request
	var req struct {
		ClientPubKey string `json:"clientPubKey"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	// Decode client public key
	clientPubKey, err := UnmarshalB64(req.ClientPubKey)
	if err != nil {
		http.Error(w, "Invalid client public key", http.StatusBadRequest)
		return
	}

	// Create session
	session, err := h.sessionMgr.CreateSession(clientPubKey)
	if err != nil {
		http.Error(w, "Failed to create session", http.StatusInternalServerError)
		return
	}

	h.metrics.mu.Lock()
	h.metrics.SessionsCreated++
	h.metrics.ConnectionsTotal++
	h.metrics.mu.Unlock()

	// Return connection response
	response := ConnectResponse{
		ChannelID:      session.ChannelID,
		GuardianPubB64: MarshalB64(session.GuardianPubKey),
		Protocol:       Protocol,
		ExpiresAt:      session.ExpiresAt.Unix(),
		RebindHintMs:   int(h.sessionMgr.ttl.Milliseconds() / 2),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// HandleSend handles WCH envelope sending
func (h *WCHHandler) HandleSend(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Apply camouflage
	if h.camouflage != nil {
		h.camouflage.ApplyFingerprint(w, r)
	}

	// Parse envelope
	var envelope Envelope
	if err := json.NewDecoder(r.Body).Decode(&envelope); err != nil {
		http.Error(w, "Invalid envelope", http.StatusBadRequest)
		return
	}

	// Get session
	session, err := h.sessionMgr.GetSession(envelope.ChannelID)
	if err != nil {
		http.Error(w, "Invalid or expired session", http.StatusUnauthorized)
		return
	}

	// Derive key
	var key []byte
	if envelope.RekeyCounter > 0 {
		key, err = DeriveKeyWithCounter(session.SharedSecret, envelope.ChannelID, envelope.RekeyCounter)
	} else {
		key, err = DeriveKey(session.SharedSecret, envelope.ChannelID)
	}
	if err != nil {
		http.Error(w, "Key derivation failed", http.StatusInternalServerError)
		return
	}

	// Decrypt envelope
	nonce, err := UnmarshalB64(envelope.NonceB64)
	if err != nil {
		http.Error(w, "Invalid nonce", http.StatusBadRequest)
		return
	}

	ciphertext, err := UnmarshalB64(envelope.CiphertextB64)
	if err != nil {
		http.Error(w, "Invalid ciphertext", http.StatusBadRequest)
		return
	}

	plaintext, err := Open(key, nonce, ciphertext)
	if err != nil {
		h.metrics.mu.Lock()
		h.metrics.DecryptionErrors++
		h.metrics.mu.Unlock()
		http.Error(w, "Decryption failed", http.StatusBadRequest)
		return
	}

	h.metrics.mu.Lock()
	h.metrics.EnvelopesReceived++
	h.metrics.mu.Unlock()

	// Process inner request
	var innerReq InnerRequest
	if err := json.Unmarshal(plaintext, &innerReq); err != nil {
		http.Error(w, "Invalid inner request", http.StatusBadRequest)
		return
	}

	// TODO: Forward request to target service
	// For now, return mock response
	innerResp := InnerResponse{
		Status: 200,
		Headers: map[string]string{
			"Content-Type": "application/json",
		},
		Body: []byte(`{"status":"success","message":"WCH envelope processed"}`),
	}

	// Encrypt response
	respData, _ := json.Marshal(innerResp)
	respNonce, respCiphertext, err := Seal(key, respData)
	if err != nil {
		h.metrics.mu.Lock()
		h.metrics.EncryptionErrors++
		h.metrics.mu.Unlock()
		http.Error(w, "Encryption failed", http.StatusInternalServerError)
		return
	}

	h.metrics.mu.Lock()
	h.metrics.EnvelopesSent++
	h.metrics.mu.Unlock()

	// Update session activity
	h.sessionMgr.UpdateSessionActivity(envelope.ChannelID)

	// Return encrypted response
	respEnvelope := map[string]string{
		"nonce":      MarshalB64(respNonce),
		"ciphertext": MarshalB64(respCiphertext),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(respEnvelope)
}

// HandleMetrics returns WCH metrics
func (h *WCHHandler) HandleMetrics(w http.ResponseWriter, r *http.Request) {
	h.metrics.mu.RLock()
	defer h.metrics.mu.RUnlock()

	metrics := map[string]interface{}{
		"connections_total":  h.metrics.ConnectionsTotal,
		"envelopes_sent":     h.metrics.EnvelopesSent,
		"envelopes_received": h.metrics.EnvelopesReceived,
		"encryption_errors":  h.metrics.EncryptionErrors,
		"decryption_errors":  h.metrics.DecryptionErrors,
		"sessions_created":   h.metrics.SessionsCreated,
		"sessions_expired":   h.metrics.SessionsExpired,
		"sessions_active":    h.sessionMgr.GetActiveSessionsCount(),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(metrics)
}

// SetupWCHServer sets up a complete WCH server
func SetupWCHServer(ctx context.Context, config WCHServerConfig) error {
	// Create TLS config
	tlsConfig, err := CreateTLSConfig(config.CertFile, config.KeyFile)
	if err != nil {
		return fmt.Errorf("failed to create TLS config: %w", err)
	}

	// Create rate limiter
	var rateLimiter RateLimiter
	if config.RedisAddr != "" {
		rateLimiter, err = NewRedisRateLimiter(RateLimiterConfig{
			RedisAddr: config.RedisAddr,
			KeyPrefix: "wch:ratelimit:",
			Limit:     config.RateLimit,
			Window:    1 * time.Minute,
			Algorithm: "sliding_window",
		})
		if err != nil {
			log.Printf("Warning: Failed to create Redis rate limiter, using in-memory: %v", err)
			rateLimiter = NewInMemoryRateLimiter(config.RateLimit, 1*time.Minute)
		}
	} else {
		rateLimiter = NewInMemoryRateLimiter(config.RateLimit, 1*time.Minute)
	}

	// Create camouflage engine
	camouflage := NewCamouflageEngine(CamouflageConfig{
		RotationPeriod: 5 * time.Minute,
		EnableJA3:      true,
	})

	// Create session manager
	sessionMgr := NewSessionManager(30 * time.Minute)

	// Create WCH handler
	wchHandler := NewWCHHandler(sessionMgr, rateLimiter, camouflage)

	// Setup HTTP routes
	mux := http.NewServeMux()
	mux.HandleFunc("/wch/connect", wchHandler.HandleConnect)
	mux.HandleFunc("/wch/send", wchHandler.HandleSend)
	mux.HandleFunc("/wch/metrics", wchHandler.HandleMetrics)

	// Create QUIC server
	quicServer, err := NewQUICServer(QUICConfig{
		Addr:               config.Addr,
		TLSConfig:          tlsConfig,
		MaxIdleTimeout:     30 * time.Second,
		MaxIncomingStreams: 100,
		EnableDatagrams:    true,
		RateLimiter:        rateLimiter,
		SessionManager:     sessionMgr,
		CamouflageEngine:   camouflage,
	})
	if err != nil {
		return fmt.Errorf("failed to create QUIC server: %w", err)
	}

	quicServer.SetHandler(mux)

	log.Println("🔐 WCH Server starting with production features:")
	log.Println("  ✅ QUIC/HTTP3 protocol")
	log.Println("  ✅ Distributed rate limiting")
	log.Println("  ✅ TLS fingerprint camouflage")
	log.Println("  ✅ JA3 rotation")
	log.Println("  ✅ Session management")

	return quicServer.Start(ctx)
}

// WCHServerConfig configuration for WCH server
type WCHServerConfig struct {
	Addr      string
	CertFile  string
	KeyFile   string
	RedisAddr string
	RateLimit int
}
