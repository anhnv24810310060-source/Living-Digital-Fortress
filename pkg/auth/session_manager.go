package auth

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"errors"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"
)

var (
	ErrSessionNotFound = errors.New("session not found")
	ErrSessionExpired  = errors.New("session has expired")
)

// SessionData represents user session information
type SessionData struct {
	SessionID  string            `json:"session_id"`
	UserID     string            `json:"user_id"`
	TenantID   string            `json:"tenant_id"`
	Email      string            `json:"email"`
	IPAddress  string            `json:"ip_address"`
	UserAgent  string            `json:"user_agent"`
	CreatedAt  time.Time         `json:"created_at"`
	LastAccess time.Time         `json:"last_access"`
	ExpiresAt  time.Time         `json:"expires_at"`
	Metadata   map[string]string `json:"metadata,omitempty"`
}

// SessionManager handles session lifecycle
type SessionManager struct {
	client    *redis.Client
	keyPrefix string
	ttl       time.Duration
}

// SessionConfig configuration for session manager
type SessionConfig struct {
	RedisAddr     string
	RedisPassword string
	RedisDB       int
	SessionTTL    time.Duration
}

// NewSessionManager creates a new session manager
func NewSessionManager(config SessionConfig) (*SessionManager, error) {
	client := redis.NewClient(&redis.Options{
		Addr:     config.RedisAddr,
		Password: config.RedisPassword,
		DB:       config.RedisDB,
	})

	// Test connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := client.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("failed to connect to Redis: %w", err)
	}

	if config.SessionTTL == 0 {
		config.SessionTTL = 24 * time.Hour
	}

	return &SessionManager{
		client:    client,
		keyPrefix: "session:",
		ttl:       config.SessionTTL,
	}, nil
}

// CreateSession creates a new user session
func (sm *SessionManager) CreateSession(ctx context.Context, userID, tenantID, email, ipAddress, userAgent string) (*SessionData, error) {
	sessionID, err := generateSessionID()
	if err != nil {
		return nil, fmt.Errorf("failed to generate session ID: %w", err)
	}

	now := time.Now()
	session := &SessionData{
		SessionID:  sessionID,
		UserID:     userID,
		TenantID:   tenantID,
		Email:      email,
		IPAddress:  ipAddress,
		UserAgent:  userAgent,
		CreatedAt:  now,
		LastAccess: now,
		ExpiresAt:  now.Add(sm.ttl),
		Metadata:   make(map[string]string),
	}

	key := sm.keyPrefix + sessionID
	err = sm.client.HSet(ctx, key,
		"session_id", session.SessionID,
		"user_id", session.UserID,
		"tenant_id", session.TenantID,
		"email", session.Email,
		"ip_address", session.IPAddress,
		"user_agent", session.UserAgent,
		"created_at", session.CreatedAt.Unix(),
		"last_access", session.LastAccess.Unix(),
		"expires_at", session.ExpiresAt.Unix(),
	).Err()

	if err != nil {
		return nil, fmt.Errorf("failed to create session: %w", err)
	}

	// Set expiration
	sm.client.Expire(ctx, key, sm.ttl)

	return session, nil
}

// GetSession retrieves session data
func (sm *SessionManager) GetSession(ctx context.Context, sessionID string) (*SessionData, error) {
	key := sm.keyPrefix + sessionID

	result, err := sm.client.HGetAll(ctx, key).Result()
	if err != nil {
		return nil, fmt.Errorf("failed to get session: %w", err)
	}

	if len(result) == 0 {
		return nil, ErrSessionNotFound
	}

	session := &SessionData{
		SessionID: result["session_id"],
		UserID:    result["user_id"],
		TenantID:  result["tenant_id"],
		Email:     result["email"],
		IPAddress: result["ip_address"],
		UserAgent: result["user_agent"],
		Metadata:  make(map[string]string),
	}

	// Parse timestamps
	if createdAt, ok := result["created_at"]; ok {
		var timestamp int64
		fmt.Sscanf(createdAt, "%d", &timestamp)
		session.CreatedAt = time.Unix(timestamp, 0)
	}
	if lastAccess, ok := result["last_access"]; ok {
		var timestamp int64
		fmt.Sscanf(lastAccess, "%d", &timestamp)
		session.LastAccess = time.Unix(timestamp, 0)
	}
	if expiresAt, ok := result["expires_at"]; ok {
		var timestamp int64
		fmt.Sscanf(expiresAt, "%d", &timestamp)
		session.ExpiresAt = time.Unix(timestamp, 0)
	}

	// Check expiration
	if time.Now().After(session.ExpiresAt) {
		sm.DeleteSession(ctx, sessionID)
		return nil, ErrSessionExpired
	}

	return session, nil
}

// UpdateSessionActivity updates last access time and extends session
func (sm *SessionManager) UpdateSessionActivity(ctx context.Context, sessionID string) error {
	key := sm.keyPrefix + sessionID

	now := time.Now()
	expiresAt := now.Add(sm.ttl)

	err := sm.client.HSet(ctx, key,
		"last_access", now.Unix(),
		"expires_at", expiresAt.Unix(),
	).Err()

	if err != nil {
		return fmt.Errorf("failed to update session: %w", err)
	}

	// Extend expiration
	sm.client.Expire(ctx, key, sm.ttl)

	return nil
}

// DeleteSession removes a session
func (sm *SessionManager) DeleteSession(ctx context.Context, sessionID string) error {
	key := sm.keyPrefix + sessionID
	err := sm.client.Del(ctx, key).Err()
	if err != nil {
		return fmt.Errorf("failed to delete session: %w", err)
	}
	return nil
}

// DeleteUserSessions removes all sessions for a user
func (sm *SessionManager) DeleteUserSessions(ctx context.Context, userID string) error {
	// Scan for all sessions
	pattern := sm.keyPrefix + "*"
	iter := sm.client.Scan(ctx, 0, pattern, 0).Iterator()

	for iter.Next(ctx) {
		key := iter.Val()
		uid, err := sm.client.HGet(ctx, key, "user_id").Result()
		if err == nil && uid == userID {
			sm.client.Del(ctx, key)
		}
	}

	return iter.Err()
}

// SetSessionMetadata adds custom metadata to session
func (sm *SessionManager) SetSessionMetadata(ctx context.Context, sessionID, key, value string) error {
	sessionKey := sm.keyPrefix + sessionID
	metaKey := "meta_" + key

	err := sm.client.HSet(ctx, sessionKey, metaKey, value).Err()
	if err != nil {
		return fmt.Errorf("failed to set metadata: %w", err)
	}
	return nil
}

// GetSessionMetadata retrieves custom metadata from session
func (sm *SessionManager) GetSessionMetadata(ctx context.Context, sessionID, key string) (string, error) {
	sessionKey := sm.keyPrefix + sessionID
	metaKey := "meta_" + key

	value, err := sm.client.HGet(ctx, sessionKey, metaKey).Result()
	if err == redis.Nil {
		return "", nil
	}
	if err != nil {
		return "", fmt.Errorf("failed to get metadata: %w", err)
	}
	return value, nil
}

// Close closes the Redis connection
func (sm *SessionManager) Close() error {
	return sm.client.Close()
}

// generateSessionID generates a cryptographically secure session ID
func generateSessionID() (string, error) {
	b := make([]byte, 32)
	_, err := rand.Read(b)
	if err != nil {
		return "", err
	}
	return base64.URLEncoding.EncodeToString(b), nil
}
