package auth

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"
)

// RevokedTokenStore interface for managing revoked tokens
type RevokedTokenStore interface {
	RevokeToken(ctx context.Context, tokenID string, expiresAt time.Time) error
	RevokeSession(ctx context.Context, sessionID string) error
	IsRevoked(ctx context.Context, tokenID string) (bool, error)
	Cleanup(ctx context.Context) error
}

// InMemoryRevokedStore stores revoked tokens in memory (for development)
type InMemoryRevokedStore struct {
	mu      sync.RWMutex
	revoked map[string]time.Time
}

func NewInMemoryRevokedStore() *InMemoryRevokedStore {
	store := &InMemoryRevokedStore{
		revoked: make(map[string]time.Time),
	}
	// Start cleanup goroutine
	go store.cleanupLoop(context.Background())
	return store
}

func (s *InMemoryRevokedStore) RevokeToken(ctx context.Context, tokenID string, expiresAt time.Time) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.revoked[tokenID] = expiresAt
	return nil
}

func (s *InMemoryRevokedStore) RevokeSession(ctx context.Context, sessionID string) error {
	// In-memory store doesn't support session tracking
	return nil
}

func (s *InMemoryRevokedStore) IsRevoked(ctx context.Context, tokenID string) (bool, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	_, exists := s.revoked[tokenID]
	return exists, nil
}

func (s *InMemoryRevokedStore) Cleanup(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	now := time.Now()
	for tokenID, expiresAt := range s.revoked {
		if now.After(expiresAt) {
			delete(s.revoked, tokenID)
		}
	}
	return nil
}

func (s *InMemoryRevokedStore) cleanupLoop(ctx context.Context) {
	ticker := time.NewTicker(1 * time.Hour)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			s.Cleanup(ctx)
		}
	}
}

// RedisRevokedStore stores revoked tokens in Redis (production)
type RedisRevokedStore struct {
	client        *redis.Client
	keyPrefix     string
	sessionPrefix string
}

type RedisConfig struct {
	Addr     string
	Password string
	DB       int
}

func NewRedisRevokedStore(config RedisConfig) *RedisRevokedStore {
	client := redis.NewClient(&redis.Options{
		Addr:     config.Addr,
		Password: config.Password,
		DB:       config.DB,
	})

	return &RedisRevokedStore{
		client:        client,
		keyPrefix:     "revoked:token:",
		sessionPrefix: "revoked:session:",
	}
}

func (s *RedisRevokedStore) RevokeToken(ctx context.Context, tokenID string, expiresAt time.Time) error {
	key := s.keyPrefix + tokenID
	ttl := time.Until(expiresAt)

	if ttl <= 0 {
		return nil // Token already expired
	}

	err := s.client.Set(ctx, key, "1", ttl).Err()
	if err != nil {
		return fmt.Errorf("failed to revoke token: %w", err)
	}
	return nil
}

func (s *RedisRevokedStore) RevokeSession(ctx context.Context, sessionID string) error {
	key := s.sessionPrefix + sessionID
	// Store session as revoked for 30 days (covers longest possible token lifetime)
	err := s.client.Set(ctx, key, "1", 30*24*time.Hour).Err()
	if err != nil {
		return fmt.Errorf("failed to revoke session: %w", err)
	}
	return nil
}

func (s *RedisRevokedStore) IsRevoked(ctx context.Context, tokenID string) (bool, error) {
	key := s.keyPrefix + tokenID
	exists, err := s.client.Exists(ctx, key).Result()
	if err != nil {
		return false, fmt.Errorf("failed to check revocation: %w", err)
	}
	return exists > 0, nil
}

func (s *RedisRevokedStore) Cleanup(ctx context.Context) error {
	// Redis automatically handles expiration, no cleanup needed
	return nil
}

func (s *RedisRevokedStore) Close() error {
	return s.client.Close()
}

func (s *RedisRevokedStore) Ping(ctx context.Context) error {
	return s.client.Ping(ctx).Err()
}
