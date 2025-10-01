package auth
package auth

import (
	"context"
	"crypto/rsa"
	"crypto/x509"
	"encoding/pem"
	"errors"
	"fmt"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/google/uuid"
)

var (
	ErrInvalidToken      = errors.New("invalid token")
	ErrExpiredToken      = errors.New("token has expired")
	ErrInvalidSignature  = errors.New("invalid token signature")
	ErrInvalidClaims     = errors.New("invalid token claims")
	ErrTokenRevoked      = errors.New("token has been revoked")
)

// JWTManager handles JWT token creation, validation, and refresh
type JWTManager struct {
	privateKey       *rsa.PrivateKey
	publicKey        *rsa.PublicKey
	accessTokenTTL   time.Duration
	refreshTokenTTL  time.Duration
	issuer           string
	revokedTokens    RevokedTokenStore
}

// JWTConfig configuration for JWT manager
type JWTConfig struct {
	PrivateKeyPEM    string
	PublicKeyPEM     string
	AccessTokenTTL   time.Duration
	RefreshTokenTTL  time.Duration
	Issuer           string
	RevokedTokenStore RevokedTokenStore
}

// CustomClaims represents JWT claims with custom fields
type CustomClaims struct {
	UserID      string            `json:"user_id"`
	TenantID    string            `json:"tenant_id"`
	Email       string            `json:"email"`
	Roles       []string          `json:"roles"`
	Permissions []string          `json:"permissions"`
	SessionID   string            `json:"session_id"`
	TokenType   string            `json:"token_type"` // "access" or "refresh"
	Metadata    map[string]string `json:"metadata,omitempty"`
	jwt.RegisteredClaims
}

// TokenPair represents access and refresh tokens
type TokenPair struct {
	AccessToken  string    `json:"access_token"`
	RefreshToken string    `json:"refresh_token"`
	ExpiresAt    time.Time `json:"expires_at"`
	TokenType    string    `json:"token_type"`
}

// NewJWTManager creates a new JWT manager instance
func NewJWTManager(config JWTConfig) (*JWTManager, error) {
	privateKey, err := parsePrivateKey(config.PrivateKeyPEM)
	if err != nil {
		return nil, fmt.Errorf("failed to parse private key: %w", err)
	}

	publicKey, err := parsePublicKey(config.PublicKeyPEM)
	if err != nil {
		return nil, fmt.Errorf("failed to parse public key: %w", err)
	}

	if config.AccessTokenTTL == 0 {
		config.AccessTokenTTL = 15 * time.Minute
	}
	if config.RefreshTokenTTL == 0 {
		config.RefreshTokenTTL = 7 * 24 * time.Hour
	}
	if config.Issuer == "" {
		config.Issuer = "shieldx-auth"
	}
	if config.RevokedTokenStore == nil {
		config.RevokedTokenStore = NewInMemoryRevokedStore()
	}

	return &JWTManager{
		privateKey:      privateKey,
		publicKey:       publicKey,
		accessTokenTTL:  config.AccessTokenTTL,
		refreshTokenTTL: config.RefreshTokenTTL,
		issuer:          config.Issuer,
		revokedTokens:   config.RevokedTokenStore,
	}, nil
}

// GenerateTokenPair creates access and refresh tokens
func (jm *JWTManager) GenerateTokenPair(ctx context.Context, userID, tenantID, email string, roles, permissions []string) (*TokenPair, error) {
	sessionID := uuid.New().String()
	now := time.Now()

	// Generate access token
	accessClaims := CustomClaims{
		UserID:      userID,
		TenantID:    tenantID,
		Email:       email,
		Roles:       roles,
		Permissions: permissions,
		SessionID:   sessionID,
		TokenType:   "access",
		RegisteredClaims: jwt.RegisteredClaims{
			Issuer:    jm.issuer,
			Subject:   userID,
			ExpiresAt: jwt.NewNumericDate(now.Add(jm.accessTokenTTL)),
			IssuedAt:  jwt.NewNumericDate(now),
			NotBefore: jwt.NewNumericDate(now),
			ID:        uuid.New().String(),
		},
	}

	accessToken, err := jm.signToken(accessClaims)
	if err != nil {
		return nil, fmt.Errorf("failed to generate access token: %w", err)
	}

	// Generate refresh token
	refreshClaims := CustomClaims{
		UserID:    userID,
		TenantID:  tenantID,
		SessionID: sessionID,
		TokenType: "refresh",
		RegisteredClaims: jwt.RegisteredClaims{
			Issuer:    jm.issuer,
			Subject:   userID,
			ExpiresAt: jwt.NewNumericDate(now.Add(jm.refreshTokenTTL)),
			IssuedAt:  jwt.NewNumericDate(now),
			NotBefore: jwt.NewNumericDate(now),
			ID:        uuid.New().String(),
		},
	}

	refreshToken, err := jm.signToken(refreshClaims)
	if err != nil {
		return nil, fmt.Errorf("failed to generate refresh token: %w", err)
	}

	return &TokenPair{
		AccessToken:  accessToken,
		RefreshToken: refreshToken,
		ExpiresAt:    now.Add(jm.accessTokenTTL),
		TokenType:    "Bearer",
	}, nil
}

// ValidateToken validates a JWT token and returns claims
func (jm *JWTManager) ValidateToken(ctx context.Context, tokenString string) (*CustomClaims, error) {
	token, err := jwt.ParseWithClaims(tokenString, &CustomClaims{}, func(token *jwt.Token) (interface{}, error) {
		// Verify signing method
		if _, ok := token.Method.(*jwt.SigningMethodRSA); !ok {
			return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
		}
		return jm.publicKey, nil
	})

	if err != nil {
		if errors.Is(err, jwt.ErrTokenExpired) {
			return nil, ErrExpiredToken
		}
		return nil, fmt.Errorf("%w: %v", ErrInvalidToken, err)
	}

	claims, ok := token.Claims.(*CustomClaims)
	if !ok || !token.Valid {
		return nil, ErrInvalidClaims
	}

	// Check if token is revoked
	isRevoked, err := jm.revokedTokens.IsRevoked(ctx, claims.ID)
	if err != nil {
		return nil, fmt.Errorf("failed to check token revocation: %w", err)
	}
	if isRevoked {
		return nil, ErrTokenRevoked
	}

	return claims, nil
}

// RefreshToken generates new token pair from refresh token
func (jm *JWTManager) RefreshToken(ctx context.Context, refreshToken string) (*TokenPair, error) {
	claims, err := jm.ValidateToken(ctx, refreshToken)
	if err != nil {
		return nil, fmt.Errorf("invalid refresh token: %w", err)
	}

	if claims.TokenType != "refresh" {
		return nil, fmt.Errorf("token is not a refresh token")
	}

	// Revoke old refresh token
	if err := jm.RevokeToken(ctx, claims.ID, claims.ExpiresAt.Time); err != nil {
		return nil, fmt.Errorf("failed to revoke old token: %w", err)
	}

	// Generate new token pair
	return jm.GenerateTokenPair(ctx, claims.UserID, claims.TenantID, claims.Email, claims.Roles, claims.Permissions)
}

// RevokeToken adds a token to the revocation list
func (jm *JWTManager) RevokeToken(ctx context.Context, tokenID string, expiresAt time.Time) error {
	return jm.revokedTokens.RevokeToken(ctx, tokenID, expiresAt)
}

// RevokeSession revokes all tokens for a session
func (jm *JWTManager) RevokeSession(ctx context.Context, sessionID string) error {
	return jm.revokedTokens.RevokeSession(ctx, sessionID)
}

// signToken creates a signed JWT token
func (jm *JWTManager) signToken(claims CustomClaims) (string, error) {
	token := jwt.NewWithClaims(jwt.SigningMethodRS256, claims)
	return token.SignedString(jm.privateKey)
}

// Helper functions for key parsing
func parsePrivateKey(pemStr string) (*rsa.PrivateKey, error) {
	block, _ := pem.Decode([]byte(pemStr))
	if block == nil {
		return nil, errors.New("failed to parse PEM block")
	}

	key, err := x509.ParsePKCS8PrivateKey(block.Bytes)
	if err != nil {
		// Try PKCS1 format
		return x509.ParsePKCS1PrivateKey(block.Bytes)
	}

	rsaKey, ok := key.(*rsa.PrivateKey)
	if !ok {
		return nil, errors.New("not an RSA private key")
	}
	return rsaKey, nil
}

func parsePublicKey(pemStr string) (*rsa.PublicKey, error) {
	block, _ := pem.Decode([]byte(pemStr))
	if block == nil {
		return nil, errors.New("failed to parse PEM block")
	}

	pub, err := x509.ParsePKIXPublicKey(block.Bytes)
	if err != nil {
		return nil, err
	}

	rsaPub, ok := pub.(*rsa.PublicKey)
	if !ok {
		return nil, errors.New("not an RSA public key")
	}
	return rsaPub, nil
}
