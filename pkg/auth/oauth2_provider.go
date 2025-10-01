package auth

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/redis/go-redis/v9"
)

var (
	ErrInvalidAuthCode     = errors.New("invalid authorization code")
	ErrInvalidClientID     = errors.New("invalid client ID")
	ErrInvalidRedirectURI  = errors.New("invalid redirect URI")
	ErrInvalidPKCE         = errors.New("invalid PKCE verifier")
)

// OAuth2Provider handles OAuth2/OIDC flows
type OAuth2Provider struct {
	issuer         string
	clients        map[string]*OAuth2Client
	authCodes      *redis.Client
	jwtManager     *JWTManager
	sessionManager *SessionManager
}

// OAuth2Client represents an OAuth2 client application
type OAuth2Client struct {
	ClientID     string   `json:"client_id"`
	ClientSecret string   `json:"client_secret"`
	RedirectURIs []string `json:"redirect_uris"`
	GrantTypes   []string `json:"grant_types"`
	Scopes       []string `json:"scopes"`
	Name         string   `json:"name"`
}

// AuthorizationRequest represents OAuth2 authorization request
type AuthorizationRequest struct {
	ClientID            string `json:"client_id"`
	RedirectURI         string `json:"redirect_uri"`
	ResponseType        string `json:"response_type"`
	Scope               string `json:"scope"`
	State               string `json:"state"`
	CodeChallenge       string `json:"code_challenge"`
	CodeChallengeMethod string `json:"code_challenge_method"`
}

// AuthorizationCode represents an authorization code
type AuthorizationCode struct {
	Code          string    `json:"code"`
	ClientID      string    `json:"client_id"`
	RedirectURI   string    `json:"redirect_uri"`
	UserID        string    `json:"user_id"`
	TenantID      string    `json:"tenant_id"`
	Scope         string    `json:"scope"`
	CodeChallenge string    `json:"code_challenge,omitempty"`
	ExpiresAt     time.Time `json:"expires_at"`
}

// TokenRequest represents OAuth2 token request
type TokenRequest struct {
	GrantType    string `json:"grant_type"`
	Code         string `json:"code"`
	RedirectURI  string `json:"redirect_uri"`
	ClientID     string `json:"client_id"`
	ClientSecret string `json:"client_secret"`
	CodeVerifier string `json:"code_verifier"`
	RefreshToken string `json:"refresh_token"`
}

// TokenResponse represents OAuth2 token response
type TokenResponse struct {
	AccessToken  string `json:"access_token"`
	TokenType    string `json:"token_type"`
	ExpiresIn    int    `json:"expires_in"`
	RefreshToken string `json:"refresh_token,omitempty"`
	IDToken      string `json:"id_token,omitempty"`
	Scope        string `json:"scope,omitempty"`
}

// OAuth2Config configuration for OAuth2 provider
type OAuth2Config struct {
	Issuer         string
	RedisClient    *redis.Client
	JWTManager     *JWTManager
	SessionManager *SessionManager
	Clients        []*OAuth2Client
}

// NewOAuth2Provider creates a new OAuth2/OIDC provider
func NewOAuth2Provider(config OAuth2Config) (*OAuth2Provider, error) {
	if config.Issuer == "" {
		return nil, errors.New("issuer is required")
	}

	provider := &OAuth2Provider{
		issuer:         config.Issuer,
		clients:        make(map[string]*OAuth2Client),
		authCodes:      config.RedisClient,
		jwtManager:     config.JWTManager,
		sessionManager: config.SessionManager,
	}

	// Register clients
	for _, client := range config.Clients {
		provider.clients[client.ClientID] = client
	}

	return provider, nil
}

// HandleAuthorize handles OAuth2 authorization endpoint
func (op *OAuth2Provider) HandleAuthorize(w http.ResponseWriter, r *http.Request) {
	// Parse authorization request
	authReq := &AuthorizationRequest{
		ClientID:            r.URL.Query().Get("client_id"),
		RedirectURI:         r.URL.Query().Get("redirect_uri"),
		ResponseType:        r.URL.Query().Get("response_type"),
		Scope:               r.URL.Query().Get("scope"),
		State:               r.URL.Query().Get("state"),
		CodeChallenge:       r.URL.Query().Get("code_challenge"),
		CodeChallengeMethod: r.URL.Query().Get("code_challenge_method"),
	}

	// Validate client
	client, exists := op.clients[authReq.ClientID]
	if !exists {
		op.errorResponse(w, r, authReq.RedirectURI, "invalid_client", "Invalid client ID", authReq.State)
		return
	}

	// Validate redirect URI
	if !op.validateRedirectURI(client, authReq.RedirectURI) {
		http.Error(w, "Invalid redirect URI", http.StatusBadRequest)
		return
	}

	// Validate response type
	if authReq.ResponseType != "code" {
		op.errorResponse(w, r, authReq.RedirectURI, "unsupported_response_type", "Only 'code' response type is supported", authReq.State)
		return
	}

	// TODO: Show login/consent page here
	// For now, we'll assume user is authenticated and consented

	// Get user from context (set by authentication middleware)
	claims := GetClaimsFromContext(r.Context())
	if claims == nil {
		// Redirect to login
		loginURL := "/login?redirect=" + url.QueryEscape(r.URL.String())
		http.Redirect(w, r, loginURL, http.StatusFound)
		return
	}

	// Generate authorization code
	code, err := op.generateAuthCode(r.Context(), authReq, claims.UserID, claims.TenantID)
	if err != nil {
		op.errorResponse(w, r, authReq.RedirectURI, "server_error", "Failed to generate authorization code", authReq.State)
		return
	}

	// Redirect back to client with code
	redirectURL, _ := url.Parse(authReq.RedirectURI)
	q := redirectURL.Query()
	q.Set("code", code)
	if authReq.State != "" {
		q.Set("state", authReq.State)
	}
	redirectURL.RawQuery = q.Encode()

	http.Redirect(w, r, redirectURL.String(), http.StatusFound)
}

// HandleToken handles OAuth2 token endpoint
func (op *OAuth2Provider) HandleToken(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Parse token request
	if err := r.ParseForm(); err != nil {
		op.jsonError(w, http.StatusBadRequest, "invalid_request", "Failed to parse form")
		return
	}

	tokenReq := &TokenRequest{
		GrantType:    r.FormValue("grant_type"),
		Code:         r.FormValue("code"),
		RedirectURI:  r.FormValue("redirect_uri"),
		ClientID:     r.FormValue("client_id"),
		ClientSecret: r.FormValue("client_secret"),
		CodeVerifier: r.FormValue("code_verifier"),
		RefreshToken: r.FormValue("refresh_token"),
	}

	switch tokenReq.GrantType {
	case "authorization_code":
		op.handleAuthorizationCodeGrant(w, r, tokenReq)
	case "refresh_token":
		op.handleRefreshTokenGrant(w, r, tokenReq)
	default:
		op.jsonError(w, http.StatusBadRequest, "unsupported_grant_type", "Grant type not supported")
	}
}

// handleAuthorizationCodeGrant handles authorization code grant
func (op *OAuth2Provider) handleAuthorizationCodeGrant(w http.ResponseWriter, r *http.Request, tokenReq *TokenRequest) {
	// Validate client
	client, exists := op.clients[tokenReq.ClientID]
	if !exists || client.ClientSecret != tokenReq.ClientSecret {
		op.jsonError(w, http.StatusUnauthorized, "invalid_client", "Invalid client credentials")
		return
	}

	// Retrieve and validate authorization code
	authCode, err := op.getAuthCode(r.Context(), tokenReq.Code)
	if err != nil {
		op.jsonError(w, http.StatusBadRequest, "invalid_grant", "Invalid authorization code")
		return
	}

	// Validate PKCE if present
	if authCode.CodeChallenge != "" {
		if !op.validatePKCE(authCode.CodeChallenge, tokenReq.CodeVerifier) {
			op.jsonError(w, http.StatusBadRequest, "invalid_grant", "Invalid PKCE verifier")
			return
		}
	}

	// Validate redirect URI
	if authCode.RedirectURI != tokenReq.RedirectURI {
		op.jsonError(w, http.StatusBadRequest, "invalid_grant", "Redirect URI mismatch")
		return
	}

	// Delete authorization code (single use)
	op.deleteAuthCode(r.Context(), tokenReq.Code)

	// Generate tokens
	scopes := strings.Split(authCode.Scope, " ")
	tokenPair, err := op.jwtManager.GenerateTokenPair(
		r.Context(),
		authCode.UserID,
		authCode.TenantID,
		"", // Email would be fetched from user store
		[]string{"user"},
		scopes,
	)
	if err != nil {
		op.jsonError(w, http.StatusInternalServerError, "server_error", "Failed to generate tokens")
		return
	}

	// Return token response
	response := &TokenResponse{
		AccessToken:  tokenPair.AccessToken,
		TokenType:    "Bearer",
		ExpiresIn:    int(op.jwtManager.accessTokenTTL.Seconds()),
		RefreshToken: tokenPair.RefreshToken,
		Scope:        authCode.Scope,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// handleRefreshTokenGrant handles refresh token grant
func (op *OAuth2Provider) handleRefreshTokenGrant(w http.ResponseWriter, r *http.Request, tokenReq *TokenRequest) {
	// Validate client
	client, exists := op.clients[tokenReq.ClientID]
	if !exists || client.ClientSecret != tokenReq.ClientSecret {
		op.jsonError(w, http.StatusUnauthorized, "invalid_client", "Invalid client credentials")
		return
	}

	// Refresh token
	tokenPair, err := op.jwtManager.RefreshToken(r.Context(), tokenReq.RefreshToken)
	if err != nil {
		op.jsonError(w, http.StatusBadRequest, "invalid_grant", "Invalid refresh token")
		return
	}

	// Return token response
	response := &TokenResponse{
		AccessToken:  tokenPair.AccessToken,
		TokenType:    "Bearer",
		ExpiresIn:    int(op.jwtManager.accessTokenTTL.Seconds()),
		RefreshToken: tokenPair.RefreshToken,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// generateAuthCode generates and stores an authorization code
func (op *OAuth2Provider) generateAuthCode(ctx context.Context, authReq *AuthorizationRequest, userID, tenantID string) (string, error) {
	code := generateRandomString(32)
	
	authCode := &AuthorizationCode{
		Code:          code,
		ClientID:      authReq.ClientID,
		RedirectURI:   authReq.RedirectURI,
		UserID:        userID,
		TenantID:      tenantID,
		Scope:         authReq.Scope,
		CodeChallenge: authReq.CodeChallenge,
		ExpiresAt:     time.Now().Add(10 * time.Minute),
	}

	data, err := json.Marshal(authCode)
	if err != nil {
		return "", err
	}

	key := "authcode:" + code
	err = op.authCodes.Set(ctx, key, data, 10*time.Minute).Err()
	if err != nil {
		return "", err
	}

	return code, nil
}

// getAuthCode retrieves an authorization code
func (op *OAuth2Provider) getAuthCode(ctx context.Context, code string) (*AuthorizationCode, error) {
	key := "authcode:" + code
	data, err := op.authCodes.Get(ctx, key).Bytes()
	if err != nil {
		return nil, err
	}

	var authCode AuthorizationCode
	err = json.Unmarshal(data, &authCode)
	if err != nil {
		return nil, err
	}

	if time.Now().After(authCode.ExpiresAt) {
		return nil, ErrInvalidAuthCode
	}

	return &authCode, nil
}

// deleteAuthCode deletes an authorization code
func (op *OAuth2Provider) deleteAuthCode(ctx context.Context, code string) error {
	key := "authcode:" + code
	return op.authCodes.Del(ctx, key).Err()
}

// validateRedirectURI validates redirect URI against registered URIs
func (op *OAuth2Provider) validateRedirectURI(client *OAuth2Client, redirectURI string) bool {
	for _, uri := range client.RedirectURIs {
		if uri == redirectURI {
			return true
		}
	}
	return false
}

// validatePKCE validates PKCE code verifier
func (op *OAuth2Provider) validatePKCE(codeChallenge, codeVerifier string) bool {
	// S256 method
	h := sha256.New()
	h.Write([]byte(codeVerifier))
	computed := base64.RawURLEncoding.EncodeToString(h.Sum(nil))
	return computed == codeChallenge
}

// Helper functions
func (op *OAuth2Provider) errorResponse(w http.ResponseWriter, r *http.Request, redirectURI, errorCode, description, state string) {
	if redirectURI == "" {
		http.Error(w, description, http.StatusBadRequest)
		return
	}

	u, _ := url.Parse(redirectURI)
	q := u.Query()
	q.Set("error", errorCode)
	q.Set("error_description", description)
	if state != "" {
		q.Set("state", state)
	}
	u.RawQuery = q.Encode()

	http.Redirect(w, r, u.String(), http.StatusFound)
}

func (op *OAuth2Provider) jsonError(w http.ResponseWriter, status int, errorCode, description string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(map[string]string{
		"error":             errorCode,
		"error_description": description,
	})
}

func generateRandomString(length int) string {
	b := make([]byte, length)
	rand.Read(b)
	return base64.RawURLEncoding.EncodeToString(b)
}
