package core

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"
)

type ContAuthClient struct {
	baseURL    string
	httpClient *http.Client
}

type AuthDecision struct {
	SessionID  string    `json:"session_id"`
	Action     string    `json:"action"`
	Confidence float64   `json:"confidence"`
	Reason     string    `json:"reason"`
	Challenge  string    `json:"challenge,omitempty"`
	ExpiresAt  time.Time `json:"expires_at"`
}

type RiskScoreRequest struct {
	SessionID string `json:"session_id"`
}

type RiskScore struct {
	SessionID       string    `json:"session_id"`
	OverallScore    float64   `json:"overall_score"`
	KeystrokeScore  float64   `json:"keystroke_score"`
	MouseScore      float64   `json:"mouse_score"`
	LocationScore   float64   `json:"location_score"`
	DeviceScore     float64   `json:"device_score"`
	BehaviorScore   float64   `json:"behavior_score"`
	ReputationScore float64   `json:"reputation_score"`
	RiskFactors     []string  `json:"risk_factors"`
	Recommendation  string    `json:"recommendation"`
	CalculatedAt    time.Time `json:"calculated_at"`
}

func NewContAuthClient(baseURL string) *ContAuthClient {
	return &ContAuthClient{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 5 * time.Second,
		},
	}
}

func (c *ContAuthClient) GetAuthDecision(sessionID string) (*AuthDecision, error) {
	url := fmt.Sprintf("%s/contauth/decision?session_id=%s", c.baseURL, sessionID)

	resp, err := c.httpClient.Get(url)
	if err != nil {
		return nil, fmt.Errorf("failed to get auth decision: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("contauth returned status %d", resp.StatusCode)
	}

	var decision AuthDecision
	if err := json.NewDecoder(resp.Body).Decode(&decision); err != nil {
		return nil, fmt.Errorf("failed to decode auth decision: %w", err)
	}

	return &decision, nil
}

func (c *ContAuthClient) CalculateRiskScore(sessionID string) (*RiskScore, error) {
	request := RiskScoreRequest{SessionID: sessionID}
	jsonData, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/contauth/score", c.baseURL)
	resp, err := c.httpClient.Post(url, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to calculate risk score: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("contauth returned status %d", resp.StatusCode)
	}

	var riskScore RiskScore
	if err := json.NewDecoder(resp.Body).Decode(&riskScore); err != nil {
		return nil, fmt.Errorf("failed to decode risk score: %w", err)
	}

	return &riskScore, nil
}

// Orchestrator integration
type Orchestrator struct {
	contAuthClient *ContAuthClient
	// ... other fields
}

func (o *Orchestrator) InitContAuth(contAuthURL string) {
	o.contAuthClient = NewContAuthClient(contAuthURL)
	log.Printf("ContAuth client initialized with URL: %s", contAuthURL)
}

func (o *Orchestrator) RouteWithContAuth(sessionID string, request *http.Request) error {
	if o.contAuthClient == nil {
		log.Printf("ContAuth client not initialized, allowing request")
		return nil
	}

	decision, err := o.contAuthClient.GetAuthDecision(sessionID)
	if err != nil {
		log.Printf("ContAuth error for session %s: %v", sessionID, err)
		// Fail open - allow request if ContAuth is unavailable
		return nil
	}

	log.Printf("ContAuth decision for session %s: %s (confidence: %.2f)",
		sessionID, decision.Action, decision.Confidence)

	switch decision.Action {
	case "block":
		return fmt.Errorf("access blocked by continuous authentication: %s", decision.Reason)
	case "challenge":
		return o.issueMFAChallenge(sessionID, decision.Challenge)
	case "monitor":
		o.increaseMonitoring(sessionID)
		return nil
	case "allow":
		return nil
	default:
		log.Printf("Unknown ContAuth action: %s", decision.Action)
		return nil
	}
}

func (o *Orchestrator) issueMFAChallenge(sessionID, challengeType string) error {
	log.Printf("Issuing MFA challenge for session %s: %s", sessionID, challengeType)

	switch challengeType {
	case "mfa_required":
		return fmt.Errorf("multi-factor authentication required")
	case "captcha_required":
		return fmt.Errorf("captcha verification required")
	case "device_verification":
		return fmt.Errorf("device verification required")
	default:
		return fmt.Errorf("additional authentication required")
	}
}

func (o *Orchestrator) increaseMonitoring(sessionID string) {
	log.Printf("Increasing monitoring for session %s", sessionID)
	// Implementation would:
	// - Enable detailed logging for this session
	// - Increase telemetry collection frequency
	// - Add session to watchlist
	// - Notify security team if configured
}

// Middleware for HTTP handlers
func (o *Orchestrator) ContAuthMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		sessionID := o.extractSessionID(r)
		if sessionID == "" {
			log.Printf("No session ID found in request")
			next(w, r)
			return
		}

		if err := o.RouteWithContAuth(sessionID, r); err != nil {
			log.Printf("ContAuth blocked request for session %s: %v", sessionID, err)

			// Return appropriate HTTP status based on error
			if err.Error() == "access blocked by continuous authentication" {
				http.Error(w, "Access denied", http.StatusForbidden)
			} else if err.Error() == "multi-factor authentication required" {
				w.Header().Set("WWW-Authenticate", "Bearer realm=\"MFA Required\"")
				http.Error(w, "MFA Required", http.StatusUnauthorized)
			} else {
				http.Error(w, "Authentication required", http.StatusUnauthorized)
			}
			return
		}

		next(w, r)
	}
}

func (o *Orchestrator) extractSessionID(r *http.Request) string {
	// Try to get session ID from various sources

	// 1. Authorization header
	if auth := r.Header.Get("Authorization"); auth != "" {
		// Extract session ID from Bearer token or custom format
		if len(auth) > 7 && auth[:7] == "Bearer " {
			return auth[7:] // Return token as session ID
		}
	}

	// 2. Custom session header
	if sessionID := r.Header.Get("X-Session-ID"); sessionID != "" {
		return sessionID
	}

	// 3. Cookie
	if cookie, err := r.Cookie("session_id"); err == nil {
		return cookie.Value
	}

	// 4. Query parameter
	if sessionID := r.URL.Query().Get("session_id"); sessionID != "" {
		return sessionID
	}

	return ""
}

// Health check for ContAuth integration
func (o *Orchestrator) CheckContAuthHealth() error {
	if o.contAuthClient == nil {
		return fmt.Errorf("ContAuth client not initialized")
	}

	// Try to make a test request
	url := fmt.Sprintf("%s/health", o.contAuthClient.baseURL)
	resp, err := o.contAuthClient.httpClient.Get(url)
	if err != nil {
		return fmt.Errorf("ContAuth health check failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("ContAuth health check returned status %d", resp.StatusCode)
	}

	return nil
}

// Metrics collection
type ContAuthMetrics struct {
	TotalRequests      int64   `json:"total_requests"`
	BlockedRequests    int64   `json:"blocked_requests"`
	ChallengedRequests int64   `json:"challenged_requests"`
	MonitoredRequests  int64   `json:"monitored_requests"`
	AllowedRequests    int64   `json:"allowed_requests"`
	AverageRiskScore   float64 `json:"average_risk_score"`
	ErrorRate          float64 `json:"error_rate"`
}

func (o *Orchestrator) GetContAuthMetrics() *ContAuthMetrics {
	// Implementation would collect and return metrics
	// This is a placeholder structure
	return &ContAuthMetrics{
		TotalRequests:      1000,
		BlockedRequests:    50,
		ChallengedRequests: 100,
		MonitoredRequests:  200,
		AllowedRequests:    650,
		AverageRiskScore:   0.3,
		ErrorRate:          0.02,
	}
}
