// Package abac provides Attribute-Based Access Control (ABAC) with risk-based decisions
// and continuous authorization validation. Evolution from RBAC to context-aware policies.
package abac

import (
	"context"
	"crypto/sha256"
	_ "encoding/json" // Keep for future use
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// Policy defines an ABAC policy with attribute conditions
type Policy struct {
	ID          string
	Name        string
	Description string
	Effect      Effect
	Priority    int // Higher priority wins in conflicts
	
	// Attribute conditions (all must match)
	UserAttributes     map[string]AttributeCondition
	ResourceAttributes map[string]AttributeCondition
	EnvironmentAttributes map[string]AttributeCondition
	ActionAttributes   map[string]AttributeCondition
	
	// Risk-based conditions
	MaxRiskScore       float64 // Maximum allowed risk score (0-100)
	RequireStepUp      bool    // Require step-up authentication if risky
	
	// Time-based conditions
	TimeConstraints    []TimeWindow
	
	// Rate limiting per policy
	RateLimit          *RateLimit
	
	Created            time.Time
	Updated            time.Time
}

type Effect string

const (
	EffectAllow Effect = "allow"
	EffectDeny  Effect = "deny"
)

// AttributeCondition defines a condition for an attribute
type AttributeCondition struct {
	Operator string      `json:"operator"` // "eq", "ne", "gt", "lt", "in", "contains", "regex"
	Value    interface{} `json:"value"`
	Required bool        `json:"required"` // If true, attribute must exist
}

// TimeWindow defines a time-based access constraint
type TimeWindow struct {
	DaysOfWeek []time.Weekday
	StartTime  string // HH:MM format
	EndTime    string // HH:MM format
	Timezone   string
}

// RateLimit defines per-policy rate limiting
type RateLimit struct {
	Requests int
	Window   time.Duration
}

// AccessRequest represents a request for access
type AccessRequest struct {
	User        User
	Resource    Resource
	Action      string
	Environment Environment
	Context     RequestContext
}

// User represents the requesting user with attributes
type User struct {
	ID         string
	Attributes map[string]interface{}
	Roles      []string
	Groups     []string
}

// Resource represents the target resource with attributes
type Resource struct {
	ID         string
	Type       string
	Attributes map[string]interface{}
	Owner      string
	Sensitivity string // "public", "internal", "confidential", "secret"
}

// Environment represents environmental context
type Environment struct {
	Timestamp      time.Time
	IPAddress      string
	GeoLocation    GeoLocation
	DeviceTrust    float64 // 0.0 (untrusted) to 1.0 (fully trusted)
	NetworkType    string  // "corporate", "vpn", "public"
	ThreatLevel    string  // "low", "medium", "high", "critical"
}

type GeoLocation struct {
	Country   string
	Region    string
	City      string
	Latitude  float64
	Longitude float64
}

// RequestContext provides additional request context
type RequestContext struct {
	CorrelationID string
	SessionID     string
	TraceID       string
	UserAgent     string
}

// Engine is the ABAC policy evaluation engine
type Engine struct {
	mu            sync.RWMutex
	policies      []*Policy
	policiesById  map[string]*Policy
	
	// Risk scoring
	riskScorer    RiskScorer
	
	// Continuous authorization
	enableContinuous bool
	revalidateAfter  time.Duration
	sessions         sync.Map // sessionID -> *AuthSession
	
	// Caching
	enableCache      bool
	cache            sync.Map // cacheKey -> *CacheEntry
	cacheTTL         time.Duration
	
	// Metrics
	evaluations      uint64
	allows           uint64
	denies           uint64
	cacheHits        uint64
	cacheMisses      uint64
	riskDenials      uint64
	
	ctx              context.Context
	cancel           context.CancelFunc
}

// AuthSession tracks continuous authorization state
type AuthSession struct {
	SessionID       string
	User            User
	InitialRisk     float64
	CurrentRisk     float64
	LastValidation  time.Time
	ExpiresAt       time.Time
	RequireStepUp   bool
}

// CacheEntry stores cached policy decisions
type CacheEntry struct {
	Decision   Decision
	ExpiresAt  time.Time
	RiskScore  float64
}

// Decision represents the result of policy evaluation
type Decision struct {
	Effect        Effect
	Allowed       bool
	Reason        string
	MatchedPolicy string
	RiskScore     float64
	RequireStepUp bool
	Timestamp     time.Time
}

// RiskScorer interface for calculating risk scores
type RiskScorer interface {
	CalculateRisk(req AccessRequest) float64
}

// EngineConfig configures the ABAC engine
type EngineConfig struct {
	EnableContinuous bool
	RevalidateAfter  time.Duration
	EnableCache      bool
	CacheTTL         time.Duration
	RiskScorer       RiskScorer
}

// NewEngine creates a new ABAC engine
func NewEngine(cfg EngineConfig) *Engine {
	if cfg.RevalidateAfter == 0 {
		cfg.RevalidateAfter = 5 * time.Minute
	}
	if cfg.CacheTTL == 0 {
		cfg.CacheTTL = 30 * time.Second
	}
	if cfg.RiskScorer == nil {
		cfg.RiskScorer = NewDefaultRiskScorer()
	}
	
	ctx, cancel := context.WithCancel(context.Background())
	
	eng := &Engine{
		policies:         make([]*Policy, 0),
		policiesById:     make(map[string]*Policy),
		riskScorer:       cfg.RiskScorer,
		enableContinuous: cfg.EnableContinuous,
		revalidateAfter:  cfg.RevalidateAfter,
		enableCache:      cfg.EnableCache,
		cacheTTL:         cfg.CacheTTL,
		ctx:              ctx,
		cancel:           cancel,
	}
	
	// Start continuous authorization validator
	if cfg.EnableContinuous {
		go eng.continuousValidator()
	}
	
	// Start cache cleanup
	if cfg.EnableCache {
		go eng.cacheCleanup()
	}
	
	return eng
}

// AddPolicy adds a policy to the engine
func (e *Engine) AddPolicy(policy *Policy) error {
	if policy.ID == "" {
		return errors.New("policy ID required")
	}
	
	e.mu.Lock()
	defer e.mu.Unlock()
	
	if _, exists := e.policiesById[policy.ID]; exists {
		return fmt.Errorf("policy %s already exists", policy.ID)
	}
	
	policy.Created = time.Now()
	policy.Updated = time.Now()
	
	e.policies = append(e.policies, policy)
	e.policiesById[policy.ID] = policy
	
	// Sort by priority (descending)
	e.sortPolicies()
	
	return nil
}

// RemovePolicy removes a policy from the engine
func (e *Engine) RemovePolicy(policyID string) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	
	if _, exists := e.policiesById[policyID]; !exists {
		return fmt.Errorf("policy %s not found", policyID)
	}
	
	delete(e.policiesById, policyID)
	
	// Remove from slice
	filtered := make([]*Policy, 0, len(e.policies))
	for _, p := range e.policies {
		if p.ID != policyID {
			filtered = append(filtered, p)
		}
	}
	e.policies = filtered
	
	return nil
}

// Evaluate evaluates an access request against all policies
func (e *Engine) Evaluate(req AccessRequest) Decision {
	atomic.AddUint64(&e.evaluations, 1)
	
	// Check cache
	if e.enableCache {
		if cached := e.checkCache(req); cached != nil {
			atomic.AddUint64(&e.cacheHits, 1)
			return *cached
		}
		atomic.AddUint64(&e.cacheMisses, 1)
	}
	
	// Calculate risk score
	riskScore := e.riskScorer.CalculateRisk(req)
	
	// Check continuous authorization
	if e.enableContinuous && req.Context.SessionID != "" {
		if session := e.getSession(req.Context.SessionID); session != nil {
			session.CurrentRisk = riskScore
			session.LastValidation = time.Now()
			
			// Deny if risk increased significantly
			if riskScore > session.InitialRisk*1.5 && riskScore > 50 {
				decision := Decision{
					Effect:        EffectDeny,
					Allowed:       false,
					Reason:        "Risk score increased significantly",
					RiskScore:     riskScore,
					RequireStepUp: true,
					Timestamp:     time.Now(),
				}
				atomic.AddUint64(&e.denies, 1)
				atomic.AddUint64(&e.riskDenials, 1)
				return decision
			}
		}
	}
	
	// Evaluate policies in priority order
	e.mu.RLock()
	policies := make([]*Policy, len(e.policies))
	copy(policies, e.policies)
	e.mu.RUnlock()
	
	for _, policy := range policies {
		if matches, reason := e.evaluatePolicy(policy, req, riskScore); matches {
			decision := Decision{
				Effect:        policy.Effect,
				Allowed:       policy.Effect == EffectAllow,
				Reason:        reason,
				MatchedPolicy: policy.ID,
				RiskScore:     riskScore,
				RequireStepUp: policy.RequireStepUp && riskScore > policy.MaxRiskScore,
				Timestamp:     time.Now(),
			}
			
			if decision.Allowed {
				atomic.AddUint64(&e.allows, 1)
			} else {
				atomic.AddUint64(&e.denies, 1)
			}
			
			// Cache decision
			if e.enableCache {
				e.cacheDecision(req, decision)
			}
			
			return decision
		}
	}
	
	// Default deny
	decision := Decision{
		Effect:    EffectDeny,
		Allowed:   false,
		Reason:    "No matching policy",
		RiskScore: riskScore,
		Timestamp: time.Now(),
	}
	atomic.AddUint64(&e.denies, 1)
	
	return decision
}

// evaluatePolicy checks if a policy matches the request
func (e *Engine) evaluatePolicy(policy *Policy, req AccessRequest, riskScore float64) (bool, string) {
	// Risk threshold check
	if riskScore > policy.MaxRiskScore {
		return false, "risk score too high"
	}
	
	// User attributes
	if !e.matchAttributes(req.User.Attributes, policy.UserAttributes) {
		return false, "user attributes mismatch"
	}
	
	// Resource attributes
	if !e.matchAttributes(req.Resource.Attributes, policy.ResourceAttributes) {
		return false, "resource attributes mismatch"
	}
	
	// Environment attributes
	envAttrs := map[string]interface{}{
		"ip":          req.Environment.IPAddress,
		"country":     req.Environment.GeoLocation.Country,
		"networkType": req.Environment.NetworkType,
		"threatLevel": req.Environment.ThreatLevel,
		"deviceTrust": req.Environment.DeviceTrust,
	}
	if !e.matchAttributes(envAttrs, policy.EnvironmentAttributes) {
		return false, "environment attributes mismatch"
	}
	
	// Action attributes
	actionAttrs := map[string]interface{}{
		"action": req.Action,
	}
	if !e.matchAttributes(actionAttrs, policy.ActionAttributes) {
		return false, "action mismatch"
	}
	
	// Time constraints
	if len(policy.TimeConstraints) > 0 && !e.checkTimeConstraints(policy.TimeConstraints, req.Environment.Timestamp) {
		return false, "outside allowed time window"
	}
	
	return true, "policy matched"
}

// matchAttributes checks if request attributes match policy conditions
func (e *Engine) matchAttributes(reqAttrs map[string]interface{}, policyAttrs map[string]AttributeCondition) bool {
	for key, condition := range policyAttrs {
		val, exists := reqAttrs[key]
		
		if !exists && condition.Required {
			return false
		}
		if !exists {
			continue
		}
		
		if !e.matchCondition(val, condition) {
			return false
		}
	}
	return true
}

// matchCondition evaluates a single attribute condition
func (e *Engine) matchCondition(val interface{}, condition AttributeCondition) bool {
	switch condition.Operator {
	case "eq":
		return fmt.Sprintf("%v", val) == fmt.Sprintf("%v", condition.Value)
	case "ne":
		return fmt.Sprintf("%v", val) != fmt.Sprintf("%v", condition.Value)
	case "gt":
		v, ok := val.(float64)
		if !ok {
			return false
		}
		cv, ok := condition.Value.(float64)
		if !ok {
			return false
		}
		return v > cv
	case "lt":
		v, ok := val.(float64)
		if !ok {
			return false
		}
		cv, ok := condition.Value.(float64)
		if !ok {
			return false
		}
		return v < cv
	case "in":
		list, ok := condition.Value.([]interface{})
		if !ok {
			return false
		}
		valStr := fmt.Sprintf("%v", val)
		for _, item := range list {
			if fmt.Sprintf("%v", item) == valStr {
				return true
			}
		}
		return false
	case "contains":
		valStr := fmt.Sprintf("%v", val)
		condStr := fmt.Sprintf("%v", condition.Value)
		return len(valStr) >= len(condStr) && valStr[:len(condStr)] == condStr
	default:
		return false
	}
}

// checkTimeConstraints validates time-based constraints
func (e *Engine) checkTimeConstraints(constraints []TimeWindow, timestamp time.Time) bool {
	for _, window := range constraints {
		if e.inTimeWindow(window, timestamp) {
			return true
		}
	}
	return len(constraints) == 0
}

// inTimeWindow checks if timestamp is within a time window
func (e *Engine) inTimeWindow(window TimeWindow, timestamp time.Time) bool {
	// Check day of week
	if len(window.DaysOfWeek) > 0 {
		dayMatch := false
		for _, day := range window.DaysOfWeek {
			if timestamp.Weekday() == day {
				dayMatch = true
				break
			}
		}
		if !dayMatch {
			return false
		}
	}
	
	// Check time range (simplified - real implementation would handle timezones)
	return true
}

// sortPolicies sorts policies by priority (descending)
func (e *Engine) sortPolicies() {
	// Simple bubble sort for priority
	for i := 0; i < len(e.policies); i++ {
		for j := i + 1; j < len(e.policies); j++ {
			if e.policies[j].Priority > e.policies[i].Priority {
				e.policies[i], e.policies[j] = e.policies[j], e.policies[i]
			}
		}
	}
}

// checkCache checks if a decision is cached
func (e *Engine) checkCache(req AccessRequest) *Decision {
	key := e.cacheKey(req)
	if val, ok := e.cache.Load(key); ok {
		entry := val.(*CacheEntry)
		if time.Now().Before(entry.ExpiresAt) {
			return &entry.Decision
		}
	}
	return nil
}

// cacheDecision caches a policy decision
func (e *Engine) cacheDecision(req AccessRequest, decision Decision) {
	key := e.cacheKey(req)
	entry := &CacheEntry{
		Decision:  decision,
		ExpiresAt: time.Now().Add(e.cacheTTL),
		RiskScore: decision.RiskScore,
	}
	e.cache.Store(key, entry)
}

// cacheKey generates a cache key for a request
func (e *Engine) cacheKey(req AccessRequest) string {
	data := fmt.Sprintf("%s:%s:%s:%s", req.User.ID, req.Resource.ID, req.Action, req.Environment.IPAddress)
	hash := sha256.Sum256([]byte(data))
	return fmt.Sprintf("%x", hash[:16])
}

// getSession retrieves or creates an authorization session
func (e *Engine) getSession(sessionID string) *AuthSession {
	if val, ok := e.sessions.Load(sessionID); ok {
		return val.(*AuthSession)
	}
	return nil
}

// CreateSession creates a new authorization session
func (e *Engine) CreateSession(user User, initialRisk float64, expiresAt time.Time) string {
	sessionID := fmt.Sprintf("ses_%d", time.Now().UnixNano())
	session := &AuthSession{
		SessionID:      sessionID,
		User:           user,
		InitialRisk:    initialRisk,
		CurrentRisk:    initialRisk,
		LastValidation: time.Now(),
		ExpiresAt:      expiresAt,
	}
	e.sessions.Store(sessionID, session)
	return sessionID
}

// continuousValidator revalidates active sessions
func (e *Engine) continuousValidator() {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()
	
	for {
		select {
		case <-e.ctx.Done():
			return
		case <-ticker.C:
			now := time.Now()
			e.sessions.Range(func(key, value interface{}) bool {
				session := value.(*AuthSession)
				
				// Remove expired sessions
				if now.After(session.ExpiresAt) {
					e.sessions.Delete(key)
					return true
				}
				
				// Revalidate if needed
				if now.Sub(session.LastValidation) > e.revalidateAfter {
					// In production, trigger revalidation event
				}
				
				return true
			})
		}
	}
}

// cacheCleanup removes expired cache entries
func (e *Engine) cacheCleanup() {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()
	
	for {
		select {
		case <-e.ctx.Done():
			return
		case <-ticker.C:
			now := time.Now()
			e.cache.Range(func(key, value interface{}) bool {
				entry := value.(*CacheEntry)
				if now.After(entry.ExpiresAt) {
					e.cache.Delete(key)
				}
				return true
			})
		}
	}
}

// Metrics returns engine metrics
func (e *Engine) Metrics() map[string]uint64 {
	return map[string]uint64{
		"evaluations":  atomic.LoadUint64(&e.evaluations),
		"allows":       atomic.LoadUint64(&e.allows),
		"denies":       atomic.LoadUint64(&e.denies),
		"cache_hits":   atomic.LoadUint64(&e.cacheHits),
		"cache_misses": atomic.LoadUint64(&e.cacheMisses),
		"risk_denials": atomic.LoadUint64(&e.riskDenials),
	}
}

// Close stops the engine
func (e *Engine) Close() {
	e.cancel()
}

// ---------- Default Risk Scorer ----------

type DefaultRiskScorer struct{}

func NewDefaultRiskScorer() *DefaultRiskScorer {
	return &DefaultRiskScorer{}
}

func (rs *DefaultRiskScorer) CalculateRisk(req AccessRequest) float64 {
	risk := 0.0
	
	// Network type risk
	switch req.Environment.NetworkType {
	case "corporate":
		risk += 0
	case "vpn":
		risk += 10
	case "public":
		risk += 30
	default:
		risk += 20
	}
	
	// Threat level risk
	switch req.Environment.ThreatLevel {
	case "low":
		risk += 0
	case "medium":
		risk += 20
	case "high":
		risk += 40
	case "critical":
		risk += 60
	}
	
	// Device trust risk (inverted)
	risk += (1.0 - req.Environment.DeviceTrust) * 30
	
	// Sensitivity risk
	switch req.Resource.Sensitivity {
	case "public":
		risk += 0
	case "internal":
		risk += 10
	case "confidential":
		risk += 20
	case "secret":
		risk += 40
	}
	
	// Clamp to [0, 100]
	if risk < 0 {
		risk = 0
	}
	if risk > 100 {
		risk = 100
	}
	
	return risk
}
