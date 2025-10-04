// Package orchestrator - Person 1 Phase 2-3 Advanced Enhancements
// AI-Powered Traffic Intelligence + Next-Gen Policy Engine
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"shieldx/pkg/analytics"
	"shieldx/pkg/graphql"
	"shieldx/pkg/policy"
	"shieldx/pkg/ratelimit"
)

// Phase2Enhancement implements AI-Powered Traffic Intelligence
type Phase2Enhancement struct {
	// Real-time behavioral analysis
	analyticsEngine *analytics.AnalyticsEngine

	// Adaptive rate limiting
	adaptiveLimiter *ratelimit.AdaptiveLimiter

	// GraphQL security
	graphqlSecurity *graphql.SecurityMiddleware

	// Metrics
	behavioralAnomalies   uint64
	ddosBlocked           uint64
	botsBlocked           uint64
	graphqlQueriesLimited uint64
}

// Phase3Enhancement implements Next-Gen Policy Engine
type Phase3Enhancement struct {
	// Dynamic policy engine
	policyEngine *policy.DynamicEngine

	// Risk-based access control (ABAC)
	abacEngine *ABACEngine

	// Policy A/B testing
	abTestFramework *PolicyABTest

	// Metrics
	policyEvaluations uint64
	abacDecisions     uint64
	abTestsRun        uint64
	hotReloads        uint64
}

// ABACEngine implements Attribute-Based Access Control
type ABACEngine struct {
	mu sync.RWMutex

	// Attribute policies
	policies []ABACPolicy

	// Real-time risk scoring
	riskScorer *RiskScorer

	// Continuous authorization validator
	authValidator *ContinuousAuthValidator

	// Metrics
	evaluations    uint64
	allowDecisions uint64
	denyDecisions  uint64
}

// ABACPolicy defines attribute-based access control policy
type ABACPolicy struct {
	ID         string
	Priority   int
	Conditions ABACConditions
	Action     string // "allow", "deny", "mfa_required"
}

// ABACConditions defines conditions for ABAC policy
type ABACConditions struct {
	// User attributes
	UserRole       []string
	UserDepartment []string
	UserLocation   []string

	// Resource attributes
	ResourceType    []string
	DataSensitivity []string

	// Environment attributes
	TimeOfDay        TimeRange
	NetworkLocation  []string
	DeviceTrustLevel []string

	// Action attributes
	ActionType []string

	// Behavioral attributes
	RiskScore RiskRange
}

type TimeRange struct {
	Start string // "09:00"
	End   string // "17:00"
}

type RiskRange struct {
	Min float64
	Max float64
}

// RiskScorer calculates real-time risk scores
type RiskScorer struct {
	mu sync.RWMutex

	// Behavioral baselines
	baselines map[string]*BehavioralBaseline

	// Real-time metrics
	recentActivity map[string][]ActivityEvent

	// Scoring model parameters
	weights RiskWeights
}

type BehavioralBaseline struct {
	UserID                 string
	TypicalLocations       []string
	TypicalDevices         []string
	TypicalHours           []int // 0-23
	AverageRequestsPerHour float64
	TypicalEndpoints       []string

	LastUpdated time.Time
}

type ActivityEvent struct {
	Timestamp time.Time
	UserID    string
	Location  string
	Device    string
	Endpoint  string
	Success   bool
}

type RiskWeights struct {
	UnusualLocation float64
	UnusualDevice   float64
	UnusualTime     float64
	RapidRequests   float64
	FailedAuth      float64
	NewEndpoint     float64
}

// ContinuousAuthValidator validates authorization continuously
type ContinuousAuthValidator struct {
	mu sync.RWMutex

	// Active sessions
	sessions map[string]*Session

	// Validation interval
	interval time.Duration

	// Adaptive requirements
	mfaThreshold float64 // Risk score threshold for MFA challenge

	// Metrics
	validations   uint64
	revocations   uint64
	mfaChallenges uint64
}

type Session struct {
	ID            string
	UserID        string
	CreatedAt     time.Time
	LastValidated time.Time
	RiskScore     float64
	Attributes    map[string]string

	// Continuous monitoring
	ActivityCount   uint64
	SuspiciousCount uint64
}

// PolicyABTest implements A/B testing for security policies
type PolicyABTest struct {
	mu sync.RWMutex

	// Active experiments
	experiments map[string]*Experiment

	// Traffic split (0.0-1.0)
	testTrafficPct float64

	// Metrics collection
	metrics map[string]*ExperimentMetrics

	// Auto-rollback on degradation
	autoRollback         bool
	degradationThreshold float64
}

type Experiment struct {
	ID          string
	Name        string
	Description string

	// Policy variants
	ControlPolicy policy.Config
	TestPolicy    policy.Config

	// Traffic allocation
	TrafficPct float64

	// Success criteria
	SuccessCriteria SuccessCriteria

	// Status
	Status    string // "running", "completed", "rolled_back"
	StartedAt time.Time
	EndedAt   *time.Time
}

type SuccessCriteria struct {
	MinSampleSize int
	MaxDuration   time.Duration

	// Target metrics
	TargetBlockRate    float64
	MaxLatencyIncrease float64
	MinThroughput      float64
}

type ExperimentMetrics struct {
	ControlGroup GroupMetrics
	TestGroup    GroupMetrics

	// Statistical significance
	PValue      float64
	Significant bool
}

type GroupMetrics struct {
	Requests     uint64
	Blocked      uint64
	AvgLatencyMs float64
	ErrorRate    float64
	Throughput   float64
}

// InitPhase2 initializes Phase 2 enhancements
func InitPhase2() (*Phase2Enhancement, error) {
	// 1. Initialize behavioral analysis engine
	analyticsEngine := analytics.NewAnalyticsEngine(analytics.EngineConfig{
		EventBufferSize:     10000,
		WindowSize:          1440, // 24 hours
		AggregationInterval: 1 * time.Minute,
		AnomalyThreshold:    3.0, // 3-sigma
	})

	// 2. Initialize adaptive rate limiter with ML
	adaptiveLimiter := ratelimit.NewAdaptiveLimiter(ratelimit.LimiterConfig{
		BasePolicies: []ratelimit.RatePolicy{
			{Dimension: ratelimit.DimIP, Limit: 200, Window: time.Minute, BurstSize: 50},
			{Dimension: ratelimit.DimUser, Limit: 500, Window: time.Minute, BurstSize: 100},
			{Dimension: ratelimit.DimEndpoint, Limit: 1000, Window: time.Minute, BurstSize: 200},
		},
		EnableML:        true,
		EnableGeo:       true,
		AdjustmentCycle: 5 * time.Minute,
	})

	// 3. Initialize GraphQL security
	graphqlSecurity := graphql.NewSecurityMiddleware(graphql.SecurityConfig{
		MaxDepth:              10,
		MaxComplexity:         1000,
		MaxAliases:            15,
		DisableIntrospection:  true,
		QueryTimeout:          30 * time.Second,
		PersistentQueriesOnly: false,
	})

	log.Printf("[phase2] AI-Powered Traffic Intelligence initialized")

	return &Phase2Enhancement{
		analyticsEngine: analyticsEngine,
		adaptiveLimiter: adaptiveLimiter,
		graphqlSecurity: graphqlSecurity,
	}, nil
}

// InitPhase3 initializes Phase 3 enhancements
func InitPhase3(ctx context.Context) (*Phase3Enhancement, error) {
	// 1. Initialize ABAC engine
	abacEngine := NewABACEngine()

	// 2. Initialize A/B testing framework
	abTestFramework := NewPolicyABTest(0.1) // 10% test traffic

	// 3. Load initial ABAC policies
	abacEngine.LoadDefaultPolicies()

	log.Printf("[phase3] Next-Gen Policy Engine initialized")

	return &Phase3Enhancement{
		abacEngine:      abacEngine,
		abTestFramework: abTestFramework,
	}, nil
}

// NewABACEngine creates a new ABAC engine
func NewABACEngine() *ABACEngine {
	return &ABACEngine{
		policies:      make([]ABACPolicy, 0),
		riskScorer:    NewRiskScorer(),
		authValidator: NewContinuousAuthValidator(5*time.Minute, 0.7),
	}
}

// LoadDefaultPolicies loads default ABAC policies
func (a *ABACEngine) LoadDefaultPolicies() {
	// High-risk: require MFA
	a.policies = append(a.policies, ABACPolicy{
		ID:       "high-risk-mfa",
		Priority: 100,
		Conditions: ABACConditions{
			RiskScore: RiskRange{Min: 0.7, Max: 1.0},
		},
		Action: "mfa_required",
	})

	// After-hours + sensitive data: deny
	a.policies = append(a.policies, ABACPolicy{
		ID:       "after-hours-sensitive",
		Priority: 90,
		Conditions: ABACConditions{
			TimeOfDay:       TimeRange{Start: "18:00", End: "08:00"},
			DataSensitivity: []string{"confidential", "restricted"},
		},
		Action: "deny",
	})

	// Unknown location + new device: MFA required
	a.policies = append(a.policies, ABACPolicy{
		ID:       "unknown-location-device",
		Priority: 80,
		Conditions: ABACConditions{
			DeviceTrustLevel: []string{"unknown", "untrusted"},
		},
		Action: "mfa_required",
	})

	// Admin role + sensitive action: allow with audit
	a.policies = append(a.policies, ABACPolicy{
		ID:       "admin-sensitive",
		Priority: 70,
		Conditions: ABACConditions{
			UserRole:   []string{"admin", "security_admin"},
			ActionType: []string{"delete", "modify_permissions"},
		},
		Action: "allow",
	})
}

// Evaluate evaluates ABAC policy for a request
func (a *ABACEngine) Evaluate(ctx context.Context, req *ABACRequest) (string, error) {
	atomic.AddUint64(&a.evaluations, 1)

	// Calculate real-time risk score
	riskScore := a.riskScorer.CalculateRisk(req)

	// Find matching policy (highest priority wins)
	a.mu.RLock()
	defer a.mu.RUnlock()

	for _, policy := range a.policies {
		if a.matchesConditions(req, policy.Conditions, riskScore) {
			if policy.Action == "allow" {
				atomic.AddUint64(&a.allowDecisions, 1)
			} else {
				atomic.AddUint64(&a.denyDecisions, 1)
			}
			return policy.Action, nil
		}
	}

	// Default deny
	atomic.AddUint64(&a.denyDecisions, 1)
	return "deny", nil
}

// ABACRequest contains attributes for ABAC evaluation
type ABACRequest struct {
	UserID         string
	UserRole       string
	UserDepartment string
	UserLocation   string

	ResourceType    string
	ResourceID      string
	DataSensitivity string

	ActionType string

	DeviceID         string
	DeviceTrustLevel string

	Timestamp       time.Time
	NetworkLocation string
}

// matchesConditions checks if request matches policy conditions
func (a *ABACEngine) matchesConditions(req *ABACRequest, cond ABACConditions, riskScore float64) bool {
	// Check user role
	if len(cond.UserRole) > 0 && !contains(cond.UserRole, req.UserRole) {
		return false
	}

	// Check data sensitivity
	if len(cond.DataSensitivity) > 0 && !contains(cond.DataSensitivity, req.DataSensitivity) {
		return false
	}

	// Check device trust level
	if len(cond.DeviceTrustLevel) > 0 && !contains(cond.DeviceTrustLevel, req.DeviceTrustLevel) {
		return false
	}

	// Check risk score range
	if cond.RiskScore.Min > 0 || cond.RiskScore.Max > 0 {
		if riskScore < cond.RiskScore.Min || riskScore > cond.RiskScore.Max {
			return false
		}
	}

	// Check time of day
	if cond.TimeOfDay.Start != "" {
		currentHour := req.Timestamp.Format("15:04")
		if !isInTimeRange(currentHour, cond.TimeOfDay.Start, cond.TimeOfDay.End) {
			return false
		}
	}

	return true
}

// NewRiskScorer creates a new risk scorer
func NewRiskScorer() *RiskScorer {
	return &RiskScorer{
		baselines:      make(map[string]*BehavioralBaseline),
		recentActivity: make(map[string][]ActivityEvent),
		weights: RiskWeights{
			UnusualLocation: 0.3,
			UnusualDevice:   0.3,
			UnusualTime:     0.1,
			RapidRequests:   0.15,
			FailedAuth:      0.10,
			NewEndpoint:     0.05,
		},
	}
}

// CalculateRisk calculates real-time risk score (0.0-1.0)
func (r *RiskScorer) CalculateRisk(req *ABACRequest) float64 {
	r.mu.RLock()
	defer r.mu.RUnlock()

	baseline, ok := r.baselines[req.UserID]
	if !ok {
		// New user = medium risk
		return 0.5
	}

	risk := 0.0

	// Factor 1: Unusual location
	if !contains(baseline.TypicalLocations, req.UserLocation) {
		risk += r.weights.UnusualLocation
	}

	// Factor 2: Unusual device
	if !contains(baseline.TypicalDevices, req.DeviceID) {
		risk += r.weights.UnusualDevice
	}

	// Factor 3: Unusual time
	currentHour := req.Timestamp.Hour()
	if !containsInt(baseline.TypicalHours, currentHour) {
		risk += r.weights.UnusualTime
	}

	// Factor 4: Rapid requests (check recent activity)
	if activity, ok := r.recentActivity[req.UserID]; ok {
		recentCount := 0
		cutoff := req.Timestamp.Add(-5 * time.Minute)
		for _, evt := range activity {
			if evt.Timestamp.After(cutoff) {
				recentCount++
			}
		}

		if float64(recentCount) > baseline.AverageRequestsPerHour/12 {
			risk += r.weights.RapidRequests
		}
	}

	// Clamp to [0, 1]
	if risk > 1.0 {
		risk = 1.0
	}

	return risk
}

// NewContinuousAuthValidator creates a new continuous auth validator
func NewContinuousAuthValidator(interval time.Duration, mfaThreshold float64) *ContinuousAuthValidator {
	return &ContinuousAuthValidator{
		sessions:     make(map[string]*Session),
		interval:     interval,
		mfaThreshold: mfaThreshold,
	}
}

// ValidateSession validates a session continuously
func (c *ContinuousAuthValidator) ValidateSession(sessionID string, currentRisk float64) (bool, string) {
	c.mu.RLock()
	session, ok := c.sessions[sessionID]
	c.mu.RUnlock()

	if !ok {
		return false, "session_not_found"
	}

	atomic.AddUint64(&c.validations, 1)

	// Check if risk exceeds MFA threshold
	if currentRisk > c.mfaThreshold {
		atomic.AddUint64(&c.mfaChallenges, 1)
		return false, "mfa_required"
	}

	// Check if too many suspicious activities
	if session.SuspiciousCount > session.ActivityCount/10 {
		atomic.AddUint64(&c.revocations, 1)
		return false, "revoked"
	}

	// Update session
	c.mu.Lock()
	session.LastValidated = time.Now()
	session.RiskScore = currentRisk
	c.mu.Unlock()

	return true, "valid"
}

// NewPolicyABTest creates a new policy A/B testing framework
func NewPolicyABTest(testTrafficPct float64) *PolicyABTest {
	return &PolicyABTest{
		experiments:          make(map[string]*Experiment),
		testTrafficPct:       testTrafficPct,
		metrics:              make(map[string]*ExperimentMetrics),
		autoRollback:         true,
		degradationThreshold: 0.1, // 10% performance degradation triggers rollback
	}
}

// StartExperiment starts a new A/B test
func (p *PolicyABTest) StartExperiment(exp *Experiment) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if _, exists := p.experiments[exp.ID]; exists {
		return fmt.Errorf("experiment already running: %s", exp.ID)
	}

	exp.Status = "running"
	exp.StartedAt = time.Now()

	p.experiments[exp.ID] = exp
	p.metrics[exp.ID] = &ExperimentMetrics{}

	log.Printf("[ab-test] Started experiment: %s", exp.Name)
	return nil
}

// ---------- Helper Functions ----------

func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

func containsInt(slice []int, item int) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

func isInTimeRange(current, start, end string) bool {
	return current >= start && current <= end
}

// Metrics endpoints
func handlePhase2Metrics(w http.ResponseWriter, r *http.Request, p2 *Phase2Enhancement) {
	metrics := map[string]interface{}{
		"analytics":            p2.analyticsEngine.Metrics(),
		"ratelimit":            p2.adaptiveLimiter.Metrics(),
		"behavioral_anomalies": atomic.LoadUint64(&p2.behavioralAnomalies),
		"ddos_blocked":         atomic.LoadUint64(&p2.ddosBlocked),
		"bots_blocked":         atomic.LoadUint64(&p2.botsBlocked),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(metrics)
}

func handlePhase3Metrics(w http.ResponseWriter, r *http.Request, p3 *Phase3Enhancement) {
	metrics := map[string]interface{}{
		"abac_evaluations": atomic.LoadUint64(&p3.abacEngine.evaluations),
		"abac_allow":       atomic.LoadUint64(&p3.abacEngine.allowDecisions),
		"abac_deny":        atomic.LoadUint64(&p3.abacEngine.denyDecisions),
		"ab_tests_run":     atomic.LoadUint64(&p3.abTestsRun),
		"hot_reloads":      atomic.LoadUint64(&p3.hotReloads),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(metrics)
}
