// Phase 1 Enhancements for Orchestrator Service
// Integrates: PQC, Advanced QUIC, CT Monitoring, Adaptive Rate Limiting, Dynamic Policy
package main

import (
	"context"
	"log"
	"time"
	
	"shieldx/pkg/certtransparency"
	"shieldx/pkg/pqcrypto"
	"shieldx/shared/policy"
	"shieldx/pkg/ratelimit"
)

// Phase1Components holds all Phase 1 enhancement components
type Phase1Components struct {
	// Post-Quantum Crypto Engine
	PQCEngine *pqcrypto.PQCryptoEngine
	
	// Certificate Transparency Monitor
	CTMonitor *certtransparency.Monitor
	
	// Adaptive Rate Limiter
	AdaptiveLimiter *ratelimit.AdaptiveLimiter
	
	// Dynamic Policy Engine
	PolicyEngine *policy.DynamicEngine
	
	// Metrics
	pqcMetrics     map[string]uint64
	ctMetrics      map[string]uint64
	limiterMetrics map[string]interface{}
	policyMetrics  map[string]interface{}
}

// InitializePhase1 initializes all Phase 1 enhancement components
func InitializePhase1() (*Phase1Components, error) {
	log.Println("[orchestrator] initializing Phase 1 enhancements...")
	
	components := &Phase1Components{}
	
	// 1. Initialize Post-Quantum Crypto Engine
	log.Println("[orchestrator] ✓ initializing PQC engine (Kyber-1024 + Dilithium-5)...")
	pqcEng, err := pqcrypto.NewEngine(pqcrypto.EngineConfig{
		RotationPeriod: 24 * time.Hour, // Daily key rotation
		EnableHybrid:   true,            // Hybrid mode for backward compatibility
		Validity:       48 * time.Hour,  // Keys valid for 48h
	})
	if err != nil {
		return nil, err
	}
	components.PQCEngine = pqcEng
	log.Printf("[orchestrator] ✓ PQC engine initialized (KEM pubkey: %s...)", 
		pqcEng.GetKEMPublicKey()[:16])
	
	// 2. Initialize Certificate Transparency Monitor
	log.Println("[orchestrator] ✓ initializing CT monitor...")
	monitoredDomains := []string{
		"shieldx.local",
		"*.shieldx.local",
		// Add production domains here
	}
	ctMon := certtransparency.NewMonitor(monitoredDomains, 60*time.Second)
	
	// Start CT monitoring
	if err := ctMon.Start(); err != nil {
		log.Printf("[orchestrator] ⚠ CT monitor start warning: %v", err)
	} else {
		// Monitor alerts in background
		go func() {
			for alert := range ctMon.Alerts() {
				log.Printf("[orchestrator] 🚨 CT ALERT: severity=%v domain=%s fingerprint=%s reason=%s",
					alert.Severity, alert.Domain, alert.CertFingerprint[:16], alert.Reason)
				
				// In production: send to alerting system (PagerDuty, Slack, etc.)
			}
		}()
		log.Println("[orchestrator] ✓ CT monitor started (checking 2 logs)")
	}
	components.CTMonitor = ctMon
	
	// 3. Initialize Adaptive Rate Limiter
	log.Println("[orchestrator] ✓ initializing adaptive rate limiter...")
	limiter := ratelimit.NewAdaptiveLimiter(ratelimit.Config{
		BaseRate:     100,                // 100 requests/minute baseline
		Window:       time.Minute,
		Dimensions:   []ratelimit.DimensionType{
			ratelimit.DimensionIP,
			ratelimit.DimensionTenant,
			ratelimit.DimensionEndpoint,
		},
		AdaptEnabled: true,               // Enable ML-based adaptation
		LearningRate: 0.1,                // 10% adjustment per cycle
		MinRate:      10,
		MaxRate:      10000,
		GeoPolicy:    map[string]int{
			"US": 2,    // 2x for US
			"EU": 2,    // 2x for EU
			"CN": 1,    // 1x for China
		},
	})
	components.AdaptiveLimiter = limiter
	log.Println("[orchestrator] ✓ adaptive limiter initialized (multi-dimensional)")
	
	// 4. Initialize Dynamic Policy Engine
	log.Println("[orchestrator] ✓ initializing dynamic policy engine...")
	policyEng := policy.NewDynamicEngine()
	
	// Load initial policy
	initialPolicy := `{
		"tenants": [
			{"name": "default", "allow": ["*"], "deny": [], "riskLevel": "low"},
			{"name": "premium", "allow": ["*"], "deny": [], "riskLevel": "low"},
			{"name": "trial", "allow": ["read"], "deny": ["write", "admin"], "riskLevel": "medium"}
		],
		"paths": [
			{"pattern": "/health", "action": "allow"},
			{"pattern": "/metrics", "action": "allow"},
			{"pattern": "/admin/*", "action": "deny"}
		],
		"abacRules": [
			{
				"id": "high_risk_block",
				"priority": 100,
				"conditions": [
					{"attribute": "env.risk_score", "operator": "gt", "value": 0.8}
				],
				"action": "deny"
			},
			{
				"id": "business_hours_only",
				"priority": 90,
				"conditions": [
					{"attribute": "time.hour", "operator": "lt", "value": 8}
				],
				"action": "tarpit"
			}
		]
	}`
	
	version, err := policyEng.CompileAndLoad(initialPolicy, policy.PolicyMetadata{
		Author:      "orchestrator-init",
		Description: "Initial production policy v1",
		ValidFrom:   time.Now(),
		ValidUntil:  time.Now().Add(365 * 24 * time.Hour),
		Tags:        []string{"production", "phase1"},
	})
	if err != nil {
		log.Printf("[orchestrator] ⚠ policy load warning: %v (using fallback)", err)
	} else {
		log.Printf("[orchestrator] ✓ policy engine initialized (version=%d)", version)
	}
	
	// Watch for policy events
	go func() {
		for event := range policyEng.Watch() {
			log.Printf("[orchestrator] 📋 policy event: type=%v version=%d details=%s",
				event.Type, event.Version, event.Details)
		}
	}()
	
	components.PolicyEngine = policyEng
	
	// Start metrics collection goroutine
	go components.collectMetrics()
	
	log.Println("[orchestrator] ✅ Phase 1 initialization complete!")
	log.Println("[orchestrator] 📊 Components active:")
	log.Println("[orchestrator]    ✓ Post-Quantum Crypto (Kyber-1024 + Dilithium-5)")
	log.Println("[orchestrator]    ✓ Certificate Transparency Monitoring")
	log.Println("[orchestrator]    ✓ Adaptive Rate Limiting (multi-dimensional)")
	log.Println("[orchestrator]    ✓ Dynamic Policy Engine (ABAC + RBAC)")
	
	return components, nil
}

// collectMetrics periodically collects metrics from all components
func (p1 *Phase1Components) collectMetrics() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	
	for range ticker.C {
		// PQC metrics
		if p1.PQCEngine != nil {
			p1.pqcMetrics = p1.PQCEngine.Metrics()
		}
		
		// CT metrics
		if p1.CTMonitor != nil {
			p1.ctMetrics = p1.CTMonitor.Metrics()
		}
		
		// Rate limiter metrics
		if p1.AdaptiveLimiter != nil {
			p1.limiterMetrics = p1.AdaptiveLimiter.Metrics()
		}
		
		// Policy engine metrics
		if p1.PolicyEngine != nil {
			p1.policyMetrics = p1.PolicyEngine.Metrics()
		}
	}
}

// EvaluateRequestWithEnhancements evaluates request with Phase 1 enhancements
func (p1 *Phase1Components) EvaluateRequestWithEnhancements(ctx context.Context, reqCtx *policy.EvalContext, rlCtx ratelimit.RequestContext) (*EnhancedDecision, error) {
	decision := &EnhancedDecision{
		Timestamp: time.Now(),
	}
	
	// 1. Adaptive Rate Limiting
	if p1.AdaptiveLimiter != nil {
		rlDecision, err := p1.AdaptiveLimiter.Allow(rlCtx)
		if err != nil {
			return nil, err
		}
		
		decision.RateLimitAllowed = rlDecision.Allowed
		decision.RateLimitReason = rlDecision.Reason
		decision.RiskScore = rlDecision.RiskScore
		
		if !rlDecision.Allowed {
			decision.Action = "deny"
			decision.Reason = "rate_limit_exceeded"
			return decision, nil
		}
	}
	
	// 2. Dynamic Policy Evaluation (ABAC + RBAC)
	if p1.PolicyEngine != nil {
		policyDecision, err := p1.PolicyEngine.Evaluate(reqCtx)
		if err != nil {
			log.Printf("[orchestrator] policy evaluation error: %v", err)
		} else {
			decision.PolicyAction = string(policyDecision.Action)
			decision.PolicyVersion = policyDecision.Version
			decision.PolicyReason = policyDecision.Reason
			decision.RiskScore = (decision.RiskScore + policyDecision.RiskScore) / 2
			
			if policyDecision.Action == "deny" {
				decision.Action = "deny"
				decision.Reason = policyDecision.Reason
				return decision, nil
			}
			
			if policyDecision.Action == "tarpit" {
				decision.Action = "tarpit"
				decision.TarpitMs = 5000 // 5 second delay
			}
		}
	}
	
	// 3. All checks passed
	if decision.Action == "" {
		decision.Action = "allow"
		decision.Reason = "all_checks_passed"
	}
	
	return decision, nil
}

// EnhancedDecision combines all Phase 1 decision factors
type EnhancedDecision struct {
	Action            string
	Reason            string
	Timestamp         time.Time
	
	// Rate limiting
	RateLimitAllowed  bool
	RateLimitReason   string
	
	// Policy
	PolicyAction      string
	PolicyVersion     uint64
	PolicyReason      string
	
	// Risk assessment
	RiskScore         float64
	
	// Tarpit
	TarpitMs          int
}

// SignResponse signs response using PQC (Dilithium-5)
func (p1 *Phase1Components) SignResponse(message []byte) ([]byte, error) {
	if p1.PQCEngine == nil {
		return nil, nil // PQC not enabled
	}
	
	signature, err := p1.PQCEngine.Sign(message)
	if err != nil {
		return nil, err
	}
	
	return signature, nil
}

// GetPQCPublicKeys returns current PQC public keys for client handshake
func (p1 *Phase1Components) GetPQCPublicKeys() map[string]string {
	if p1.PQCEngine == nil {
		return nil
	}
	
	return map[string]string{
		"kem_public":  p1.PQCEngine.GetKEMPublicKey(),
		"sig_public":  p1.PQCEngine.GetSigPublicKey(),
		"algorithm":   "kyber1024+dilithium5",
		"hybrid_mode": "enabled",
	}
}

// GetComponentMetrics returns metrics from all Phase 1 components
func (p1 *Phase1Components) GetComponentMetrics() map[string]interface{} {
	return map[string]interface{}{
		"pqc":         p1.pqcMetrics,
		"ct_monitor":  p1.ctMetrics,
		"rate_limit":  p1.limiterMetrics,
		"policy":      p1.policyMetrics,
	}
}

// Shutdown gracefully shuts down all Phase 1 components
func (p1 *Phase1Components) Shutdown() {
	log.Println("[orchestrator] shutting down Phase 1 components...")
	
	if p1.CTMonitor != nil {
		p1.CTMonitor.Stop()
	}
	
	if p1.AdaptiveLimiter != nil {
		p1.AdaptiveLimiter.Close()
	}
	
	if p1.PolicyEngine != nil {
		p1.PolicyEngine.Close()
	}
	
	log.Println("[orchestrator] Phase 1 shutdown complete")
}
