// Enhanced handlers for Orchestrator with Phase 1-3 improvements
package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"
	
	"shieldx/pkg/abac"
	"shieldx/pkg/adaptive"
	"shieldx/pkg/certtransparency"
	"shieldx/pkg/pqcrypto"
)

// Global enhancement state
var (
	// Post-quantum crypto engine
	pqEngine *pqcrypto.PQCryptoEngine
	
	// Adaptive rate limiter
	adaptiveLimiter *adaptive.Limiter
	
	// ABAC engine
	abacEngine *abac.Engine
	
	// CT monitor
	ctMonitor *certtransparency.Monitor
)

// initEnhancements initializes Phase 1-3 enhancements
func initEnhancements() error {
	// Initialize Post-Quantum Crypto
	pqCfg := pqcrypto.EngineConfig{
		RotationPeriod: 24 * time.Hour,
		EnableHybrid:   true,
		Validity:       48 * time.Hour,
	}
	var err error
	pqEngine, err = pqcrypto.NewEngine(pqCfg)
	if err != nil {
		return fmt.Errorf("pqcrypto init: %w", err)
	}
	
	// Initialize Adaptive Rate Limiter
	limiterCfg := adaptive.LimiterConfig{
		BaseCapacity:         getenvInt("ADAPTIVE_BASE_CAPACITY", 200),
		Window:               envDur("ADAPTIVE_WINDOW", time.Minute),
		EnableMLAdaptive:     true,
		AdaptInterval:        envDur("ADAPTIVE_INTERVAL", time.Minute),
		MinCapacity:          50,
		MaxCapacity:          10000,
		EnableIPLimit:        true,
		EnableUserLimit:      true,
		EnableEndpointLimit:  true,
		EnablePayloadLimit:   true,
		EnableGeoPolicy:      true,
		GeoMultipliers: map[string]float64{
			"US": 1.5, // US gets 1.5x capacity
			"EU": 1.2,
			"CN": 0.8,
		},
	}
	adaptiveLimiter = adaptive.NewLimiter(limiterCfg)
	
	// Initialize ABAC Engine
	abacCfg := abac.EngineConfig{
		EnableContinuous: true,
		RevalidateAfter:  5 * time.Minute,
		EnableCache:      true,
		CacheTTL:         30 * time.Second,
		RiskScorer:       abac.NewDefaultRiskScorer(),
	}
	abacEngine = abac.NewEngine(abacCfg)
	
	// Load default ABAC policies
	if err := loadDefaultABACPolicies(); err != nil {
		return fmt.Errorf("abac policies: %w", err)
	}
	
	// Initialize Certificate Transparency Monitor
	ctCfg := certtransparency.MonitorConfig{
		Domains:        []string{"shieldx.local", "*.shieldx.local"},
		CTLogs:         []string{certtransparency.GooglePilotLog},
		CheckInterval:  5 * time.Minute,
		AlertThreshold: 1 * time.Hour,
		EnableOCSP:     true,
		EnablePinning:  true,
	}
	ctMonitor, err = certtransparency.NewMonitor(ctCfg)
	if err != nil {
		return fmt.Errorf("ct monitor: %w", err)
	}
	
	// Start CT monitoring
	if err := ctMonitor.Start(); err != nil {
		return fmt.Errorf("ct monitor start: %w", err)
	}
	
	// Start CT alert handler
	go handleCTAlerts()
	
	return nil
}

// loadDefaultABACPolicies loads production-ready ABAC policies
func loadDefaultABACPolicies() error {
	// Policy 1: Allow internal corporate network access
	policy1 := &abac.Policy{
		ID:          "policy-internal-allow",
		Name:        "Internal Corporate Access",
		Description: "Allow access from corporate network with low risk",
		Effect:      abac.EffectAllow,
		Priority:    100,
		EnvironmentAttributes: map[string]abac.AttributeCondition{
			"networkType": {Operator: "eq", Value: "corporate", Required: true},
		},
		MaxRiskScore: 50.0,
	}
	if err := abacEngine.AddPolicy(policy1); err != nil {
		return err
	}
	
	// Policy 2: Deny high-risk public network access to confidential resources
	policy2 := &abac.Policy{
		ID:          "policy-highrisk-deny",
		Name:        "High Risk Public Network Deny",
		Description: "Deny high-risk access from public networks to sensitive resources",
		Effect:      abac.EffectDeny,
		Priority:    200,
		EnvironmentAttributes: map[string]abac.AttributeCondition{
			"networkType": {Operator: "eq", Value: "public", Required: true},
			"threatLevel": {Operator: "in", Value: []interface{}{"high", "critical"}, Required: false},
		},
		ResourceAttributes: map[string]abac.AttributeCondition{
			"sensitivity": {Operator: "in", Value: []interface{}{"confidential", "secret"}, Required: false},
		},
		MaxRiskScore:  70.0,
		RequireStepUp: true,
	}
	if err := abacEngine.AddPolicy(policy2); err != nil {
		return err
	}
	
	// Policy 3: Allow VPN access with step-up for medium risk
	policy3 := &abac.Policy{
		ID:          "policy-vpn-stepup",
		Name:        "VPN Access with Step-Up",
		Description: "Allow VPN access but require step-up auth for medium/high risk",
		Effect:      abac.EffectAllow,
		Priority:    150,
		EnvironmentAttributes: map[string]abac.AttributeCondition{
			"networkType": {Operator: "eq", Value: "vpn", Required: true},
		},
		MaxRiskScore:  80.0,
		RequireStepUp: true,
	}
	if err := abacEngine.AddPolicy(policy3); err != nil {
		return err
	}
	
	return nil
}

// handleCTAlerts monitors certificate transparency alerts
func handleCTAlerts() {
	for alert := range ctMonitor.Alerts() {
		// Log alert
		_ = ledger.AppendJSONLine(ledgerPath, serviceName, "ct.alert", map[string]any{
			"type":        alert.Type,
			"domain":      alert.Domain,
			"issuer":      alert.Issuer,
			"fingerprint": alert.Fingerprint,
			"message":     alert.Message,
			"timestamp":   alert.Timestamp,
		})
		
		// In production: trigger incident response workflow
		// For critical alerts (mis-issuance, pin violation), escalate immediately
		if alert.Type == certtransparency.AlertTypeMissIssuance || 
		   alert.Type == certtransparency.AlertTypePinViolation {
			// Escalate to security team
		}
	}
}

// handleRouteEnhanced is the enhanced /route handler with all improvements
func handleRouteEnhanced(w http.ResponseWriter, r *http.Request, logger *accesslog.Logger) {
	// Phase 2: Adaptive rate limiting with multi-dimensional checks
	ip := clientIP(r)
	userID := r.Header.Get("X-User-ID")
	
	attrs := adaptive.RequestAttributes{
		UserID:      userID,
		Endpoint:    r.URL.Path,
		PayloadSize: int(r.ContentLength),
		GeoCountry:  extractGeoCountry(r), // Extract from GeoIP lookup
	}
	
	if !adaptiveLimiter.Allow(ip, "ip", attrs) {
		mIPRateLimitHit.Inc()
		http.Error(w, "rate limit exceeded", http.StatusTooManyRequests)
		return
	}
	
	// Continue with original routing logic
	var req routeRequest
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&req); err != nil {
		http.Error(w, "bad request", 400)
		return
	}
	
	// Phase 3: ABAC policy evaluation with risk scoring
	abacReq := abac.AccessRequest{
		User: abac.User{
			ID:         userID,
			Attributes: map[string]interface{}{"tenant": req.Tenant},
		},
		Resource: abac.Resource{
			ID:          req.Service,
			Type:        "service",
			Attributes:  map[string]interface{}{"scope": req.Scope},
			Sensitivity: "internal",
		},
		Action: "route",
		Environment: abac.Environment{
			Timestamp:   time.Now(),
			IPAddress:   ip,
			GeoLocation: extractGeoLocation(r),
			DeviceTrust: 0.8, // Would be calculated from device fingerprint
			NetworkType: detectNetworkType(ip),
			ThreatLevel: "low", // Would be from threat intelligence feed
		},
		Context: abac.RequestContext{
			CorrelationID: getCorrID(r),
			SessionID:     r.Header.Get("X-Session-ID"),
		},
	}
	
	decision := abacEngine.Evaluate(abacReq)
	if !decision.Allowed {
		mRouteDenied.Inc()
		_ = ledger.AppendJSONLine(ledgerPath, serviceName, "route.abac_denied", map[string]any{
			"reason":    decision.Reason,
			"riskScore": decision.RiskScore,
			"corrId":    getCorrID(r),
		})
		
		if decision.RequireStepUp {
			w.Header().Set("X-Require-StepUp", "true")
			http.Error(w, "step-up authentication required", http.StatusForbidden)
		} else {
			http.Error(w, "access denied: "+decision.Reason, http.StatusForbidden)
		}
		return
	}
	
	// Original routing logic
	p := buildPoolForRequest(req)
	if p == nil || len(p.backends) == 0 {
		mRouteErr.Inc()
		http.Error(w, "no backends", http.StatusServiceUnavailable)
		return
	}
	
	algo := defaultAlgo
	if req.Algo != "" {
		algo = parseLBAlgo(req.Algo, defaultAlgo)
	}
	
	b, err := pickBackend(p, algo, req.HashKey)
	if err != nil {
		mRouteErr.Inc()
		http.Error(w, err.Error(), http.StatusServiceUnavailable)
		return
	}
	
	// Add PQ crypto handshake hint if available
	resp := routeResponse{
		Target:  b.URL,
		Algo:    string(algo),
		Policy:  string(decision.Effect),
		Healthy: b.Healthy.Load(),
	}
	
	// Phase 1: Include PQ public keys for client handshake
	w.Header().Set("X-PQ-KEM-Public", pqEngine.GetKEMPublicKey())
	w.Header().Set("X-PQ-Sig-Public", pqEngine.GetSigPublicKey())
	w.Header().Set("X-Risk-Score", fmt.Sprintf("%.2f", decision.RiskScore))
	
	writeJSON(w, 200, resp)
}

// handleHealthEnhanced provides enhanced health check with all subsystems
func handleHealthEnhanced(w http.ResponseWriter, r *http.Request) {
	poolsMu.RLock()
	defer poolsMu.RUnlock()
	
	out := map[string]any{
		"service": serviceName,
		"time":    time.Now().UTC(),
		"version": "2.0.0-enhanced",
	}
	
	// Original pool stats
	stats := map[string]any{}
	for name, p := range pools {
		p.mu.RLock()
		healthy := 0
		for _, b := range p.backends {
			if b.Healthy.Load() {
				healthy++
			}
		}
		stats[name] = map[string]any{
			"healthy": healthy,
			"total":   len(p.backends),
		}
		p.mu.RUnlock()
	}
	out["pools"] = stats
	
	// Phase 1: PQ Crypto metrics
	out["pqcrypto"] = pqEngine.Metrics()
	
	// Phase 2: Adaptive limiter metrics
	out["adaptive_limiter"] = adaptiveLimiter.Metrics()
	
	// Phase 3: ABAC metrics
	out["abac"] = abacEngine.Metrics()
	
	// CT Monitor metrics
	out["cert_transparency"] = ctMonitor.Metrics()
	
	writeJSON(w, 200, out)
}

// newSecurityMiddleware wraps requests with enhanced security logging
func newSecurityMiddleware(next http.Handler, logger *accesslog.Logger) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		
		// Log security-relevant request attributes
		secEvent := map[string]interface{}{
			"timestamp":   start,
			"method":      r.Method,
			"path":        r.URL.Path,
			"ip":          clientIP(r),
			"userAgent":   r.UserAgent(),
			"corrId":      getCorrID(r),
		}
		
		// Check reputation score
		ip := clientIP(r)
		if adaptiveLimiter != nil {
			repScore := adaptiveLimiter.GetReputationScore(ip)
			secEvent["reputationScore"] = repScore
			
			// Flag low reputation
			if repScore < 0.3 {
				secEvent["lowReputation"] = true
			}
		}
		
		// Wrap response writer to capture status
		wrapped := &statusRecorder{ResponseWriter: w, statusCode: 200}
		
		next.ServeHTTP(wrapped, r)
		
		// Log completion
		secEvent["duration"] = time.Since(start).Milliseconds()
		secEvent["statusCode"] = wrapped.statusCode
		
		if logger != nil {
			logger.LogAccess(secEvent)
		}
	})
}

type statusRecorder struct {
	http.ResponseWriter
	statusCode int
}

func (sr *statusRecorder) WriteHeader(code int) {
	sr.statusCode = code
	sr.ResponseWriter.WriteHeader(code)
}

// Helper functions

func extractGeoCountry(r *http.Request) string {
	// In production, use MaxMind GeoIP2 or similar
	// For now, return default
	if cf := r.Header.Get("CF-IPCountry"); cf != "" {
		return cf
	}
	return "US"
}

func extractGeoLocation(r *http.Request) abac.GeoLocation {
	return abac.GeoLocation{
		Country: extractGeoCountry(r),
		Region:  "",
		City:    "",
	}
}

func detectNetworkType(ip string) string {
	// In production, check against corporate IP ranges
	// For now, simple heuristic
	if isPrivateIP(ip) {
		return "corporate"
	}
	return "public"
}

func isPrivateIP(ip string) bool {
	// RFC 1918 private addresses
	if len(ip) >= 3 {
		if ip[:3] == "10." || ip[:4] == "172." || ip[:8] == "192.168." {
			return true
		}
	}
	return false
}

func startQUICServer(addr string) error {
	// QUIC server initialization would go here
	// For now, placeholder
	return nil
}
