// Phase 1 Enhanced Handlers for Orchestrator
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"sync/atomic"
	"time"

	"shieldx/pkg/accesslog"
	"shieldx/pkg/ledger"
)

var (
	// Phase 1 global instance
	phase1 *Phase1Enhancement
	
	// Enhanced metrics for Phase 1
	mRouteLatency    = metrics.NewHistogram("orchestrator_route_latency_ms", "Route decision latency in milliseconds", nil)
	mPQCOperations   = metrics.NewCounter("orchestrator_pqc_operations_total", "Total PQC operations")
	mAdaptiveChanges = metrics.NewCounter("orchestrator_adaptive_changes_total", "Adaptive rate limit changes")
)

// initPhase1 initializes Phase 1 enhancements
func initPhase1() error {
	config := LoadPhase1Config()
	
	var err error
	phase1, err = NewPhase1Enhancement(config)
	if err != nil {
		return fmt.Errorf("Phase 1 init: %w", err)
	}
	
	phase1.Start()
	
	// Register Phase 1 metrics
	if reg != nil {
		reg.RegisterHistogram(mRouteLatency)
		reg.Register(mPQCOperations)
		reg.Register(mAdaptiveChanges)
	}
	
	return nil
}

// handleRouteEnhanced provides enhanced routing with Phase 1 features
func handleRouteEnhanced(w http.ResponseWriter, r *http.Request, logger *accesslog.Logger) {
	t0 := time.Now()
	mRoute.Inc()
	
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	
	// Limit request body size
	r.Body = http.MaxBytesReader(w, r.Body, int64(envInt("ORCH_MAX_ROUTE_BYTES", 16*1024)))
	
	// Extract client ID for behavioral analysis
	clientID := clientIP(r)
	corrID := getCorrID(r)
	
	// DPoP optional verify
	if jws := r.Header.Get("DPoP"); jws != "" {
		if _, jti, _, err := dpop.VerifyEdDSA(jws, r.Method, normalizeHTU(r), time.Now(), 60); err != nil {
			_ = logger.LogSecurity(map[string]interface{}{
				"event":         "dpop_invalid",
				"client_ip":     clientID,
				"correlation_id": corrID,
				"error":         err.Error(),
			})
			http.Error(w, "dpop invalid", http.StatusUnauthorized)
			return
		} else {
			// anti-replay window 2 minutes
			dpopStoreMu.Lock()
			now := time.Now().Unix()
			exp := now + 120
			if old, ok := dpopStore[jti]; ok && old >= now {
				dpopStoreMu.Unlock()
				http.Error(w, "dpop replay", http.StatusUnauthorized)
				return
			}
			dpopStore[jti] = exp
			dpopStoreMu.Unlock()
		}
	}
	
	var req routeRequest
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&req); err != nil {
		http.Error(w, "bad request", 400)
		return
	}
	
	if req.Service == "" && len(req.Candidates) == 0 {
		http.Error(w, "missing service or candidates", 400)
		return
	}
	
	if req.Service != "" && !validServiceName(req.Service) {
		http.Error(w, "invalid service", 400)
		return
	}
	
	// Behavioral analysis - record request features
	if phase1 != nil && phase1.config.EnableBehaviorAnalysis {
		features := map[string]float64{
			"request_size":   float64(r.ContentLength),
			"hour_of_day":    float64(time.Now().Hour()),
			"has_hash_key":   boolToFloat(req.HashKey != ""),
			"has_candidates": boolToFloat(len(req.Candidates) > 0),
			"path_length":    float64(len(req.Path)),
		}
		phase1.RecordBehavior(clientID, features)
		
		// Check for anomalies
		if isAnom, score := phase1.CheckBehaviorAnomaly(clientID); isAnom {
			_ = logger.LogSecurity(map[string]interface{}{
				"event":          "behavioral_anomaly",
				"client_ip":      clientID,
				"anomaly_score":  score,
				"correlation_id": corrID,
			})
			
			// Optional: add delay or block for highly anomalous behavior
			if score > 5.0 {
				time.Sleep(time.Duration(score) * 100 * time.Millisecond) // Adaptive tarpit
			}
		}
	}
	
	// Policy evaluation
	action := policy.Evaluate(loadBasePolicy(), req.Tenant, req.Scope, req.Path)
	
	// OPA evaluation with cache
	if opaEng != nil {
		if dec, ok := evaluateOPAWithCache(req.Tenant, req.Scope, req.Path, clientID); ok {
			if opaEnforce {
				action = dec
			}
		}
	}
	
	if action == policy.ActionDeny {
		mRouteDenied.Inc()
		_ = ledger.AppendJSONLine(ledgerPath, serviceName, "route.denied", map[string]any{
			"tenant": req.Tenant, "scope": req.Scope, "path": req.Path, "corrId": corrID,
		})
		_ = logger.LogSecurity(map[string]interface{}{
			"event":          "route_denied",
			"tenant":         req.Tenant,
			"scope":          req.Scope,
			"path":           req.Path,
			"correlation_id": corrID,
		})
		http.Error(w, "policy denied", http.StatusForbidden)
		return
	}
	
	// Optional: get route hints from OPA
	if opaEng != nil {
		if hint, ok, err := opaEng.EvaluateRoute(map[string]any{
			"tenant": req.Tenant, "scope": req.Scope, "path": req.Path, "ip": clientID,
		}); err == nil && ok {
			if v, ok := hint["algo"].(string); ok && v != "" {
				req.Algo = v
			}
			if v, ok := hint["service"].(string); ok && v != "" {
				req.Service = v
			}
			if arr, ok := hint["candidates"].([]any); ok && len(arr) > 0 {
				req.Candidates = req.Candidates[:0]
				for _, x := range arr {
					if s, ok := x.(string); ok {
						req.Candidates = append(req.Candidates, s)
					}
				}
			}
		}
	}
	
	// Choose pool
	p := buildPoolForRequest(req)
	if p == nil || len(p.backends) == 0 {
		mRouteErr.Inc()
		http.Error(w, "no backends", http.StatusServiceUnavailable)
		return
	}
	
	// Determine algorithm (allow per-request override via header)
	algo := defaultAlgo
	if req.Algo != "" {
		algo = parseLBAlgo(req.Algo, defaultAlgo)
	} else if p.algo != "" {
		algo = p.algo
	}
	
	// Allow X-LB-Algo header override for experimentation
	if headerAlgo := r.Header.Get("X-LB-Algo"); headerAlgo != "" {
		if parsedAlgo := parseLBAlgo(headerAlgo, ""); parsedAlgo != "" {
			algo = parsedAlgo
		}
	}
	
	// Pick backend
	b, err := pickBackend(p, algo, req.HashKey)
	if err != nil {
		mRouteErr.Inc()
		http.Error(w, err.Error(), http.StatusServiceUnavailable)
		return
	}
	
	// Increment connection counter for load balancing
	atomic.AddInt64(&b.Conns, 1)
	defer atomic.AddInt64(&b.Conns, -1)
	
	// Metric for selection
	mLBPick.Inc(map[string]string{
		"pool":    p.name,
		"algo":    string(algo),
		"healthy": strconv.FormatBool(b.Healthy.Load()),
	})
	
	// Record latency
	latency := time.Since(t0)
	if mRouteLatency != nil {
		mRouteLatency.Observe(float64(latency.Milliseconds()))
	}
	
	// Audit log
	_ = ledger.AppendJSONLine(ledgerPath, serviceName, "route.ok", map[string]any{
		"tenant":    req.Tenant,
		"scope":     req.Scope,
		"path":      req.Path,
		"algo":      string(algo),
		"target":    b.URL,
		"healthy":   b.Healthy.Load(),
		"latency_ms": latency.Milliseconds(),
		"corrId":    corrID,
	})
	
	// Access log
	_ = logger.LogAccess(map[string]interface{}{
		"timestamp":      time.Now().UTC(),
		"correlation_id": corrID,
		"method":         r.Method,
		"path":           r.URL.Path,
		"client_ip":      clientID,
		"tenant":         req.Tenant,
		"service":        req.Service,
		"backend":        b.URL,
		"algo":           string(algo),
		"latency_ms":     latency.Milliseconds(),
		"status":         200,
	})
	
	writeJSON(w, 200, routeResponse{
		Target:  b.URL,
		Algo:    string(algo),
		Policy:  string(action),
		Healthy: b.Healthy.Load(),
	})
}

// handleHealthEnhanced provides enhanced health endpoint with Phase 1 metrics
func handleHealthEnhanced(w http.ResponseWriter, r *http.Request) {
	poolsMu.RLock()
	defer poolsMu.RUnlock()
	
	out := map[string]any{
		"service":   serviceName,
		"time":      time.Now().UTC(),
		"version":   policyVersion.Load(),
		"uptime":    time.Since(startTime).String(),
	}
	
	// Pool health stats
	stats := map[string]any{}
	totalHealthy := 0
	totalBackends := 0
	
	for name, p := range pools {
		p.mu.RLock()
		healthy := 0
		arr := make([]map[string]any, 0, len(p.backends))
		
		for _, b := range p.backends {
			isHealthy := b.Healthy.Load()
			if isHealthy {
				healthy++
			}
			
			totalBackends++
			if isHealthy {
				totalHealthy++
			}
			
			arr = append(arr, map[string]any{
				"url":       b.URL,
				"healthy":   isHealthy,
				"ewma":      b.getEWMA(),
				"conns":     atomic.LoadInt64(&b.Conns),
				"lastErr":   asString(b.LastErr.Load()),
				"lastLatMs": atomic.LoadUint64(&b.LastLatMs),
				"weight":    b.Weight,
			})
		}
		
		stats[name] = map[string]any{
			"backends": arr,
			"healthy":  healthy,
			"total":    len(p.backends),
			"algo":     p.algo,
		}
		p.mu.RUnlock()
	}
	
	out["pools"] = stats
	
	// Overall health ratio
	var healthRatio float64
	if totalBackends > 0 {
		healthRatio = float64(totalHealthy) / float64(totalBackends)
	}
	out["health_ratio"] = healthRatio
	
	// Phase 1 metrics
	if phase1 != nil {
		phase1Metrics := phase1.GetPhase1Metrics()
		out["phase1"] = phase1Metrics
		
		// Adaptive rate limiting feedback
		if phase1.config.EnableAdaptiveRL {
			out["rate_limit"] = phase1.GetCurrentRateLimit()
			
			// Update adaptive rate limit based on health
			phase1.AdaptRateLimit(healthRatio)
		}
	}
	
	// PQC public key (if enabled)
	if phase1 != nil && phase1.config.EnablePQC {
		if pubKey, err := phase1.GetPQCPublicKey(); err == nil && len(pubKey) > 0 {
			out["pqc_public_key"] = fmt.Sprintf("%x", pubKey[:32]) // First 32 bytes for display
		}
	}
	
	writeJSON(w, 200, out)
}

// handlePQCEndpoint provides PQC key exchange endpoint
func handlePQCEndpoint(w http.ResponseWriter, r *http.Request) {
	if phase1 == nil || !phase1.config.EnablePQC {
		http.Error(w, "PQC not enabled", http.StatusNotImplemented)
		return
	}
	
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	
	// Perform key encapsulation
	ciphertext, sharedSecret, err := phase1.PQCEncapsulate()
	if err != nil {
		http.Error(w, "encapsulation failed", http.StatusInternalServerError)
		return
	}
	
	mPQCOperations.Inc()
	
	// In production, use sharedSecret for session key derivation
	// For now, just return ciphertext
	writeJSON(w, 200, map[string]any{
		"ciphertext":      fmt.Sprintf("%x", ciphertext[:64]), // Truncated for display
		"shared_secret":   fmt.Sprintf("%x", sharedSecret[:16]), // Truncated
		"algorithm":       phase1.config.PQCAlgorithm,
	})
}

// handleGraphQLEndpoint provides GraphQL with security validation
func handleGraphQLEndpoint(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	
	var gqlReq struct {
		Query         string                 `json:"query"`
		Variables     map[string]interface{} `json:"variables"`
		OperationName string                 `json:"operationName"`
	}
	
	if err := json.NewDecoder(r.Body).Decode(&gqlReq); err != nil {
		http.Error(w, "bad request", 400)
		return
	}
	
	clientID := clientIP(r)
	
	// Phase 1 GraphQL security validation
	if phase1 != nil && phase1.config.EnableGraphQLSec {
		ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
		defer cancel()
		
		if err := phase1.ValidateGraphQLQuery(ctx, clientID, gqlReq.Query, gqlReq.Variables); err != nil {
			http.Error(w, fmt.Sprintf("GraphQL security: %s", err), http.StatusBadRequest)
			return
		}
	}
	
	// Forward to GraphQL backend (simplified - in production use proper GraphQL library)
	writeJSON(w, 200, map[string]any{
		"data": map[string]any{
			"message": "GraphQL query validated successfully",
		},
	})
}

// newSecurityMiddleware wraps handlers with security logging
func newSecurityMiddleware(next http.Handler, logger *accesslog.Logger) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Detect suspicious patterns
		suspicious := false
		reason := ""
		
		// Check for SQL injection patterns
		if hasSQLInjection(r.URL.RawQuery) {
			suspicious = true
			reason = "sql_injection_attempt"
		}
		
		// Check for XSS patterns
		if hasXSS(r.URL.RawQuery) {
			suspicious = true
			reason = "xss_attempt"
		}
		
		// Check for path traversal
		if hasPathTraversal(r.URL.Path) {
			suspicious = true
			reason = "path_traversal_attempt"
		}
		
		if suspicious {
			_ = logger.LogSecurity(map[string]interface{}{
				"event":          "suspicious_request",
				"reason":         reason,
				"path":           r.URL.Path,
				"query":          r.URL.RawQuery,
				"client_ip":      clientIP(r),
				"user_agent":     r.UserAgent(),
				"correlation_id": getCorrID(r),
			})
			
			// In production: block, tarpit, or divert to honeypot
			// For now, log and continue
		}
		
		next.ServeHTTP(w, r)
	})
}

// Security check helpers
func hasSQLInjection(s string) bool {
	patterns := []string{"' OR ", "'; DROP ", "UNION SELECT", "1=1--", "admin'--"}
	sl := strings.ToUpper(s)
	for _, p := range patterns {
		if strings.Contains(sl, p) {
			return true
		}
	}
	return false
}

func hasXSS(s string) bool {
	patterns := []string{"<script", "javascript:", "onerror=", "onload="}
	sl := strings.ToLower(s)
	for _, p := range patterns {
		if strings.Contains(sl, p) {
			return true
		}
	}
	return false
}

func hasPathTraversal(s string) bool {
	return strings.Contains(s, "../") || strings.Contains(s, "..\\")
}

func boolToFloat(b bool) float64 {
	if b {
		return 1.0
	}
	return 0.0
}

var startTime = time.Now()
