package main

// P0 Enhancement: Advanced input validation middleware for Orchestrator
// This file adds production-ready validation layer on top of existing handlers

import (
	"context"
	crand "crypto/rand"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"net/http"
	"sync/atomic"
	"time"

	"shieldx/shared/shieldx-common/pkg/accesslog"
	"shieldx/shared/shieldx-common/pkg/dpop"
	"shieldx/shared/shieldx-common/pkg/policy"
	"shieldx/shared/shieldx-common/pkg/validation"
)

// P0: Enhanced request validation with detailed error responses
func validateRouteRequestEnhanced(req *routeRequest) error {
	// Validate using our comprehensive validation package
	if err := validation.ValidateRouteRequest(req.Service, req.Tenant, req.Path, req.Scope); err != nil {
		return err
	}

	// Additional checks for candidates
	if len(req.Candidates) > 0 {
		if len(req.Candidates) > 100 {
			return fmt.Errorf("too many candidates: max 100")
		}
		for i, candidate := range req.Candidates {
			if err := validation.ValidateURL(candidate); err != nil {
				return fmt.Errorf("invalid candidate[%d]: %w", i, err)
			}
		}
	}

	// Validate algorithm if specified
	if req.Algo != "" {
		validAlgos := map[string]bool{
			"round_robin": true,
			"least_conn":  true,
			"ewma":        true,
			"rendezvous":  true,
			"p2c":         true,
		}
		if !validAlgos[req.Algo] {
			return fmt.Errorf("invalid algorithm: %s", req.Algo)
		}
	}

	// Validate hash key if present
	if req.HashKey != "" && len(req.HashKey) > 256 {
		return fmt.Errorf("hash key too long: max 256 chars")
	}

	return nil
}

// P0: Security middleware wrapper
type securityMiddleware struct {
	handler http.Handler
	logger  *accesslog.Logger
}

func newSecurityMiddleware(h http.Handler, logger *accesslog.Logger) *securityMiddleware {
	return &securityMiddleware{handler: h, logger: logger}
}

func (sm *securityMiddleware) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	start := time.Now()

	// Get correlation ID from context
	corrID := getCorrID(r)
	if corrID == "" {
		corrID = generateCorrelationID()
		r = r.WithContext(context.WithValue(r.Context(), ctxKeyCorrID{}, corrID))
	}

	// Wrap response writer to capture status and bytes
	rw := &responseWriter{ResponseWriter: w, statusCode: 200}

	// Execute handler
	sm.handler.ServeHTTP(rw, r)

	// Log access
	duration := time.Since(start)
	_ = sm.logger.LogHTTPRequest(r, rw.statusCode, duration, rw.bytesWritten, nil, corrID)
}

type responseWriter struct {
	http.ResponseWriter
	statusCode   int
	bytesWritten int64
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.statusCode = code
	rw.ResponseWriter.WriteHeader(code)
}

func (rw *responseWriter) Write(b []byte) (int, error) {
	n, err := rw.ResponseWriter.Write(b)
	rw.bytesWritten += int64(n)
	return n, err
}

// P0: Enhanced route handler with validation
func handleRouteEnhanced(w http.ResponseWriter, r *http.Request, logger *accesslog.Logger) {
	mRoute.Inc()
	corrID := getCorrID(r)

	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// P0: Strict size limit (16KB default)
	maxBytes := int64(envInt("ORCH_MAX_ROUTE_BYTES", 16*1024))
	r.Body = http.MaxBytesReader(w, r.Body, maxBytes)

	// P0: DPoP verification (if present)
	if jws := r.Header.Get("DPoP"); jws != "" {
		if err := verifyDPoP(r, jws); err != nil {
			_ = logger.LogAuthenticationFailure(clientIP(r), "DPoP invalid", corrID)
			http.Error(w, "dpop invalid", http.StatusUnauthorized)
			return
		}
	}

	// P0: Parse and validate JSON
	var req routeRequest
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields() // P0: Strict parsing
	if err := dec.Decode(&req); err != nil {
		_ = logger.LogSecurityEvent("invalid_json", "low", "deny", clientIP(r), corrID, map[string]string{
			"error": err.Error(),
		})
		http.Error(w, "invalid JSON", http.StatusBadRequest)
		return
	}

	// P0: Enhanced validation
	if err := validateRouteRequestEnhanced(&req); err != nil {
		_ = logger.LogSecurityEvent("validation_failure", "medium", "deny", clientIP(r), corrID, map[string]string{
			"error": err.Error(),
		})
		http.Error(w, fmt.Sprintf("validation error: %v", err), http.StatusBadRequest)
		return
	}

	// P0: Check for injection attempts
	if err := validation.CheckSQLInjection(req.Path); err != nil {
		_ = logger.LogInjectionAttempt(clientIP(r), "sql_injection", req.Path, corrID)
		http.Error(w, "request blocked", http.StatusForbidden)
		return
	}

	if err := validation.CheckXSS(req.Path); err != nil {
		_ = logger.LogInjectionAttempt(clientIP(r), "xss", req.Path, corrID)
		http.Error(w, "request blocked", http.StatusForbidden)
		return
	}

	// Continue with existing route logic...
	handleRouteLogic(w, r, &req, logger, corrID)
}

// P0: Separated routing logic for better testability
func handleRouteLogic(w http.ResponseWriter, r *http.Request, req *routeRequest, logger *accesslog.Logger, corrID string) {
	// Check if service or candidates provided
	if req.Service == "" && len(req.Candidates) == 0 {
		http.Error(w, "missing service or candidates", http.StatusBadRequest)
		return
	}

	// P0: Policy evaluation
	action := evaluatePolicy(req, r, logger, corrID)
	if action == "deny" {
		mRouteDenied.Inc()
		_ = logger.LogPolicyDeny(clientIP(r), req.Tenant, req.Path, "policy denied", corrID)
		http.Error(w, "forbidden by policy", http.StatusForbidden)
		return
	}

	// P0: Load balancing with circuit breaker check
	backend, algo, err := selectBackend(req)
	if err != nil {
		mRouteErr.Inc()
		_ = logger.LogSecurityEvent("route_error", "low", "error", clientIP(r), corrID, map[string]string{
			"error": err.Error(),
		})
		http.Error(w, "no healthy backend", http.StatusServiceUnavailable)
		return
	}

	// Success response
	resp := routeResponse{
		Target:  backend.URL,
		Algo:    string(algo),
		Policy:  string(action),
		Healthy: backend.Healthy.Load(),
	}

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Correlation-ID", corrID)
	_ = json.NewEncoder(w).Encode(resp)
}

// P0: DPoP verification with anti-replay
func verifyDPoP(r *http.Request, jws string) error {
	_, jti, _, err := dpop.VerifyEdDSA(jws, r.Method, normalizeHTU(r), time.Now(), 60)
	if err != nil {
		return err
	}

	// P0: Anti-replay check (2-minute window)
	dpopStoreMu.Lock()
	defer dpopStoreMu.Unlock()

	now := time.Now().Unix()
	exp := now + 120

	if old, ok := dpopStore[jti]; ok && old >= now {
		return fmt.Errorf("dpop replay detected")
	}

	dpopStore[jti] = exp

	// Cleanup old entries (every 1000 requests)
	if len(dpopStore) > 10000 {
		for k, v := range dpopStore {
			if v < now {
				delete(dpopStore, k)
			}
		}
	}

	return nil
}

// P0: Policy evaluation with OPA integration
func evaluatePolicy(req *routeRequest, r *http.Request, logger *accesslog.Logger, corrID string) string {
	// Base policy check
	action := policy.Evaluate(basePolicy, req.Tenant, req.Scope, req.Path)

	// OPA check (if configured)
	if opaEng != nil {
		if opaAction, ok := evaluateOPAWithCache(req.Tenant, req.Scope, req.Path, clientIP(r)); ok {
			if opaEnforce {
				action = opaAction
			}
			// Log if OPA disagrees with base policy
			if action != opaAction {
				_ = logger.LogSecurityEvent("policy_mismatch", "low", "alert", clientIP(r), corrID, map[string]string{
					"base_policy": string(action),
					"opa_policy":  string(opaAction),
				})
			}
		}
	}

	return string(action)
}

// P0: Backend selection with enhanced circuit breaker
func selectBackend(req *routeRequest) (*Backend, LBAlgo, error) {
	var pool *Pool
	var algo LBAlgo

	// Determine pool and algorithm
	if len(req.Candidates) > 0 {
		// Ad-hoc pool from candidates
		pool = newPool("adhoc", req.Candidates)
		algo = parseLBAlgo(req.Algo, defaultAlgo)
	} else {
		// Named service pool
		poolsMu.RLock()
		pool = pools[req.Service]
		poolsMu.RUnlock()

		if pool == nil {
			return nil, "", fmt.Errorf("service not found: %s", req.Service)
		}

		algo = parseLBAlgo(req.Algo, pool.algo)
		if algo == "" {
			algo = defaultAlgo
		}
	}

	// Get healthy backends (P0: circuit breaker filtering)
	pool.mu.RLock()
	candidates := filterHealthyBackends(pool.backends)
	pool.mu.RUnlock()

	if len(candidates) == 0 {
		return nil, algo, fmt.Errorf("no healthy backends available")
	}

	// Select backend using algorithm (P0: Use optimized pickBackend)
	backend, err := pickBackendFromCandidates(candidates, algo, req.HashKey)
	if err != nil {
		return nil, algo, err
	}

	// Record metrics (P0: Fixed signature for labeled counter)
	mLBPick.Inc(map[string]string{
		"pool":    pool.name,
		"algo":    string(algo),
		"healthy": fmt.Sprintf("%t", backend.Healthy.Load()),
	})

	return backend, algo, nil
}

// P0: Filter backends with circuit breaker state
func filterHealthyBackends(backends []*Backend) []*Backend {
	var healthy []*Backend
	now := time.Now().UnixNano()

	for _, b := range backends {
		cbState := b.cbState.Load()

		switch cbState {
		case 0: // CLOSED - healthy
			if b.Healthy.Load() {
				healthy = append(healthy, b)
			}
		case 1: // OPEN - check if probe time reached
			if now >= b.cbNextProbe.Load() {
				// Allow one probe attempt (half-open state)
				b.cbState.Store(2)
				mCBHalfOpen.Inc()
				healthy = append(healthy, b)
			}
		case 2: // HALF-OPEN - allow probing
			if b.Healthy.Load() {
				healthy = append(healthy, b)
			}
		}
	}

	return healthy
}

// P0: Helper to select backend from pre-filtered candidates
func pickBackendFromCandidates(candidates []*Backend, algo LBAlgo, hashKey string) (*Backend, error) {
	if len(candidates) == 0 {
		return nil, fmt.Errorf("no candidates")
	}

	switch algo {
	case LBRoundRobin:
		// Use time-based rotation for adhoc pools
		idx := int(time.Now().UnixNano()/1000000) % len(candidates)
		return candidates[idx], nil

	case LBLeastConnections:
		best := candidates[0]
		bestC := atomic.LoadInt64(&best.Conns)
		for _, b := range candidates[1:] {
			if c := atomic.LoadInt64(&b.Conns); c < bestC {
				best = b
				bestC = c
			}
		}
		return best, nil

	case LBEWMA:
		best := candidates[0]
		bestE := best.getEWMA()
		if bestE == 0 {
			bestE = math.MaxFloat64
		}
		for _, b := range candidates[1:] {
			e := b.getEWMA()
			if e == 0 {
				e = math.MaxFloat64
			}
			if e < bestE {
				best = b
				bestE = e
			}
		}
		return best, nil

	case LBP2CEWMA:
		if len(candidates) == 1 {
			return candidates[0], nil
		}
		i := rand.Intn(len(candidates))
		j := rand.Intn(len(candidates) - 1)
		if j >= i {
			j++
		}
		a := candidates[i]
		b := candidates[j]
		ca := lbCost(a)
		cb := lbCost(b)
		if ca <= cb {
			return a, nil
		}
		return b, nil

	case LBConsistentHash:
		key := hashKey
		if key == "" {
			key = fmt.Sprintf("time:%d", time.Now().UnixNano())
		}
		return chooseRendezvousWeighted(candidates, key), nil

	default:
		return candidates[0], nil
	}
}

// P0: Generate cryptographically secure correlation ID
func generateCorrelationID() string {
	b := make([]byte, 16)
	if _, err := crand.Read(b); err != nil {
		// Fallback to timestamp-based ID
		return fmt.Sprintf("orch-%d", time.Now().UnixNano())
	}
	return fmt.Sprintf("orch-%x", b)
}

// P0: Enhanced health check with detailed backend status
func handleHealthEnhanced(w http.ResponseWriter, r *http.Request) {
	poolsMu.RLock()
	defer poolsMu.RUnlock()

	out := map[string]interface{}{
		"service":   serviceName,
		"timestamp": time.Now().UTC().Format(time.RFC3339),
		"status":    "healthy",
	}

	stats := make(map[string]interface{})
	totalHealthy := 0
	totalBackends := 0

	for name, p := range pools {
		p.mu.RLock()
		healthy := 0
		backends := make([]map[string]interface{}, 0, len(p.backends))

		for _, b := range p.backends {
			isHealthy := b.Healthy.Load()
			if isHealthy {
				healthy++
			}
			totalBackends++

			cbState := "closed"
			switch b.cbState.Load() {
			case 1:
				cbState = "open"
			case 2:
				cbState = "half-open"
			}

			backends = append(backends, map[string]interface{}{
				"url":         b.URL,
				"healthy":     isHealthy,
				"ewma_ms":     b.getEWMA(),
				"connections": atomic.LoadInt64(&b.Conns),
				"last_error":  asString(b.LastErr.Load()),
				"latency_ms":  atomic.LoadUint64(&b.LastLatMs),
				"cb_state":    cbState,
				"weight":      b.Weight,
			})
		}

		totalHealthy += healthy
		stats[name] = map[string]interface{}{
			"backends": backends,
			"healthy":  healthy,
			"total":    len(p.backends),
			"ratio":    float64(healthy) / float64(len(p.backends)),
		}
		p.mu.RUnlock()
	}

	// Overall health status
	if totalBackends > 0 {
		healthRatio := float64(totalHealthy) / float64(totalBackends)
		if healthRatio < 0.5 {
			out["status"] = "degraded"
			w.WriteHeader(http.StatusServiceUnavailable)
		}
		out["health_ratio"] = healthRatio
	}

	out["pools"] = stats
	out["total_healthy"] = totalHealthy
	out["total_backends"] = totalBackends

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(out); err != nil {
		http.Error(w, "encoding error", http.StatusInternalServerError)
	}
}
