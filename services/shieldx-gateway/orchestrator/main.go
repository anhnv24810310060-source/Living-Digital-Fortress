package main

import (
	"context"
	crand "crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"hash/fnv"
	"log"
	"math"
	"math/rand"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	redis "github.com/redis/go-redis/v9"

	"shieldx/shared/shieldx-common/pkg/accesslog"
	"shieldx/shared/shieldx-common/pkg/dpop"
	"shieldx/shared/shieldx-common/pkg/guard"
	"shieldx/shared/shieldx-common/pkg/ledger"
	"shieldx/shared/shieldx-common/pkg/metrics"
	otelobs "shieldx/shared/shieldx-common/pkg/observability/otel"
	"shieldx/shared/shieldx-common/pkg/policy"
	"shieldx/shared/shieldx-common/pkg/ratls"
	tlsutil "shieldx/shared/shieldx-common/pkg/security/tls"
)

// Service constants
const (
	serviceName = "orchestrator"
	ledgerPath  = "data/ledger-orchestrator.log"
)

// LB algorithms
type LBAlgo string

const (
	LBRoundRobin       LBAlgo = "round_robin"
	LBLeastConnections LBAlgo = "least_conn"
	LBEWMA             LBAlgo = "ewma"
	LBConsistentHash   LBAlgo = "rendezvous"
	LBP2CEWMA          LBAlgo = "p2c"
)

// Backend and Pool
type Backend struct {
	URL       string
	Healthy   atomic.Bool
	EWMA      uint64       // atomically store math.Float64bits
	Conns     int64        // in-flight connections
	LastErr   atomic.Value // string
	LastLatMs uint64       // last observed latency in ms
	// circuit breaker
	cbState     atomic.Uint32 // 0=closed,1=open,2=half-open
	cbFails     atomic.Uint32 // consecutive failures
	cbNextProbe atomic.Int64  // unix nano when next probe is allowed in OPEN state
	// weight: relative capacity (>=0.1). Default 1.0. Used by EWMA/P2C cost and rendezvous hashing
	Weight float64
}

func (b *Backend) getEWMA() float64  { return math.Float64frombits(atomic.LoadUint64(&b.EWMA)) }
func (b *Backend) setEWMA(v float64) { atomic.StoreUint64(&b.EWMA, math.Float64bits(v)) }

type Pool struct {
	name     string
	backends []*Backend
	rr       uint64
	mu       sync.RWMutex
	algo     LBAlgo // optional default algo for this pool
}

func newPool(name string, urls []string) *Pool {
	p := &Pool{name: name}
	for _, u := range urls {
		if strings.TrimSpace(u) == "" {
			continue
		}
		raw := strings.TrimSpace(u)
		// Accept bare hosts by assuming http://
		candidate := raw
		if !strings.Contains(raw, "://") {
			candidate = "http://" + raw
		}
		if parsed, err := url.Parse(candidate); err == nil && (parsed.Scheme == "http" || parsed.Scheme == "https") && parsed.Host != "" {
			p.backends = append(p.backends, &Backend{URL: strings.TrimRight(candidate, "/"), Weight: 1.0})
		}
	}
	p.rebuildHasher()
	return p
}

func (p *Pool) rebuildHasher() {}

// global state
var (
	reg         = metrics.NewRegistry()
	rdb         *redis.Client
	poolsMu     sync.RWMutex
	pools       = map[string]*Pool{}
	defaultAlgo = envLBAlgo("ORCH_LB_ALGO", LBEWMA)
	// penalty per in-flight connection (ms equivalent added to EWMA) for p2c cost
	p2cPenalty  = envFloat("ORCH_P2C_CONN_PENALTY", 5.0)
	opaEng      *policy.OPAEngine
	basePolicy  policy.Config // retained for backward compatibility; reads should use loadBasePolicy()
	opaEnforce  bool
	issuer      *ratls.AutoIssuer
	gCertExpiry = metrics.NewGauge("ratls_cert_expiry_seconds", "Seconds until current RA-TLS cert expiry")
	// metrics
	mRoute         = metrics.NewCounter("orchestrator_route_total", "Total route decisions")
	mRouteDenied   = metrics.NewCounter("orchestrator_route_denied_total", "Route denied by policy")
	mRouteErr      = metrics.NewCounter("orchestrator_route_error_total", "Route errors")
	mHealthOK      = metrics.NewCounter("orchestrator_health_ok_total", "Health probes OK")
	mHealthBad     = metrics.NewCounter("orchestrator_health_bad_total", "Health probes failures")
	mCBOpen        = metrics.NewCounter("orchestrator_cb_open_total", "Circuit breaker opened")
	mCBHalfOpen    = metrics.NewCounter("orchestrator_cb_halfopen_total", "Circuit breaker half-open probes")
	mCBClose       = metrics.NewCounter("orchestrator_cb_close_total", "Circuit breaker closed")
	mLBPick        = metrics.NewLabeledCounter("orchestrator_lb_pick_total", "LB selections by pool and algo", []string{"pool", "algo", "healthy"})
	mProbeDur      = metrics.NewHistogram("orchestrator_health_probe_seconds", "Duration of health probes (seconds)", nil)
	mPolicyReloads = metrics.NewCounter("orchestrator_policy_reload_total", "Number of policy reload events")
	gPolicyVersion = metrics.NewGauge("orchestrator_policy_version", "Current loaded policy version")
	gHealthRatio   = metrics.NewGauge("orchestrator_health_ratio_x10000", "Overall backend health ratio * 10000")
	orchLogger     *accesslog.Logger
)

// DPoP anti-replay store: jti -> expiry unix
var (
	dpopStoreMu sync.Mutex
	dpopStore   = map[string]int64{}
)

// OPA decision cache (best-effort TTL map)
type opaCacheEntry struct {
	action policy.Action
	exp    int64
}

var (
	opaCacheMu    sync.RWMutex
	opaCache      = map[string]opaCacheEntry{}
	mOPACacheHit  = metrics.NewCounter("orchestrator_opa_cache_hit_total", "OPA cache hits")
	mOPACacheMiss = metrics.NewCounter("orchestrator_opa_cache_miss_total", "OPA cache misses")
)

// dynamic policy version & atomic storage
var (
	policyVersion  atomic.Uint64
	basePolicyVal  atomic.Value // stores policy.Config
	opaPolicyPath  string
	basePolicyPath string
)

// adaptive global IP burst (auto tuned by healthProber)
var (
	ipBurstBase     = envInt("ORCH_IP_BURST", 200)
	ipBurstDegraded = envInt("ORCH_IP_BURST_DEGRADED", 50)
	healthDegradedT = envFloat("ORCH_HEALTH_DEGRADED_RATIO", 0.5) // degrade threshold (0..1)
	currentIPBurst  atomic.Int64
)

func init() {
	currentIPBurst.Store(int64(ipBurstBase))
}

type routeRequest struct {
	Service    string   `json:"service"`
	Tenant     string   `json:"tenant"`
	Scope      string   `json:"scope"`
	Path       string   `json:"path"`
	HashKey    string   `json:"hashKey,omitempty"`
	Algo       string   `json:"algo,omitempty"`       // override per request
	Candidates []string `json:"candidates,omitempty"` // optional adhoc pool
}

type routeResponse struct {
	Target  string `json:"target"`
	Algo    string `json:"algo"`
	Policy  string `json:"policy"`
	Healthy bool   `json:"healthy"`
}

// rate limiter (simple token bucket per key)
type rateLimiter struct {
	capacity int
	window   time.Duration
	mu       sync.Mutex
	store    map[string]bucket
}
type bucket struct {
	remaining int
	reset     time.Time
}

func newRateLimiter(capacity int, window time.Duration) *rateLimiter {
	return &rateLimiter{capacity: capacity, window: window, store: map[string]bucket{}}
}
func (r *rateLimiter) Allow(key string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	b, ok := r.store[key]
	now := time.Now()
	if !ok || now.After(b.reset) {
		r.store[key] = bucket{remaining: r.capacity - 1, reset: now.Add(r.window)}
		return true
	}
	if b.remaining <= 0 {
		r.store[key] = b
		return false
	}
	b.remaining--
	r.store[key] = b
	return true
}

var ipLimiter = newRateLimiter(envInt("ORCH_IP_BURST", 200), time.Minute)

func main() {
	port := envInt("ORCH_PORT", 8080)
	rand.Seed(time.Now().UnixNano())
	// tracing
	shutdown := otelobs.InitTracer(serviceName)
	defer shutdown(context.Background())

	// structured access/security logger setup
	// Structured access/security logger. If filesystem permissions block file creation
	// you can set ACCESSLOG_FALLBACK_STDOUT=1 (or ACCESSLOG_STDOUT_ONLY=1) to avoid
	// crash loops in minimal container FS environments.
	orchLogger, logErr := accesslog.NewLogger(serviceName, "data/orchestrator-access.log", "data/orchestrator-security.log")
	if logErr != nil {
		log.Fatalf("[orchestrator] accesslog init: %v (set ACCESSLOG_FALLBACK_STDOUT=1 to fallback to stdout)", logErr)
	}
	defer orchLogger.Close()

	// optional redis for distributed rate limiting/health sharing
	if addr := os.Getenv("REDIS_ADDR"); addr != "" {
		rdb = redis.NewClient(&redis.Options{Addr: addr})
	}

	// Load policy
	ppath := os.Getenv("ORCH_POLICY_PATH")
	var err error
	basePolicy, err = policy.Load(ppath)
	if err != nil {
		log.Printf("[orchestrator] policy load error: %v (default allow)", err)
		basePolicy = policy.Config{AllowAll: true}
	}
	basePolicyPath = ppath
	basePolicyVal.Store(basePolicy)
	policyVersion.Store(1)
	gPolicyVersion.Set(1)
	opaEng, _ = policy.LoadOPA(os.Getenv("ORCH_OPA_POLICY_PATH"))
	opaPolicyPath = os.Getenv("ORCH_OPA_POLICY_PATH")
	if os.Getenv("ORCH_OPA_ENFORCE") == "1" {
		opaEnforce = true
	}
	// start hot reload watchers (poll-based)
	go watchBasePolicy()
	go watchOPAPolicy()

	// Load pools
	loadPoolsFromEnv()
	// start health probe loop
	go healthProber()

	// RA-TLS optional
	reg.RegisterGauge(gCertExpiry)
	if os.Getenv("RATLS_ENABLE") == "true" {
		td := envStr("RATLS_TRUST_DOMAIN", "shieldx.local")
		ns := envStr("RATLS_NAMESPACE", "default")
		svc := envStr("RATLS_SERVICE", serviceName)
		rotate := envDur("RATLS_ROTATE_EVERY", 45*time.Minute)
		valid := envDur("RATLS_VALIDITY", 60*time.Minute)
		issuer, err = ratls.NewDevIssuer(ratls.Identity{TrustDomain: td, Namespace: ns, Service: svc}, rotate, valid)
		if err != nil {
			log.Fatalf("[orchestrator] RA-TLS init: %v", err)
		}
		go func() {
			for {
				if t, err := issuer.LeafNotAfter(); err == nil {
					gCertExpiry.Set(uint64(time.Until(t).Seconds()))
				}
				time.Sleep(time.Minute)
			}
		}()
	} else {
		log.Printf("[orchestrator] WARNING: RATLS_ENABLE not set; TLS 1.3 not enforced in dev mode")
	}

	// HTTP server
	mux := http.NewServeMux()
	// enhanced endpoints
	mux.HandleFunc("/health", handleHealthEnhanced)
	mux.HandleFunc("/healthz", handleHealthEnhanced)
	mux.HandleFunc("/policy", handlePolicy)
	mux.HandleFunc("/route", func(w http.ResponseWriter, r *http.Request) {
		// allow algo override via header (X-LB-Algo) for experimentation (validated in enhanced handler)
		if alg := r.Header.Get("X-LB-Algo"); alg != "" && r.Method == http.MethodPost {
			r.Header.Set("X-Orch-Requested-Algo", alg)
		}
		handleRouteEnhanced(w, r, orchLogger)
	})
	// admin (secured by admission header if configured)
	mux.HandleFunc("/admin/pools", handleAdminPools)
	mux.HandleFunc("/admin/pools/", handleAdminPoolOne) // /admin/pools/{name}
	// metrics
	reg.Register(mRoute)
	reg.Register(mRouteDenied)
	reg.Register(mRouteErr)
	reg.Register(mHealthOK)
	reg.Register(mHealthBad)
	reg.Register(mCBOpen)
	reg.Register(mCBHalfOpen)
	reg.Register(mCBClose)
	reg.Register(mOPACacheHit)
	reg.Register(mOPACacheMiss)
	reg.RegisterLabeledCounter(mLBPick)
	reg.RegisterHistogram(mProbeDur)
	reg.Register(mPolicyReloads)
	reg.RegisterGauge(gPolicyVersion)
	reg.RegisterGauge(gHealthRatio)
	mux.Handle("/metrics", reg)

	// wrap middlewares: metrics + admission + rate limit + tracing
	httpMetrics := metrics.NewHTTPMetrics(reg, serviceName)
	base := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Correlation-ID: accept incoming or generate, and propagate via header + context
		cid := getOrMakeCorrelationID(r)
		w.Header().Set("X-Correlation-ID", cid)
		r = r.WithContext(context.WithValue(r.Context(), ctxKeyCorrID{}, cid))
		// Admission header check if configured
		if sec := os.Getenv("ADMISSION_SECRET"); sec != "" {
			if !guard.VerifyHeader(r, sec, os.Getenv("ADMISSION_HEADER"), serviceName) {
				http.Error(w, "unauthorized", http.StatusUnauthorized)
				return
			}
		}
		// Basic IP rate limit (use Redis if configured)
		ip := clientIP(r)
		if !allowRate(ip) {
			_ = ledger.AppendJSONLine(ledgerPath, serviceName, "ratelimit.ip", map[string]any{"ip": ip, "path": r.URL.Path, "corrId": getCorrID(r)})
			http.Error(w, "rate limit", http.StatusTooManyRequests)
			return
		}
		// Continue
		mux.ServeHTTP(w, r)
	})
	// structured access + security middleware
	baseWithLog := newSecurityMiddleware(httpMetrics.Middleware(base), orchLogger)
	h := otelobs.WrapHTTPHandler(serviceName, baseWithLog)

	addr := fmt.Sprintf(":%d", port)
	// Listen on all interfaces so other containers (ingress, gateway, etc.) and
	// host port publishing can reach the orchestrator. (Explicit note added after
	// guardian fix which was previously loopback-bound.)
	log.Printf("[orchestrator] listening on %s (algo=%s, policyVersion=%d)", addr, defaultAlgo, policyVersion.Load())
	// Periodic DPoP replay-store GC
	go func() {
		t := time.NewTicker(2 * time.Minute)
		defer t.Stop()
		for range t.C {
			gcDPoPStore()
		}
	}()
	// Unified TLS server setup with graceful shutdown
	var tlsCfg *tls.Config
	var certFile, keyFile string
	var useStatic bool
	if issuer != nil {
		// Enforce client SPIFFE trust domain and optional SAN allowlist prefixes
		allowed := envStr("ORCH_ALLOWED_CLIENT_SAN_PREFIXES", "")
		if strings.TrimSpace(allowed) != "" {
			verifier := func(rawCerts [][]byte, chains [][]*x509.Certificate) error {
				// best-effort: check URI SAN prefixes
				if len(rawCerts) == 0 {
					return fmt.Errorf("no peer cert")
				}
				cert, err := x509.ParseCertificate(rawCerts[0])
				if err != nil {
					return err
				}
				prefs := splitCSV(allowed)
				for _, u := range cert.URIs {
					s := u.String()
					for _, p := range prefs {
						if strings.HasPrefix(s, p) {
							return nil
						}
					}
				}
				for _, dns := range cert.DNSNames {
					for _, p := range prefs {
						if strings.HasPrefix(dns, p) {
							return nil
						}
					}
				}
				for _, ip := range cert.IPAddresses {
					s := ip.String()
					for _, p := range prefs {
						if strings.HasPrefix(s, p) {
							return nil
						}
					}
				}
				if cert.Subject.CommonName != "" {
					for _, p := range prefs {
						if strings.HasPrefix(cert.Subject.CommonName, p) {
							return nil
						}
					}
				}
				return fmt.Errorf("client SAN not allowed")
			}
			tlsCfg = issuer.ServerTLSConfigWithPeerVerifier(true, envStr("RATLS_TRUST_DOMAIN", "shieldx.local"), verifier)
		} else {
			tlsCfg = issuer.ServerTLSConfig(true, envStr("RATLS_TRUST_DOMAIN", "shieldx.local"))
		}
	} else if cert := os.Getenv("ORCH_TLS_CERT_FILE"); cert != "" && os.Getenv("ORCH_TLS_KEY_FILE") != "" {
		// Static TLS mode: enforce mTLS and SAN allowlist if provided
		allowed := envStr("ORCH_ALLOWED_CLIENT_SAN_PREFIXES", "")
		ca := os.Getenv("ORCH_TLS_CLIENT_CA_FILE")
		if ca == "" {
			log.Fatalf("ORCH_TLS_CLIENT_CA_FILE required for mTLS static mode")
		}
		var err error
		if strings.TrimSpace(allowed) != "" {
			tlsCfg, err = tlsutil.LoadServerMTLSWithSANAllow(os.Getenv("ORCH_TLS_CERT_FILE"), os.Getenv("ORCH_TLS_KEY_FILE"), ca, allowed)
		} else {
			tlsCfg, err = tlsutil.LoadServerMTLS(os.Getenv("ORCH_TLS_CERT_FILE"), os.Getenv("ORCH_TLS_KEY_FILE"), ca)
		}
		if err != nil {
			log.Fatalf("TLS load error: %v", err)
		}
		certFile, keyFile = os.Getenv("ORCH_TLS_CERT_FILE"), os.Getenv("ORCH_TLS_KEY_FILE")
		useStatic = true
	} else {
		// Development fallback: allow plain HTTP if explicitly enabled via ORCH_ALLOW_INSECURE=1.
		// Previously this exited hard which prevented local compose / health tests when
		// no RA-TLS or static certs configured. In production, set RATLS_ENABLE=true
		// or configure ORCH_TLS_CERT_FILE / ORCH_TLS_KEY_FILE / ORCH_TLS_CLIENT_CA_FILE.
		if os.Getenv("ORCH_ALLOW_INSECURE") == "1" {
			log.Printf("[orchestrator] WARNING: starting without TLS (ORCH_ALLOW_INSECURE=1)")
			// Use a minimal TLS config placeholder to avoid nil deref later; but we will
			// actually start an HTTP (non-TLS) server below.
			// Set a dummy config so later code does not panic
			// Provide a non-nil placeholder TLS config (not used in insecure mode)
			tlsCfg = &tls.Config{MinVersion: tls.VersionTLS12}
		} else {
			log.Fatalf("TLS required: set RATLS_ENABLE=true or provide ORCH_TLS_CERT_FILE/ORCH_TLS_KEY_FILE env vars (or ORCH_ALLOW_INSECURE=1 for dev)")
		}
	}
	if tlsCfg.MinVersion == 0 {
		tlsCfg.MinVersion = tls.VersionTLS13
	}
	srv := &http.Server{Addr: addr, Handler: h, TLSConfig: tlsCfg, ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 30 * time.Second, WriteTimeout: 30 * time.Second, IdleTimeout: 60 * time.Second}
	// graceful shutdown
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-stop
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := srv.Shutdown(ctx); err != nil {
			log.Printf("[orchestrator] shutdown error: %v", err)
		} else {
			log.Printf("[orchestrator] shutdown complete")
		}
	}()
	var errSrv error
	if os.Getenv("ORCH_ALLOW_INSECURE") == "1" && issuer == nil && !useStatic {
		// Plain HTTP dev mode
		errSrv = srv.ListenAndServe()
	} else if useStatic {
		errSrv = srv.ListenAndServeTLS(certFile, keyFile)
	} else {
		errSrv = srv.ListenAndServeTLS("", "")
	}
	if errSrv != nil && errSrv != http.ErrServerClosed {
		log.Fatalf("[orchestrator] server error: %v", errSrv)
	}
}

// loadBasePolicy returns the current policy.Config atomically
func loadBasePolicy() policy.Config {
	if v := basePolicyVal.Load(); v != nil {
		if pc, ok := v.(policy.Config); ok {
			return pc
		}
	}
	return basePolicy
}

// watchBasePolicy polls for changes to the JSON policy file and hot-reloads it
func watchBasePolicy() {
	if basePolicyPath == "" {
		return
	}
	interval := envDur("ORCH_POLICY_WATCH_EVERY", 3*time.Second)
	var lastMod int64
	for {
		fi, err := os.Stat(basePolicyPath)
		if err == nil {
			mod := fi.ModTime().UnixNano()
			if mod != 0 && mod != lastMod {
				if pc, err := policy.Load(basePolicyPath); err == nil {
					basePolicyVal.Store(pc)
					basePolicy = pc // maintain legacy variable for any direct reads
					last := policyVersion.Add(1)
					gPolicyVersion.Set(last)
					mPolicyReloads.Inc()
					log.Printf("[orchestrator] policy reloaded version=%d", last)
				} else {
					log.Printf("[orchestrator] policy reload error: %v", err)
				}
				lastMod = mod
			}
		}
		time.Sleep(interval)
	}
}

// watchOPAPolicy reloads the OPA policy bundle/regos on file change if a path is provided
func watchOPAPolicy() {
	if opaPolicyPath == "" {
		return
	}
	interval := envDur("ORCH_OPA_WATCH_EVERY", 5*time.Second)
	var lastMod int64
	for {
		fi, err := os.Stat(opaPolicyPath)
		if err == nil {
			mod := fi.ModTime().UnixNano()
			if mod != 0 && mod != lastMod {
				if eng, err := policy.LoadOPA(opaPolicyPath); err == nil {
					opaEng = eng
					log.Printf("[orchestrator] OPA policy reloaded")
				} else {
					log.Printf("[orchestrator] OPA reload error: %v", err)
				}
				lastMod = mod
			}
		}
		time.Sleep(interval)
	}
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	poolsMu.RLock()
	defer poolsMu.RUnlock()
	out := map[string]any{"service": serviceName, "time": time.Now().UTC()}
	stats := map[string]any{}
	for name, p := range pools {
		p.mu.RLock()
		healthy := 0
		arr := make([]map[string]any, 0, len(p.backends))
		for _, b := range p.backends {
			if b.Healthy.Load() {
				healthy++
			}
			arr = append(arr, map[string]any{
				"url": b.URL, "healthy": b.Healthy.Load(), "ewma": b.getEWMA(), "conns": atomic.LoadInt64(&b.Conns), "lastErr": asString(b.LastErr.Load()), "lastLatMs": atomic.LoadUint64(&b.LastLatMs),
			})
		}
		stats[name] = map[string]any{"backends": arr, "healthy": healthy, "total": len(p.backends)}
		p.mu.RUnlock()
	}
	out["pools"] = stats
	writeJSON(w, 200, out)
}

func handlePolicy(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, 200, map[string]any{
		"allowAll":   basePolicy.AllowAll,
		"allowed":    basePolicy.Allowed,
		"advanced":   basePolicy.Advanced,
		"opaLoaded":  opaEng != nil,
		"opaEnforce": opaEnforce,
	})
}

func handleRoute(w http.ResponseWriter, r *http.Request) {
	mRoute.Inc()
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	// limit request body size
	r.Body = http.MaxBytesReader(w, r.Body, int64(envInt("ORCH_MAX_ROUTE_BYTES", 16*1024)))
	// DPoP optional verify
	if jws := r.Header.Get("DPoP"); jws != "" {
		if _, jti, _, err := dpop.VerifyEdDSA(jws, r.Method, normalizeHTU(r), time.Now(), 60); err != nil {
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
	// Policy allow/deny
	action := policy.Evaluate(basePolicy, req.Tenant, req.Scope, req.Path)
	if opaEng != nil {
		if dec, ok := evaluateOPAWithCache(req.Tenant, req.Scope, req.Path, clientIP(r)); ok {
			if opaEnforce {
				action = dec
			}
		}
	}
	if action == policy.ActionDeny {
		mRouteDenied.Inc()
		_ = ledger.AppendJSONLine(ledgerPath, serviceName, "route.denied", map[string]any{"tenant": req.Tenant, "scope": req.Scope, "path": req.Path, "corrId": getCorrID(r)})
		http.Error(w, "policy denied", http.StatusForbidden)
		return
	}
	// Optional: get route hints from OPA (pool/algo/candidates override/augment)
	if opaEng != nil {
		if hint, ok, err := opaEng.EvaluateRoute(map[string]any{"tenant": req.Tenant, "scope": req.Scope, "path": req.Path, "ip": clientIP(r)}); err == nil && ok {
			if v, ok := hint["algo"].(string); ok && v != "" {
				req.Algo = v
			}
			if v, ok := hint["service"].(string); ok && v != "" {
				req.Service = v
			}
			if arr, ok := hint["candidates"].([]any); ok && len(arr) > 0 {
				req.Candidates = req.Candidates[:0]
				for _, x := range arr {
					if s, ok := x.(string); ok && strings.TrimSpace(s) != "" {
						req.Candidates = append(req.Candidates, s)
					}
				}
			}
		}
	}
	// choose pool
	p := buildPoolForRequest(req)
	if p == nil || len(p.backends) == 0 {
		mRouteErr.Inc()
		http.Error(w, "no backends", http.StatusServiceUnavailable)
		return
	}
	algo := defaultAlgo
	if req.Algo != "" {
		algo = parseLBAlgo(req.Algo, defaultAlgo)
	} else if p.algo != "" {
		algo = p.algo
	}
	b, err := pickBackend(p, algo, req.HashKey)
	if err != nil {
		mRouteErr.Inc()
		http.Error(w, err.Error(), http.StatusServiceUnavailable)
		return
	}
	// metric for selection
	mLBPick.Inc(map[string]string{"pool": p.name, "algo": string(algo), "healthy": strconv.FormatBool(b.Healthy.Load())})
	// Audit log successful routing decision
	_ = ledger.AppendJSONLine(ledgerPath, serviceName, "route.ok", map[string]any{
		"tenant": req.Tenant, "scope": req.Scope, "path": req.Path, "algo": string(algo), "target": b.URL, "healthy": b.Healthy.Load(), "corrId": getCorrID(r),
	})
	writeJSON(w, 200, routeResponse{Target: b.URL, Algo: string(algo), Policy: string(action), Healthy: b.Healthy.Load()})
}

func buildPoolForRequest(req routeRequest) *Pool {
	if len(req.Candidates) > 0 {
		return newPool("adhoc", req.Candidates)
	}
	poolsMu.RLock()
	p := pools[strings.ToLower(req.Service)]
	poolsMu.RUnlock()
	return p
}

func pickBackend(p *Pool, algo LBAlgo, hashKey string) (*Backend, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	if len(p.backends) == 0 {
		return nil, errors.New("empty pool")
	}
	// prefer healthy backends; fallback to any if none healthy
	healthy := make([]*Backend, 0, len(p.backends))
	for _, b := range p.backends {
		if b.Healthy.Load() {
			healthy = append(healthy, b)
		}
	}
	candidates := healthy
	if len(candidates) == 0 {
		candidates = p.backends
	}
	switch algo {
	case LBRoundRobin:
		idx := int(atomic.AddUint64(&p.rr, 1)-1) % len(candidates)
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
		// Power-of-two choices with EWMA + in-flight penalty: pick two random candidates and choose lower cost
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

// lbCost computes a selection cost for a backend combining EWMA latency and in-flight connections.
// Lower is better. If EWMA is 0 (uninitialized), treat as very high to prefer warmed backends.
func lbCost(b *Backend) float64 {
	ew := b.getEWMA()
	if ew == 0 {
		ew = math.MaxFloat64
	}
	conns := float64(atomic.LoadInt64(&b.Conns))
	w := b.Weight
	if w < 0.1 {
		w = 0.1
	}
	// lower is better; apply weight as capacity multiplier
	return (ew + p2cPenalty*conns) / w
}

func findByURL(arr []*Backend, url string) (*Backend, error) {
	for _, b := range arr {
		if b.URL == url {
			return b, nil
		}
	}
	// fallback first
	if len(arr) > 0 {
		return arr[0], nil
	}
	return nil, errors.New("no candidate")
}

// chooseRendezvous selects backend with highest FNV-1a 64 hash of key+URL
// chooseRendezvousWeighted selects backend using Rendezvous (highest-random-weight) hashing.
// Uses weighted scoring: score = weight / -ln(u), where u is a deterministic uniform (0,1] derived from hash(key+URL).
func chooseRendezvousWeighted(arr []*Backend, key string) *Backend {
	if len(arr) == 0 {
		return nil
	}
	best := arr[0]
	bestScore := rendezvousScore(best, key)
	for i := 1; i < len(arr); i++ {
		sc := rendezvousScore(arr[i], key)
		if sc > bestScore {
			best = arr[i]
			bestScore = sc
		}
	}
	return best
}

func hash64(s string) uint64 { h := fnv.New64a(); _, _ = h.Write([]byte(s)); return h.Sum64() }

// u01 returns a deterministic uniform (0,1] from a 64-bit hash using the top 53 bits.
func u01(h uint64) float64 {
	// take top 53 bits to form a float64 mantissa
	const denom = 1 << 53
	v := (h >> 11) & ((1 << 53) - 1)
	// Map to (0,1]: add 1 to avoid zero
	return (float64(v) + 1.0) / (denom + 1.0)
}

func rendezvousScore(b *Backend, key string) float64 {
	w := b.Weight
	if w < 0.0001 {
		w = 0.0001
	}
	h := hash64(key + b.URL)
	u := u01(h)
	// score increases with weight; -ln(u) in (0, +inf)
	return w / -math.Log(u)
}

// validServiceName ensures service name is lowercase [a-z0-9-_]{1,64}
func validServiceName(s string) bool {
	if len(s) == 0 || len(s) > 64 {
		return false
	}
	for i := 0; i < len(s); i++ {
		c := s[i]
		if (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-' || c == '_' {
			continue
		}
		return false
	}
	return true
}

// health probing loop with EWMA update
func healthProber() {
	interval := envDur("ORCH_HEALTH_EVERY", 5*time.Second)
	decay := envFloat("ORCH_EWMA_DECAY", 0.3) // 0<alpha<=1
	c := &http.Client{Timeout: envDur("ORCH_HEALTH_TIMEOUT", 1500*time.Millisecond)}
	tick := time.NewTicker(interval)
	defer tick.Stop()
	for range tick.C {
		var total, healthy uint64
		poolsMu.RLock()
		for _, p := range pools {
			p.mu.RLock()
			for _, b := range p.backends {
				total++
				bb := b
				// add small jitter to avoid thundering herd on backends
				go func() {
					time.Sleep(time.Duration(rand.Intn(250)) * time.Millisecond)
					probeOne(bb, c, decay)
				}()
				if b.Healthy.Load() {
					healthy++
				}
			}
			p.mu.RUnlock()
		}
		poolsMu.RUnlock()
		if total > 0 {
			ratio := float64(healthy) / float64(total)
			gHealthRatio.Set(uint64(ratio * 10000))
			// adaptive rate limit: if degraded, reduce IP burst; else restore
			if ratio < healthDegradedT {
				currentIPBurst.Store(int64(ipBurstDegraded))
			} else {
				currentIPBurst.Store(int64(ipBurstBase))
			}
		}
	}
}

func probeOne(b *Backend, c *http.Client, decay float64) {
	// circuit breaker parameters
	maxFails := envInt("ORCH_CB_FAILS", 3)
	openFor := envDur("ORCH_CB_OPEN_FOR", 15*time.Second)
	now := time.Now()
	// if OPEN and not time to try, skip heavy probe
	if b.cbState.Load() == 1 {
		next := time.Unix(0, b.cbNextProbe.Load())
		if now.Before(next) {
			return
		}
		// allow half-open single probe
		b.cbState.Store(2)
		mCBHalfOpen.Inc()
	}
	t0 := time.Now()
	req, _ := http.NewRequest(http.MethodGet, strings.TrimRight(b.URL, "/")+"/healthz", nil)
	resp, err := c.Do(req)
	dur := time.Since(t0)
	// record probe duration in seconds
	if mProbeDur != nil {
		mProbeDur.Observe(dur.Seconds())
	}
	atomic.StoreUint64(&b.LastLatMs, uint64(dur.Milliseconds()))
	if err != nil || resp.StatusCode >= 400 {
		b.Healthy.Store(false)
		b.LastErr.Store(asErrString(err, resp))
		mHealthBad.Inc()
		if resp != nil {
			_ = resp.Body.Close()
		}
		// increase EWMA slightly to penalize
		ew := b.getEWMA()
		if ew == 0 {
			ew = float64(dur.Milliseconds())
		}
		b.setEWMA(ew + 10)
		// circuit breaker: count failures
		fails := b.cbFails.Add(1)
		if int(fails) >= maxFails {
			if b.cbState.Load() != 1 {
				mCBOpen.Inc()
			}
			b.cbState.Store(1) // OPEN
			b.cbNextProbe.Store(time.Now().Add(openFor).UnixNano())
		}
		return
	}
	_ = resp.Body.Close()
	b.Healthy.Store(true)
	b.LastErr.Store("")
	mHealthOK.Inc()
	// EWMA update
	cur := float64(dur.Milliseconds())
	prev := b.getEWMA()
	if prev == 0 {
		prev = cur
	}
	ewma := decay*cur + (1.0-decay)*prev
	b.setEWMA(ewma)
	// circuit breaker reset/close
	if b.cbState.Load() != 0 {
		// success in half-open -> close
		b.cbState.Store(0)
		mCBClose.Inc()
	}
	b.cbFails.Store(0)
}

// helpers
func allowRate(key string) bool {
	if rdb == nil {
		// adjust limiter capacity adaptively
		ipLimiter.capacity = int(currentIPBurst.Load())
		return ipLimiter.Allow(key)
	}
	ctx := context.Background()
	k := "orl:" + key
	cnt, _ := rdb.Incr(ctx, k).Result()
	if cnt == 1 {
		_ = rdb.Expire(ctx, k, time.Minute).Err()
	}
	limit := int(currentIPBurst.Load())
	return int(cnt) <= limit
}

func asErrString(err error, resp *http.Response) string {
	if err != nil {
		return err.Error()
	}
	if resp != nil {
		return fmt.Sprintf("status %d", resp.StatusCode)
	}
	return ""
}
func asString(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	return ""
}

func envInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}
func envStr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
func envDur(key string, def time.Duration) time.Duration {
	if v := os.Getenv(key); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			return d
		}
	}
	return def
}
func envFloat(key string, def float64) float64 {
	if v := os.Getenv(key); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return def
}

func envLBAlgo(key string, def LBAlgo) LBAlgo {
	if v := os.Getenv(key); v != "" {
		return parseLBAlgo(v, def)
	}
	return def
}
func parseLBAlgo(v string, def LBAlgo) LBAlgo {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case string(LBRoundRobin):
		return LBRoundRobin
	case string(LBLeastConnections):
		return LBLeastConnections
	case string(LBEWMA):
		return LBEWMA
	case string(LBP2CEWMA):
		return LBP2CEWMA
	case string(LBConsistentHash):
		return LBConsistentHash
	default:
		return def
	}
}

func loadPoolsFromEnv() {
	// ORCH_BACKENDS_JSON = {"guardian":["http://127.0.0.1:9090"], "ingress":["http://127.0.0.1:8081"]}
	if js := os.Getenv("ORCH_BACKENDS_JSON"); js != "" {
		var m map[string][]string
		if json.Unmarshal([]byte(js), &m) == nil {
			for name, urls := range m {
				upsertPool(name, urls)
			}
		}
	}
	// ORCH_POOL_<NAME>=url1,url2
	for _, e := range os.Environ() {
		if !strings.HasPrefix(e, "ORCH_POOL_") {
			continue
		}
		parts := strings.SplitN(e, "=", 2)
		if len(parts) != 2 {
			continue
		}
		key := parts[0]
		val := parts[1]
		name := strings.ToLower(strings.TrimPrefix(key, "ORCH_POOL_"))
		urls := splitCSV(val)
		upsertPool(name, urls)
	}
}

func upsertPool(name string, urls []string) {
	poolsMu.Lock()
	defer poolsMu.Unlock()
	p := newPool(name, urls)
	// default algo per pool via env ORCH_POOL_ALGO_<NAME>
	if v := os.Getenv("ORCH_POOL_ALGO_" + strings.ToUpper(name)); v != "" {
		p.algo = parseLBAlgo(v, defaultAlgo)
	}
	// optional weights via env ORCH_POOL_WEIGHTS_<NAME>
	if wspec := os.Getenv("ORCH_POOL_WEIGHTS_" + strings.ToUpper(name)); wspec != "" {
		wm := parseWeightsSpec(wspec)
		for _, b := range p.backends {
			if w, ok := wm[b.URL]; ok && w > 0 {
				b.Weight = w
			}
		}
	}
	pools[strings.ToLower(name)] = p
}

func splitCSV(s string) []string {
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if t := strings.TrimSpace(p); t != "" {
			out = append(out, t)
		}
	}
	return out
}

// parseWeightsSpec parses either a JSON object {"url": weight} or a CSV "url=2.0,url2=0.5" into a map.
func parseWeightsSpec(s string) map[string]float64 {
	res := map[string]float64{}
	st := strings.TrimSpace(s)
	if st == "" {
		return res
	}
	if strings.HasPrefix(st, "{") {
		var m map[string]float64
		if json.Unmarshal([]byte(st), &m) == nil {
			for k, v := range m {
				u := strings.TrimRight(strings.TrimSpace(k), "/")
				if u != "" && v > 0 {
					res[u] = v
				}
			}
		}
		return res
	}
	// CSV form url=weight,...
	parts := strings.Split(st, ",")
	for _, p := range parts {
		kv := strings.SplitN(strings.TrimSpace(p), "=", 2)
		if len(kv) != 2 {
			continue
		}
		u := strings.TrimRight(strings.TrimSpace(kv[0]), "/")
		if u == "" {
			continue
		}
		if w, err := strconv.ParseFloat(strings.TrimSpace(kv[1]), 64); err == nil && w > 0 {
			res[u] = w
		}
	}
	return res
}

func clientIP(r *http.Request) string {
	if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
		parts := strings.Split(xff, ",")
		if len(parts) > 0 {
			return strings.TrimSpace(parts[0])
		}
	}
	if rip := r.Header.Get("X-Real-IP"); rip != "" {
		return rip
	}
	host := r.RemoteAddr
	if i := strings.LastIndex(host, ":"); i > 0 {
		return host[:i]
	}
	return host
}

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	// security headers for APIs
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Header().Set("X-Frame-Options", "DENY")
	w.Header().Set("Referrer-Policy", "no-referrer")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func normalizeHTU(r *http.Request) string {
	u := *r.URL
	u.RawQuery = ""
	scheme := "http"
	// prefer forwarded proto if present
	if p := r.Header.Get("X-Forwarded-Proto"); p != "" {
		scheme = p
	}
	if r.TLS != nil {
		scheme = "https"
	}
	return fmt.Sprintf("%s://%s%s", scheme, r.Host, u.Path)
}

// Admin: list all pools (GET /admin/pools)
func handleAdminPools(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	poolsMu.RLock()
	defer poolsMu.RUnlock()
	out := map[string]any{}
	for name, p := range pools {
		p.mu.RLock()
		arr := make([]map[string]any, 0, len(p.backends))
		for _, b := range p.backends {
			arr = append(arr, map[string]any{"url": b.URL, "healthy": b.Healthy.Load(), "ewma": b.getEWMA(), "conns": atomic.LoadInt64(&b.Conns), "weight": b.Weight})
		}
		out[name] = map[string]any{"algo": p.algo, "backends": arr}
		p.mu.RUnlock()
	}
	writeJSON(w, 200, out)
}

// Admin: upsert/delete one pool (/admin/pools/{name})
func handleAdminPoolOne(w http.ResponseWriter, r *http.Request) {
	path := strings.TrimPrefix(r.URL.Path, "/admin/pools/")
	name := strings.TrimSpace(path)
	if name == "" {
		http.Error(w, "missing name", 400)
		return
	}
	switch r.Method {
	case http.MethodPut, http.MethodPost:
		var req struct {
			URLs []string `json:"urls"`
			Algo string   `json:"algo"`
			// optional: weights mapping URL -> weight
			Weights map[string]float64 `json:"weights,omitempty"`
		}
		dec := json.NewDecoder(r.Body)
		dec.DisallowUnknownFields()
		if err := dec.Decode(&req); err != nil {
			http.Error(w, "bad request", 400)
			return
		}
		upsertPool(name, req.URLs)
		// allow overriding algo via request
		if req.Algo != "" {
			poolsMu.Lock()
			if p, ok := pools[strings.ToLower(name)]; ok {
				p.algo = parseLBAlgo(req.Algo, defaultAlgo)
			}
			poolsMu.Unlock()
		}
		// apply weights if provided
		if len(req.Weights) > 0 {
			poolsMu.Lock()
			if p, ok := pools[strings.ToLower(name)]; ok {
				p.mu.Lock()
				for _, b := range p.backends {
					if w, ok := req.Weights[b.URL]; ok && w > 0 {
						b.Weight = w
					}
				}
				p.mu.Unlock()
			}
			poolsMu.Unlock()
		}
		writeJSON(w, 200, map[string]string{"status": "ok"})
	case http.MethodDelete:
		poolsMu.Lock()
		delete(pools, strings.ToLower(name))
		poolsMu.Unlock()
		writeJSON(w, 200, map[string]string{"status": "deleted"})
	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

// OPA cache helpers
func evaluateOPAWithCache(tenant, scope, path, ip string) (policy.Action, bool) {
	if opaEng == nil {
		return "", false
	}
	ttl := envDur("ORCH_OPA_CACHE_TTL", 2*time.Second)
	max := envInt("ORCH_OPA_CACHE_MAX", 10000)
	key := tenant + "|" + scope + "|" + path + "|" + ip
	now := time.Now().UnixNano()
	// read path
	opaCacheMu.RLock()
	if ent, ok := opaCache[key]; ok && now < ent.exp {
		opaCacheMu.RUnlock()
		mOPACacheHit.Inc()
		return ent.action, true
	}
	opaCacheMu.RUnlock()
	// miss -> evaluate
	dec, ok, err := opaEng.Evaluate(map[string]any{"tenant": tenant, "scope": scope, "path": path, "ip": ip})
	if err != nil || !ok {
		mOPACacheMiss.Inc()
		return "", false
	}
	mOPACacheMiss.Inc()
	// write-back with TTL
	opaCacheMu.Lock()
	// simple size cap: drop random when above max
	if len(opaCache) >= max {
		// evict arbitrary 1% to make room
		evict := len(opaCache)/100 + 1
		for k := range opaCache {
			delete(opaCache, k)
			evict--
			if evict <= 0 {
				break
			}
		}
	}
	opaCache[key] = opaCacheEntry{action: dec, exp: time.Now().Add(ttl).UnixNano()}
	opaCacheMu.Unlock()
	return dec, true
}

func gcDPoPStore() {
	dpopStoreMu.Lock()
	now := time.Now().Unix()
	for k, v := range dpopStore {
		if v < now {
			delete(dpopStore, k)
		}
	}
	dpopStoreMu.Unlock()
}

// end of file

// Correlation-ID helpers
type ctxKeyCorrID struct{}

func getCorrID(r *http.Request) string {
	if v := r.Context().Value(ctxKeyCorrID{}); v != nil {
		if s, ok := v.(string); ok {
			return s
		}
	}
	// fallback to header if context missing
	if h := r.Header.Get("X-Correlation-ID"); h != "" {
		return h
	}
	return ""
}

func getOrMakeCorrelationID(r *http.Request) string {
	if cid := r.Header.Get("X-Correlation-ID"); cid != "" {
		return cid
	}
	// generate cryptographically random 16 bytes, hex-encode
	var b [16]byte
	if _, err := crand.Read(b[:]); err == nil {
		return hex.EncodeToString(b[:])
	}
	// fallback: time-based pseudo id
	return fmt.Sprintf("cid-%d", time.Now().UnixNano())
}
