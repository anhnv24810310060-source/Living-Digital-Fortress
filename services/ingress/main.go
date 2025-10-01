package main

import (
    "bytes"
    "context"
    "encoding/json"
    "fmt"
    "io"
    "log"
    "net"
    "net/http"
    "os"
    "strconv"
    "strings"
    "sync"
    "time"

    "shieldx/pkg/ledger"
    "shieldx/pkg/policy"
    "shieldx/pkg/wch"
    "shieldx/pkg/wgcfg"
    "shieldx/pkg/wgctrlmgr"
    "shieldx/pkg/audit"
    "shieldx/pkg/metrics"
	"shieldx/pkg/forensics"
	"shieldx/pkg/guard"
	"shieldx/pkg/xdpguard"
    "shieldx/pkg/dpop"
    redis "github.com/redis/go-redis/v9"
    "os/exec"
    "runtime"
    "shieldx/pkg/wgstate"
    "os/signal"
    "syscall"

    // optional tracing
    otelobs "shieldx/pkg/observability/otel"
    "shieldx/pkg/ratls"
)

type connectRequest struct {
    Token string `json:"token"`
}

type locatorIntrospectResponse struct {
    Active bool                   `json:"active"`
    Claims map[string]interface{} `json:"claims"`
    Error  string                 `json:"error,omitempty"`
}

type connectResponse struct {
    ChannelID string    `json:"channelId"`
    ExpiresAt time.Time `json:"expiresAt"`
}

var (
	serviceName      = "ingress"
	ledgerPath       = "data/ledger-ingress.log"
	pol              policy.Config
	shadowPol        policy.Config
    // naive in-memory rate limiting per ip
	rl             = newRateLimiter(100, time.Minute)
	reg            = metrics.NewRegistry()
	mConnect       = metrics.NewCounter("ingress_connect_total", "Total connect requests")
	mSend          = metrics.NewCounter("ingress_wch_send_total", "Total sealed send relays")
	mDivert        = metrics.NewCounter("ingress_divert_total", "Total suspicious divert requests")
	mConnectDenied = metrics.NewCounter("ingress_connect_denied_total", "Connect denied by policy or token")
	mSendRejected  = metrics.NewCounter("ingress_wch_send_rejected_total", "Rejected sealed sends (size/invalid/expired)")
	mShadowAllow   = metrics.NewCounter("ingress_shadow_allow_total", "Shadow policy allow decisions")
	mShadowDeny    = metrics.NewCounter("ingress_shadow_deny_total", "Shadow policy deny decisions")
	mShadowDivert  = metrics.NewCounter("ingress_shadow_divert_total", "Shadow policy divert decisions")
	mShadowTarpit  = metrics.NewCounter("ingress_shadow_tarpit_total", "Shadow policy tarpit decisions")
	opaEng         *policy.OPAEngine
	opaEnforce     bool
	// DPoP anti-replay store: jti -> expiry unix
	dpopStoreMu    sync.Mutex
	dpopStore      = map[string]int64{}
	// WireGuard server key (base64) held in-memory (for linux wgctrl EnsureDevice)
	wgServerPrivB64 string
	wgServerPubB64  string
	wgSt            *wgstate.Store
	// WG metrics
	mWGAdd      = metrics.NewCounter("wg_add_peer_total", "WG add peer ops")
	mWGRemove   = metrics.NewCounter("wg_remove_peer_total", "WG remove peer ops")
	mWGRotate   = metrics.NewCounter("wg_rotate_total", "WG rotate ops")
	mWGRouteErr = metrics.NewCounter("wg_route_errors_total", "WG route/NAT errors")
	mWGThrottle = metrics.NewCounter("wg_throttle_hits_total", "WG throttle hits")
	gWGPeers    = metrics.NewGauge("wg_peers", "Current active WG peers")
	gWGRxBytes  = metrics.NewGauge("wg_rx_bytes", "Aggregated RX bytes of all peers")
	gWGTxBytes  = metrics.NewGauge("wg_tx_bytes", "Aggregated TX bytes of all peers")
	gWGHandshakeStale = metrics.NewGauge("wg_handshake_stale", "Number of peers with stale handshakes")
	gWGStaleRatio = metrics.NewGauge("wg_handshake_stale_ratio", "Stale handshake peers ratio x10000")
	mWGStateRep = metrics.NewCounter("wg_state_replicate_total", "WG state replication ops")
	mWGRotateFail = metrics.NewCounter("wg_rotate_fail_total", "WG rotate failures")
	mWGRotateStart = metrics.NewCounter("wg_rotate_start_total", "WG rotate two-phase start")
	mWGRotateCommit = metrics.NewCounter("wg_rotate_commit_total", "WG rotate two-phase commit")
	// Allowed subnets per scope (from env WG_SCOPE_SUBNETS as JSON: {"api":["10.20.0.0/16"], ...})
	wgScopeSubnets = map[string][]string{}
	// New: route/quota metrics
	mWGRouteSetup = metrics.NewCounter("wg_route_setup_total", "WG route/NAT setups")
	mWGRouteTeardown = metrics.NewCounter("wg_route_teardown_total", "WG route/NAT teardowns")
	mWGTCConfigured = metrics.NewCounter("wg_tc_config_total", "WG TC bandwidth class configured")
	mWGPPSConfigured = metrics.NewCounter("wg_pps_config_total", "WG PPS nft limit configured")
    // RA-TLS
    ratlsIssuer *ratls.AutoIssuer
    gCertExpiry = metrics.NewGauge("ratls_cert_expiry_seconds", "Seconds until current RA-TLS cert expiry")
)

func loadWGScopeSubnets() {
	if js := os.Getenv("WG_SCOPE_SUBNETS"); js != "" {
		_ = json.Unmarshal([]byte(js), &wgScopeSubnets)
	}
}

func computeAllowedIPs(scope, clientCIDR string) string {
	allowed := []string{clientCIDR}
	if subs, ok := wgScopeSubnets[scope]; ok {
		allowed = append(allowed, subs...)
	} else {
		// default allow guardian endpoint only
		allowed = append(allowed, "10.10.0.1/32")
	}
	return strings.Join(allowed, ",")
}

func applyTCBandwidthLimit(iface, tenant, clientCIDR string, kbps int) {
	if runtime.GOOS != "linux" || kbps <= 0 { return }
	ipOnly := strings.Split(clientCIDR, "/")[0]
	// root qdisc
	_ = exec.Command("sh", "-c", fmt.Sprintf("tc qdisc show dev %s | grep -q 'htb 1:' || tc qdisc add dev %s root handle 1: htb default 30", iface, iface)).Run()
	// deterministic class id per tenant
	hash := 0
	for i:=0;i<len(tenant);i++{ hash = (hash*31 + int(tenant[i])) & 0xFFF }
	if hash < 10 { hash = 10 }
	classid := fmt.Sprintf("1:%x", hash)
	_ = exec.Command("sh", "-c", fmt.Sprintf("tc class replace dev %s parent 1: classid %s htb rate %dkbit ceil %dkbit", iface, classid, kbps, kbps)).Run()
	_ = exec.Command("sh", "-c", fmt.Sprintf("tc filter replace dev %s protocol ip parent 1:0 prio 1 u32 match ip src %s flowid %s", iface, ipOnly, classid)).Run()
	mWGTCConfigured.Inc()
}

// channelRegistry stores short-lived channel state to validate sealed relays
type channelRegistry struct {
	mu   sync.RWMutex
	m    map[string]chanInfo
}
type chanInfo struct {
	Tenant string
	Scope  string
	Expiry time.Time
}

func newChannelRegistry() *channelRegistry { return &channelRegistry{m: make(map[string]chanInfo)} }
func (c *channelRegistry) add(id, tenant, scope string, exp time.Time) {
	c.mu.Lock(); c.m[id] = chanInfo{Tenant: tenant, Scope: scope, Expiry: exp}; c.mu.Unlock()
}
func (c *channelRegistry) get(id string) (chanInfo, bool) {
	c.mu.RLock(); v, ok := c.m[id]; c.mu.RUnlock(); return v, ok
}
func (c *channelRegistry) purgeExpired(now time.Time) {
	c.mu.Lock()
	for k, v := range c.m { if now.After(v.Expiry) { delete(c.m, k) } }
	c.mu.Unlock()
}

var chanReg = newChannelRegistry()

// tenant-level rate limit
var (
	tenantRL   = map[string]*rateLimiter{}
	tenantMu   sync.Mutex
	// redis client for distributed limiting (optional)
	rdb        *redis.Client
)

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(status)
    _ = json.NewEncoder(w).Encode(v)
}

func connectHandler(w http.ResponseWriter, r *http.Request) {
    mConnect.Inc()
    if r.Method != http.MethodPost {
        http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
        return
    }
    var req connectRequest
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, "bad request", http.StatusBadRequest)
        return
    }
    if req.Token == "" {
        http.Error(w, "missing token", http.StatusBadRequest)
        return
    }
	// Enforce DPoP if provided
	if jws := r.Header.Get("DPoP"); jws != "" {
		if jkt, jti, iat, err := dpop.VerifyEdDSA(jws, r.Method, r.URL.String(), time.Now(), 60); err != nil {
			http.Error(w, "dpop invalid", http.StatusUnauthorized)
			return
		} else {
			_ = jkt; _ = iat
			// anti-replay window 2 minutes
			dpopStoreMu.Lock()
			exp := time.Now().Add(2 * time.Minute).Unix()
			if old, ok := dpopStore[jti]; ok && old >= time.Now().Unix() {
				dpopStoreMu.Unlock()
				http.Error(w, "dpop replay", http.StatusUnauthorized)
				return
			}
			dpopStore[jti] = exp
			dpopStoreMu.Unlock()
		}
    }

    // Introspect token against Locator
    loc := os.Getenv("LOCATOR_URL")
    if loc == "" {
        loc = "http://localhost:8080"
    }
    body, _ := json.Marshal(map[string]string{"token": req.Token})
    httpResp, err := getHTTPClient().Post(loc+"/introspect", "application/json", bytes.NewReader(body))
    if err != nil {
        http.Error(w, "locator unavailable", http.StatusBadGateway)
        _ = ledger.AppendJSONLine(ledgerPath, serviceName, "connect.locator_error", map[string]any{"error": err.Error()})
        return
    }
    defer httpResp.Body.Close()
    respBytes, _ := io.ReadAll(httpResp.Body)
    if httpResp.StatusCode != 200 {
        http.Error(w, "introspection failed", http.StatusUnauthorized)
        _ = ledger.AppendJSONLine(ledgerPath, serviceName, "connect.introspect_http", map[string]any{"status": httpResp.StatusCode})
        return
    }
    var introspect locatorIntrospectResponse
    if err := json.Unmarshal(respBytes, &introspect); err != nil {
        http.Error(w, "introspection parse error", http.StatusUnauthorized)
        return
    }
    if !introspect.Active {
		mConnectDenied.Inc()
        http.Error(w, "invalid token", http.StatusUnauthorized)
        return
    }
    tenant, _ := introspect.Claims["tenant"].(string)
    scope, _ := introspect.Claims["scope"].(string)
    // Advanced policy evaluation (first-match)
    action := policy.Evaluate(pol, tenant, scope, "/")
	// OPA enforcement (if configured) overrides base action when decision exists
	if opaEng != nil && opaEnforce {
		if dec, ok, err := opaEng.Evaluate(map[string]any{"tenant": tenant, "scope": scope, "path": "/", "ip": clientIP(r)}); err == nil && ok {
			action = dec
		}
	}
	// Shadow-eval: OPA first, fallback to shadow JSON rules if provided
	if opaEng != nil {
		if dec, ok, err := opaEng.Evaluate(map[string]any{"tenant": tenant, "scope": scope, "path": "/", "ip": clientIP(r)}); err == nil && ok {
			switch dec {
			case policy.ActionAllow: mShadowAllow.Inc()
			case policy.ActionDeny: mShadowDeny.Inc()
			case policy.ActionDivert: mShadowDivert.Inc()
			case policy.ActionTarpit: mShadowTarpit.Inc()
			}
			_ = ledger.AppendJSONLine(ledgerPath, serviceName, "policy.shadow_opa", map[string]any{"tenant": tenant, "scope": scope, "decision": dec})
		}
    } else if shadowPol.AllowAll || len(shadowPol.Allowed) > 0 || len(shadowPol.Advanced) > 0 {
        shadowAction := policy.Evaluate(shadowPol, tenant, scope, "/")
		switch shadowAction {
		case policy.ActionAllow: mShadowAllow.Inc()
		case policy.ActionDeny: mShadowDeny.Inc()
		case policy.ActionDivert: mShadowDivert.Inc()
		case policy.ActionTarpit: mShadowTarpit.Inc()
		}
        _ = ledger.AppendJSONLine(ledgerPath, serviceName, "policy.shadow_decision", map[string]any{"tenant": tenant, "scope": scope, "action": shadowAction})
    }
    if action == policy.ActionDeny {
		mConnectDenied.Inc()
        http.Error(w, "policy denied", http.StatusForbidden)
        _ = ledger.AppendJSONLine(ledgerPath, serviceName, "connect.policy_denied", map[string]any{"tenant": tenant, "scope": scope})
        return
    }
	if action == policy.ActionDivert {
		mDivert.Inc()
		// best-effort divert to decoy manager
		dcm := os.Getenv("DECOY_MGR_URL")
		if dcm == "" { dcm = "http://localhost:8083" }
        getHTTPClient().Post(dcm+"/spawn", "application/json", bytes.NewReader([]byte(fmt.Sprintf(`{"tenant":"%s","kind":"http"}`, tenant))))
        getHTTPClient().Post(dcm+"/analyze", "application/json", bytes.NewReader([]byte(`{"decoyId":"auto","event":"diverted","payload":"connect"}`)))
		w.WriteHeader(http.StatusAccepted)
		_, _ = w.Write([]byte("diverted"))
        return
    }
    if action == policy.ActionTarpit && pol.TarpitMs > 0 {
        time.Sleep(time.Duration(pol.TarpitMs) * time.Millisecond)
    }

    // Create Whisper Channel info (app-layer). Guardian generates static pubkey for PoC, but here we fetch from /wch/pubkey.
    channelID := fmt.Sprintf("wch_%d", time.Now().UnixNano())
    exp := time.Now().Add(5 * time.Minute)
    guardian := fmt.Sprintf("http://127.0.0.1:%d", getenvInt("GUARDIAN_PORT", 9090))
    resp, err := getHTTPClient().Get(guardian + "/wch/pubkey")
    if err != nil || resp.StatusCode != 200 {
        http.Error(w, "guardian pubkey unavailable", http.StatusBadGateway)
        return
    }
    var pub struct{ PubKey string `json:"pubKey"` }
    if err := json.NewDecoder(resp.Body).Decode(&pub); err != nil { http.Error(w, "guardian pubkey parse", 502); return }
    _ = resp.Body.Close()
	// Record channel for validation on /wch/send
	chanReg.add(channelID, tenant, scope, exp)
    // Reply with channel info and guardian pubkey for client ECDH
    cr := wch.ConnectResponse{ChannelID: channelID, GuardianPubB64: pub.PubKey, Protocol: wch.Protocol, ExpiresAt: exp.Unix(), RebindHintMs: 500}
    _ = ledger.AppendJSONLine(ledgerPath, serviceName, "channel.created", map[string]any{"channelId": channelID, "exp": exp})
	// Optional one-time redemption: revoke token after successful connect to prevent reuse
	if os.Getenv("INGRESS_REDEEM_ONCE") == "1" {
		go func(tok string) {
			loc := os.Getenv("LOCATOR_URL")
			if loc == "" { loc = "http://localhost:8080" }
			body, _ := json.Marshal(map[string]string{"token": tok})
            resp, err := getHTTPClient().Post(loc+"/revoke", "application/json", bytes.NewReader(body))
			if err == nil && resp != nil { resp.Body.Close() }
			_ = ledger.AppendJSONLine(ledgerPath, serviceName, "connect.redeem_once", map[string]any{"channelId": channelID})
		}(req.Token)
	}
    writeJSON(w, http.StatusOK, cr)
}

func main() {
    port := getenvInt("INGRESS_PORT", 8081)
    // OpenTelemetry tracing (no-op if OTEL_EXPORTER_OTLP_ENDPOINT unset)
    shutdown := otelobs.InitTracer(serviceName)
    defer shutdown(context.Background())
    // Load policy from file if provided
    cfgPath := os.Getenv("INGRESS_POLICY_PATH")
    var err error
    pol, err = policy.Load(cfgPath)
    if err != nil {
        log.Printf("[ingress] policy load error: %v (default allow)", err)
        pol = policy.Config{AllowAll: true}
    }
	// Load shadow policy (optional)
	shadowPath := os.Getenv("INGRESS_SHADOW_POLICY_PATH")
	shadowPol, _ = policy.Load(shadowPath)
	// Load OPA policy for shadow-eval (optional)
	var opaPath = os.Getenv("INGRESS_OPA_POLICY_PATH")
	var errOPA error
	opaEng, errOPA = policy.LoadOPA(opaPath)
	if errOPA != nil {
		log.Printf("[ingress] OPA load error: %v (shadow OPA disabled)", errOPA)
	}
    // OPA bundle poller (optional): not available in this build; static policy load only
	// init redis if provided
	if addr := os.Getenv("REDIS_ADDR"); addr != "" {
		rdb = redis.NewClient(&redis.Options{Addr: addr})
	}
	// open wg state
	if p := os.Getenv("WG_STATE_PATH"); p != "" {
		if st, err := wgstate.Open(p); err == nil { wgSt = st } else { log.Printf("[ingress] wgstate open: %v", err) }
	}
	loadWGScopeSubnets()
	// set public endpoint (host:port) for this PoP and replicate
	if wgSt != nil {
		ep := os.Getenv("WG_PUBLIC_ENDPOINT")
		if ep != "" { wgSt.SetEndpoint(ep) }
		hurl := os.Getenv("WG_PUBLIC_HEALTH_URL")
		if hurl != "" { wgSt.SetHealthURL(hurl) }
		_ = wgSt.Save(); replicateWGState()
	}
	// auto route/NAT (Linux only)
	if runtime.GOOS == "linux" {
		iface := os.Getenv("WG_IFACE"); if iface=="" { iface = "wg-shieldx" }
		ifaceAddr := os.Getenv("WG_ADDR"); if ifaceAddr=="" { ifaceAddr = "10.10.0.1/24" }
		if err := ensureLinuxWGRoute(iface, ifaceAddr); err != nil { mWGRouteErr.Inc() } else { mWGRouteSetup.Inc() }
		// graceful teardown on shutdown
		go func(iface string){
			ch := make(chan os.Signal, 1)
			signal.Notify(ch, syscall.SIGINT, syscall.SIGTERM)
			<-ch
			if err := teardownLinuxWGRoute(iface); err == nil { mWGRouteTeardown.Inc() }
			os.Exit(0)
		}(iface)
	}
	if os.Getenv("INGRESS_OPA_ENFORCE") == "1" { opaEnforce = true }
    mux := http.NewServeMux()
    mux.HandleFunc("/connect", connectHandler)
    // Suspicious traffic stub: forward to decoy manager to spawn/analyze
    mux.HandleFunc("/suspicious", func(w http.ResponseWriter, r *http.Request) {
        tenant := r.URL.Query().Get("tenant")
        if tenant == "" { tenant = "unknown" }
        dcm := os.Getenv("DECOY_MGR_URL")
        if dcm == "" { dcm = "http://localhost:8083" }
        // spawn (no-op in PoC) and log
    getHTTPClient().Post(dcm+"/spawn", "application/json", bytes.NewReader([]byte(fmt.Sprintf(`{"tenant":"%s","kind":"http"}`, tenant))))
    getHTTPClient().Post(dcm+"/analyze", "application/json", bytes.NewReader([]byte(`{"decoyId":"auto","event":"suspect","payload":"sample"}`)))
        w.WriteHeader(202)
        _, _ = w.Write([]byte("diverted to decoy"))
    })
    // WireGuard health endpoint
    mux.HandleFunc("/wg/health", func(w http.ResponseWriter, r *http.Request) {
        iface := os.Getenv("WG_IFACE"); if iface=="" { iface = "wg-shieldx" }
        if mgr, err2 := wgctrlmgr.New(); err2 == nil {
            if arr, err := mgr.ListPeers(iface); err == nil {
                gWGPeers.Set(uint64(len(arr)))
                var rx, tx uint64
                var stale uint64
                cutoff := time.Now().Add(-2 * time.Minute)
                for _, p := range arr {
                    if p.LastHandshake.Before(cutoff) { stale++ }
                    if p.RxBytes > 0 { rx += uint64(p.RxBytes) }
                    if p.TxBytes > 0 { tx += uint64(p.TxBytes) }
                }
                gWGRxBytes.Set(rx)
                gWGTxBytes.Set(tx)
                gWGHandshakeStale.Set(stale)
                ratio := uint64(0)
                if len(arr) > 0 { ratio = (stale * 10000) / uint64(len(arr)) }
                gWGStaleRatio.Set(ratio)
                writeJSON(w, 200, map[string]any{"peers": len(arr), "stale": stale, "rx": rx, "tx": tx, "stale_ratio_x10000": ratio, "ok": ratio < 3000})
                return
            }
        }
        writeJSON(w, 200, map[string]any{"peers": 0})
    })
    // E2E sealed request relay: client posts an envelope; ingress forwards to guardian without decryption
    mux.HandleFunc("/wch/send", func(w http.ResponseWriter, r *http.Request) {
        if r.Method != http.MethodPost { http.Error(w, "method not allowed", 405); return }
        mSend.Inc()
		// Distributed limiter (optional): key by tenant+jkt+ip
		if rdb != nil {
			tenantKey := "unk"
			if infoT, ok := r.Context().Value("tenant").(string); ok { tenantKey = infoT }
			jkt := r.Header.Get("DPoP-JKT")
			ip := clientIP(r)
			key := fmt.Sprintf("lim:%s:%s:%s", tenantKey, jkt, ip)
			ctx := context.Background()
			cnt, _ := rdb.Incr(ctx, key).Result()
			if cnt == 1 { _ = rdb.Expire(ctx, key, time.Minute).Err() }
			limit := getenvInt("LIMIT_PER_MIN", 600)
			if int(cnt) > limit { http.Error(w, "rate limit", http.StatusTooManyRequests); return }
		}
		// Enforce max envelope size and validate channel id exists and not expired
		maxBytes := getenvInt("WCH_MAX_ENVELOPE_BYTES", 65536)
		r.Body = http.MaxBytesReader(w, r.Body, int64(maxBytes))
		var env wch.Envelope
		if err := json.NewDecoder(r.Body).Decode(&env); err != nil {
			mSendRejected.Inc()
			http.Error(w, "invalid envelope", http.StatusBadRequest)
			return
		}
		if env.ChannelID == "" || env.EphemeralPubB64 == "" || env.NonceB64 == "" || env.CiphertextB64 == "" {
			mSendRejected.Inc()
			http.Error(w, "missing fields", http.StatusBadRequest)
			return
		}
		info, ok := chanReg.get(env.ChannelID)
		if !ok {
			mSendRejected.Inc()
			http.Error(w, "unknown channel", http.StatusUnauthorized)
			return
		}
		if time.Now().After(info.Expiry) {
			mSendRejected.Inc()
			http.Error(w, "channel expired", http.StatusUnauthorized)
			return
		}
		// Per-tenant rate limit for sealed sends
		tenantMu.Lock()
		tlim, ok := tenantRL[info.Tenant]
		if !ok { tlim = newRateLimiter(300, time.Minute); tenantRL[info.Tenant] = tlim }
		tenantMu.Unlock()
		if !tlim.Allow(info.Tenant) {
			mSendRejected.Inc()
			http.Error(w, "rate limit", http.StatusTooManyRequests)
			return
		}
		// Forward validated envelope to guardian
        guardian := fmt.Sprintf("http://127.0.0.1:%d/wch/recv", getenvInt("GUARDIAN_PORT", 9090))
		buf, _ := json.Marshal(env)
        resp, err := getHTTPClient().Post(guardian, "application/json", bytes.NewReader(buf))
        if err != nil { http.Error(w, "guardian unavailable", 502); return }
        defer resp.Body.Close()
        for k, vv := range resp.Header { for _, v := range vv { w.Header().Add(k, v) } }
        w.WriteHeader(resp.StatusCode)
        io.Copy(w, resp.Body)
    })
    // E2E sealed UDP relay via guardian
    mux.HandleFunc("/wch/send-udp", func(w http.ResponseWriter, r *http.Request) {
        if r.Method != http.MethodPost { http.Error(w, "method not allowed", 405); return }
        mSend.Inc()
        maxBytes := getenvInt("WCH_MAX_ENVELOPE_BYTES", 65536)
        r.Body = http.MaxBytesReader(w, r.Body, int64(maxBytes))
        var env wch.Envelope
        if err := json.NewDecoder(r.Body).Decode(&env); err != nil { mSendRejected.Inc(); http.Error(w, "invalid envelope", 400); return }
        if env.ChannelID == "" || env.EphemeralPubB64 == "" || env.NonceB64 == "" || env.CiphertextB64 == "" { mSendRejected.Inc(); http.Error(w, "missing fields", 400); return }
        info, ok := chanReg.get(env.ChannelID)
        if !ok || time.Now().After(info.Expiry) { mSendRejected.Inc(); http.Error(w, "channel invalid", 401); return }
        guardian := fmt.Sprintf("http://127.0.0.1:%d/wch/recv-udp", getenvInt("GUARDIAN_PORT", 9090))
        buf, _ := json.Marshal(env)
    resp, err := getHTTPClient().Post(guardian, "application/json", bytes.NewReader(buf))
        if err != nil { http.Error(w, "guardian unavailable", 502); return }
        defer resp.Body.Close()
        for k, vv := range resp.Header { for _, v := range vv { w.Header().Add(k, v) } }
        w.WriteHeader(resp.StatusCode)
        io.Copy(w, resp.Body)
    })
    // WireGuard config generation + peer register with multi-PoP endpoints
    mux.HandleFunc("/wg/client-config", func(w http.ResponseWriter, r *http.Request) {
        if r.Method != http.MethodPost { http.Error(w, "method not allowed", 405); return }
        type req struct { Tenant string `json:"tenant"`; Scope string `json:"scope"` }
        var rr req
        if err := json.NewDecoder(r.Body).Decode(&rr); err != nil { http.Error(w, "bad request", 400); return }
        // X25519 keys and optional wgctrl server peer registration (linux only)
        cPriv, cPub, _ := wgcfg.GenerateKeypair()
        var sPriv, sPub string
        if wgServerPrivB64 == "" || wgServerPubB64 == "" {
            sPriv, sPub, _ = wgcfg.GenerateKeypair()
            wgServerPrivB64, wgServerPubB64 = sPriv, sPub
        } else { sPriv = wgServerPrivB64; sPub = wgServerPubB64 }
        iface := os.Getenv("WG_IFACE"); if iface=="" { iface = "wg-shieldx" }
        // Allocate IP via wgstate
        clientCIDR := "10.10.0.2/32"
        if wgSt != nil {
            ip := wgSt.AllocateIP(cPub)
            if ip != "" { clientCIDR = ip }
        }
        // AllowedIPs from scope
        allowed := computeAllowedIPs(rr.Scope, clientCIDR)
        // Ensure device (userspace fallback on non-Linux) and add peer
        ensureUserspaceWG(iface, sPriv, getenvInt("WG_LISTEN_PORT", 51820))
        if mgr, err := wgctrlmgr.New(); err == nil {
            _ = mgr.EnsureDevice(iface, sPriv, getenvInt("WG_LISTEN_PORT", 51820))
            if err := mgr.AddPeer(iface, cPub, clientCIDR, "", 25, 30*time.Minute); err == nil { mWGAdd.Inc() }
        }
        // Persist state
        if wgSt != nil { wgSt.AddPeer(wgstate.Peer{Tenant: rr.Tenant, Scope: rr.Scope, ClientPub: cPub, AssignedIP: clientCIDR}); _ = wgSt.Save(); replicateWGState() }
        // Apply bandwidth limit (TC) and PPS limit (nftables) per tenant (Linux only)
        if runtime.GOOS == "linux" {
            // TC kbps
            kbps := getenvInt("TENANT_BW_KBPS", 0)
            if v := os.Getenv("TENANT_"+strings.ToUpper(rr.Tenant)+"_KBPS"); v != "" { if n, err := strconv.Atoi(v); err==nil { kbps = n } }
            applyTCBandwidthLimit(iface, rr.Tenant, clientCIDR, kbps)
            // PPS via nft
            pps := getenvInt("TENANT_PPS_LIMIT", 0)
            if v := os.Getenv("TENANT_"+strings.ToUpper(rr.Tenant)+"_PPS"); v != "" { if n, err := strconv.Atoi(v); err==nil { pps = n } }
            if pps > 0 {
                ipOnly := strings.Split(clientCIDR, "/")[0]
                cmd := fmt.Sprintf("nft add rule ip filter FORWARD ip saddr %s limit rate over %d/second drop", ipOnly, pps)
                if err := exec.Command("sh", "-c", cmd).Run(); err != nil { mWGRouteErr.Inc() } else { mWGThrottle.Inc(); mWGPPSConfigured.Inc() }
            }
        }
        // Discover multi-PoP endpoints with health URLs
        type popInfo struct { Endpoint string; Health string }
        pops := []popInfo{}
        if rdb != nil {
            ctx := context.Background()
            keys, _ := rdb.Keys(ctx, "wgstate:*").Result()
            for _, k := range keys {
                if js, err := rdb.Get(ctx, k).Result(); err == nil {
                    var st wgstate.State
                    if json.Unmarshal([]byte(js), &st) == nil && st.PublicEndpoint != "" {
                        pops = append(pops, popInfo{Endpoint: st.PublicEndpoint, Health: st.PublicHealthURL})
                    }
                }
            }
        }
        localEp := os.Getenv("WG_PUBLIC_ENDPOINT")
        if localEp == "" { localEp = "127.0.0.1:"+strconv.Itoa(getenvInt("WG_LISTEN_PORT", 51820)) }
        // If a staged alt key exists locally, add alt endpoint (same host, alt port)
        if wgSt != nil {
            st := wgSt.Get()
            if st.AltListenPort > 0 && localEp != "" {
                if alt := replacePort(localEp, st.AltListenPort); alt != "" { pops = append(pops, popInfo{Endpoint: alt}) }
            }
        }
        uniq := map[string]struct{}{}
        outEP := []string{}
        outHealth := []string{}
        // local first
        if localEp != "" { if _,ok:=uniq[localEp]; !ok { uniq[localEp]=struct{}{}; outEP = append(outEP, localEp); outHealth = append(outHealth, os.Getenv("WG_PUBLIC_HEALTH_URL")) } }
        for _, pi := range pops {
            if _, ok := uniq[pi.Endpoint]; !ok {
                uniq[pi.Endpoint] = struct{}{}
                outEP = append(outEP, pi.Endpoint)
                outHealth = append(outHealth, pi.Health)
            }
        }
        // Return JSON config for intelligent clients
        writeJSON(w, 200, map[string]any{
            "privateKey": cPriv,
            "assignedCIDR": clientCIDR,
            "serverPub": sPub,
            "endpoints": outEP,
            "stagedServerPub": func() string { if wgSt==nil { return "" }; st:=wgSt.Get(); return st.AltServerPubB64 }(),
            "health": outHealth,
            "allowedIPs": allowed,
            "listenPort": getenvInt("WG_LISTEN_PORT", 51820),
        })
        _ = ledger.AppendJSONLine(ledgerPath, serviceName, "wg.client_config", map[string]any{"tenant": rr.Tenant, "scope": rr.Scope, "clientPub": cPub, "endpoints": outEP})
    })
    // WireGuard remove peer
    mux.HandleFunc("/wg/remove", func(w http.ResponseWriter, r *http.Request) {
        if r.Method != http.MethodPost { http.Error(w, "method not allowed", 405); return }
        var req struct{ Pub string `json:"pub"` }
        if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Pub == "" { http.Error(w, "bad request", 400); return }
        iface := os.Getenv("WG_IFACE"); if iface=="" { iface = "wg-shieldx" }
        if mgr, err := wgctrlmgr.New(); err == nil { if mgr.RemovePeer(iface, req.Pub) == nil { mWGRemove.Inc() } }
        if wgSt != nil { wgSt.RemovePeer(req.Pub); _ = wgSt.Save(); replicateWGState() }
        _ = ledger.AppendJSONLine(ledgerPath, serviceName, "wg.remove", map[string]any{"pub": req.Pub})
        w.WriteHeader(204)
    })
    // WireGuard rotate server key (2-phase)
    mux.HandleFunc("/wg/rotate", func(w http.ResponseWriter, r *http.Request) {
        if r.Method != http.MethodPost { http.Error(w, "method not allowed", 405); return }
        iface := os.Getenv("WG_IFACE"); if iface=="" { iface = "wg-shieldx" }
        // phase 1: generate new key, apply and stage in wgstate
        sPrivNew, sPubNew, _ := wgcfg.GenerateKeypair()
        mWGRotateStart.Inc()
        if mgr, err := wgctrlmgr.New(); err == nil {
            // Configure staged ALT interface to run alongside current one (Linux only route config)
            altIface := os.Getenv("WG_ALT_IFACE"); if altIface=="" { altIface = "wg-shieldx2" }
            altPort := getenvInt("WG_ALT_LISTEN_PORT", 51822)
            if err := mgr.EnsureDevice(altIface, sPrivNew, altPort); err != nil { mWGRotateFail.Inc(); http.Error(w, "rotate failed", 500); return }
            if runtime.GOOS == "linux" {
                altAddr := os.Getenv("WG_ALT_ADDR"); if altAddr=="" { altAddr = "10.11.0.1/24" }
                _ = ensureLinuxWGRoute(altIface, altAddr)
            }
        }
        if wgSt != nil { wgSt.SetAltServer(sPrivNew, sPubNew, getenvInt("WG_ALT_LISTEN_PORT", 51822)); _ = wgSt.Save(); replicateWGState() }
        _ = ledger.AppendJSONLine(ledgerPath, serviceName, "wg.rotate_start", map[string]any{"serverPubNew": sPubNew, "altPort": getenvInt("WG_ALT_LISTEN_PORT", 51822)})
        writeJSON(w, 200, map[string]string{"serverPubNew": sPubNew})
    })

    // commit rotation: promote staged key to primary
    mux.HandleFunc("/wg/rotate/commit", func(w http.ResponseWriter, r *http.Request) {
        if r.Method != http.MethodPost { http.Error(w, "method not allowed", 405); return }
        iface := os.Getenv("WG_IFACE"); if iface=="" { iface = "wg-shieldx" }
        if wgSt == nil { http.Error(w, "no state", 500); return }
        st := wgSt.Get()
        if st.AltServerPrivB64 == "" || st.AltServerPubB64 == "" { http.Error(w, "no alt key", 400); return }
        wgServerPrivB64, wgServerPubB64 = st.AltServerPrivB64, st.AltServerPubB64
        wgSt.SetServer(st.AltServerPrivB64, st.AltServerPubB64)
        wgSt.ClearAltServer()
        _ = wgSt.Save(); replicateWGState()
        if mgr, err := wgctrlmgr.New(); err == nil {
            _ = mgr.EnsureDevice(iface, wgServerPrivB64, getenvInt("WG_LISTEN_PORT", 51820))
            // Teardown ALT interface after promotion (best-effort)
            altIface := os.Getenv("WG_ALT_IFACE"); if altIface=="" { altIface = "wg-shieldx2" }
            if runtime.GOOS == "linux" { _ = exec.Command("sh", "-c", fmt.Sprintf("ip link del dev %s", altIface)).Run() }
        }
        _ = ledger.AppendJSONLine(ledgerPath, serviceName, "wg.rotate_commit", map[string]any{"serverPub": wgServerPubB64})
        mWGRotateCommit.Inc(); mWGRotate.Inc()
        writeJSON(w, 200, map[string]string{"serverPub": wgServerPubB64})
    })
    // WireGuard list peers
    mux.HandleFunc("/wg/peers", func(w http.ResponseWriter, r *http.Request) {
        iface := os.Getenv("WG_IFACE"); if iface=="" { iface = "wg-shieldx" }
        if mgr, err := wgctrlmgr.New(); err == nil {
            if arr, err2 := mgr.ListPeers(iface); err2 == nil {
                writeJSON(w, 200, arr)
                return
            }
        }
        writeJSON(w, 200, []any{})
    })
    // Simple reverse proxy endpoint to guardian for PoC (no real tunnel yet)
    mux.HandleFunc("/proxy/", func(w http.ResponseWriter, r *http.Request) {
        target := fmt.Sprintf("http://127.0.0.1:%d%s", getenvInt("GUARDIAN_PORT", 9090), r.URL.Path[len("/proxy"):])
    resp, err := getHTTPClient().Get(target)
        if err != nil {
            http.Error(w, "upstream error", http.StatusBadGateway)
            return
        }
        defer resp.Body.Close()
        for k, vv := range resp.Header {
            for _, v := range vv { w.Header().Add(k, v) }
        }
        w.WriteHeader(resp.StatusCode)
        io.Copy(w, resp.Body)
    })
    // /metrics
    reg.Register(mConnect); reg.Register(mSend); reg.Register(mDivert); reg.Register(mConnectDenied); reg.Register(mSendRejected); reg.Register(mShadowAllow); reg.Register(mShadowDeny); reg.Register(mShadowDivert); reg.Register(mShadowTarpit)
    reg.Register(mWGAdd); reg.Register(mWGRemove); reg.Register(mWGRotate); reg.Register(mWGRouteErr); reg.Register(mWGThrottle); reg.Register(mWGStateRep); reg.Register(mWGRotateFail)
    reg.RegisterGauge(gWGPeers); reg.RegisterGauge(gWGRxBytes); reg.RegisterGauge(gWGTxBytes); reg.RegisterGauge(gWGHandshakeStale); reg.RegisterGauge(gWGStaleRatio)
    reg.Register(mWGRouteSetup); reg.Register(mWGRouteTeardown); reg.Register(mWGTCConfigured); reg.Register(mWGPPSConfigured)
    reg.RegisterGauge(gCertExpiry)
    mux.Handle("/metrics", reg)
    mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(200); _, _ = w.Write([]byte("ok")) })

    addr := fmt.Sprintf(":%d", port)
    log.Printf("[ingress] listening on %s", addr)
	// optional XDP guard attach (linux only)
	if obj := os.Getenv("XDP_OBJ"); obj != "" {
		iface := os.Getenv("XDP_IFACE")
		if l, err := xdpguard.New(); err == nil {
			if err := l.Attach(iface, obj); err != nil {
				log.Printf("[ingress] XDP attach: %v", err)
			} else {
				log.Printf("[ingress] XDP attached iface=%s", iface)
				if sec := os.Getenv("ADMISSION_SECRET"); sec != "" {
					_ = l.UpdateMap("admission_map", []byte("secret"), []byte(sec))
				}
				defer l.Detach(iface)
			}
		}
	}
    // periodic audit anchoring
    go func() {
        ticker := time.NewTicker(60 * time.Second)
        defer ticker.Stop()
        for range ticker.C {
            if h, err := audit.HashChain(ledgerPath); err == nil {
                _ = ledger.AppendJSONLine(ledgerPath, serviceName, "anchor", map[string]any{"hash": h})
                _, _, _ = forensics.SaveJSON(serviceName, "anchors", map[string]any{"hashchain": h, "ts": time.Now().UTC()})
            }
        }
    }()
	// periodic channel purge
	go func() {
		ticker := time.NewTicker(30 * time.Second)
		defer ticker.Stop()
		for now := range ticker.C { chanReg.purgeExpired(now) }
	}()
	// optional QUIC tunnel server
	if os.Getenv("INGRESS_QUIC_ADDR") != "" {
		if err := startQUICServer(os.Getenv("INGRESS_QUIC_ADDR")); err != nil {
			log.Printf("[ingress] QUIC start error: %v", err)
		}
	}
	// periodic DPoP GC
	go func(){ t:=time.NewTicker(2*time.Minute); defer t.Stop(); for range t.C { gcDPoPStore() } }()
    // wrap with basic rate limiting middleware
    // HTTP metrics middleware
    httpMetrics := metrics.NewHTTPMetrics(reg, serviceName)
    baseH := httpMetrics.Middleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Early L4-ish token guard (header-based) to drop before heavy work
		if adm := os.Getenv("ADMISSION_SECRET"); adm != "" {
			if !guard.VerifyHeader(r, adm, os.Getenv("ADMISSION_HEADER"), "ingress") {
				http.Error(w, "unauthorized", http.StatusUnauthorized)
				return
			}
		}
		ip := clientIP(r)
        if !rl.Allow(ip) {
            http.Error(w, "rate limit", http.StatusTooManyRequests)
            return
        }
        mux.ServeHTTP(w, r)
    }))
    h := otelobs.WrapHTTPHandler(serviceName, baseH)
    // RA-TLS optional enable
    if os.Getenv("RATLS_ENABLE") == "true" {
        td := getenvDefault("RATLS_TRUST_DOMAIN", "shieldx.local")
        ns := getenvDefault("RATLS_NAMESPACE", "default")
        svc := getenvDefault("RATLS_SERVICE", serviceName)
        rotate := parseDurationDefault("RATLS_ROTATE_EVERY", 45*time.Minute)
        valid := parseDurationDefault("RATLS_VALIDITY", 60*time.Minute)
        issuer, err := ratls.NewDevIssuer(ratls.Identity{TrustDomain: td, Namespace: ns, Service: svc}, rotate, valid)
        if err != nil { log.Fatalf("[ingress] RA-TLS init: %v", err) }
        ratlsIssuer = issuer
        // update cert expiry gauge
        go func(){ for { if t, err := ratlsIssuer.LeafNotAfter(); err==nil { secs := uint64(time.Until(t).Seconds()); gCertExpiry.Set(secs) }; time.Sleep(time.Minute) } }()
    }
    // Use RA-TLS if enabled
    if ratlsIssuer != nil {
        srv := &http.Server{ Addr: addr, Handler: h, TLSConfig: ratlsIssuer.ServerTLSConfig(true, getenvDefault("RATLS_TRUST_DOMAIN", "shieldx.local")) }
        log.Fatal(srv.ListenAndServeTLS("", ""))
    } else {
        log.Fatal(http.ListenAndServe(addr, h))
    }
}

func normalizeHTU(r *http.Request) string {
	u := *r.URL
	u.RawQuery = ""
	return fmt.Sprintf("%s://%s%s", "http", r.Host, u.Path)
}

func gcDPoPStore() {
	dpopStoreMu.Lock()
	now := time.Now().Unix()
	for k, v := range dpopStore { if v < now { delete(dpopStore, k) } }
	dpopStoreMu.Unlock()
}

func replicateWGState() {
    if rdb == nil || wgSt == nil { return }
    b, _ := json.Marshal(wgSt.Get())
    _ = rdb.Set(context.Background(), "wgstate:"+os.Getenv("POP_ID"), string(b), 0).Err()
    mWGStateRep.Inc()
}

// simple token bucket per key
type rateLimiter struct {
    capacity int
    refill   int
    window   time.Duration
	mu       sync.Mutex
    store    map[string]bucket
}
type bucket struct { remaining int; reset time.Time }

func newRateLimiter(capacity int, window time.Duration) *rateLimiter {
    return &rateLimiter{capacity: capacity, refill: capacity, window: window, store: map[string]bucket{}}
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

func getenvInt(key string, def int) int {
    v := os.Getenv(key)
    if v == "" {
        return def
    }
    n, err := strconv.Atoi(v)
    if err != nil {
        return def
    }
    return n
}

func getenvDefault(key, def string) string {
    if v := os.Getenv(key); v != "" { return v }
    return def
}

func parseDurationDefault(key string, def time.Duration) time.Duration {
    if v := os.Getenv(key); v != "" { if d, err := time.ParseDuration(v); err==nil { return d } }
    return def
}

// clientIP extracts client IP from X-Forwarded-For / X-Real-IP / RemoteAddr
func clientIP(r *http.Request) string {
	if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
		parts := strings.Split(xff, ",")
		if len(parts) > 0 { return strings.TrimSpace(parts[0]) }
	}
	if rip := r.Header.Get("X-Real-IP"); rip != "" { return rip }
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err == nil { return host }
	return r.RemoteAddr
}

func ensureLinuxWGRoute(iface, cidr string) error {
	// ip link add && addr add && up (idempotent)
	exec.Command("sh", "-c", fmt.Sprintf("ip link show %s || ip link add dev %s type wireguard", iface, iface)).Run()
	exec.Command("sh", "-c", fmt.Sprintf("ip address show dev %s | grep -q %s || ip address add %s dev %s", iface, cidr, cidr, iface)).Run()
	exec.Command("sh", "-c", fmt.Sprintf("ip link set up dev %s", iface)).Run()
	// forwarding
	exec.Command("sh", "-c", "sysctl -w net.ipv4.ip_forward=1").Run()
	// nftables: dedicated chains for WG iface; idempotent creation
	exec.Command("sh", "-c", "nft add table ip nat 2>/dev/null || true").Run()
	exec.Command("sh", "-c", "nft add chain ip nat SHIELDX_POST '{ type nat hook postrouting priority 100 ; }' 2>/dev/null || true").Run()
	exec.Command("sh", "-c", fmt.Sprintf("nft add rule ip nat SHIELDX_POST ip saddr %s oifname != %s counter masquerade 2>/dev/null || true", cidr, iface)).Run()
	exec.Command("sh", "-c", "nft add table ip filter 2>/dev/null || true").Run()
	exec.Command("sh", "-c", "nft add chain ip filter SHIELDX_FWD '{ type filter hook forward priority 0 ; }' 2>/dev/null || true").Run()
	// Allow forward in and out for WG iface
	exec.Command("sh", "-c", fmt.Sprintf("nft add rule ip filter SHIELDX_FWD iifname \"%s\" ct state new,established,related accept 2>/dev/null || true", iface)).Run()
	exec.Command("sh", "-c", fmt.Sprintf("nft add rule ip filter SHIELDX_FWD oifname \"%s\" ct state established,related accept 2>/dev/null || true", iface)).Run()
	return nil
}

func ensureUserspaceWG(iface, privB64 string, listenPort int) {
	if runtime.GOOS == "linux" {
		return
	}
	// Start wireguard-go userspace interface if not exists
	exec.Command("wireguard-go", iface).Run()
	// Set private key and listen port
	_ = exec.Command("sh", "-c", fmt.Sprintf("WG_PRIV=$(mktemp); echo %s | base64 -d > $WG_PRIV; wg set %s private-key $WG_PRIV listen-port %d; rm -f $WG_PRIV", privB64, iface, listenPort)).Run()
}

// replacePort takes host:port and returns host:newPort (best-effort).
func replacePort(endpoint string, newPort int) string {
	host, _, err := net.SplitHostPort(endpoint)
    if err != nil { return "" }
    return net.JoinHostPort(host, strconv.Itoa(newPort))
}

// New: teardown Linux WG route/NAT
func teardownLinuxWGRoute(iface string) error {
	exec.Command("sh", "-c", "nft delete chain ip nat SHIELDX_POST 2>/dev/null || true").Run()
	exec.Command("sh", "-c", "nft delete chain ip filter SHIELDX_FWD 2>/dev/null || true").Run()
	exec.Command("sh", "-c", fmt.Sprintf("ip link set down dev %s 2>/dev/null || true", iface)).Run()
	exec.Command("sh", "-c", fmt.Sprintf("ip link del dev %s 2>/dev/null || true", iface)).Run()
	return nil
}


// Shared HTTP client (wraps with OpenTelemetry and RA-TLS if enabled)
var httpClientOnce sync.Once
var httpClient *http.Client

func getHTTPClient() *http.Client {
    httpClientOnce.Do(func(){
        tr := &http.Transport{ MaxIdleConns: 100, MaxIdleConnsPerHost: 10, IdleConnTimeout: 90 * time.Second }
        if ratlsIssuer != nil { tr.TLSClientConfig = ratlsIssuer.ClientTLSConfig() }
        httpClient = &http.Client{ Transport: otelobs.WrapHTTPTransport(tr), Timeout: 30 * time.Second }
    })
    return httpClient
}


