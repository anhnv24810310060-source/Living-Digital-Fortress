package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"strconv"
	"time"

	"crypto/ecdh"
	crand "crypto/rand"
	"crypto/sha256"
	"crypto/tls"
	"shieldx/shared/shieldx-common/pkg/metrics"
	"shieldx/shared/shieldx-common/pkg/sandbox"
	"shieldx/shared/shieldx-common/pkg/wch"
	"strings"
	"sync"

	"github.com/google/uuid"
	quic "github.com/quic-go/quic-go"
)

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

// getenv returns the environment variable value or empty string if unset.
func getenv(key string) string { return os.Getenv(key) }

func main() {
	// This is the real server protected behind ingress. It only listens on loopback.
	port := getenvInt("GUARDIAN_PORT", 9090)
	// Static X25519 key for guardian (PoC). In production rotate per process and per channel.
	curve := ecdh.X25519()
	guardianPriv, err := curve.GenerateKey(crand.Reader)
	if err != nil {
		log.Fatalf("guardian key: %v", err)
	}
	guardianPub := guardianPriv.PublicKey().Bytes()

	mux := http.NewServeMux()
	reg := metrics.NewRegistry()
	mRecv := metrics.NewCounter("guardian_wch_recv_total", "Total sealed envelopes received")
	mUDPRecv := metrics.NewCounter("guardian_wch_udp_recv_total", "Total sealed UDP envelopes received")
	mMasqueOK := metrics.NewCounter("guardian_masque_success_total", "MASQUE QUIC UDP relays succeeded")
	mMasqueFB := metrics.NewCounter("guardian_masque_fallback_total", "MASQUE QUIC fallback to direct UDP")
	mUDPDir := metrics.NewCounter("guardian_udp_direct_total", "Direct UDP relays succeeded")
	mUDPErr := metrics.NewCounter("guardian_udp_error_total", "UDP relay errors")
	mRekeyGrace := metrics.NewCounter("guardian_rekey_grace_hits_total", "Decrypt succeeded using counter-1 grace window")
	// job lifecycle metrics
	mJobsCreated := metrics.NewCounter("guardian_jobs_created_total", "Jobs created")
	mJobsCompleted := metrics.NewCounter("guardian_jobs_completed_total", "Jobs completed successfully")
	mJobsTimeout := metrics.NewCounter("guardian_jobs_timeout_total", "Jobs timed out")
	mJobsError := metrics.NewCounter("guardian_jobs_error_total", "Jobs ended with error")
	gJobsActive := metrics.NewGauge("guardian_jobs_active", "Currently active jobs")
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, "Guardian real service OK. Path=%s\n", r.URL.Path)
	})
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(200); _, _ = w.Write([]byte("ok")) })
	// Compatibility alias per system design docs
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(200); _, _ = w.Write([]byte("ok")) })
	// Expose minimal eBPF metrics summary if available (labels are static for guardian)
	mux.HandleFunc("/guardian/metrics/summary", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"service": "guardian",
			"prom": map[string]string{
				"syscalls":    "ebpf_syscall_total{service=\"guardian\"}",
				"latency":     "ebpf_syscall_duration_seconds{service=\"guardian\"}",
				"network_in":  "ebpf_network_bytes_received_total{service=\"guardian\"}",
				"network_out": "ebpf_network_bytes_sent_total{service=\"guardian\"}",
				"file_ops":    "ebpf_file_operations_total{service=\"guardian\"}",
				"dangerous":   "ebpf_dangerous_syscalls_total{service=\"guardian\"}",
			},
		})
	})

	// --- Guardian job manager (sandboxed execution) ---
	type jobStatus string
	const (
		jobQueued  jobStatus = "queued"
		jobRunning jobStatus = "running"
		jobDone    jobStatus = "done"
		jobError   jobStatus = "error"
		jobTimeout jobStatus = "timeout"
	)
	type job struct {
		ID          string         `json:"id"`
		Status      jobStatus      `json:"status"`
		CreatedAt   time.Time      `json:"created_at"`
		CompletedAt time.Time      `json:"completed_at,omitempty"`
		Error       string         `json:"error,omitempty"`
		Hash        string         `json:"hash,omitempty"`
		Output      string         `json:"output,omitempty"`
		Threat      float64        `json:"threat_score,omitempty"`     // normalized 0..1 (back-compat)
		Threat100   int            `json:"threat_score_100,omitempty"` // 0..100 preferred for reports
		Features    map[string]any `json:"features,omitempty"`         // summarized eBPF features if available
		Backend     string         `json:"sandbox_backend,omitempty"`  // firecracker|noop|docker|wasm
		Duration    string         `json:"duration,omitempty"`
	}
	var (
		jobsMu sync.RWMutex
		jobs   = map[string]*job{}
	)
	// simple id counter
	var idCtr uint64
	nextID := func() string { idCtr++; return fmt.Sprintf("j-%d", idCtr) }
	// TTL-based cleanup to avoid unbounded memory use
	jobTTL := time.Duration(getenvInt("GUARDIAN_JOB_TTL_SEC", 600)) * time.Second
	jobMax := getenvInt("GUARDIAN_JOB_MAX", 10000)
	go func() {
		ticker := time.NewTicker(30 * time.Second)
		defer ticker.Stop()
		for range ticker.C {
			now := time.Now()
			removed := 0
			jobsMu.Lock()
			// Size guard first
			if len(jobs) > jobMax {
				// remove oldest completed first
				type kv struct {
					id   string
					t    time.Time
					done bool
				}
				arr := make([]kv, 0, len(jobs))
				for id, j := range jobs {
					arr = append(arr, kv{id: id, t: j.CompletedAt, done: j.Status == jobDone || j.Status == jobError || j.Status == jobTimeout})
				}
				// simple selection: remove up to overflow count prioritizing done and oldest
				overflow := len(jobs) - jobMax
				// naive O(n^2) is fine for small overflow; else selection could be optimized
				for overflow > 0 {
					// pick victim
					vi := -1
					var vt time.Time
					for i := range arr {
						if arr[i].id == "" {
							continue
						}
						if vi == -1 || (arr[i].done && (!arr[vi].done || arr[i].t.Before(vt))) || (!arr[vi].done && !arr[i].done && arr[i].t.Before(vt)) {
							vi = i
							vt = arr[i].t
						}
					}
					if vi == -1 {
						break
					}
					delete(jobs, arr[vi].id)
					arr[vi].id = ""
					overflow--
					removed++
				}
			}
			// TTL sweep
			for id, j := range jobs {
				if !j.CompletedAt.IsZero() && now.Sub(j.CompletedAt) > jobTTL {
					delete(jobs, id)
					removed++
				}
			}
			// update active gauge
			active := 0
			for _, j := range jobs {
				if j.Status == jobQueued || j.Status == jobRunning {
					active++
				}
			}
			jobsMu.Unlock()
			gJobsActive.Set(uint64(active))
			if removed > 0 {
				log.Printf("[guardian] jobs cleanup removed=%d active=%d", removed, active)
			}
		}
	}()
	// POST /guardian/execute { payload: "..." } with simple per-IP RL
	execLimiter := makeRLLimiter(getenvInt("GUARDIAN_RL_PER_MIN", 60))
	// Concurrency limiter & simple circuit breaker
	maxConcurrent := getenvInt("GUARDIAN_MAX_CONCURRENT", 32)
	sem := make(chan struct{}, maxConcurrent)
	var breakerMu sync.Mutex
	var breakerOpen bool
	var breakerFail, breakerSuccess int
	openThreshold := getenvInt("GUARDIAN_BREAKER_FAIL", 10)
	closeAfter := getenvInt("GUARDIAN_BREAKER_SUCCESS", 50)
	breakerState := metrics.NewGauge("guardian_breaker_state", "Circuit breaker state (0=closed,1=open)")
	reg.RegisterGauge(breakerState)

	mux.HandleFunc("/guardian/execute", execLimiter(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		breakerMu.Lock()
		if breakerOpen {
			breakerMu.Unlock()
			http.Error(w, "service temporarily unavailable", http.StatusServiceUnavailable)
			return
		}
		breakerMu.Unlock()
		// Acquire slot
		select {
		case sem <- struct{}{}:
			defer func() { <-sem }()
		default:
			http.Error(w, "too many concurrent executions", http.StatusTooManyRequests)
			return
		}
		var body struct {
			Payload  string `json:"payload"`
			TenantID string `json:"tenant_id"`
			Cost     int64  `json:"cost"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, "bad json", 400)
			return
		}
		if len(body.Payload) == 0 {
			http.Error(w, "payload required", 400)
			return
		}
		if len(body.Payload) > getenvInt("GUARDIAN_MAX_PAYLOAD", 64*1024) {
			http.Error(w, "payload too large", http.StatusRequestEntityTooLarge)
			return
		}
		// Optional credits pre-check
		if body.TenantID != "" && getenv("GUARDIAN_CREDITS_URL") != "" {
			cost := body.Cost
			if cost <= 0 {
				cost = int64(getenvInt("GUARDIAN_DEFAULT_COST", 1))
			}
			if ok, code := consumeCredits(getenv("GUARDIAN_CREDITS_URL"), body.TenantID, cost); !ok {
				if code == 402 {
					http.Error(w, "insufficient credits", http.StatusPaymentRequired)
					return
				}
				http.Error(w, "credits service unavailable", http.StatusServiceUnavailable)
				return
			}
		}

		id := nextID()
		jb := &job{ID: id, Status: jobQueued, CreatedAt: time.Now()}
		jobsMu.Lock()
		jobs[id] = jb
		jobsMu.Unlock()
		mJobsCreated.Add(1)
		gJobsActive.Set(uint64(len(jobs)))
		// run asynchronously with 30s timeout (enforced)
		go func(j *job, payload string) {
			j.Status = jobRunning
			t0 := time.Now()
			// Use secure sandbox with hard timeout 30s
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
			out, threat, sres, backend, err := runSecured(ctx, payload)
			if err != nil {
				breakerMu.Lock()
				breakerFail++
				if !breakerOpen && breakerFail >= openThreshold {
					breakerOpen = true
					breakerFail = 0
					breakerSuccess = 0
					breakerState.Set(1)
					log.Printf("[guardian] circuit breaker OPEN after failures")
					// auto half-open after cooldown
					go func() {
						time.Sleep(10 * time.Second)
						breakerMu.Lock()
						breakerOpen = false
						breakerState.Set(0)
						breakerMu.Unlock()
						log.Printf("[guardian] circuit breaker HALF-OPEN (trial)")
					}()
				}
				breakerMu.Unlock()
				if ctx.Err() == context.DeadlineExceeded {
					j.Status = jobTimeout
					mJobsTimeout.Add(1)
				} else {
					j.Status = jobError
					mJobsError.Add(1)
				}
				j.Error = err.Error()
				log.Printf("[guardian] execute id=%s status=%s err=%v", j.ID, j.Status, err)
			} else {
				breakerMu.Lock()
				breakerSuccess++
				if breakerOpen && breakerSuccess >= closeAfter { // close after enough successes
					breakerOpen = false
					breakerState.Set(0)
					breakerSuccess = 0
					breakerFail = 0
					log.Printf("[guardian] circuit breaker CLOSED")
				}
				breakerMu.Unlock()
				h := sha256.Sum256([]byte(out))
				j.Hash = fmt.Sprintf("%x", h[:])
				j.Output = out
				// Prefer sandbox-provided threat score; fallback to heuristic if unavailable
				thr := threat
				if thr < 0 {
					thr = 0
				}
				if thr == 0 {
					if len(out) > 1024 {
						thr += 0.2
					}
					if strings.Contains(out, "exec") || strings.Contains(out, "/bin/sh") {
						thr += 0.5
					}
					if thr > 1.0 {
						thr = 1.0
					}
				}
				j.Threat = thr
				j.Threat100 = int(thr * 100.0)
				j.Backend = backend
				// If we have a sandbox result (eBPF), summarize features for the report
				if sres != nil {
					feats := map[string]any{}
					// Basic counts
					feats["syscalls_total"] = len(sres.Syscalls)
					// Count dangerous syscalls
					dang := 0
					for _, ev := range sres.Syscalls {
						if ev.Dangerous {
							dang++
						}
					}
					feats["dangerous_syscalls"] = dang
					feats["network_events"] = len(sres.NetworkIO)
					// Count file writes
					writes := 0
					for _, fe := range sres.FileAccess {
						if fe.Operation == "write" && fe.Success {
							writes++
						}
					}
					feats["file_writes"] = writes
					// If ThreatScore provided on 0..100, prefer it for Threat100
					if sres.ThreatScore >= 0 {
						j.Threat100 = int(sres.ThreatScore)
						if j.Threat100 > 100 {
							j.Threat100 = 100
						}
						j.Threat = float64(j.Threat100) / 100.0
					}
					j.Features = feats
				}
				j.Status = jobDone
				mJobsCompleted.Add(1)
				log.Printf("[guardian] execute id=%s status=done threat=%.2f dur=%s", j.ID, j.Threat, time.Since(t0))
			}
			j.Duration = time.Since(t0).String()
			j.CompletedAt = time.Now()
		}(jb, body.Payload)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]string{"id": id, "status": string(jb.Status)})
	}))
	// GET /guardian/status/:id
	mux.HandleFunc("/guardian/status/", func(w http.ResponseWriter, r *http.Request) {
		id := strings.TrimPrefix(r.URL.Path, "/guardian/status/")
		jobsMu.RLock()
		j := jobs[id]
		jobsMu.RUnlock()
		if j == nil {
			http.Error(w, "not found", 404)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(j)
	})
	// GET /guardian/report/:id (redacts output length, does not expose internals)
	mux.HandleFunc("/guardian/report/", func(w http.ResponseWriter, r *http.Request) {
		id := strings.TrimPrefix(r.URL.Path, "/guardian/report/")
		jobsMu.RLock()
		j := jobs[id]
		jobsMu.RUnlock()
		if j == nil {
			http.Error(w, "not found", 404)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		// avoid exposing raw output fully; provide hash and limited preview
		preview := j.Output
		if len(preview) > 256 {
			preview = preview[:256] + "..."
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id":               j.ID,
			"status":           j.Status,
			"created_at":       j.CreatedAt,
			"completed_at":     j.CompletedAt,
			"hash":             j.Hash,
			"threat_score":     j.Threat,
			"threat_score_100": j.Threat100,
			"features":         j.Features,
			"backend":          j.Backend,
			"duration":         j.Duration,
			"output_preview":   preview,
		})
	})
	// Publish guardian public key (base64)
	mux.HandleFunc("/wch/pubkey", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]string{"pubKey": wch.MarshalB64(guardianPub)})
	})
	// Receive sealed envelope, decrypt, route to local handler, encrypt response
	mux.HandleFunc("/wch/recv", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		// Limit envelope size to prevent memory abuse
		maxBytes := getenvInt("WCH_MAX_ENVELOPE_BYTES", 65536)
		r.Body = http.MaxBytesReader(w, r.Body, int64(maxBytes))
		mRecv.Inc()
		var env wch.Envelope
		if err := json.NewDecoder(r.Body).Decode(&env); err != nil {
			http.Error(w, "bad request", 400)
			return
		}
		if env.ChannelID == "" || env.EphemeralPubB64 == "" || env.NonceB64 == "" || env.CiphertextB64 == "" {
			http.Error(w, "missing fields", 400)
			return
		}
		clientPubBytes, err := wch.UnmarshalB64(env.EphemeralPubB64)
		if err != nil {
			http.Error(w, "bad client key", 400)
			return
		}
		nonce, err := wch.UnmarshalB64(env.NonceB64)
		if err != nil {
			http.Error(w, "bad nonce", 400)
			return
		}
		ct, err := wch.UnmarshalB64(env.CiphertextB64)
		if err != nil {
			http.Error(w, "bad ciphertext", 400)
			return
		}
		clientPub, err := curve.NewPublicKey(clientPubBytes)
		if err != nil {
			http.Error(w, "bad client pub", 400)
			return
		}
		shared, _ := guardianPriv.ECDH(clientPub)
		var key []byte
		var pt []byte
		// Try current counter and a grace window of counter-1 to avoid race during rekey
		tryCounters := []int{env.RekeyCounter}
		if env.RekeyCounter > 0 {
			tryCounters = append(tryCounters, env.RekeyCounter-1)
		}
		var decErr error
		for _, c := range tryCounters {
			if c > 0 {
				key, err = wch.DeriveKeyWithCounter(shared, env.ChannelID, c)
			} else {
				key, err = wch.DeriveKey(shared, env.ChannelID)
			}
			if err != nil {
				decErr = err
				continue
			}
			pt, decErr = wch.Open(key, nonce, ct)
			if decErr == nil {
				if c == env.RekeyCounter-1 {
					mRekeyGrace.Inc()
				}
				break
			}
		}
		if decErr != nil {
			http.Error(w, "decrypt error", 400)
			return
		}
		// Unmarshal inner request and handle locally
		var inner wch.InnerRequest
		if err := json.Unmarshal(pt, &inner); err != nil {
			http.Error(w, "inner parse", 400)
			return
		}
		// For PoC, only GET / is supported
		status := 200
		body := []byte(fmt.Sprintf("Hello from Guardian over Whisper. Path=%s", inner.Path))
		resp := wch.InnerResponse{Status: status, Headers: map[string]string{"Content-Type": "text/plain"}, Body: body}
		respJSON := wch.ToJSON(resp)
		nonce2, ct2, err := wch.Seal(key, respJSON)
		if err != nil {
			http.Error(w, "encrypt error", 500)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]string{
			"nonce":      wch.MarshalB64(nonce2),
			"ciphertext": wch.MarshalB64(ct2),
		})
	})
	// Receive sealed UDP request, relay via MASQUE QUIC and seal response
	mux.HandleFunc("/wch/recv-udp", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		// Limit envelope size to prevent memory abuse
		maxBytes := getenvInt("WCH_MAX_ENVELOPE_BYTES", 65536)
		r.Body = http.MaxBytesReader(w, r.Body, int64(maxBytes))
		mUDPRecv.Inc()
		var env wch.Envelope
		if err := json.NewDecoder(r.Body).Decode(&env); err != nil {
			http.Error(w, "bad request", 400)
			return
		}
		clientPubBytes, err := wch.UnmarshalB64(env.EphemeralPubB64)
		if err != nil {
			http.Error(w, "bad client key", 400)
			return
		}
		nonce, err := wch.UnmarshalB64(env.NonceB64)
		if err != nil {
			http.Error(w, "bad nonce", 400)
			return
		}
		ct, err := wch.UnmarshalB64(env.CiphertextB64)
		if err != nil {
			http.Error(w, "bad ciphertext", 400)
			return
		}
		clientPub, err := curve.NewPublicKey(clientPubBytes)
		if err != nil {
			http.Error(w, "bad client pub", 400)
			return
		}
		shared, _ := guardianPriv.ECDH(clientPub)
		var key []byte
		// Grace window for counter-1 when decrypting
		tryCounters := []int{env.RekeyCounter}
		if env.RekeyCounter > 0 {
			tryCounters = append(tryCounters, env.RekeyCounter-1)
		}
		var pt []byte
		var decErr error
		for _, c := range tryCounters {
			if c > 0 {
				key, err = wch.DeriveKeyWithCounter(shared, env.ChannelID, c)
			} else {
				key, err = wch.DeriveKey(shared, env.ChannelID)
			}
			if err != nil {
				decErr = err
				continue
			}
			pt, decErr = wch.Open(key, nonce, ct)
			if decErr == nil {
				if c == env.RekeyCounter-1 {
					mRekeyGrace.Inc()
				}
				break
			}
		}
		if decErr != nil {
			http.Error(w, "decrypt error", 400)
			return
		}
		// Decrypt inner UDP request done in grace loop above; pt holds plaintext
		var udpReq wch.InnerUDPRequest
		if err := json.Unmarshal(pt, &udpReq); err != nil {
			http.Error(w, "inner parse", 400)
			return
		}
		// Relay via MASQUE QUIC if configured, else direct UDP
		var respBytes []byte
		if masque := os.Getenv("MASQUE_QUIC_ADDR"); masque != "" {
			if b, err := masqueSingleExchange(masque, udpReq.Target, udpReq.Data, udpReq.TimeoutMs); err == nil {
				respBytes = b
				mMasqueOK.Inc()
			} else {
				respBytes, _ = directUDPSingleExchange(udpReq.Target, udpReq.Data, udpReq.TimeoutMs)
				mMasqueFB.Inc()
			}
		} else {
			respBytes, _ = directUDPSingleExchange(udpReq.Target, udpReq.Data, udpReq.TimeoutMs)
			if len(respBytes) > 0 {
				mUDPDir.Inc()
			} else {
				mUDPErr.Inc()
			}
		}
		udpResp := wch.UDPResponse{Data: respBytes}
		respJSON := wch.ToJSON(udpResp)
		nonce2, ct2, err := wch.Seal(key, respJSON)
		if err != nil {
			http.Error(w, "encrypt error", 500)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]string{"nonce": wch.MarshalB64(nonce2), "ciphertext": wch.MarshalB64(ct2)})
	})
	reg.Register(mRecv)
	reg.Register(mUDPRecv)
	reg.Register(mMasqueOK)
	reg.Register(mMasqueFB)
	reg.Register(mUDPDir)
	reg.Register(mUDPErr)
	reg.Register(mRekeyGrace)
	reg.Register(mJobsCreated)
	reg.Register(mJobsCompleted)
	reg.Register(mJobsTimeout)
	reg.Register(mJobsError)
	mux.Handle("/metrics", reg)

	// HTTP metrics middleware
	httpMetrics := metrics.NewHTTPMetrics(reg, "guardian")
	// NOTE: Previously bound only to 127.0.0.1 which made the guardian service
	// unreachable from other containers (and host mapped port) because Docker's
	// port publishing/NAT directs traffic to the container's eth0 interface, not
	// its loopback. Binding to 0.0.0.0 ensures health checks and dependent
	// services (ingress, gateway, etc.) can reach guardian on GUARDIAN_PORT.
	addr := fmt.Sprintf(":%d", port) // listen on all interfaces inside container
	log.Printf("[guardian] listening on http://0.0.0.0:%d", port)
	log.Fatal(http.ListenAndServe(addr, httpMetrics.Middleware(mux)))
}

type masqueTarget struct {
	Addr string `json:"addr"`
}

func directUDPSingleExchange(target string, payload []byte, timeoutMs int) ([]byte, error) {
	raddr, err := net.ResolveUDPAddr("udp", target)
	if err != nil {
		return nil, err
	}
	c, err := net.DialUDP("udp", nil, raddr)
	if err != nil {
		return nil, err
	}
	defer c.Close()
	_, _ = c.Write(payload)
	_ = c.SetReadDeadline(time.Now().Add(time.Duration(timeoutMs+500) * time.Millisecond))
	buf := make([]byte, 64*1024)
	n, _, err := c.ReadFromUDP(buf)
	if err != nil {
		return nil, err
	}
	return buf[:n], nil
}

func masqueSingleExchange(addrList, target string, payload []byte, timeoutMs int) ([]byte, error) {
	// Support multiple addresses (comma separated) with simple retry policy
	addrs := []string{}
	for _, a := range strings.Split(addrList, ",") {
		if s := strings.TrimSpace(a); s != "" {
			addrs = append(addrs, s)
		}
	}
	if len(addrs) == 0 {
		addrs = []string{addrList}
	}
	retries := getenvInt("MASQUE_RETRY", 2)
	tout := getenvInt("MASQUE_TIMEOUT_MS", timeoutMs+1000)
	if tout < 500 {
		tout = 500
	}
	var lastErr error
	for attempt := 0; attempt <= retries; attempt++ {
		for _, addr := range addrs {
			tlsConf := &tls.Config{InsecureSkipVerify: true, NextProtos: []string{"shieldx-masque"}}
			conn, err := quic.DialAddr(context.Background(), addr, tlsConf, &quic.Config{})
			if err != nil {
				lastErr = err
				continue
			}
			func() {
				defer conn.CloseWithError(0, "bye")
				ctx, cancel := context.WithTimeout(context.Background(), time.Duration(tout)*time.Millisecond)
				defer cancel()
				st, err := conn.OpenStreamSync(ctx)
				if err != nil {
					lastErr = err
					return
				}
				defer st.Close()
				// send target header (length-prefixed JSON)
				hdr := make([]byte, 2)
				tgt, _ := json.Marshal(masqueTarget{Addr: target})
				if len(tgt) > 65535 {
					lastErr = fmt.Errorf("target too long")
					return
				}
				binary.BigEndian.PutUint16(hdr, uint16(len(tgt)))
				if _, err := st.Write(hdr); err != nil {
					lastErr = err
					return
				}
				if _, err := st.Write(tgt); err != nil {
					lastErr = err
					return
				}
				// send payload frame
				if len(payload) > 65535 {
					payload = payload[:65535]
				}
				binary.BigEndian.PutUint16(hdr, uint16(len(payload)))
				if _, err := st.Write(hdr); err != nil {
					lastErr = err
					return
				}
				if _, err := st.Write(payload); err != nil {
					lastErr = err
					return
				}
				// read one response frame
				if _, err := io.ReadFull(st, hdr); err != nil {
					lastErr = err
					return
				}
				nlen := binary.BigEndian.Uint16(hdr)
				if nlen == 0 {
					lastErr = nil
					return
				}
				buf := make([]byte, int(nlen))
				if _, err := io.ReadFull(st, buf); err != nil {
					lastErr = err
					return
				}
				// success
				lastErr = nil
				payload = buf
			}()
			if lastErr == nil {
				return payload, nil
			}
		}
		time.Sleep(time.Duration(100*(attempt+1)) * time.Millisecond)
	}
	return nil, lastErr
}

// sandboxRun is separated to avoid importing sandbox at top-level to keep existing build behavior.
// sandboxRun keeps backward-compatibility with basic Runner
// sandboxRun adapts to sandbox.Run returning either (string,error) or (*SandboxResult,error)
func sandboxRun(ctx context.Context, payload string) (string, error) {
	r := sandbox.NewFromEnv()
	// The sandbox package offers different Runner types under build tags.
	// Try a type switch to obtain stdout in both cases.
	switch rr := any(r).(type) {
	case interface {
		Run(context.Context, string) (*sandbox.SandboxResult, error)
	}:
		res, err := rr.Run(ctx, payload)
		if err != nil {
			return "", err
		}
		if res == nil {
			return "", fmt.Errorf("nil sandbox result")
		}
		return res.Stdout, nil
	case interface {
		Run(context.Context, string) (string, error)
	}:
		return rr.Run(ctx, payload)
	default:
		// Fallback attempt
		type runStr interface {
			Run(context.Context, string) (string, error)
		}
		if rr2, ok := any(r).(runStr); ok {
			return rr2.Run(ctx, payload)
		}
		return "", fmt.Errorf("unsupported sandbox runner type")
	}
}

// runSecured executes payload in the most secure available sandbox:
// - If GUARDIAN_SANDBOX_BACKEND=firecracker and kernel/rootfs configured, run inside Firecracker with strict limits.
// - Else, fallback to environment-provided runner.
// Returns (stdout, threatScore[0..1], sandboxResult, backend, error)
func runSecured(ctx context.Context, payload string) (string, float64, *sandbox.SandboxResult, string, error) {
	// Initialize threat scorer with optimized weights
	scorer := sandbox.NewThreatScorer()

	if os.Getenv("GUARDIAN_SANDBOX_BACKEND") == "firecracker" {
		k := os.Getenv("FC_KERNEL_PATH")
		rfs := os.Getenv("FC_ROOTFS_PATH")
		if k != "" && rfs != "" {
			vcpus := int64(getenvInt("FC_VCPU", 1))
			mem := int64(getenvInt("FC_MEM_MIB", 128))
			tout := getenvInt("FC_TIMEOUT_SEC", 30)
			limits := sandbox.ResourceLimits{VCPUCount: vcpus, MemSizeMib: mem, TimeoutSec: tout, NetworkDeny: true, FilesystemRO: true}
			fcr := sandbox.NewFirecrackerRunner(k, rfs, limits)
			res, err := fcr.Run(ctx, payload)
			if err != nil {
				return "", 0, nil, "firecracker", err
			}

			// Use advanced threat scoring pipeline (P0 requirement)
			threatScore100, explanation := scorer.CalculateScore(res)
			res.ThreatScore = float64(threatScore100)

			// Store explanation in artifacts for audit
			if res.Artifacts == nil {
				res.Artifacts = make(map[string][]byte)
			}
			res.Artifacts["threat_explanation"] = []byte(explanation)
			res.Artifacts["risk_level"] = []byte(sandbox.RiskLevel(threatScore100))

			// Normalize to 0..1 for backward compatibility
			ts := float64(threatScore100) / 100.0
			if ts < 0 {
				ts = 0
			}
			if ts > 1 {
				ts = 1
			}

			log.Printf("[guardian] sandbox execution: threat=%d/100 risk=%s reasons=%s",
				threatScore100, sandbox.RiskLevel(threatScore100), explanation)

			return res.Stdout, ts, res, "firecracker", nil
		}
		// If misconfigured, fall through to default runner
	}

	// Default runner (fallback with basic threat analysis)
	out, err := sandboxRun(ctx, payload)
	if err != nil {
		return "", 0, nil, "default", err
	}

	// Apply threat scoring even for default runner
	basicResult := &sandbox.SandboxResult{
		Stdout:   out,
		Duration: 0,                        // will be set by caller
		Syscalls: []sandbox.SyscallEvent{}, // no eBPF in default mode
	}
	threatScore100, explanation := scorer.CalculateScore(basicResult)
	basicResult.ThreatScore = float64(threatScore100)

	log.Printf("[guardian] default execution: threat=%d/100 risk=%s reasons=%s",
		threatScore100, sandbox.RiskLevel(threatScore100), explanation)

	return out, float64(threatScore100) / 100.0, basicResult, "default", nil
}

// lightweight per-IP rate limiter for endpoints (req/min)
func makeRLLimiter(reqPerMin int) func(http.HandlerFunc) http.HandlerFunc {
	if reqPerMin <= 0 {
		reqPerMin = 60
	}
	type bucket struct {
		c int
		w int64
	}
	var mu sync.Mutex
	m := map[string]*bucket{}
	return func(next http.HandlerFunc) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			ip := r.Header.Get("X-Forwarded-For")
			if ip == "" {
				ip = strings.Split(r.RemoteAddr, ":")[0]
			}
			now := time.Now().Unix() / 60
			mu.Lock()
			b := m[ip]
			if b == nil || b.w != now {
				b = &bucket{c: 0, w: now}
				m[ip] = b
			}
			b.c++
			c := b.c
			mu.Unlock()
			if c > reqPerMin {
				http.Error(w, "rate limit exceeded", http.StatusTooManyRequests)
				return
			}
			next(w, r)
		}
	}
}

// consumeCredits attempts to consume "amount" credits for the given tenant using the Credits service.
// Returns (ok, code). If the Credits service indicates insufficient funds, code will be 402.
// baseURL example: http://localhost:5004
func consumeCredits(baseURL, tenantID string, amount int64) (bool, int) {
	if baseURL == "" || tenantID == "" || amount <= 0 {
		return false, http.StatusBadRequest
	}
	// Build request payload
	payload := map[string]any{
		"tenant_id":       tenantID,
		"amount":          amount,
		"description":     "guardian_execute",
		"reference":       "guardian",
		"idempotency_key": uuid.New().String(),
	}
	b, _ := json.Marshal(payload)
	req, err := http.NewRequest(http.MethodPost, strings.TrimRight(baseURL, "/")+"/credits/consume", bytes.NewReader(b))
	if err != nil {
		return false, http.StatusServiceUnavailable
	}
	req.Header.Set("Content-Type", "application/json")
	// Tight timeout to avoid coupling Guardian to Credits latency
	cli := &http.Client{Timeout: 2 * time.Second}
	resp, err := cli.Do(req)
	if err != nil {
		return false, http.StatusServiceUnavailable
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return false, resp.StatusCode
	}
	var cr struct {
		Success bool   `json:"success"`
		Error   string `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&cr); err != nil {
		return false, http.StatusBadGateway
	}
	if cr.Success {
		return true, http.StatusOK
	}
	if strings.Contains(strings.ToLower(cr.Error), "insufficient") {
		return false, http.StatusPaymentRequired
	}
	return false, http.StatusBadGateway
}
