package main

import (
	"archive/zip"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net/http"
	"os"
	"strings"
	"sync/atomic"
	"time"

	otelobs "shieldx/shared/shieldx-common/pkg/observability/otel"
	"shieldx/shared/shieldx-common/pkg/metrics"
	"shieldx/shared/shieldx-common/pkg/policy"
)

type state struct {
	currentDigest atomic.Value // string
	driftCount    uint64
	verifyOK      uint64
	verifyFail    uint64
	rolloutPct    uint64       // 0..100
	lastSource    atomic.Value // string (url or manual)
	// metrics
	reg           *metrics.Registry
	mVerifyOK     *metrics.Counter
	mVerifyFail   *metrics.Counter
	mDriftTotal   *metrics.Counter
	mRolloutGauge *metrics.Gauge
}

func (s *state) digest() string {
	v := s.currentDigest.Load()
	if v == nil {
		return ""
	}
	return v.(string)
}

func main() {
	addr := ":8099"
	if v := os.Getenv("POLICY_ROLLOUT_PORT"); v != "" {
		addr = ":" + v
	}
	// metrics registry
	reg := metrics.NewRegistry()
	st := &state{reg: reg}
	st.mVerifyOK = metrics.NewCounter("policy_verify_success_total", "Total successful policy verifications")
	st.mVerifyFail = metrics.NewCounter("policy_verify_failure_total", "Total failed policy verifications")
	st.mDriftTotal = metrics.NewCounter("policy_drift_events_total", "Total detected policy drift events")
	st.mRolloutGauge = metrics.NewGauge("policy_rollout_percentage", "Current canary rollout percentage")
	reg.Register(st.mVerifyOK)
	reg.Register(st.mVerifyFail)
	reg.Register(st.mDriftTotal)
	reg.RegisterGauge(st.mRolloutGauge)
	st.currentDigest.Store("")
	atomic.StoreUint64(&st.rolloutPct, 10)
	// Optional registry digest polling for drift detection
	regURL := os.Getenv("REGISTRY_DIGEST_URL") // expects plain text body = digest
	pollEvery := 30 * time.Second
	if v := os.Getenv("REGISTRY_DIGEST_INTERVAL"); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			pollEvery = d
		}
	}

	// Expose metrics via registry
	http.Handle("/metrics", reg)

	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(200) })

	// Endpoint to apply a new bundle digest and simulate canary rollout.
	http.HandleFunc("/apply", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Digest string `json:"digest"`
			URL    string `json:"url"`
			Sig    string `json:"sig"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		if req.URL != "" {
			// Fetch zip bundle and (optional) signature; compute digest and verify via Cosign if sig provided.
			start := time.Now()
			digest, verr := fetchAndVerify(req.URL, req.Sig)
			if verr != nil {
				// set simple trace attributes via response headers for demo
				w.Header().Set("x-verify-status", "fail")
				w.Header().Set("x-verify-source", req.URL)
				atomic.AddUint64(&st.verifyFail, 1)
				st.mVerifyFail.Inc()
				http.Error(w, "verify failed: "+verr.Error(), 400)
				return
			}
			w.Header().Set("x-verify-status", "ok")
			w.Header().Set("x-verify-source", req.URL)
			w.Header().Set("x-verify-digest", digest)
			atomic.AddUint64(&st.verifyOK, 1)
			st.mVerifyOK.Inc()
			st.currentDigest.Store(digest)
			st.lastSource.Store(req.URL)
			atomic.StoreUint64(&st.rolloutPct, 10)
			st.mRolloutGauge.Set(10)
			w.WriteHeader(202)
			_, _ = w.Write([]byte(fmt.Sprintf("canary started (digest %s) in %s", digest, time.Since(start))))
			return
		}
		if req.Digest == "" {
			http.Error(w, "missing digest or url", 400)
			return
		}
		// fallback: plain digest + self-signed demo (Noop)
		if err := policy.VerifyDigest(policy.NoopVerifier{}, req.Digest, []byte(req.Digest)); err != nil {
			atomic.AddUint64(&st.verifyFail, 1)
			st.mVerifyFail.Inc()
			http.Error(w, "verify failed", 400)
			return
		}
		atomic.AddUint64(&st.verifyOK, 1)
		st.mVerifyOK.Inc()
		st.currentDigest.Store(req.Digest)
		st.lastSource.Store("manual")
		atomic.StoreUint64(&st.rolloutPct, 10)
		st.mRolloutGauge.Set(10)
		w.WriteHeader(202)
		_, _ = w.Write([]byte("canary started"))
	})

	// Background drift detector: random tick simulating detection and promotion/rollback.
	go func() {
		ticker := time.NewTicker(5 * time.Second)
		defer ticker.Stop()
		for range ticker.C {
			// simulate drift 5% chance
			if rand.Intn(100) < 5 {
				atomic.AddUint64(&st.driftCount, 1)
				st.mDriftTotal.Inc()
			}
			// simulate canary promotion
			p := atomic.LoadUint64(&st.rolloutPct)
			if p > 0 && p < 100 {
				// simulate SLO check pass 80%
				if rand.Intn(100) < 80 {
					atomic.StoreUint64(&st.rolloutPct, p+10)
					st.mRolloutGauge.Set(p + 10)
				} else {
					// rollback
					atomic.StoreUint64(&st.rolloutPct, 0)
					st.mRolloutGauge.Set(0)
					st.currentDigest.Store("")
				}
			}
		}
	}()

	// Real drift detector against registry digest (if configured)
	if regURL != "" {
		go func() {
			t := time.NewTicker(pollEvery)
			defer t.Stop()
			for range t.C {
				resp, err := http.Get(regURL)
				if err != nil {
					continue
				}
				b, err := io.ReadAll(resp.Body)
				resp.Body.Close()
				if err != nil {
					continue
				}
				remote := strings.TrimSpace(string(b))
				if remote == "" {
					continue
				}
				local := st.digest()
				if local != "" && local != remote {
					atomic.AddUint64(&st.driftCount, 1)
					st.mDriftTotal.Inc()
				}
			}
		}()
	}

	// Wrap with OTEL if enabled
	shutdown := otelobs.InitTracer("policy_rollout")
	defer shutdown(nil)
	mux := http.DefaultServeMux
	// Add HTTP metrics middleware
	httpMetrics := metrics.NewHTTPMetrics(reg, "policy_rollout")
	h := httpMetrics.Middleware(mux)
	h = otelobs.WrapHTTPHandler("policy_rollout", h)
	log.Printf("policy-rollout listening on %s", addr)
	log.Fatal(http.ListenAndServe(addr, h))
}

func fetchAndVerify(url, sigURL string) (string, error) {
	// download bundle zip
	resp, err := http.Get(url)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return "", fmt.Errorf("fetch bundle: status %d", resp.StatusCode)
	}
	bz, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	zr, err := zip.NewReader(bytes.NewReader(bz), int64(len(bz)))
	if err != nil {
		return "", err
	}
	b, err := policy.LoadFromZipReader(zr)
	if err != nil {
		return "", err
	}
	digest, err := b.Hash()
	if err != nil {
		return "", err
	}
	if strings.TrimSpace(sigURL) != "" {
		sresp, err := http.Get(sigURL)
		if err != nil {
			return "", err
		}
		defer sresp.Body.Close()
		if sresp.StatusCode != 200 {
			return "", fmt.Errorf("fetch sig: status %d", sresp.StatusCode)
		}
		sig, err := io.ReadAll(sresp.Body)
		if err != nil {
			return "", err
		}
		if err := policy.VerifyDigest(policy.CosignCLI{}, digest, sig); err != nil {
			return "", err
		}
	}
	return digest, nil
}
