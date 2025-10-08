package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"shieldx/shared/shieldx-common/pkg/metrics"
	logcorr "shieldx/shared/shieldx-common/pkg/observability/logcorr"
	otelobs "shieldx/shared/shieldx-common/pkg/observability/otel"
	"shieldx/shared/shieldx-common/pkg/ratls"
	"strconv"
	"strings"
	"sync"
	"time"
)

func main() {
	dbURL := getEnv("DATABASE_URL", "postgres://contauth_user:contauth_pass2024@localhost:5432/contauth")
	port := getEnv("PORT", "5002")

	var collector interface {
		Close() error
		CollectTelemetry(http.ResponseWriter, *http.Request)
		CalculateRiskScore(http.ResponseWriter, *http.Request)
		GetAuthDecision(http.ResponseWriter, *http.Request)
	}
	if os.Getenv("DISABLE_DB") == "true" {
		log.Printf("DISABLE_DB=true detected; using DummyCollector (no database)")
		collector = NewDummyCollector()
	} else {
		c, err := NewContAuthCollector(dbURL)
		if err != nil {
			log.Fatalf("Failed to initialize collector: %v", err)
		}
		collector = c
	}
	defer collector.Close()

	// Setup routes with metrics registry
	mux := http.NewServeMux()
	reg := metrics.NewRegistry()
	httpMetrics := metrics.NewHTTPMetrics(reg, "contauth")
	// Simple per-IP rate limiter (req/min)
	rl := makeRateLimiter(parseIntDefault("CONTAUTH_RL_REQS_PER_MIN", 240))
	// Body size guard for POST endpoints
	guard := func(max int64, h http.HandlerFunc) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			if max > 0 && (r.Method == http.MethodPost || r.Method == http.MethodPut) {
				r.Body = http.MaxBytesReader(w, r.Body, max)
			}
			h(w, r)
		}
	}
	mux.HandleFunc("/contauth/telemetry", rl(guard(1<<20, collector.CollectTelemetry))) // 1MB cap
	mux.HandleFunc("/contauth/collect", rl(guard(1<<20, collector.CollectTelemetry)))
	mux.HandleFunc("/contauth/score", rl(guard(256<<10, collector.CalculateRiskScore))) // legacy comprehensive scorer

	// High-performance scorer (stateless, hashed features only) - optimized path
	hps := NewHighPerformanceScorer(nil)
	mux.HandleFunc("/contauth/scorefast", rl(guard(256<<10, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var raw map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&raw); err != nil {
			http.Error(w, "bad json", http.StatusBadRequest)
			return
		}
		features, err := hps.ExtractHashedFeatures(raw)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		res := hps.CalculateRiskScore(features)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(res)
	})))
	mux.HandleFunc("/contauth/decision", rl(collector.GetAuthDecision))
	mux.Handle("/metrics", reg)
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(200)
		w.Write([]byte(`{"status":"healthy","service":"contauth"}`))
	})

	// OpenTelemetry tracing (no-op unless built with otelotlp and endpoint set)
	shutdown := otelobs.InitTracer("contauth")
	defer shutdown(context.Background())

	// Wrap with metrics, log-correlation, then tracing (outermost) so span is in context for logging
	h := httpMetrics.Middleware(mux)
	h = logcorr.Middleware(h)
	h = otelobs.WrapHTTPHandler("contauth", h)
	// RA-TLS optional enablement
	gCertExpiry := metrics.NewGauge("ratls_cert_expiry_seconds", "Seconds until current RA-TLS cert expiry")
	reg.RegisterGauge(gCertExpiry)
	var issuer *ratls.AutoIssuer
	if os.Getenv("RATLS_ENABLE") == "true" {
		td := getenvDefault("RATLS_TRUST_DOMAIN", "shieldx.local")
		ns := getenvDefault("RATLS_NAMESPACE", "default")
		svc := getenvDefault("RATLS_SERVICE", "contauth")
		rotate := parseDurationDefault("RATLS_ROTATE_EVERY", 45*time.Minute)
		valid := parseDurationDefault("RATLS_VALIDITY", 60*time.Minute)
		ai, err := ratls.NewDevIssuer(ratls.Identity{TrustDomain: td, Namespace: ns, Service: svc}, rotate, valid)
		if err != nil {
			log.Fatalf("[contauth] RA-TLS init: %v", err)
		}
		issuer = ai
		go func() {
			for {
				if t, err := issuer.LeafNotAfter(); err == nil {
					gCertExpiry.Set(uint64(time.Until(t).Seconds()))
				}
				time.Sleep(1 * time.Minute)
			}
		}()
	}
	// Configure server with optional mTLS
	addr := fmt.Sprintf(":%s", port)
	srv := &http.Server{Addr: addr, Handler: h}
	if issuer != nil {
		srv.TLSConfig = issuer.ServerTLSConfig(true, getenvDefault("RATLS_TRUST_DOMAIN", "shieldx.local"))
		log.Printf("Continuous Auth service (RA-TLS) starting on %s", addr)
		log.Fatal(srv.ListenAndServeTLS("", ""))
	} else {
		log.Printf("Continuous Auth service starting on %s", addr)
		log.Fatal(srv.ListenAndServe())
	}
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

// small env helpers (match other services)
func getenvDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
func parseDurationDefault(key string, def time.Duration) time.Duration {
	if v := os.Getenv(key); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			return d
		}
	}
	return def
}
func parseIntDefault(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			return i
		}
	}
	return def
}

// Simple per-IP token bucket (count per minute window)
func makeRateLimiter(reqPerMin int) func(http.HandlerFunc) http.HandlerFunc {
	if reqPerMin <= 0 {
		reqPerMin = 240
	}
	type bucket struct {
		count int
		win   int64
	}
	var mu sync.Mutex
	buckets := map[string]*bucket{}
	return func(next http.HandlerFunc) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			ip := r.Header.Get("X-Forwarded-For")
			if ip == "" {
				ip = strings.Split(r.RemoteAddr, ":")[0]
			}
			now := time.Now().Unix() / 60
			mu.Lock()
			b := buckets[ip]
			if b == nil || b.win != now {
				b = &bucket{count: 0, win: now}
				buckets[ip] = b
			}
			b.count++
			cnt := b.count
			mu.Unlock()
			if cnt > reqPerMin {
				http.Error(w, "rate limit exceeded", http.StatusTooManyRequests)
				return
			}
			next(w, r)
		}
	}
}
