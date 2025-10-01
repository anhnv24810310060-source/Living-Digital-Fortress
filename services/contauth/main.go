package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"
	"shieldx/pkg/metrics"
	otelobs "shieldx/pkg/observability/otel"
	logcorr "shieldx/pkg/observability/logcorr"
	"shieldx/pkg/ratls"
)

func main() {
	dbURL := getEnv("DATABASE_URL", "postgres://contauth_user:contauth_pass2024@localhost:5432/contauth")
	port := getEnv("PORT", "5002")

	var collector interface{ Close() error; CollectTelemetry(http.ResponseWriter, *http.Request); CalculateRiskScore(http.ResponseWriter, *http.Request); GetAuthDecision(http.ResponseWriter, *http.Request) }
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
	mux.HandleFunc("/contauth/telemetry", collector.CollectTelemetry)
	mux.HandleFunc("/contauth/score", collector.CalculateRiskScore)
	mux.HandleFunc("/contauth/decision", collector.GetAuthDecision)
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
		if err != nil { log.Fatalf("[contauth] RA-TLS init: %v", err) }
		issuer = ai
		go func(){ for { if t, err := issuer.LeafNotAfter(); err==nil { gCertExpiry.Set(uint64(time.Until(t).Seconds())) }; time.Sleep(1*time.Minute) } }()
	}
	// Configure server with optional mTLS
	addr := fmt.Sprintf(":%s", port)
	srv := &http.Server{ Addr: addr, Handler: h }
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
	if v := os.Getenv(key); v != "" { return v }
	return def
}
func parseDurationDefault(key string, def time.Duration) time.Duration {
	if v := os.Getenv(key); v != "" { if d, err := time.ParseDuration(v); err == nil { return d } }
	return def
}