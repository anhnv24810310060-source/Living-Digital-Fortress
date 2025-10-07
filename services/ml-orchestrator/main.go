package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"mime/multipart"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"shieldx/shared/metrics"
	logcorr "shieldx/shared/observability/logcorr"
	otelobs "shieldx/shared/observability/otel"
	"shieldx/pkg/ratls"
)

type MLOrchestrator struct {
	anomalyDetector  *AnomalyDetector
	featureExtractor *FeatureExtractor
	forest           *IsolationForest
	ensembleWeight   float64
	// A/B testing: alternate ensemble weight and traffic split (%0-100)
	abAltWeight float64
	abPercent   int
	_abA        *metrics.Counter
	_abB        *metrics.Counter
	// simple in-memory model registry: version -> serialized bytes
	versions       map[string][]byte
	currentVersion string
}

// AnomalyDetector implements a robust multivariate anomaly detector using
// Mahalanobis distance with covariance regularization and a data-driven threshold (p99).
type AnomalyDetector struct {
	Mu              []float64   `json:"mu"`
	Cov             [][]float64 `json:"cov"`
	InvCov          [][]float64 `json:"inv_cov"`
	Dim             int         `json:"dim"`
	Trained         bool        `json:"trained"`
	ThreshD2        float64     `json:"thresh_d2"`
	Eps             float64     `json:"eps"`
	MuMutex         sync.RWMutex
	onMetricsUpdate func(float64)
	onTrain         func(int)
	onAnomaly       func()
}

type FeatureExtractor struct {
	windowSize time.Duration
}

type TelemetryEvent struct {
	Timestamp   time.Time `json:"timestamp"`
	Source      string    `json:"source"`
	EventType   string    `json:"event_type"`
	TenantID    string    `json:"tenant_id"`
	Features    []float64 `json:"features"`
	ThreatScore float64   `json:"threat_score"`
}

type AnomalyResult struct {
	IsAnomaly   bool    `json:"is_anomaly"`
	Score       float64 `json:"score"`
	Confidence  float64 `json:"confidence"`
	Explanation string  `json:"explanation"`
}

func main() {
	orchestrator := &MLOrchestrator{
		anomalyDetector:  &AnomalyDetector{Eps: 1e-6},
		featureExtractor: &FeatureExtractor{windowSize: 30 * time.Second},
		forest:           NewIsolationForest(100, 256),
		ensembleWeight:   parseFloatDefault("ML_ENSEMBLE_WEIGHT", 0.6),
		abAltWeight:      parseFloatDefault("ML_AB_WEIGHT_ALT", 0.5),
		abPercent:        parseIntDefault("ML_AB_PERCENT", 0),
		versions:         make(map[string][]byte),
	}

	mux := http.NewServeMux()
	reg := metrics.NewRegistry()
	httpMetrics := metrics.NewHTTPMetrics(reg, "ml_orchestrator")
	// Security: admin-gate model management and training APIs; lightweight RL for public analyze
	adminOnly := makeAdminMiddleware()
	rateLimit := makeRateLimiter(parseIntDefault("ML_RL_REQS_PER_MIN", 120))
	mux.HandleFunc("/analyze", rateLimit(orchestrator.handleAnalyze))
	mux.HandleFunc("/train", adminOnly(orchestrator.handleTrain))
	mux.HandleFunc("/model/save", adminOnly(orchestrator.handleSave))
	mux.HandleFunc("/model/load", adminOnly(orchestrator.handleLoad))
	mux.HandleFunc("/model/version/save", adminOnly(orchestrator.handleSaveVersion))
	mux.HandleFunc("/model/version/rollback", adminOnly(orchestrator.handleRollback))
	mux.HandleFunc("/model/version/list", adminOnly(orchestrator.handleListVersions))
	mux.HandleFunc("/model/mode", adminOnly(orchestrator.handleMode))
	// Federated learning aggregation (secure aggregation placeholder)
	mux.HandleFunc("/federated/aggregate", adminOnly(orchestrator.handleFederatedAggregate))
	// Adversarial example generation (PGD/FGSM placeholder)
	mux.HandleFunc("/adversarial/generate", adminOnly(orchestrator.handleAdversarialGenerate))
	mux.HandleFunc("/health", orchestrator.handleHealth)
	mux.HandleFunc("/whoami", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"service":       "ml-orchestrator",
			"ratls_enabled": os.Getenv("RATLS_ENABLE") == "true",
		})
	})
	mux.Handle("/metrics", reg)

	port := os.Getenv("ML_ORCHESTRATOR_PORT")
	if port == "" {
		port = "8087"
	}

	// OpenTelemetry tracing (no-op unless built with otelotlp and endpoint set)
	shutdown := otelobs.InitTracer("ml_orchestrator")
	defer shutdown(context.Background())

	// Wrap with metrics, log-correlation, then tracing
	h := httpMetrics.Middleware(mux)
	h = logcorr.Middleware(h)
	h = otelobs.WrapHTTPHandler("ml_orchestrator", h)

	// RA-TLS optional enablement
	gCertExpiry := metrics.NewGauge("ratls_cert_expiry_seconds", "Seconds until current RA-TLS cert expiry")
	reg.RegisterGauge(gCertExpiry)
	// ML metrics
	mAnalyze := metrics.NewCounter("ml_analyze_total", "Total analyze requests")
	mTrain := metrics.NewCounter("ml_train_total", "Total training calls")
	mAnomaly := metrics.NewCounter("ml_anomalies_total", "Total anomalies detected")
	gThreshold := metrics.NewGauge("ml_threshold_d2", "Current MD^2 anomaly threshold")
	// A/B testing counters
	mABGroupA := metrics.NewCounter("ml_ab_group_a_total", "Analyze requests in group A (control)")
	mABGroupB := metrics.NewCounter("ml_ab_group_b_total", "Analyze requests in group B (alt)")
	reg.Register(mAnalyze)
	reg.Register(mTrain)
	reg.Register(mAnomaly)
	reg.RegisterGauge(gThreshold)
	reg.Register(mABGroupA)
	reg.Register(mABGroupB)
	orchestrator.anomalyDetector.onMetricsUpdate = func(th float64) { gThreshold.Set(uint64(th)) }
	orchestrator.anomalyDetector.onTrain = func(n int) { mTrain.Add(1) }
	orchestrator.anomalyDetector.onAnomaly = func() { mAnomaly.Add(1) }
	// Scheduled retrain worker (optional)
	if d := parseDurationDefault("ML_RETRAIN_EVERY", 0); d > 0 {
		go func() {
			ticker := time.NewTicker(d)
			defer ticker.Stop()
			for range ticker.C {
				orchestrator.runScheduledRetrain()
			}
		}()
	}

	var issuer *ratls.AutoIssuer
	if os.Getenv("RATLS_ENABLE") == "true" {
		td := getenvDefault("RATLS_TRUST_DOMAIN", "shieldx.local")
		ns := getenvDefault("RATLS_NAMESPACE", "default")
		svc := getenvDefault("RATLS_SERVICE", "ml-orchestrator")
		rotate := parseDurationDefault("RATLS_ROTATE_EVERY", 45*time.Minute)
		valid := parseDurationDefault("RATLS_VALIDITY", 60*time.Minute)
		ai, err := ratls.NewDevIssuer(ratls.Identity{TrustDomain: td, Namespace: ns, Service: svc}, rotate, valid)
		if err != nil {
			log.Fatalf("[ml-orchestrator] RA-TLS init: %v", err)
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
	addr := fmt.Sprintf(":%s", port)
	// expose counters inside orchestrator for handler usage
	srv := &http.Server{Addr: addr, Handler: h}
	// attach counters via closure fields using package-level vars? We'll pass via receiver state
	orchestrator._abA = mABGroupA
	orchestrator._abB = mABGroupB
	if issuer != nil {
		srv.TLSConfig = issuer.ServerTLSConfig(true, getenvDefault("RATLS_TRUST_DOMAIN", "shieldx.local"))
		log.Printf("ML Orchestrator (RA-TLS) starting on %s", addr)
		log.Fatal(srv.ListenAndServeTLS("", ""))
	} else {
		log.Printf("ML Orchestrator starting on %s", addr)
		log.Fatal(srv.ListenAndServe())
	}
}

// env helpers aligned with other services
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
func parseFloatDefault(key string, def float64) float64 {
	if v := os.Getenv(key); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
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
func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

// --- Middleware helpers ---
// Admin middleware: require header "X-Admin-Token" to match env ML_API_ADMIN_TOKEN; else 403.
func makeAdminMiddleware() func(http.HandlerFunc) http.HandlerFunc {
	token := os.Getenv("ML_API_ADMIN_TOKEN")
	if token == "" {
		// If not set, keep endpoints disabled by default in production: respond 403
		return func(next http.HandlerFunc) http.HandlerFunc {
			return func(w http.ResponseWriter, r *http.Request) {
				http.Error(w, "admin endpoint disabled", http.StatusForbidden)
			}
		}
	}
	return func(next http.HandlerFunc) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			if r.Header.Get("X-Admin-Token") != token {
				http.Error(w, "forbidden", http.StatusForbidden)
				return
			}
			next(w, r)
		}
	}
}

// Simple per-IP token bucket rate limiter (req/minute)
func makeRateLimiter(reqPerMin int) func(http.HandlerFunc) http.HandlerFunc {
	if reqPerMin <= 0 {
		reqPerMin = 120
	}
	type bucket struct {
		count  int
		window int64
	}
	var mu sync.Mutex
	buckets := map[string]*bucket{}
	return func(next http.HandlerFunc) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			ip := r.Header.Get("X-Forwarded-For")
			if ip == "" {
				ip = strings.Split(r.RemoteAddr, ":")[0]
			}
			nowMin := time.Now().Unix() / 60
			mu.Lock()
			b := buckets[ip]
			if b == nil || b.window != nowMin {
				b = &bucket{count: 0, window: nowMin}
				buckets[ip] = b
			}
			b.count++
			c := b.count
			mu.Unlock()
			if c > reqPerMin {
				http.Error(w, "rate limit exceeded", http.StatusTooManyRequests)
				return
			}
			next(w, r)
		}
	}
}

func (m *MLOrchestrator) handleAnalyze(w http.ResponseWriter, r *http.Request) {
	var event TelemetryEvent
	if err := json.NewDecoder(r.Body).Decode(&event); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	// If A/B testing enabled, select group based on TenantID hash for stickiness
	weight := m.ensembleWeight
	if m.abPercent > 0 {
		bucket := 0
		for i := 0; i < len(event.TenantID); i++ {
			bucket = (bucket*31 + int(event.TenantID[i])) % 100
		}
		if bucket < m.abPercent { // group B
			weight = m.abAltWeight
			if m._abB != nil {
				m._abB.Add(1)
			}
		} else {
			if m._abA != nil {
				m._abA.Add(1)
			}
		}
	}

	// Mahalanobis-based score
	mdRes := m.anomalyDetector.Predict(event)
	// Isolation Forest score in [0,1]
	ifScore := 0.0
	if m.forest != nil {
		ifScore = m.forest.Score(event.Features)
	}
	// Ensemble: weighted average; conservative anomaly if either flags strongly
	ensScore := weight*mdRes.Score + (1-weight)*ifScore
	isAnom := ensScore >= 0.5 || mdRes.IsAnomaly
	conf := 0.5 + 0.5*math.Tanh(2*(ensScore-0.5))
	result := AnomalyResult{IsAnomaly: isAnom, Score: ensScore, Confidence: conf, Explanation: "Ensemble(MD^2+IF)"}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

func (m *MLOrchestrator) handleTrain(w http.ResponseWriter, r *http.Request) {
	var events []TelemetryEvent
	if err := json.NewDecoder(r.Body).Decode(&events); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	if err := m.anomalyDetector.Train(events); err != nil {
		http.Error(w, fmt.Sprintf("train error: %v", err), http.StatusBadRequest)
		return
	}
	// Train Isolation Forest on same batch
	X := make([][]float64, 0, len(events))
	for _, ev := range events {
		X = append(X, append([]float64(nil), ev.Features...))
	}
	if m.forest == nil {
		m.forest = NewIsolationForest(100, 256)
	}
	m.forest.Train(X)
	// Save model snapshot and log to MLflow if configured
	_ = os.MkdirAll("data", 0o755)
	modelPath := getenvDefault("ML_MODEL_PATH", "data/ml_model.json")
	// persist current model
	_ = m.persistModel(modelPath)
	// Log to MLflow (metrics/params + artifact)
	m.logTrainingToMLflow(len(events), m.anomalyDetector.Dim, m.anomalyDetector.ThreshD2)
	m.logArtifactToMLflow(modelPath, "models")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]interface{}{"status": "trained", "dim": m.anomalyDetector.Dim, "thresh_d2": m.anomalyDetector.ThreshD2, "if_trees": m.forest.NumTrees, "saved": modelPath})
}

func (m *MLOrchestrator) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":          "healthy",
		"timestamp":       time.Now(),
		"service":         "ml-orchestrator",
		"trained":         m.anomalyDetector.Trained,
		"dim":             m.anomalyDetector.Dim,
		"ensemble_weight": m.ensembleWeight,
	})
}

// Prometheus metrics moved to /metrics via Registry

func (ad *AnomalyDetector) Predict(event TelemetryEvent) AnomalyResult {
	ad.MuMutex.RLock()
	trained := ad.Trained
	mu := append([]float64(nil), ad.Mu...)
	inv := copyMatrix(ad.InvCov)
	thresh := ad.ThreshD2
	ad.MuMutex.RUnlock()

	if !trained {
		return AnomalyResult{IsAnomaly: false, Score: 0.0, Confidence: 0.0, Explanation: "Model not trained"}
	}
	x := event.Features
	if len(x) != len(mu) {
		return AnomalyResult{IsAnomaly: false, Score: 0.0, Confidence: 0.0, Explanation: "feature dimension mismatch"}
	}
	d2 := mahalanobis2(x, mu, inv)
	isAnom := d2 > thresh
	if isAnom && ad.onAnomaly != nil {
		ad.onAnomaly()
	}
	// Map d2 to [0,1] via logistic on sqrt distance relative to threshold
	score := 1.0 / (1.0 + math.Exp(-2*(math.Sqrt(d2)-math.Sqrt(thresh))))
	conf := 0.5 + 0.5*math.Tanh((d2-thresh)/(thresh+1e-9))
	return AnomalyResult{IsAnomaly: isAnom, Score: score, Confidence: conf, Explanation: "Mahalanobis distance detector"}
}

// Train estimates mean/cov and selects a data-driven threshold (p99 of training MD^2).
func (ad *AnomalyDetector) Train(events []TelemetryEvent) error {
	if len(events) == 0 {
		return errors.New("no events to train")
	}
	// Extract feature matrix
	X := make([][]float64, 0, len(events))
	var d int
	for i, ev := range events {
		if i == 0 {
			d = len(ev.Features)
			if d == 0 {
				return errors.New("empty feature vector")
			}
		}
		if len(ev.Features) != d {
			return fmt.Errorf("inconsistent feature dims: want %d got %d at row %d", d, len(ev.Features), i)
		}
		row := append([]float64(nil), ev.Features...)
		X = append(X, row)
	}
	mu := meanVector(X)
	cov := covarianceMatrix(X, mu)
	// Regularize
	for i := 0; i < d; i++ {
		cov[i][i] += ad.Eps
	}
	inv, ok := invertMatrix(cov)
	if !ok {
		// Fallback to diagonal inverse
		inv = make([][]float64, d)
		for i := 0; i < d; i++ {
			inv[i] = make([]float64, d)
			v := cov[i][i]
			if v < ad.Eps {
				v = ad.Eps
			}
			inv[i][i] = 1.0 / v
		}
	}
	// Compute training distances and set p99 threshold
	dists := make([]float64, len(X))
	for i := range X {
		dists[i] = mahalanobis2(X[i], mu, inv)
	}
	thr := percentile(dists, 0.99)
	if thr <= 0 || math.IsNaN(thr) || math.IsInf(thr, 0) {
		thr = float64(d) * 9.0
	} // 3-sigma heuristic fallback

	ad.MuMutex.Lock()
	ad.Mu = mu
	ad.Cov = cov
	ad.InvCov = inv
	ad.Dim = d
	ad.Trained = true
	ad.ThreshD2 = thr
	ad.MuMutex.Unlock()
	if ad.onMetricsUpdate != nil {
		ad.onMetricsUpdate(thr)
	}
	if ad.onTrain != nil {
		ad.onTrain(len(events))
	}
	return nil
}

// add fields to struct above

// Persistence: save/load model params (mu, cov, invCov, dim, thresh)
func (m *MLOrchestrator) handleSave(w http.ResponseWriter, r *http.Request) {
	path := getenvDefault("ML_MODEL_PATH", "data/ml_model.json")
	os.MkdirAll("data", 0o755)
	m.anomalyDetector.MuMutex.RLock()
	payload := map[string]interface{}{
		"mu":        m.anomalyDetector.Mu,
		"cov":       m.anomalyDetector.Cov,
		"inv_cov":   m.anomalyDetector.InvCov,
		"dim":       m.anomalyDetector.Dim,
		"trained":   m.anomalyDetector.Trained,
		"thresh_d2": m.anomalyDetector.ThreshD2,
	}
	m.anomalyDetector.MuMutex.RUnlock()
	f, err := os.Create(path)
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	defer f.Close()
	if err := json.NewEncoder(f).Encode(payload); err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	w.WriteHeader(200)
	_ = json.NewEncoder(w).Encode(map[string]string{"status": "saved", "path": path})
}

func (m *MLOrchestrator) handleLoad(w http.ResponseWriter, r *http.Request) {
	path := getenvDefault("ML_MODEL_PATH", "data/ml_model.json")
	f, err := os.Open(path)
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	defer f.Close()
	var payload struct {
		Mu       []float64   `json:"mu"`
		Cov      [][]float64 `json:"cov"`
		InvCov   [][]float64 `json:"inv_cov"`
		Dim      int         `json:"dim"`
		Trained  bool        `json:"trained"`
		ThreshD2 float64     `json:"thresh_d2"`
	}
	if err := json.NewDecoder(f).Decode(&payload); err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	m.anomalyDetector.MuMutex.Lock()
	m.anomalyDetector.Mu = payload.Mu
	m.anomalyDetector.Cov = payload.Cov
	m.anomalyDetector.InvCov = payload.InvCov
	m.anomalyDetector.Dim = payload.Dim
	m.anomalyDetector.Trained = payload.Trained
	m.anomalyDetector.ThreshD2 = payload.ThreshD2
	m.anomalyDetector.MuMutex.Unlock()
	w.WriteHeader(200)
	_ = json.NewEncoder(w).Encode(map[string]string{"status": "loaded", "path": path})
}

// Save current model into version registry with version id provided or timestamp-based.
func (m *MLOrchestrator) handleSaveVersion(w http.ResponseWriter, r *http.Request) {
	ver := r.URL.Query().Get("ver")
	if ver == "" {
		ver = time.Now().Format("20060102-150405")
	}
	// persist current model to bytes
	buf, err := m.saveModelToBytes()
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	if m.versions == nil {
		m.versions = make(map[string][]byte)
	}
	m.versions[ver] = buf
	m.currentVersion = ver
	writeJSON(w, 200, map[string]any{"saved": ver, "bytes": len(buf)})
}

func (m *MLOrchestrator) handleListVersions(w http.ResponseWriter, r *http.Request) {
	vs := make([]string, 0, len(m.versions))
	for k := range m.versions {
		vs = append(vs, k)
	}
	sort.Strings(vs)
	writeJSON(w, 200, map[string]any{"versions": vs, "current": m.currentVersion})
}

// Rollback to a previous version id. If not found in memory, optionally load from disk path=ML_MODEL_PATH.ver
func (m *MLOrchestrator) handleRollback(w http.ResponseWriter, r *http.Request) {
	ver := r.URL.Query().Get("ver")
	if ver == "" {
		http.Error(w, "ver required", 400)
		return
	}
	var data []byte
	if b, ok := m.versions[ver]; ok {
		data = append([]byte(nil), b...)
	} else {
		// best-effort: try file with suffix
		path := getenvDefault("ML_MODEL_PATH", "data/ml_model.json") + "." + ver
		if b, err := os.ReadFile(path); err == nil {
			data = b
		}
	}
	if len(data) == 0 {
		http.Error(w, "version not found", 404)
		return
	}
	if err := m.loadModelFromBytes(data); err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	m.currentVersion = ver
	writeJSON(w, 200, map[string]any{"rollback": ver})
}

func (m *MLOrchestrator) saveModelToBytes() ([]byte, error) {
	payload := map[string]any{
		"mu":        m.anomalyDetector.Mu,
		"cov":       m.anomalyDetector.Cov,
		"inv_cov":   m.anomalyDetector.InvCov,
		"dim":       m.anomalyDetector.Dim,
		"trained":   m.anomalyDetector.Trained,
		"thresh_d2": m.anomalyDetector.ThreshD2,
		"if_trees":  m.forest.NumTrees,
		"if_sample": m.forest.SampleSize,
	}
	return json.Marshal(payload)
}

func (m *MLOrchestrator) loadModelFromBytes(b []byte) error {
	var payload struct {
		Mu       []float64   `json:"mu"`
		Cov      [][]float64 `json:"cov"`
		InvCov   [][]float64 `json:"inv_cov"`
		Dim      int         `json:"dim"`
		Trained  bool        `json:"trained"`
		ThreshD2 float64     `json:"thresh_d2"`
		IFTrees  int         `json:"if_trees"`
		IFSample int         `json:"if_sample"`
	}
	if err := json.Unmarshal(b, &payload); err != nil {
		return err
	}
	m.anomalyDetector.MuMutex.Lock()
	m.anomalyDetector.Mu = payload.Mu
	m.anomalyDetector.Cov = payload.Cov
	m.anomalyDetector.InvCov = payload.InvCov
	m.anomalyDetector.Dim = payload.Dim
	m.anomalyDetector.Trained = payload.Trained
	m.anomalyDetector.ThreshD2 = payload.ThreshD2
	m.anomalyDetector.MuMutex.Unlock()
	if payload.IFTrees > 0 {
		m.forest.NumTrees = payload.IFTrees
	}
	if payload.IFSample > 0 {
		m.forest.SampleSize = payload.IFSample
	}
	return nil
}

// Switch/inspect ensemble mode and IF params
func (m *MLOrchestrator) handleMode(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		writeJSON(w, 200, map[string]interface{}{"ensemble_weight": m.ensembleWeight, "if_trees": m.forest.NumTrees, "if_sample": m.forest.SampleSize})
	case http.MethodPost:
		var body struct {
			EnsembleWeight float64 `json:"ensemble_weight"`
			Trees          int     `json:"if_trees"`
			Sample         int     `json:"if_sample"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			writeJSON(w, 400, map[string]string{"error": "bad json"})
			return
		}
		if body.EnsembleWeight > 0 {
			m.ensembleWeight = body.EnsembleWeight
		}
		if body.Trees > 0 {
			m.forest.NumTrees = body.Trees
		}
		if body.Sample > 0 {
			m.forest.SampleSize = body.Sample
			if m.forest.SampleSize < 2 {
				m.forest.SampleSize = 2
			}
		}
		writeJSON(w, 200, map[string]interface{}{"ok": true, "ensemble_weight": m.ensembleWeight})
	default:
		w.WriteHeader(405)
	}
}

// scheduled retrain from a local JSON file if present
func (m *MLOrchestrator) runScheduledRetrain() {
	path := getenvDefault("ML_RETRAIN_DATA", "data/retrain.json")
	b, err := os.ReadFile(path)
	if err != nil {
		return
	}
	var events []TelemetryEvent
	if err := json.Unmarshal(b, &events); err != nil {
		return
	}
	if len(events) == 0 {
		return
	}
	_ = m.anomalyDetector.Train(events)
	X := make([][]float64, 0, len(events))
	for _, ev := range events {
		X = append(X, append([]float64(nil), ev.Features...))
	}
	if m.forest == nil {
		m.forest = NewIsolationForest(100, 256)
	}
	m.forest.Train(X)
	modelPath := getenvDefault("ML_MODEL_PATH", "data/ml_model.json")
	_ = m.persistModel(modelPath)
	m.logTrainingToMLflow(len(events), m.anomalyDetector.Dim, m.anomalyDetector.ThreshD2)
	m.logArtifactToMLflow(modelPath, "models")
}

// MLflow logging (metrics/params only; artifacts handled by MLflow server to MinIO)
func (m *MLOrchestrator) logTrainingToMLflow(numEvents int, dim int, thresh float64) {
	if c := newMLflowClientFromEnv(); c != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = c.LogTrainingRun(ctx, map[string]float64{
			"num_events": float64(numEvents),
			"dim":        float64(dim),
			"thresh_d2":  thresh,
			"if_trees":   float64(m.forest.NumTrees),
			"if_sample":  float64(m.forest.SampleSize),
		}, map[string]string{"detector": "ensemble_md2_iforest"})
	}
}

// Upload a local artifact file to MLflow under artifactPath (best-effort)
func (m *MLOrchestrator) logArtifactToMLflow(localPath, artifactPath string) {
	c := newMLflowClientFromEnv()
	if c == nil {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
	defer cancel()
	// Create a new run for artifact-or reuse experiment/run approach: create a short-lived run
	expID, err := c.getOrCreateExperiment(ctx)
	if err != nil {
		return
	}
	runID, err := c.createRun(ctx, expID)
	if err != nil {
		return
	}
	_ = c.uploadArtifact(ctx, runID, localPath, artifactPath)
	_ = c.setRunTerminated(ctx, runID)
}

// persist current model to a JSON file
func (m *MLOrchestrator) persistModel(path string) error {
	payload := map[string]interface{}{
		"mu":        m.anomalyDetector.Mu,
		"cov":       m.anomalyDetector.Cov,
		"inv_cov":   m.anomalyDetector.InvCov,
		"dim":       m.anomalyDetector.Dim,
		"trained":   m.anomalyDetector.Trained,
		"thresh_d2": m.anomalyDetector.ThreshD2,
		"if_trees":  m.forest.NumTrees,
		"if_sample": m.forest.SampleSize,
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return json.NewEncoder(f).Encode(payload)
}

// ---- Minimal MLflow REST client ----
type mlflowClient struct {
	base       string
	experiment string
	token      string
}

func newMLflowClientFromEnv() *mlflowClient {
	base := os.Getenv("MLFLOW_TRACKING_URI")
	if base == "" {
		return nil
	}
	exp := os.Getenv("MLFLOW_EXPERIMENT")
	if exp == "" {
		exp = "ml-orchestrator"
	}
	return &mlflowClient{base: strings.TrimRight(base, "/"), experiment: exp, token: os.Getenv("MLFLOW_TOKEN")}
}

func (c *mlflowClient) LogTrainingRun(ctx context.Context, metrics map[string]float64, params map[string]string) error {
	// 1) Get or create experiment
	expID, err := c.getOrCreateExperiment(ctx)
	if err != nil {
		return err
	}
	// 2) Create run
	runID, err := c.createRun(ctx, expID)
	if err != nil {
		return err
	}
	// 3) Log params
	for k, v := range params {
		_ = c.logParam(ctx, runID, k, v)
	}
	// 4) Log metrics
	ts := time.Now().UnixMilli()
	for k, v := range metrics {
		_ = c.logMetric(ctx, runID, k, v, ts)
	}
	// 5) Set terminated
	_ = c.setRunTerminated(ctx, runID)
	return nil
}

func (c *mlflowClient) doJSON(ctx context.Context, method, path string, payload any, out any) error {
	var body io.Reader
	if payload != nil {
		b, _ := json.Marshal(payload)
		body = bytes.NewReader(b)
	}
	req, _ := http.NewRequestWithContext(ctx, method, c.base+path, body)
	req.Header.Set("Content-Type", "application/json")
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("mlflow %s: %s", path, string(b))
	}
	if out != nil {
		return json.NewDecoder(resp.Body).Decode(out)
	}
	return nil
}

func (c *mlflowClient) getOrCreateExperiment(ctx context.Context) (string, error) {
	// try get by name
	var getResp struct {
		Experiment struct {
			ExperimentId string `json:"experiment_id"`
		} `json:"experiment"`
	}
	err := c.doJSON(ctx, http.MethodGet, "/api/2.0/mlflow/experiments/get-by-name?experiment_name="+url.QueryEscape(c.experiment), nil, &getResp)
	if err == nil && getResp.Experiment.ExperimentId != "" {
		return getResp.Experiment.ExperimentId, nil
	}
	// create
	var createResp struct {
		ExperimentId string `json:"experiment_id"`
	}
	err = c.doJSON(ctx, http.MethodPost, "/api/2.0/mlflow/experiments/create", map[string]string{"name": c.experiment}, &createResp)
	if err != nil {
		return "", err
	}
	return createResp.ExperimentId, nil
}

func (c *mlflowClient) createRun(ctx context.Context, expID string) (string, error) {
	var resp struct {
		Run struct {
			Info struct {
				RunId string `json:"run_id"`
			} `json:"info"`
		} `json:"run"`
	}
	payload := map[string]any{"experiment_id": expID, "start_time": time.Now().UnixMilli(), "tags": []map[string]string{{"key": "orchestrator", "value": "go"}}}
	if err := c.doJSON(ctx, http.MethodPost, "/api/2.0/mlflow/runs/create", payload, &resp); err != nil {
		return "", err
	}
	return resp.Run.Info.RunId, nil
}

func (c *mlflowClient) logParam(ctx context.Context, runID, key, value string) error {
	payload := map[string]any{"run_id": runID, "key": key, "value": value}
	return c.doJSON(ctx, http.MethodPost, "/api/2.0/mlflow/runs/log-parameter", payload, nil)
}

func (c *mlflowClient) logMetric(ctx context.Context, runID, key string, value float64, ts int64) error {
	payload := map[string]any{"run_id": runID, "key": key, "value": value, "timestamp": ts, "step": 0}
	return c.doJSON(ctx, http.MethodPost, "/api/2.0/mlflow/runs/log-metric", payload, nil)
}

func (c *mlflowClient) setRunTerminated(ctx context.Context, runID string) error {
	payload := map[string]any{"run_id": runID, "status": "FINISHED", "end_time": time.Now().UnixMilli()}
	return c.doJSON(ctx, http.MethodPost, "/api/2.0/mlflow/runs/update", payload, nil)
}

// uploadArtifact uploads a single file to a run's artifact path.
func (c *mlflowClient) uploadArtifact(ctx context.Context, runID, localPath, artifactPath string) error {
	f, err := os.Open(localPath)
	if err != nil {
		return err
	}
	defer f.Close()
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	part, err := mw.CreateFormFile("file", filepath.Base(localPath))
	if err != nil {
		return err
	}
	if _, err := io.Copy(part, f); err != nil {
		return err
	}
	_ = mw.WriteField("path", artifactPath)
	_ = mw.WriteField("run_id", runID)
	mw.Close()
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, c.base+"/api/2.0/mlflow/artifacts/log-artifact", &buf)
	req.Header.Set("Content-Type", mw.FormDataContentType())
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("mlflow upload: %s", string(b))
	}
	return nil
}

// ---- Math helpers ----
func meanVector(X [][]float64) []float64 {
	if len(X) == 0 {
		return nil
	}
	d := len(X[0])
	mu := make([]float64, d)
	for _, row := range X {
		for j := 0; j < d; j++ {
			mu[j] += row[j]
		}
	}
	n := float64(len(X))
	for j := 0; j < d; j++ {
		mu[j] /= n
	}
	return mu
}

func covarianceMatrix(X [][]float64, mu []float64) [][]float64 {
	n := float64(len(X))
	d := len(mu)
	cov := make([][]float64, d)
	for i := 0; i < d; i++ {
		cov[i] = make([]float64, d)
	}
	for _, row := range X {
		for i := 0; i < d; i++ {
			xi := row[i] - mu[i]
			for j := 0; j < d; j++ {
				xj := row[j] - mu[j]
				cov[i][j] += xi * xj
			}
		}
	}
	// unbiased estimator
	denom := n - 1
	if denom < 1 {
		denom = 1
	}
	for i := 0; i < d; i++ {
		for j := 0; j < d; j++ {
			cov[i][j] /= denom
		}
	}
	return cov
}

func copyMatrix(a [][]float64) [][]float64 {
	if a == nil {
		return nil
	}
	m := make([][]float64, len(a))
	for i := range a {
		m[i] = append([]float64(nil), a[i]...)
	}
	return m
}

func invertMatrix(a [][]float64) ([][]float64, bool) {
	n := len(a)
	// Augment with identity
	aug := make([][]float64, n)
	for i := 0; i < n; i++ {
		aug[i] = make([]float64, 2*n)
		copy(aug[i][:n], a[i])
		aug[i][n+i] = 1
	}
	// Gauss-Jordan elimination with partial pivoting
	for i := 0; i < n; i++ {
		// pivot
		maxRow := i
		maxVal := math.Abs(aug[i][i])
		for r := i + 1; r < n; r++ {
			if v := math.Abs(aug[r][i]); v > maxVal {
				maxVal = v
				maxRow = r
			}
		}
		if maxVal < 1e-12 {
			return nil, false
		}
		if maxRow != i {
			aug[i], aug[maxRow] = aug[maxRow], aug[i]
		}
		// normalize
		piv := aug[i][i]
		invPiv := 1.0 / piv
		for c := 0; c < 2*n; c++ {
			aug[i][c] *= invPiv
		}
		// eliminate others
		for r := 0; r < n; r++ {
			if r == i {
				continue
			}
			factor := aug[r][i]
			if factor == 0 {
				continue
			}
			for c := 0; c < 2*n; c++ {
				aug[r][c] -= factor * aug[i][c]
			}
		}
	}
	inv := make([][]float64, n)
	for i := 0; i < n; i++ {
		inv[i] = append([]float64(nil), aug[i][n:]...)
	}
	return inv, true
}

func mahalanobis2(x, mu []float64, inv [][]float64) float64 {
	d := len(mu)
	// v = x - mu
	v := make([]float64, d)
	for i := 0; i < d; i++ {
		v[i] = x[i] - mu[i]
	}
	// t = inv * v
	t := make([]float64, d)
	for i := 0; i < d; i++ {
		s := 0.0
		row := inv[i]
		for j := 0; j < d; j++ {
			s += row[j] * v[j]
		}
		t[i] = s
	}
	// d2 = v^T * t
	s := 0.0
	for i := 0; i < d; i++ {
		s += v[i] * t[i]
	}
	return s
}

func percentile(vals []float64, p float64) float64 {
	if len(vals) == 0 {
		return math.NaN()
	}
	cp := append([]float64(nil), vals...)
	sort.Float64s(cp)
	if p <= 0 {
		return cp[0]
	}
	if p >= 1 {
		return cp[len(cp)-1]
	}
	idx := p * float64(len(cp)-1)
	i := int(math.Floor(idx))
	frac := idx - float64(i)
	if i+1 >= len(cp) {
		return cp[i]
	}
	return cp[i]*(1-frac) + cp[i+1]*frac
}

// handleFederatedAggregate performs secure aggregation of client model updates.
// Accepts JSON: {"updates":[{"weights":[..],"count":123}, ...], "epsilon":1.0}
// Applies differential privacy noise (Laplace) if epsilon > 0.
func (m *MLOrchestrator) handleFederatedAggregate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req struct {
		Updates []struct {
			Weights []float64 `json:"weights"`
			Count   int       `json:"count"`
		} `json:"updates"`
		Epsilon float64 `json:"epsilon"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad json", http.StatusBadRequest)
		return
	}
	if len(req.Updates) == 0 {
		http.Error(w, "no updates", http.StatusBadRequest)
		return
	}
	// Determine dimension
	dim := len(req.Updates[0].Weights)
	if dim == 0 {
		http.Error(w, "empty weights", http.StatusBadRequest)
		return
	}
	agg := make([]float64, dim)
	total := 0
	for _, u := range req.Updates {
		if len(u.Weights) != dim || u.Count <= 0 {
			http.Error(w, "inconsistent update", http.StatusBadRequest)
			return
		}
		for i := 0; i < dim; i++ {
			agg[i] += u.Weights[i] * float64(u.Count)
		}
		total += u.Count
	}
	if total == 0 {
		http.Error(w, "zero total count", http.StatusBadRequest)
		return
	}
	for i := 0; i < dim; i++ {
		agg[i] /= float64(total)
	}
	// Differential privacy noise (Laplace with scale = 1/epsilon) simplistic
	if req.Epsilon > 0 {
		scale := 1.0 / req.Epsilon
		for i := 0; i < dim; i++ {
			// Laplace sampling via inverse CDF
			u := randFloat64()*2 - 1 // (-1,1)
			if u == 0 {
				u = 1e-9
			}
			sign := 1.0
			if u < 0 {
				sign = -1.0
			}
			agg[i] += -scale * sign * math.Log(1-2*math.Abs(u))
		}
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{"aggregated": agg, "total_count": total, "epsilon": req.Epsilon})
}

// handleAdversarialGenerate returns adversarially perturbed feature vectors using FGSM.
// Input: {"features":[...],"epsilon":0.1,"gradient":[...]}
// For PoC security: gradient must be supplied (no raw model exposure).
func (m *MLOrchestrator) handleAdversarialGenerate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req struct {
		Features []float64 `json:"features"`
		Gradient []float64 `json:"gradient"`
		Epsilon  float64   `json:"epsilon"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad json", http.StatusBadRequest)
		return
	}
	if len(req.Features) == 0 || len(req.Gradient) != len(req.Features) {
		http.Error(w, "dimension mismatch", http.StatusBadRequest)
		return
	}
	if req.Epsilon <= 0 || req.Epsilon > 5 {
		req.Epsilon = 0.1
	}
	adv := make([]float64, len(req.Features))
	for i, f := range req.Features {
		g := req.Gradient[i]
		sign := 1.0
		if g < 0 { sign = -1.0 }
		adv[i] = f + req.Epsilon*sign
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{"adversarial": adv, "epsilon": req.Epsilon})
}

func randFloat64() float64 { return float64(time.Now().UnixNano()%1_000_000)/1_000_000 }
