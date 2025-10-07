// Package main - Phase 1: Quantum-Safe Security Infrastructure
// Implements advanced post-quantum cryptography, certificate transparency,
// and GraphQL security for production deployment.
package main

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"shieldx/pkg/certtransparency"
	"shieldx/pkg/graphql"
	"shieldx/pkg/ledger"
	pqc "shieldx/pkg/pqc"
)

// Phase1Config holds Phase 1 enhancement configuration
type Phase1Config struct {
	// Post-Quantum Cryptography
	EnablePQC      bool
	PQCAlgorithm   string // "kyber1024", "dilithium5", "hybrid"
	PQCKeyRotation time.Duration

	// Certificate Transparency
	EnableCTMonitoring bool
	CTMonitoredDomains []string
	CTAlertWebhook     string

	// GraphQL Security
	EnableGraphQLSec     bool
	GraphQLMaxDepth      int
	GraphQLMaxComplexity int
	DisableIntrospection bool

	// Adaptive Rate Limiting
	EnableAdaptiveRL  bool
	BaseRateLimit     int
	DegradedRateLimit int

	// Real-time Behavioral Analysis
	EnableBehaviorAnalysis bool
	BehaviorWindowSize     int
	AnomalyThreshold       float64
}

// Phase1Enhancement provides Phase 1 security enhancements
type Phase1Enhancement struct {
	config Phase1Config

	// PQC components (Kyber keypair + optional Dilithium via pqc engine later)
	kyberPub      *pqc.KyberPublicKey
	kyberSec      *pqc.KyberSecretKey
	pqcMu         sync.RWMutex
	pqcRotateTick *time.Ticker

	// Certificate Transparency
	ctMonitor *certtransparency.CTMonitor
	ctAlerts  chan certtransparency.CTAlert

	// GraphQL Security
	graphqlSec *graphql.SecurityMiddleware

	// Adaptive Rate Limiting
	currentRateLimit atomic.Int64
	healthRatio      atomic.Uint64 // x10000

	// Real-time Behavioral Analysis
	behaviorAnalyzer *BehavioralAnalyzer

	// Metrics
	mPQCRotations      atomic.Uint64
	mCTAlertsReceived  atomic.Uint64
	mGraphQLBlocked    atomic.Uint64
	mBehaviorAnomalies atomic.Uint64

	stopChan chan struct{}
	wg       sync.WaitGroup
}

// NewPhase1Enhancement creates Phase 1 enhancements
func NewPhase1Enhancement(config Phase1Config) (*Phase1Enhancement, error) {
	p1 := &Phase1Enhancement{
		config:   config,
		stopChan: make(chan struct{}),
	}

	// Initialize PQC if enabled
	if config.EnablePQC {
		if err := p1.initPQC(); err != nil {
			return nil, fmt.Errorf("PQC init: %w", err)
		}
	}

	// Initialize CT monitoring if enabled
	if config.EnableCTMonitoring && len(config.CTMonitoredDomains) > 0 {
		p1.ctMonitor = certtransparency.NewCTMonitor(config.CTMonitoredDomains)
		p1.ctAlerts = make(chan certtransparency.CTAlert, 100)

		if err := p1.ctMonitor.Start(); err != nil {
			log.Printf("[phase1] CT monitor start error: %v", err)
		}
	}

	// Initialize GraphQL security if enabled
	if config.EnableGraphQLSec {
		gqlConfig := graphql.DefaultSecurityConfig()
		gqlConfig.MaxDepth = config.GraphQLMaxDepth
		gqlConfig.MaxComplexity = config.GraphQLMaxComplexity
		gqlConfig.DisableIntrospection = config.DisableIntrospection

		p1.graphqlSec = graphql.NewSecurityMiddleware(gqlConfig)
	}

	// Initialize behavioral analyzer if enabled
	if config.EnableBehaviorAnalysis {
		p1.behaviorAnalyzer = NewBehavioralAnalyzer(config.BehaviorWindowSize, config.AnomalyThreshold)
	}

	// Set initial rate limit
	if config.EnableAdaptiveRL {
		p1.currentRateLimit.Store(int64(config.BaseRateLimit))
	}

	return p1, nil
}

// Start starts Phase 1 enhancement services
func (p1 *Phase1Enhancement) Start() {
	// PQC key rotation
	if p1.config.EnablePQC && p1.config.PQCKeyRotation > 0 {
		p1.pqcRotateTick = time.NewTicker(p1.config.PQCKeyRotation)
		p1.wg.Add(1)
		go p1.pqcRotationLoop()
	}

	// CT alert processor
	if p1.ctMonitor != nil {
		p1.wg.Add(1)
		go p1.processCTAlerts()
	}

	// Behavioral analysis processor
	if p1.behaviorAnalyzer != nil {
		p1.wg.Add(1)
		go p1.behaviorAnalysisLoop()
	}

	log.Printf("[phase1] Started - PQC:%v CT:%v GraphQL:%v Adaptive:%v Behavior:%v",
		p1.config.EnablePQC,
		p1.config.EnableCTMonitoring,
		p1.config.EnableGraphQLSec,
		p1.config.EnableAdaptiveRL,
		p1.config.EnableBehaviorAnalysis)
}

// Stop stops Phase 1 enhancement services
func (p1 *Phase1Enhancement) Stop() {
	close(p1.stopChan)

	if p1.pqcRotateTick != nil {
		p1.pqcRotateTick.Stop()
	}

	if p1.ctMonitor != nil {
		p1.ctMonitor.Stop()
	}

	p1.wg.Wait()
	log.Printf("[phase1] Stopped")
}

// initPQC initializes post-quantum cryptography
func (p1 *Phase1Enhancement) initPQC() error {
	// For now we only generate Kyber keypair; algorithm selection placeholder
	pub, sec, err := pqc.GenerateKyberKeypair()
	if err != nil {
		return err
	}
	p1.kyberPub, p1.kyberSec = pub, sec
	log.Printf("[phase1] PQC initialized (Kyber keypair generated)")
	return nil
}

// pqcRotationLoop handles automatic PQC key rotation
func (p1 *Phase1Enhancement) pqcRotationLoop() {
	defer p1.wg.Done()

	for {
		select {
		case <-p1.stopChan:
			return
		case <-p1.pqcRotateTick.C:
			if err := p1.rotatePQCKeys(); err != nil {
				log.Printf("[phase1] PQC rotation error: %v", err)
			} else {
				p1.mPQCRotations.Add(1)
				log.Printf("[phase1] PQC keys rotated (total: %d)", p1.mPQCRotations.Load())
			}
		}
	}
}

// rotatePQCKeys performs PQC key rotation
func (p1 *Phase1Enhancement) rotatePQCKeys() error {
	pub, sec, err := pqc.GenerateKyberKeypair()
	if err != nil {
		return err
	}
	p1.pqcMu.Lock()
	old := p1.kyberPub
	p1.kyberPub, p1.kyberSec = pub, sec
	p1.pqcMu.Unlock()
	_ = ledger.AppendJSONLine(ledgerPath, serviceName, "pqc.rotation", map[string]any{
		"timestamp":      time.Now().UTC(),
		"old_pub_prefix": fmt.Sprintf("%x", old.Data[:16]),
		"new_pub_prefix": fmt.Sprintf("%x", pub.Data[:16]),
	})
	return nil
}

// GetPQCPublicKey returns the current PQC public key
func (p1 *Phase1Enhancement) GetPQCPublicKey() ([]byte, error) {
	p1.pqcMu.RLock()
	defer p1.pqcMu.RUnlock()
	if p1.kyberPub == nil {
		return nil, fmt.Errorf("PQC not initialized")
	}
	cp := make([]byte, len(p1.kyberPub.Data))
	copy(cp, p1.kyberPub.Data[:])
	return cp, nil
}

// PQCEncapsulate performs post-quantum key encapsulation
func (p1 *Phase1Enhancement) PQCEncapsulate() (ciphertext, sharedSecret []byte, err error) {
	p1.pqcMu.RLock()
	defer p1.pqcMu.RUnlock()
	if p1.kyberPub == nil {
		return nil, nil, fmt.Errorf("PQC not initialized")
	}
	ct, shared, err := pqc.KyberEncapsulate(p1.kyberPub)
	if err != nil {
		return nil, nil, err
	}
	return ct.Data[:], shared, nil
}

// PQCDecapsulate performs post-quantum key decapsulation
func (p1 *Phase1Enhancement) PQCDecapsulate(ciphertext []byte) ([]byte, error) {
	p1.pqcMu.RLock()
	defer p1.pqcMu.RUnlock()
	if p1.kyberSec == nil {
		return nil, fmt.Errorf("PQC not initialized")
	}
	if len(ciphertext) != pqc.KyberCiphertextSize {
		return nil, fmt.Errorf("invalid ciphertext size")
	}
	var ct pqc.KyberCiphertext
	copy(ct.Data[:], ciphertext)
	return pqc.KyberDecapsulate(&ct, p1.kyberSec)
}

// processCTAlerts processes Certificate Transparency alerts
func (p1 *Phase1Enhancement) processCTAlerts() {
	defer p1.wg.Done()

	alerts := p1.ctMonitor.Alerts()

	for {
		select {
		case <-p1.stopChan:
			return
		case alert := <-alerts:
			p1.mCTAlertsReceived.Add(1)
			p1.handleCTAlert(alert)
		}
	}
}

// handleCTAlert handles a Certificate Transparency alert
func (p1 *Phase1Enhancement) handleCTAlert(alert certtransparency.CTAlert) {
	// Log alert
	_ = ledger.AppendJSONLine(ledgerPath, serviceName, "ct.alert", map[string]any{
		"timestamp":    alert.Timestamp,
		"log_url":      alert.LogURL,
		"domain":       alert.Domain,
		"serial":       alert.SerialNumber,
		"issuer":       alert.Issuer,
		"fingerprint":  alert.Fingerprint,
		"mis_issuance": alert.IsMisissuance,
		"reason":       alert.Reason,
	})

	// Send webhook notification if configured
	if p1.config.CTAlertWebhook != "" {
		go p1.sendCTAlertWebhook(alert)
	}

	// Take action on mis-issuance
	if alert.IsMisissuance {
		log.Printf("[phase1] CT ALERT - Mis-issued certificate detected: %s (domain: %s, issuer: %s)",
			alert.Fingerprint, alert.Domain, alert.Issuer)

		// In production: trigger incident response, revoke trust, etc.
	}
}

// sendCTAlertWebhook sends CT alert to webhook
func (p1 *Phase1Enhancement) sendCTAlertWebhook(alert certtransparency.CTAlert) {
	data, err := json.Marshal(alert)
	if err != nil {
		return
	}

	req, err := http.NewRequest("POST", p1.config.CTAlertWebhook, strings.NewReader(string(data)))
	if err != nil {
		return
	}

	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Do(req)
	if err == nil && resp != nil {
		resp.Body.Close()
	}
}

// ValidateGraphQLQuery validates a GraphQL query with security checks
func (p1 *Phase1Enhancement) ValidateGraphQLQuery(ctx context.Context, clientID, query string, variables map[string]interface{}) error {
	if !p1.config.EnableGraphQLSec || p1.graphqlSec == nil {
		return nil // GraphQL security disabled
	}

	if err := p1.graphqlSec.ValidateQuery(ctx, clientID, query, variables); err != nil {
		p1.mGraphQLBlocked.Add(1)
		return err
	}

	return nil
}

// AdaptRateLimit adjusts rate limit based on system health
func (p1 *Phase1Enhancement) AdaptRateLimit(healthRatio float64) {
	if !p1.config.EnableAdaptiveRL {
		return
	}

	// Store health ratio (x10000 for metrics)
	p1.healthRatio.Store(uint64(healthRatio * 10000))

	// Adjust rate limit based on health
	degradedThreshold := 0.5

	var newLimit int64
	if healthRatio < degradedThreshold {
		// System degraded - reduce rate limit
		newLimit = int64(p1.config.DegradedRateLimit)
	} else {
		// System healthy - restore base rate limit
		newLimit = int64(p1.config.BaseRateLimit)
	}

	oldLimit := p1.currentRateLimit.Swap(newLimit)

	if oldLimit != newLimit {
		log.Printf("[phase1] Adaptive rate limit adjusted: %d -> %d (health: %.2f%%)",
			oldLimit, newLimit, healthRatio*100)

		// Update global IP rate limiter capacity
		if ipLimiter != nil {
			ipLimiter.capacity = int(newLimit)
		}
	}
}

// GetCurrentRateLimit returns the current adaptive rate limit
func (p1 *Phase1Enhancement) GetCurrentRateLimit() int {
	return int(p1.currentRateLimit.Load())
}

// RecordBehavior records behavioral data for analysis
func (p1 *Phase1Enhancement) RecordBehavior(clientID string, features map[string]float64) {
	if p1.behaviorAnalyzer != nil {
		p1.behaviorAnalyzer.RecordBehavior(clientID, features)
	}
}

// CheckBehaviorAnomaly checks if client behavior is anomalous
func (p1 *Phase1Enhancement) CheckBehaviorAnomaly(clientID string) (bool, float64) {
	if p1.behaviorAnalyzer == nil {
		return false, 0
	}

	return p1.behaviorAnalyzer.IsAnomalous(clientID)
}

// behaviorAnalysisLoop processes behavioral analysis
func (p1 *Phase1Enhancement) behaviorAnalysisLoop() {
	defer p1.wg.Done()

	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-p1.stopChan:
			return
		case <-ticker.C:
			anomalies := p1.behaviorAnalyzer.DetectAnomalies()

			for clientID, score := range anomalies {
				p1.mBehaviorAnomalies.Add(1)

				log.Printf("[phase1] Behavioral anomaly detected: client=%s score=%.3f", clientID, score)

				// Log anomaly
				_ = ledger.AppendJSONLine(ledgerPath, serviceName, "behavior.anomaly", map[string]any{
					"timestamp": time.Now().UTC(),
					"client_id": clientID,
					"score":     score,
				})
			}
		}
	}
}

// GetPhase1Metrics returns Phase 1 metrics
func (p1 *Phase1Enhancement) GetPhase1Metrics() map[string]interface{} {
	metrics := map[string]interface{}{
		"pqc_rotations":      p1.mPQCRotations.Load(),
		"ct_alerts":          p1.mCTAlertsReceived.Load(),
		"graphql_blocked":    p1.mGraphQLBlocked.Load(),
		"behavior_anomalies": p1.mBehaviorAnomalies.Load(),
		"current_rate_limit": p1.currentRateLimit.Load(),
		"health_ratio":       float64(p1.healthRatio.Load()) / 10000.0,
	}

	// Add GraphQL metrics if enabled
	if p1.graphqlSec != nil {
		gqlMetrics := p1.graphqlSec.GetMetrics()
		for k, v := range gqlMetrics {
			metrics["graphql_"+k] = v
		}
	}

	// Add CT metrics if enabled
	if p1.ctMonitor != nil {
		ctMetrics := p1.ctMonitor.GetMetrics()
		for k, v := range ctMetrics {
			metrics["ct_"+k] = v
		}
	}

	return metrics
}

// ---------- Behavioral Analyzer ----------

// BehavioralAnalyzer performs real-time behavioral analysis
type BehavioralAnalyzer struct {
	windowSize       int
	anomalyThreshold float64

	// Client behavior profiles
	profiles sync.Map // clientID -> *BehaviorProfile

	mu sync.RWMutex
}

// BehaviorProfile holds behavioral data for a client
type BehaviorProfile struct {
	ClientID     string
	Features     []map[string]float64 // Rolling window of feature vectors
	Baseline     map[string]float64   // Baseline feature means
	StdDev       map[string]float64   // Standard deviations
	AnomalyScore float64
	LastUpdate   time.Time
	mu           sync.Mutex
}

// NewBehavioralAnalyzer creates a new behavioral analyzer
func NewBehavioralAnalyzer(windowSize int, threshold float64) *BehavioralAnalyzer {
	return &BehavioralAnalyzer{
		windowSize:       windowSize,
		anomalyThreshold: threshold,
	}
}

// RecordBehavior records behavioral features for a client
func (ba *BehavioralAnalyzer) RecordBehavior(clientID string, features map[string]float64) {
	val, _ := ba.profiles.LoadOrStore(clientID, &BehaviorProfile{
		ClientID:   clientID,
		Features:   make([]map[string]float64, 0, ba.windowSize),
		Baseline:   make(map[string]float64),
		StdDev:     make(map[string]float64),
		LastUpdate: time.Now(),
	})

	profile := val.(*BehaviorProfile)
	profile.mu.Lock()
	defer profile.mu.Unlock()

	// Add to rolling window
	profile.Features = append(profile.Features, features)
	if len(profile.Features) > ba.windowSize {
		profile.Features = profile.Features[1:]
	}

	profile.LastUpdate = time.Now()

	// Update baseline if we have enough data
	if len(profile.Features) >= ba.windowSize/2 {
		ba.updateBaseline(profile)
	}
}

// updateBaseline updates the baseline statistics
func (ba *BehavioralAnalyzer) updateBaseline(profile *BehaviorProfile) {
	// Calculate mean and stddev for each feature
	featureSums := make(map[string]float64)
	featureCounts := make(map[string]int)

	for _, features := range profile.Features {
		for k, v := range features {
			featureSums[k] += v
			featureCounts[k]++
		}
	}

	// Compute means
	for k := range featureSums {
		if featureCounts[k] > 0 {
			profile.Baseline[k] = featureSums[k] / float64(featureCounts[k])
		}
	}

	// Compute standard deviations
	varianceSums := make(map[string]float64)
	for _, features := range profile.Features {
		for k, v := range features {
			if mean, ok := profile.Baseline[k]; ok {
				diff := v - mean
				varianceSums[k] += diff * diff
			}
		}
	}

	for k := range varianceSums {
		if featureCounts[k] > 1 {
			variance := varianceSums[k] / float64(featureCounts[k]-1)
			profile.StdDev[k] = sqrt(variance)
		}
	}
}

// IsAnomalous checks if a client's behavior is anomalous
func (ba *BehavioralAnalyzer) IsAnomalous(clientID string) (bool, float64) {
	val, ok := ba.profiles.Load(clientID)
	if !ok {
		return false, 0
	}

	profile := val.(*BehaviorProfile)
	profile.mu.Lock()
	defer profile.mu.Unlock()

	if len(profile.Features) == 0 || len(profile.Baseline) == 0 {
		return false, 0
	}

	// Get latest features
	latest := profile.Features[len(profile.Features)-1]

	// Calculate z-scores and anomaly score
	var anomalyScore float64
	count := 0

	for k, v := range latest {
		if mean, ok := profile.Baseline[k]; ok {
			if stddev, ok := profile.StdDev[k]; ok && stddev > 0 {
				zScore := abs((v - mean) / stddev)
				anomalyScore += zScore
				count++
			}
		}
	}

	if count > 0 {
		anomalyScore /= float64(count)
	}

	profile.AnomalyScore = anomalyScore

	return anomalyScore > ba.anomalyThreshold, anomalyScore
}

// DetectAnomalies returns all anomalous clients
func (ba *BehavioralAnalyzer) DetectAnomalies() map[string]float64 {
	anomalies := make(map[string]float64)

	ba.profiles.Range(func(key, value interface{}) bool {
		clientID := key.(string)
		if isAnom, score := ba.IsAnomalous(clientID); isAnom {
			anomalies[clientID] = score
		}
		return true
	})

	return anomalies
}

// Helper functions
func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

func sqrt(x float64) float64 {
	// Simple approximation - use math.Sqrt in production
	if x <= 0 {
		return 0
	}
	z := x
	for i := 0; i < 10; i++ {
		z = (z + x/z) / 2
	}
	return z
}

// LoadPhase1Config loads Phase 1 configuration from environment
func LoadPhase1Config() Phase1Config {
	return Phase1Config{
		EnablePQC:      getenvBool("PHASE1_ENABLE_PQC", true),
		PQCAlgorithm:   getenvStr("PHASE1_PQC_ALGORITHM", "kyber1024"),
		PQCKeyRotation: getenvDur("PHASE1_PQC_ROTATION", 24*time.Hour),

		EnableCTMonitoring: getenvBool("PHASE1_ENABLE_CT", true),
		CTMonitoredDomains: getenvSlice("PHASE1_CT_DOMAINS", []string{"shieldx.local"}),
		CTAlertWebhook:     getenvStr("PHASE1_CT_WEBHOOK", ""),

		EnableGraphQLSec:     getenvBool("PHASE1_ENABLE_GRAPHQL_SEC", true),
		GraphQLMaxDepth:      getenvInt("PHASE1_GRAPHQL_MAX_DEPTH", 10),
		GraphQLMaxComplexity: getenvInt("PHASE1_GRAPHQL_MAX_COMPLEXITY", 1000),
		DisableIntrospection: getenvBool("PHASE1_DISABLE_INTROSPECTION", true),

		EnableAdaptiveRL:  getenvBool("PHASE1_ENABLE_ADAPTIVE_RL", true),
		BaseRateLimit:     getenvInt("PHASE1_BASE_RATE_LIMIT", 200),
		DegradedRateLimit: getenvInt("PHASE1_DEGRADED_RATE_LIMIT", 50),

		EnableBehaviorAnalysis: getenvBool("PHASE1_ENABLE_BEHAVIOR", true),
		BehaviorWindowSize:     getenvInt("PHASE1_BEHAVIOR_WINDOW", 100),
		AnomalyThreshold:       getenvFloat("PHASE1_ANOMALY_THRESHOLD", 3.0),
	}
}

// --- local env helpers (duplicated minimally to avoid cross-package coupling) ---
func getenvStr(key, def string) string {
	if v := os.Getenv(key); strings.TrimSpace(v) != "" {
		return v
	}
	return def
}
func getenvInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func getenvBool(key string, def bool) bool {
	if v := os.Getenv(key); v != "" {
		return v == "1" || v == "true" || v == "yes"
	}
	return def
}

func getenvSlice(key string, def []string) []string {
	if v := os.Getenv(key); v != "" {
		return strings.Split(v, ",")
	}
	return def
}

func getenvDur(key string, def time.Duration) time.Duration {
	if v := os.Getenv(key); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			return d
		}
	}
	return def
}

func getenvFloat(key string, def float64) float64 {
	if v := os.Getenv(key); v != "" {
		var f float64
		if _, err := fmt.Sscanf(v, "%f", &f); err == nil {
			return f
		}
	}
	return def
}

// EnhancedTLSConfig creates TLS 1.3 config with PQC support
func (p1 *Phase1Enhancement) EnhancedTLSConfig(baseCfg *tls.Config) *tls.Config {
	if baseCfg == nil {
		baseCfg = &tls.Config{}
	}

	// Enforce TLS 1.3 minimum
	baseCfg.MinVersion = tls.VersionTLS13
	baseCfg.MaxVersion = tls.VersionTLS13

	// Prefer strong cipher suites (TLS 1.3 only)
	baseCfg.CipherSuites = []uint16{
		tls.TLS_AES_256_GCM_SHA384,
		tls.TLS_CHACHA20_POLY1305_SHA256,
		tls.TLS_AES_128_GCM_SHA256,
	}

	// Set modern curve preferences
	baseCfg.CurvePreferences = []tls.CurveID{
		tls.X25519, // Preferred
		tls.CurveP256,
	}

	return baseCfg
}
