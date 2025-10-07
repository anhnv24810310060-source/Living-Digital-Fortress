package main

import (
	"context"
	crand "crypto/rand"
	"crypto/sha256"
	"fmt"
	"log"
	"math"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
)

// AdaptiveCamouflageEngine implements intelligent deception with:
// 1. Multi-Armed Bandit algorithm for optimal decoy selection
// 2. Threat-level adaptive responses
// 3. Decoy effectiveness learning
// 4. Resource-efficient decoy management
// 5. A/B testing for deception strategies
type AdaptiveCamouflageEngine struct {
	decoys         map[string]*SmartDecoy
	mu             sync.RWMutex
	bandit         *MultiArmedBandit
	threatAnalyzer *ThreatLevelAnalyzer
	metricsStore   *DecoyMetricsStore
}

type SmartDecoy struct {
	ID          string            `json:"id"`
	Type        string            `json:"type"` // web_server, ssh, database, api, etc.
	Profile     DecoyProfile      `json:"profile"`
	Fingerprint ServerFingerprint `json:"fingerprint"`

	// Learning metrics
	InteractionCount  int64   `json:"interaction_count"`
	SuccessRate       float64 `json:"success_rate"`
	AvgEngagementTime float64 `json:"avg_engagement_time"` // seconds
	DetectionRate     float64 `json:"detection_rate"`

	// Bandit metrics
	Reward float64 `json:"reward"`
	Pulls  int64   `json:"pulls"`

	// Configuration
	Enabled   bool      `json:"enabled"`
	LastUsed  time.Time `json:"last_used"`
	CreatedAt time.Time `json:"created_at"`
}

type DecoyProfile struct {
	// Server characteristics
	ServerType   string `json:"server_type"`
	Version      string `json:"version"`
	Architecture string `json:"architecture"`

	// Response templates
	Templates map[string]string `json:"templates"`
	Headers   map[string]string `json:"headers"`

	// Behavior parameters
	ResponseDelay int     `json:"response_delay_ms"`
	ErrorRate     float64 `json:"error_rate"`
	Complexity    int     `json:"complexity"` // 1-10 scale
}

type ServerFingerprint struct {
	// HTTP fingerprints
	ServerHeader  string `json:"server_header"`
	PoweredBy     string `json:"powered_by"`
	XFrameOptions string `json:"x_frame_options"`
	ContentType   string `json:"content_type"`

	// TLS fingerprints (JA3)
	TLSVersion   string   `json:"tls_version"`
	CipherSuites []string `json:"cipher_suites"`

	// Application fingerprints
	Cookies      []string `json:"cookies"`
	HiddenFields []string `json:"hidden_fields"`

	// Behavioral fingerprints
	ResponseTiming int      `json:"response_timing_ms"`
	ErrorPages     []string `json:"error_pages"`
}

// Multi-Armed Bandit for optimal decoy selection
type MultiArmedBandit struct {
	epsilon    float64 // Exploration rate
	mu         sync.RWMutex
	arms       map[string]*BanditArm
	totalPulls int64
}

type BanditArm struct {
	DecoyID string
	Reward  float64
	Pulls   int64
	UCB     float64 // Upper Confidence Bound
}

type ThreatLevelAnalyzer struct {
	mu               sync.RWMutex
	threatLevels     map[string]ThreatLevel // IP -> ThreatLevel
	behaviorPatterns map[string]*BehaviorProfile
}

type ThreatLevel struct {
	Level      int       `json:"level"` // 0=benign, 1-10=threat
	Confidence float64   `json:"confidence"`
	Indicators []string  `json:"indicators"`
	LastUpdate time.Time `json:"last_update"`
}

type BehaviorProfile struct {
	RequestCount     int64
	FailedAuthCount  int64
	ScanBehavior     bool
	AutomatedTraffic bool
	ThreatScore      float64
	FirstSeen        time.Time
	LastSeen         time.Time
}

type DecoyMetricsStore struct {
	mu      sync.RWMutex
	metrics map[string]*DecoyMetrics
}

type DecoyMetrics struct {
	DecoyID          string
	Interactions     int64
	UniqueVisitors   int64
	AvgEngagementSec float64
	ConversionRate   float64 // % that triggered honeypot
	DetectionEvents  int64
	LastInteraction  time.Time
}

type DecoySelectionRequest struct {
	SourceIP    string                 `json:"source_ip"`
	UserAgent   string                 `json:"user_agent"`
	RequestPath string                 `json:"request_path"`
	Headers     map[string]string      `json:"headers"`
	ThreatScore int                    `json:"threat_score"`
	Context     map[string]interface{} `json:"context"`
}

type DecoyResponse struct {
	DecoyID      string            `json:"decoy_id"`
	Strategy     string            `json:"strategy"` // "engage", "delay", "redirect", "block"
	Profile      DecoyProfile      `json:"profile"`
	Fingerprint  ServerFingerprint `json:"fingerprint"`
	ResponseTime int               `json:"response_time_ms"`
	Confidence   float64           `json:"confidence"`
}

// NewAdaptiveCamouflageEngine creates an intelligent deception engine
func NewAdaptiveCamouflageEngine() *AdaptiveCamouflageEngine {
	ace := &AdaptiveCamouflageEngine{
		decoys: make(map[string]*SmartDecoy),
		bandit: &MultiArmedBandit{
			epsilon: 0.1, // 10% exploration
			arms:    make(map[string]*BanditArm),
		},
		threatAnalyzer: &ThreatLevelAnalyzer{
			threatLevels:     make(map[string]ThreatLevel),
			behaviorPatterns: make(map[string]*BehaviorProfile),
		},
		metricsStore: &DecoyMetricsStore{
			metrics: make(map[string]*DecoyMetrics),
		},
	}

	// Initialize with diverse decoy pool
	ace.initializeDecoyPool()

	return ace
}

// initializeDecoyPool creates a diverse set of decoys
func (ace *AdaptiveCamouflageEngine) initializeDecoyPool() {
	decoyConfigs := []struct {
		decoyType   string
		profile     DecoyProfile
		fingerprint ServerFingerprint
	}{
		{
			decoyType: "nginx_web",
			profile: DecoyProfile{
				ServerType: "nginx",
				Version:    "1.21.6",
				Templates: map[string]string{
					"index": "nginx_default_page",
					"404":   "nginx_404",
					"500":   "nginx_500",
				},
				Headers: map[string]string{
					"Server":          "nginx/1.21.6",
					"X-Frame-Options": "SAMEORIGIN",
				},
				ResponseDelay: 50,
				ErrorRate:     0.01,
				Complexity:    3,
			},
			fingerprint: ServerFingerprint{
				ServerHeader:   "nginx/1.21.6",
				TLSVersion:     "TLSv1.3",
				CipherSuites:   []string{"TLS_AES_256_GCM_SHA384", "TLS_CHACHA20_POLY1305_SHA256"},
				ResponseTiming: 50,
			},
		},
		{
			decoyType: "apache_web",
			profile: DecoyProfile{
				ServerType: "apache",
				Version:    "2.4.54",
				Templates: map[string]string{
					"index": "apache_default",
					"404":   "apache_404",
				},
				Headers: map[string]string{
					"Server":       "Apache/2.4.54 (Ubuntu)",
					"X-Powered-By": "PHP/7.4.30",
				},
				ResponseDelay: 75,
				ErrorRate:     0.02,
				Complexity:    5,
			},
			fingerprint: ServerFingerprint{
				ServerHeader: "Apache/2.4.54 (Ubuntu)",
				PoweredBy:    "PHP/7.4.30",
				TLSVersion:   "TLSv1.2",
			},
		},
		{
			decoyType: "iis_web",
			profile: DecoyProfile{
				ServerType: "iis",
				Version:    "10.0",
				Templates: map[string]string{
					"index": "iis_default",
					"404":   "iis_404",
				},
				Headers: map[string]string{
					"Server":       "Microsoft-IIS/10.0",
					"X-Powered-By": "ASP.NET",
				},
				ResponseDelay: 100,
				ErrorRate:     0.015,
				Complexity:    7,
			},
			fingerprint: ServerFingerprint{
				ServerHeader: "Microsoft-IIS/10.0",
				PoweredBy:    "ASP.NET",
			},
		},
		{
			decoyType: "express_api",
			profile: DecoyProfile{
				ServerType: "express",
				Version:    "4.18.2",
				Templates: map[string]string{
					"api_root": "express_api_root",
					"health":   "express_health",
				},
				Headers: map[string]string{
					"X-Powered-By": "Express",
					"Content-Type": "application/json",
				},
				ResponseDelay: 30,
				ErrorRate:     0.005,
				Complexity:    6,
			},
			fingerprint: ServerFingerprint{
				PoweredBy:      "Express",
				ContentType:    "application/json",
				ResponseTiming: 30,
			},
		},
		{
			decoyType: "spring_api",
			profile: DecoyProfile{
				ServerType: "spring",
				Version:    "2.7.5",
				Templates: map[string]string{
					"api_root": "spring_api_root",
					"error":    "spring_whitelabel_error",
				},
				Headers: map[string]string{
					"Server":       "Apache-Coyote/1.1",
					"Content-Type": "application/json",
				},
				ResponseDelay: 80,
				ErrorRate:     0.01,
				Complexity:    8,
			},
			fingerprint: ServerFingerprint{
				ServerHeader: "Apache-Coyote/1.1",
				ContentType:  "application/json",
			},
		},
	}

	for _, cfg := range decoyConfigs {
		decoyID := uuid.New().String()
		decoy := &SmartDecoy{
			ID:          decoyID,
			Type:        cfg.decoyType,
			Profile:     cfg.profile,
			Fingerprint: cfg.fingerprint,
			Enabled:     true,
			CreatedAt:   time.Now(),
		}

		ace.mu.Lock()
		ace.decoys[decoyID] = decoy
		ace.mu.Unlock()

		// Initialize bandit arm
		ace.bandit.mu.Lock()
		ace.bandit.arms[decoyID] = &BanditArm{
			DecoyID: decoyID,
			Reward:  0.5, // Neutral initial reward
			Pulls:   0,
		}
		ace.bandit.mu.Unlock()

		// Initialize metrics
		ace.metricsStore.mu.Lock()
		ace.metricsStore.metrics[decoyID] = &DecoyMetrics{
			DecoyID: decoyID,
		}
		ace.metricsStore.mu.Unlock()
	}

	log.Printf("Initialized %d decoys in camouflage pool", len(decoyConfigs))
}

// SelectOptimalDecoy uses Multi-Armed Bandit to choose best decoy
func (ace *AdaptiveCamouflageEngine) SelectOptimalDecoy(ctx context.Context, req DecoySelectionRequest) (*DecoyResponse, error) {
	// Analyze threat level
	threatLevel := ace.analyzeThreat(req)

	// Update behavior profile
	ace.updateBehaviorProfile(req.SourceIP, req)

	// Select decoy using bandit algorithm
	var selectedDecoy *SmartDecoy
	var strategy string

	if threatLevel.Level >= 8 {
		// High threat: Use most effective decoy or delay
		strategy = "delay"
		selectedDecoy = ace.selectHighEngagementDecoy()
	} else if threatLevel.Level >= 5 {
		// Medium threat: Balance exploration/exploitation
		strategy = "engage"
		selectedDecoy = ace.bandit.selectDecoy(ace.decoys)
	} else if threatLevel.Level >= 2 {
		// Low threat: Mostly exploit best performer
		strategy = "redirect"
		selectedDecoy = ace.selectBestPerformer()
	} else {
		// Benign: Fast response with lightweight decoy
		strategy = "engage"
		selectedDecoy = ace.selectLightweightDecoy()
	}

	if selectedDecoy == nil {
		return nil, fmt.Errorf("no suitable decoy available")
	}

	// Update usage metrics
	ace.recordDecoyUsage(selectedDecoy.ID)

	// Generate response
	response := &DecoyResponse{
		DecoyID:      selectedDecoy.ID,
		Strategy:     strategy,
		Profile:      selectedDecoy.Profile,
		Fingerprint:  selectedDecoy.Fingerprint,
		ResponseTime: selectedDecoy.Profile.ResponseDelay,
		Confidence:   threatLevel.Confidence,
	}

	// Add jitter to response time for realism
	response.ResponseTime += int(float64(response.ResponseTime) * (0.1 - 0.2*float64(threatLevel.Level)/10.0))

	return response, nil
}

// analyzeThreat computes threat level from request characteristics
func (ace *AdaptiveCamouflageEngine) analyzeThreat(req DecoySelectionRequest) ThreatLevel {
	ace.threatAnalyzer.mu.RLock()
	existing, found := ace.threatAnalyzer.threatLevels[req.SourceIP]
	ace.threatAnalyzer.mu.RUnlock()

	if found && time.Since(existing.LastUpdate) < 5*time.Minute {
		return existing
	}

	// Compute threat indicators
	indicators := make([]string, 0)
	threatScore := 0.0

	// User-Agent analysis
	if req.UserAgent == "" || containsSuspiciousUA(req.UserAgent) {
		indicators = append(indicators, "suspicious_user_agent")
		threatScore += 2.0
	}

	// Path analysis
	if containsScanPattern(req.RequestPath) {
		indicators = append(indicators, "scan_pattern")
		threatScore += 3.0
	}

	// Behavior history
	ace.threatAnalyzer.mu.RLock()
	if profile, ok := ace.threatAnalyzer.behaviorPatterns[req.SourceIP]; ok {
		if profile.ScanBehavior {
			indicators = append(indicators, "known_scanner")
			threatScore += 4.0
		}
		if profile.FailedAuthCount > 5 {
			indicators = append(indicators, "brute_force")
			threatScore += 3.0
		}
		if profile.AutomatedTraffic {
			indicators = append(indicators, "bot_traffic")
			threatScore += 2.0
		}
	}
	ace.threatAnalyzer.mu.RUnlock()

	// Provided threat score
	if req.ThreatScore > 0 {
		threatScore += float64(req.ThreatScore) / 10.0
	}

	// Normalize to 0-10 scale
	level := int(math.Min(10, threatScore))
	confidence := math.Min(1.0, threatScore/10.0)

	threatLevel := ThreatLevel{
		Level:      level,
		Confidence: confidence,
		Indicators: indicators,
		LastUpdate: time.Now(),
	}

	// Cache result
	ace.threatAnalyzer.mu.Lock()
	ace.threatAnalyzer.threatLevels[req.SourceIP] = threatLevel
	ace.threatAnalyzer.mu.Unlock()

	return threatLevel
}

// updateBehaviorProfile tracks attacker behavior patterns
func (ace *AdaptiveCamouflageEngine) updateBehaviorProfile(sourceIP string, req DecoySelectionRequest) {
	ace.threatAnalyzer.mu.Lock()
	defer ace.threatAnalyzer.mu.Unlock()

	profile, exists := ace.threatAnalyzer.behaviorPatterns[sourceIP]
	if !exists {
		profile = &BehaviorProfile{
			FirstSeen: time.Now(),
		}
		ace.threatAnalyzer.behaviorPatterns[sourceIP] = profile
	}

	profile.RequestCount++
	profile.LastSeen = time.Now()

	// Detect scan behavior
	if containsScanPattern(req.RequestPath) {
		profile.ScanBehavior = true
	}

	// Detect automation
	if req.UserAgent == "" || time.Since(profile.FirstSeen) < 1*time.Second && profile.RequestCount > 10 {
		profile.AutomatedTraffic = true
	}

	// Update threat score
	profile.ThreatScore = float64(profile.RequestCount)/100.0 +
		float64(profile.FailedAuthCount)/10.0
	if profile.ScanBehavior {
		profile.ThreatScore += 3.0
	}
	if profile.AutomatedTraffic {
		profile.ThreatScore += 2.0
	}
}

// Multi-Armed Bandit: UCB1 algorithm
func (mab *MultiArmedBandit) selectDecoy(decoys map[string]*SmartDecoy) *SmartDecoy {
	mab.mu.Lock()
	defer mab.mu.Unlock()

	mab.totalPulls++

	// Epsilon-greedy exploration
	if randFloat() < mab.epsilon {
		// Explore: random decoy
		for _, decoy := range decoys {
			if decoy.Enabled {
				return decoy
			}
		}
	}

	// Exploit: UCB1 selection
	bestDecoyID := ""
	bestUCB := -1.0

	for decoyID, arm := range mab.arms {
		if arm.Pulls == 0 {
			// Always try unpulled arms first
			bestDecoyID = decoyID
			break
		}

		// UCB1 formula
		avgReward := arm.Reward / float64(arm.Pulls)
		exploration := math.Sqrt(2 * math.Log(float64(mab.totalPulls)) / float64(arm.Pulls))
		ucb := avgReward + exploration

		arm.UCB = ucb

		if ucb > bestUCB {
			bestUCB = ucb
			bestDecoyID = decoyID
		}
	}

	if bestDecoyID == "" {
		// Fallback to first available
		for _, decoy := range decoys {
			if decoy.Enabled {
				return decoy
			}
		}
	}

	return decoys[bestDecoyID]
}

// RecordDecoyFeedback updates bandit rewards based on interaction quality
func (ace *AdaptiveCamouflageEngine) RecordDecoyFeedback(decoyID string, engagementTime float64, triggered bool) {
	ace.bandit.mu.Lock()
	defer ace.bandit.mu.Unlock()

	arm, exists := ace.bandit.arms[decoyID]
	if !exists {
		return
	}

	arm.Pulls++

	// Reward formula: longer engagement + triggering = higher reward
	reward := 0.0
	if triggered {
		reward += 1.0
	}
	reward += math.Min(1.0, engagementTime/30.0) // Cap at 30 seconds

	// Update moving average reward
	arm.Reward = (arm.Reward*float64(arm.Pulls-1) + reward) / float64(arm.Pulls)

	// Update decoy metrics
	ace.mu.Lock()
	if decoy, ok := ace.decoys[decoyID]; ok {
		decoy.Pulls = arm.Pulls
		decoy.Reward = arm.Reward
		decoy.InteractionCount++
		if triggered {
			decoy.SuccessRate = (decoy.SuccessRate*float64(decoy.InteractionCount-1) + 1.0) / float64(decoy.InteractionCount)
		} else {
			decoy.SuccessRate = (decoy.SuccessRate * float64(decoy.InteractionCount-1)) / float64(decoy.InteractionCount)
		}
		decoy.AvgEngagementTime = (decoy.AvgEngagementTime*float64(decoy.InteractionCount-1) + engagementTime) / float64(decoy.InteractionCount)
	}
	ace.mu.Unlock()

	log.Printf("Decoy %s feedback: engagement=%.1fs, triggered=%v, reward=%.2f",
		decoyID, engagementTime, triggered, reward)
}

// Helper functions
func (ace *AdaptiveCamouflageEngine) selectHighEngagementDecoy() *SmartDecoy {
	ace.mu.RLock()
	defer ace.mu.RUnlock()

	var best *SmartDecoy
	maxEngagement := 0.0

	for _, decoy := range ace.decoys {
		if decoy.Enabled && decoy.AvgEngagementTime > maxEngagement {
			maxEngagement = decoy.AvgEngagementTime
			best = decoy
		}
	}

	return best
}

func (ace *AdaptiveCamouflageEngine) selectBestPerformer() *SmartDecoy {
	ace.mu.RLock()
	defer ace.mu.RUnlock()

	var best *SmartDecoy
	maxReward := 0.0

	for _, decoy := range ace.decoys {
		if decoy.Enabled && decoy.Reward > maxReward {
			maxReward = decoy.Reward
			best = decoy
		}
	}

	return best
}

func (ace *AdaptiveCamouflageEngine) selectLightweightDecoy() *SmartDecoy {
	ace.mu.RLock()
	defer ace.mu.RUnlock()

	var best *SmartDecoy
	minComplexity := 100

	for _, decoy := range ace.decoys {
		if decoy.Enabled && decoy.Profile.Complexity < minComplexity {
			minComplexity = decoy.Profile.Complexity
			best = decoy
		}
	}

	return best
}

func (ace *AdaptiveCamouflageEngine) recordDecoyUsage(decoyID string) {
	ace.mu.Lock()
	defer ace.mu.Unlock()

	if decoy, ok := ace.decoys[decoyID]; ok {
		decoy.LastUsed = time.Now()
	}

	ace.metricsStore.mu.Lock()
	if metrics, ok := ace.metricsStore.metrics[decoyID]; ok {
		metrics.Interactions++
		metrics.LastInteraction = time.Now()
	}
	ace.metricsStore.mu.Unlock()
}

func containsSuspiciousUA(ua string) bool {
	suspicious := []string{"sqlmap", "nmap", "nikto", "scanner", "bot", "curl", "wget", "python-requests"}
	uaLower := toLower(ua)
	for _, s := range suspicious {
		if contains(uaLower, s) {
			return true
		}
	}
	return false
}

func containsScanPattern(path string) bool {
	patterns := []string{"admin", "wp-admin", ".env", "config", "backup", ".git", "phpmyadmin", "sql"}
	pathLower := toLower(path)
	for _, p := range patterns {
		if contains(pathLower, p) {
			return true
		}
	}
	return false
}

// contains reports whether substr is within s (case-sensitive). Wrapper around strings.Contains
func contains(s, substr string) bool { return strings.Contains(s, substr) }

// toLower returns a lowercase copy of s (ASCII + Unicode aware via strings.ToLower)
func toLower(s string) string { return strings.ToLower(s) }

// randFloat returns a cryptographically strong uniform float64 in [0,1)
func randFloat() float64 {
	b := make([]byte, 8)
	if _, err := crand.Read(b); err != nil {
		// Fallback: deterministic but should almost never happen
		h := sha256.Sum256([]byte(time.Now().String()))
		copy(b, h[:8])
	}
	h := sha256.Sum256(b)
	val := uint64(0)
	for i := 0; i < 8; i++ {
		val = (val << 8) | uint64(h[i])
	}
	return float64(val) / float64(^uint64(0))
}

// GetMetrics returns current performance metrics
func (ace *AdaptiveCamouflageEngine) GetMetrics() map[string]interface{} {
	ace.mu.RLock()
	defer ace.mu.RUnlock()

	metrics := make(map[string]interface{})

	decoyMetrics := make([]map[string]interface{}, 0, len(ace.decoys))
	for _, decoy := range ace.decoys {
		decoyMetrics = append(decoyMetrics, map[string]interface{}{
			"id":             decoy.ID,
			"type":           decoy.Type,
			"interactions":   decoy.InteractionCount,
			"success_rate":   decoy.SuccessRate,
			"avg_engagement": decoy.AvgEngagementTime,
			"reward":         decoy.Reward,
			"pulls":          decoy.Pulls,
		})
	}

	metrics["decoys"] = decoyMetrics
	metrics["total_decoys"] = len(ace.decoys)
	metrics["bandit_total_pulls"] = ace.bandit.totalPulls
	metrics["threat_levels_tracked"] = len(ace.threatAnalyzer.threatLevels)

	return metrics
}
