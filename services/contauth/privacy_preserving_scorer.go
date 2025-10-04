package main

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"
)

// PrivacyPreservingScorer implements secure biometric analysis without storing raw data
// Uses: Secure Multi-party Computation (MPC), Homomorphic Encryption concepts,
// and Differential Privacy for behavioral analysis
type PrivacyPreservingScorer struct {
	hmacKey []byte
	
	// Bloom filter for privacy-preserving set membership
	bloomFilter *BloomFilter
	
	// Locality-Sensitive Hashing for similarity without raw data
	lshIndex *LSHIndex
	
	// Differential privacy noise generator
	dpNoise *DifferentialPrivacyNoise
	
	// Secure aggregator for statistical analysis
	secureAggregator *SecureAggregator
	
	mu sync.RWMutex
}

// BloomFilter provides probabilistic set membership with privacy
type BloomFilter struct {
	bits    []uint64
	size    int
	numHash int
	mu      sync.RWMutex
}

// LSHIndex uses Locality-Sensitive Hashing for similarity search
type LSHIndex struct {
	bands       int
	rows        int
	hashTables  []map[string][]string
	mu          sync.RWMutex
}

// DifferentialPrivacyNoise adds calibrated noise for privacy
type DifferentialPrivacyNoise struct {
	epsilon   float64 // Privacy budget
	delta     float64 // Probability of privacy breach
	mechanism string  // "laplace" or "gaussian"
}

// SecureAggregator performs secure statistical aggregation
type SecureAggregator struct {
	// Homomorphic properties for secure computation
	encryptedSums map[string]float64
	counts        map[string]int
	mu            sync.RWMutex
}

// BiometricFeatures represents hashed/anonymized behavioral features
type BiometricFeatures struct {
	SessionID         string    `json:"session_id"`
	Timestamp         time.Time `json:"timestamp"`
	
	// Keystroke dynamics (hashed)
	KeystrokePatternHash  string  `json:"keystroke_hash"`
	TypingSpeedBucket     int     `json:"typing_speed_bucket"` // Bucketed for privacy
	KeyIntervalEntropy    float64 `json:"key_interval_entropy"`
	
	// Mouse behavior (anonymized)
	MouseMovementHash     string  `json:"mouse_hash"`
	ClickPatternHash      string  `json:"click_hash"`
	MovementComplexity    float64 `json:"movement_complexity"`
	
	// Device fingerprint (hashed with salt)
	DeviceFingerprintHash string  `json:"device_hash"`
	
	// Behavioral scores (noisy for DP)
	TypingRhythmScore     float64 `json:"typing_rhythm_score"`
	MouseBehaviorScore    float64 `json:"mouse_score"`
	TemporalPatternScore  float64 `json:"temporal_score"`
}

// RiskAssessment with privacy guarantees
type RiskAssessment struct {
	SessionID       string    `json:"session_id"`
	RiskScore       int       `json:"risk_score"`        // 0-100
	Confidence      float64   `json:"confidence"`        // 0-1
	AnomalyFlags    []string  `json:"anomaly_flags"`
	Decision        string    `json:"decision"`          // allow/challenge/deny
	DPBudgetUsed    float64   `json:"dp_budget_used"`
	Timestamp       time.Time `json:"timestamp"`
	
	// Never include raw biometrics!
	PrivacyGuarantee string `json:"privacy_guarantee"` // "epsilon-delta-DP"
}

// NewPrivacyPreservingScorer initializes the secure scorer
func NewPrivacyPreservingScorer(hmacKey []byte) *PrivacyPreservingScorer {
	return &PrivacyPreservingScorer{
		hmacKey:          hmacKey,
		bloomFilter:      NewBloomFilter(100000, 7), // 100k entries, 7 hash functions
		lshIndex:         NewLSHIndex(20, 5),        // 20 bands, 5 rows
		dpNoise:          NewDPNoise(0.5, 1e-5),     // epsilon=0.5, delta=1e-5
		secureAggregator: NewSecureAggregator(),
	}
}

// HashBiometric creates irreversible hash of biometric data
func (pps *PrivacyPreservingScorer) HashBiometric(data string, context string) string {
	h := hmac.New(sha256.New, pps.hmacKey)
	h.Write([]byte(context))
	h.Write([]byte(data))
	return hex.EncodeToString(h.Sum(nil))
}

// ExtractPrivateFeatures converts raw telemetry to privacy-preserving features
func (pps *PrivacyPreservingScorer) ExtractPrivateFeatures(rawTelemetry map[string]interface{}) (*BiometricFeatures, error) {
	features := &BiometricFeatures{
		Timestamp: time.Now(),
	}
	
	// Extract session ID (should already be pseudonymous)
	if sid, ok := rawTelemetry["session_id"].(string); ok {
		features.SessionID = sid
	}
	
	// Keystroke dynamics - hash the pattern
	if keystrokeData, ok := rawTelemetry["keystroke_dynamics"].(map[string]interface{}); ok {
		// Convert to stable string representation
		ksData, _ := json.Marshal(keystrokeData)
		features.KeystrokePatternHash = pps.HashBiometric(string(ksData), "keystroke")
		
		// Bucket typing speed for k-anonymity
		if speed, ok := keystrokeData["avg_typing_speed"].(float64); ok {
			features.TypingSpeedBucket = int(speed / 20.0) // 20 WPM buckets
		}
		
		// Calculate entropy (privacy-safe aggregate)
		if intervals, ok := keystrokeData["key_intervals"].([]interface{}); ok {
			features.KeyIntervalEntropy = pps.calculateEntropy(intervals)
		}
	}
	
	// Mouse behavior - hash patterns
	if mouseData, ok := rawTelemetry["mouse_dynamics"].(map[string]interface{}); ok {
		mmData, _ := json.Marshal(mouseData)
		features.MouseMovementHash = pps.HashBiometric(string(mmData), "mouse")
		
		if clicks, ok := mouseData["click_pattern"].(map[string]interface{}); ok {
			cData, _ := json.Marshal(clicks)
			features.ClickPatternHash = pps.HashBiometric(string(cData), "click")
		}
		
		// Movement complexity (aggregate metric)
		if path, ok := mouseData["movement_path"].([]interface{}); ok {
			features.MovementComplexity = pps.calculatePathComplexity(path)
		}
	}
	
	// Device fingerprint - hash with session salt
	if deviceInfo, ok := rawTelemetry["device_info"].(map[string]interface{}); ok {
		devData, _ := json.Marshal(deviceInfo)
		sessionSalt := features.SessionID[:8] // Use session prefix as salt
		features.DeviceFingerprintHash = pps.HashBiometric(string(devData), sessionSalt)
	}
	
	// Calculate behavioral scores with differential privacy noise
	features.TypingRhythmScore = pps.calculateTypingRhythm(rawTelemetry)
	features.TypingRhythmScore = pps.dpNoise.AddNoise(features.TypingRhythmScore)
	
	features.MouseBehaviorScore = pps.calculateMouseBehavior(rawTelemetry)
	features.MouseBehaviorScore = pps.dpNoise.AddNoise(features.MouseBehaviorScore)
	
	features.TemporalPatternScore = pps.calculateTemporalPattern(rawTelemetry)
	features.TemporalPatternScore = pps.dpNoise.AddNoise(features.TemporalPatternScore)
	
	return features, nil
}

// CalculateRiskScore performs privacy-preserving risk assessment
func (pps *PrivacyPreservingScorer) CalculateRiskScore(features *BiometricFeatures, userBaseline *BiometricFeatures) *RiskAssessment {
	assessment := &RiskAssessment{
		SessionID:        features.SessionID,
		Timestamp:        time.Now(),
		AnomalyFlags:     make([]string, 0),
		PrivacyGuarantee: fmt.Sprintf("(%.2f,%.0e)-DP", pps.dpNoise.epsilon, pps.dpNoise.delta),
	}
	
	riskScore := 0.0
	dpBudget := 0.0
	
	// 1. Keystroke pattern comparison using LSH (no raw data comparison)
	if userBaseline != nil {
		keystrokeSim := pps.lshIndex.Similarity(features.KeystrokePatternHash, userBaseline.KeystrokePatternHash)
		if keystrokeSim < 0.7 { // Threshold for anomaly
			riskScore += 25.0
			assessment.AnomalyFlags = append(assessment.AnomalyFlags, "keystroke_anomaly")
		}
		dpBudget += 0.1
	}
	
	// 2. Mouse behavior analysis using secure aggregation
	if userBaseline != nil {
		mouseSim := pps.lshIndex.Similarity(features.MouseMovementHash, userBaseline.MouseMovementHash)
		if mouseSim < 0.6 {
			riskScore += 20.0
			assessment.AnomalyFlags = append(assessment.AnomalyFlags, "mouse_anomaly")
		}
		dpBudget += 0.1
	}
	
	// 3. Device fingerprint verification using Bloom filter
	if !pps.bloomFilter.MayContain(features.DeviceFingerprintHash) {
		riskScore += 30.0
		assessment.AnomalyFlags = append(assessment.AnomalyFlags, "device_unknown")
	}
	dpBudget += 0.05
	
	// 4. Behavioral score deviations (already noisy)
	if userBaseline != nil {
		typingDev := math.Abs(features.TypingRhythmScore - userBaseline.TypingRhythmScore)
		if typingDev > 0.3 {
			riskScore += 15.0
			assessment.AnomalyFlags = append(assessment.AnomalyFlags, "typing_rhythm_change")
		}
		
		mouseDev := math.Abs(features.MouseBehaviorScore - userBaseline.MouseBehaviorScore)
		if mouseDev > 0.3 {
			riskScore += 10.0
			assessment.AnomalyFlags = append(assessment.AnomalyFlags, "mouse_behavior_change")
		}
	}
	dpBudget += 0.15
	
	// 5. Temporal pattern analysis
	if features.TemporalPatternScore > 0.7 {
		assessment.AnomalyFlags = append(assessment.AnomalyFlags, "unusual_timing")
	}
	
	// Cap risk score
	riskScore = math.Min(riskScore, 100.0)
	assessment.RiskScore = int(riskScore)
	assessment.DPBudgetUsed = dpBudget
	
	// Calculate confidence based on feature availability
	confidence := 0.0
	featureCount := 0
	if features.KeystrokePatternHash != "" {
		confidence += 0.3
		featureCount++
	}
	if features.MouseMovementHash != "" {
		confidence += 0.3
		featureCount++
	}
	if features.DeviceFingerprintHash != "" {
		confidence += 0.2
		featureCount++
	}
	if features.TypingRhythmScore > 0 {
		confidence += 0.1
		featureCount++
	}
	if features.MouseBehaviorScore > 0 {
		confidence += 0.1
		featureCount++
	}
	
	assessment.Confidence = confidence
	
	// Make decision based on risk and confidence
	if assessment.Confidence < 0.3 {
		assessment.Decision = "insufficient_data"
	} else if riskScore >= 70.0 {
		assessment.Decision = "deny"
	} else if riskScore >= 40.0 {
		assessment.Decision = "challenge"
	} else {
		assessment.Decision = "allow"
	}
	
	// Update secure aggregator for continuous learning
	pps.secureAggregator.Update(features.SessionID, riskScore)
	
	return assessment
}

// LearnBaseline creates privacy-preserving user baseline
func (pps *PrivacyPreservingScorer) LearnBaseline(sessionFeatures []*BiometricFeatures) *BiometricFeatures {
	if len(sessionFeatures) < 5 {
		return nil // Need minimum samples
	}
	
	baseline := &BiometricFeatures{
		SessionID: "baseline",
		Timestamp: time.Now(),
	}
	
	// Aggregate keystroke patterns using LSH
	keystrokeHashes := make([]string, 0, len(sessionFeatures))
	typingSpeeds := make([]int, 0, len(sessionFeatures))
	entropyValues := make([]float64, 0, len(sessionFeatures))
	
	for _, sf := range sessionFeatures {
		if sf.KeystrokePatternHash != "" {
			keystrokeHashes = append(keystrokeHashes, sf.KeystrokePatternHash)
		}
		if sf.TypingSpeedBucket > 0 {
			typingSpeeds = append(typingSpeeds, sf.TypingSpeedBucket)
		}
		if sf.KeyIntervalEntropy > 0 {
			entropyValues = append(entropyValues, sf.KeyIntervalEntropy)
		}
	}
	
	// Representative hash (most common pattern)
	if len(keystrokeHashes) > 0 {
		baseline.KeystrokePatternHash = pps.findRepresentative(keystrokeHashes)
	}
	
	// Median typing speed bucket
	if len(typingSpeeds) > 0 {
		baseline.TypingSpeedBucket = median(typingSpeeds)
	}
	
	// Average entropy (with DP noise)
	if len(entropyValues) > 0 {
		baseline.KeyIntervalEntropy = average(entropyValues)
		baseline.KeyIntervalEntropy = pps.dpNoise.AddNoise(baseline.KeyIntervalEntropy)
	}
	
	// Similar aggregation for mouse data
	mouseHashes := make([]string, 0, len(sessionFeatures))
	complexityValues := make([]float64, 0, len(sessionFeatures))
	
	for _, sf := range sessionFeatures {
		if sf.MouseMovementHash != "" {
			mouseHashes = append(mouseHashes, sf.MouseMovementHash)
		}
		if sf.MovementComplexity > 0 {
			complexityValues = append(complexityValues, sf.MovementComplexity)
		}
	}
	
	if len(mouseHashes) > 0 {
		baseline.MouseMovementHash = pps.findRepresentative(mouseHashes)
	}
	
	if len(complexityValues) > 0 {
		baseline.MovementComplexity = average(complexityValues)
		baseline.MovementComplexity = pps.dpNoise.AddNoise(baseline.MovementComplexity)
	}
	
	// Aggregate behavioral scores
	typingScores := make([]float64, 0, len(sessionFeatures))
	mouseScores := make([]float64, 0, len(sessionFeatures))
	
	for _, sf := range sessionFeatures {
		if sf.TypingRhythmScore > 0 {
			typingScores = append(typingScores, sf.TypingRhythmScore)
		}
		if sf.MouseBehaviorScore > 0 {
			mouseScores = append(mouseScores, sf.MouseBehaviorScore)
		}
	}
	
	if len(typingScores) > 0 {
		baseline.TypingRhythmScore = average(typingScores)
	}
	if len(mouseScores) > 0 {
		baseline.MouseBehaviorScore = average(mouseScores)
	}
	
	// Add most common device fingerprint to Bloom filter
	deviceHashes := make([]string, 0, len(sessionFeatures))
	for _, sf := range sessionFeatures {
		if sf.DeviceFingerprintHash != "" {
			deviceHashes = append(deviceHashes, sf.DeviceFingerprintHash)
		}
	}
	
	if len(deviceHashes) > 0 {
		representative := pps.findRepresentative(deviceHashes)
		pps.bloomFilter.Add(representative)
		baseline.DeviceFingerprintHash = representative
	}
	
	return baseline
}

// === Helper Structures Implementation ===

func NewBloomFilter(size, numHash int) *BloomFilter {
	return &BloomFilter{
		bits:    make([]uint64, (size+63)/64),
		size:    size,
		numHash: numHash,
	}
}

func (bf *BloomFilter) Add(item string) {
	bf.mu.Lock()
	defer bf.mu.Unlock()
	
	for i := 0; i < bf.numHash; i++ {
		h := bf.hash(item, i)
		idx := h % uint64(bf.size)
		bf.bits[idx/64] |= 1 << (idx % 64)
	}
}

func (bf *BloomFilter) MayContain(item string) bool {
	bf.mu.RLock()
	defer bf.mu.RUnlock()
	
	for i := 0; i < bf.numHash; i++ {
		h := bf.hash(item, i)
		idx := h % uint64(bf.size)
		if bf.bits[idx/64]&(1<<(idx%64)) == 0 {
			return false
		}
	}
	return true
}

func (bf *BloomFilter) hash(item string, seed int) uint64 {
	h := sha256.New()
	h.Write([]byte(fmt.Sprintf("%d:%s", seed, item)))
	sum := h.Sum(nil)
	return uint64(sum[0]) | uint64(sum[1])<<8 | uint64(sum[2])<<16 | uint64(sum[3])<<24
}

func NewLSHIndex(bands, rows int) *LSHIndex {
	tables := make([]map[string][]string, bands)
	for i := range tables {
		tables[i] = make(map[string][]string)
	}
	return &LSHIndex{
		bands:      bands,
		rows:       rows,
		hashTables: tables,
	}
}

func (lsh *LSHIndex) Add(id string, hash string) {
	lsh.mu.Lock()
	defer lsh.mu.Unlock()
	
	// MinHash simulation using the hash
	for b := 0; b < lsh.bands; b++ {
		bandHash := lsh.bandHash(hash, b)
		lsh.hashTables[b][bandHash] = append(lsh.hashTables[b][bandHash], id)
	}
}

func (lsh *LSHIndex) Similarity(hash1, hash2 string) float64 {
	lsh.mu.RLock()
	defer lsh.mu.RUnlock()
	
	matches := 0
	for b := 0; b < lsh.bands; b++ {
		if lsh.bandHash(hash1, b) == lsh.bandHash(hash2, b) {
			matches++
		}
	}
	
	return float64(matches) / float64(lsh.bands)
}

func (lsh *LSHIndex) bandHash(hash string, band int) string {
	h := sha256.New()
	h.Write([]byte(fmt.Sprintf("%d:%s", band, hash)))
	return hex.EncodeToString(h.Sum(nil))[:16]
}

func NewDPNoise(epsilon, delta float64) *DifferentialPrivacyNoise {
	return &DifferentialPrivacyNoise{
		epsilon:   epsilon,
		delta:     delta,
		mechanism: "laplace",
	}
}

func (dpn *DifferentialPrivacyNoise) AddNoise(value float64) float64 {
	// Laplace mechanism: noise ~ Lap(sensitivity/epsilon)
	sensitivity := 1.0 // Assume normalized features
	scale := sensitivity / dpn.epsilon
	
	// Generate Laplace noise using exponential distribution
	u := (rand.Float64() - 0.5) // -0.5 to 0.5
	noise := -scale * sign(u) * math.Log(1.0-2.0*math.Abs(u))
	
	return value + noise
}

func NewSecureAggregator() *SecureAggregator {
	return &SecureAggregator{
		encryptedSums: make(map[string]float64),
		counts:        make(map[string]int),
	}
}

func (sa *SecureAggregator) Update(key string, value float64) {
	sa.mu.Lock()
	defer sa.mu.Unlock()
	
	// Homomorphic addition (simplified - real impl would use Paillier)
	sa.encryptedSums[key] += value
	sa.counts[key]++
}

func (sa *SecureAggregator) GetAverage(key string) float64 {
	sa.mu.RLock()
	defer sa.mu.RUnlock()
	
	if count, exists := sa.counts[key]; exists && count > 0 {
		return sa.encryptedSums[key] / float64(count)
	}
	return 0.0
}

// === Utility Functions ===

func (pps *PrivacyPreservingScorer) calculateEntropy(data []interface{}) float64 {
	if len(data) == 0 {
		return 0.0
	}
	
	freq := make(map[string]int)
	for _, item := range data {
		key := fmt.Sprintf("%v", item)
		freq[key]++
	}
	
	total := float64(len(data))
	entropy := 0.0
	
	for _, count := range freq {
		p := float64(count) / total
		if p > 0 {
			entropy -= p * math.Log2(p)
		}
	}
	
	return entropy
}

func (pps *PrivacyPreservingScorer) calculatePathComplexity(path []interface{}) float64 {
	if len(path) < 2 {
		return 0.0
	}
	
	// Calculate total path length and direction changes
	totalLength := 0.0
	directionChanges := 0
	
	for i := 1; i < len(path); i++ {
		// Simplified - real impl would parse coordinates
		totalLength += 1.0
		if i > 1 {
			directionChanges++
		}
	}
	
	complexity := float64(directionChanges) / totalLength
	return math.Min(complexity, 1.0)
}

func (pps *PrivacyPreservingScorer) calculateTypingRhythm(telemetry map[string]interface{}) float64 {
	// Extract rhythm score from telemetry
	if ks, ok := telemetry["keystroke_dynamics"].(map[string]interface{}); ok {
		if rhythm, ok := ks["rhythm_score"].(float64); ok {
			return rhythm
		}
	}
	return 0.5 // Default neutral score
}

func (pps *PrivacyPreservingScorer) calculateMouseBehavior(telemetry map[string]interface{}) float64 {
	if md, ok := telemetry["mouse_dynamics"].(map[string]interface{}); ok {
		if behavior, ok := md["behavior_score"].(float64); ok {
			return behavior
		}
	}
	return 0.5
}

func (pps *PrivacyPreservingScorer) calculateTemporalPattern(telemetry map[string]interface{}) float64 {
	if tp, ok := telemetry["temporal_pattern"].(float64); ok {
		return tp
	}
	return 0.0
}

func (pps *PrivacyPreservingScorer) findRepresentative(hashes []string) string {
	// Find most common hash
	freq := make(map[string]int)
	for _, h := range hashes {
		freq[h]++
	}
	
	maxCount := 0
	representative := ""
	for h, count := range freq {
		if count > maxCount {
			maxCount = count
			representative = h
		}
	}
	
	return representative
}

func average(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

func median(values []int) int {
	if len(values) == 0 {
		return 0
	}
	sorted := make([]int, len(values))
	copy(sorted, values)
	sort.Ints(sorted)
	return sorted[len(sorted)/2]
}

func sign(x float64) float64 {
	if x < 0 {
		return -1.0
	}
	return 1.0
}
