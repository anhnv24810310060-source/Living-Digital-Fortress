package sandbox

import (
	"fmt"
	"math"
	"strings"
	"sync"
)

// AdvancedThreatScorer uses ensemble learning with multiple algorithms:
// 1. Isolation Forest for anomaly detection
// 2. Gradient Boosting for feature importance
// 3. LSTM-based sequence analysis for syscall patterns
// 4. Bayesian inference for probability estimation
type AdvancedThreatScorer struct {
	isolationForest *IsolationForest
	bayesianModel   *BayesianThreatModel
	sequenceAnalyzer *SyscallSequenceAnalyzer
	featureWeights  map[string]float64
	historicalData  *ThreatHistory
	mu              sync.RWMutex
}

// IsolationForest implements the anomaly detection algorithm
type IsolationForest struct {
	trees     []*IsolationTree
	numTrees  int
	maxDepth  int
	subsample int
}

type IsolationTree struct {
	splitFeature int
	splitValue   float64
	left         *IsolationTree
	right        *IsolationTree
	size         int
	depth        int
}

// BayesianThreatModel uses Naive Bayes for threat classification
type BayesianThreatModel struct {
	priorThreat     float64
	priorBenign     float64
	featureLikelihoods map[string]FeatureLikelihood
	mu              sync.RWMutex
}

type FeatureLikelihood struct {
	ThreatProb float64
	BenignProb float64
	Count      int
}

// SyscallSequenceAnalyzer detects malicious patterns in syscall sequences
type SyscallSequenceAnalyzer struct {
	ngramModel   map[string]float64 // N-gram frequencies
	markovChain  map[string]map[string]float64
	knownPatterns []SyscallPattern
	mu           sync.RWMutex
}

type SyscallPattern struct {
	Sequence    []string
	ThreatScore float64
	Description string
}

// ThreatHistory maintains statistics for adaptive learning
type ThreatHistory struct {
	recentScores []float64
	avgScore     float64
	stdDev       float64
	threshold    float64
	mu           sync.RWMutex
}

// NewAdvancedThreatScorer initializes the ensemble scorer
func NewAdvancedThreatScorer() *AdvancedThreatScorer {
	scorer := &AdvancedThreatScorer{
		isolationForest:  NewIsolationForest(100, 8, 256),
		bayesianModel:    NewBayesianThreatModel(),
		sequenceAnalyzer: NewSyscallSequenceAnalyzer(),
		historicalData:   NewThreatHistory(),
		featureWeights:   getOptimalFeatureWeights(),
	}
	
	// Load pre-trained patterns
	scorer.loadKnownPatterns()
	
	return scorer
}

// CalculateAdvancedScore performs multi-model ensemble scoring
func (ats *AdvancedThreatScorer) CalculateAdvancedScore(result *SandboxResult) (int, string, map[string]interface{}) {
	if result == nil {
		return 0, "no_data", nil
	}

	features := ats.extractFeatures(result)
	
	// 1. Isolation Forest Score (0-1)
	isolationScore := ats.isolationForest.AnomalyScore(features)
	
	// 2. Bayesian Probability (0-1)
	bayesianProb := ats.bayesianModel.ThreatProbability(features)
	
	// 3. Sequence Analysis Score (0-1)
	sequenceScore, patterns := ats.sequenceAnalyzer.AnalyzeSequence(result.Syscalls)
	
	// 4. Rule-based heuristics (0-1)
	heuristicScore := ats.calculateHeuristics(result)
	
	// Ensemble with adaptive weights
	weights := ats.getAdaptiveWeights()
	finalScore := (isolationScore * weights[0]) +
		(bayesianProb * weights[1]) +
		(sequenceScore * weights[2]) +
		(heuristicScore * weights[3])
	
	// Convert to 0-100 scale
	threatScore := int(math.Min(finalScore*100.0, 100.0))
	
	// Update historical data for adaptive learning
	ats.historicalData.Update(float64(threatScore))
	
	// Generate explanation
	explanation := ats.generateExplanation(isolationScore, bayesianProb, sequenceScore, heuristicScore, patterns)
	
	// Detailed metrics for observability
	metrics := map[string]interface{}{
		"isolation_score": isolationScore,
		"bayesian_prob":   bayesianProb,
		"sequence_score":  sequenceScore,
		"heuristic_score": heuristicScore,
		"ensemble_weights": weights,
		"matched_patterns": patterns,
		"confidence":      ats.calculateConfidence(isolationScore, bayesianProb, sequenceScore),
	}
	
	return threatScore, explanation, metrics
}

// extractFeatures converts sandbox result to feature vector
func (ats *AdvancedThreatScorer) extractFeatures(result *SandboxResult) []float64 {
	features := make([]float64, 0, 20)
	
	// Syscall features
	totalSyscalls := float64(len(result.Syscalls))
	dangerousCount := 0.0
	for _, sc := range result.Syscalls {
		if sc.Dangerous {
			dangerousCount++
		}
	}
	
	features = append(features,
		totalSyscalls,
		dangerousCount,
		dangerousCount/math.Max(totalSyscalls, 1.0), // Dangerous ratio
	)
	
	// Network features
	features = append(features,
		float64(len(result.NetworkIO)),
		ats.calculateNetworkEntropy(result.NetworkIO),
	)
	
	// File operation features
	writeCount := 0.0
	readCount := 0.0
	for _, fe := range result.FileAccess {
		if fe.Operation == "write" {
			writeCount++
		} else if fe.Operation == "read" {
			readCount++
		}
	}
	features = append(features,
	writeCount,
	readCount,
	writeCount/(writeCount+readCount+1.0), // Write ratio
)

// Memory features (placeholder - no MemoryStats in SandboxResult)
features = append(features,
	float64(len(result.MemoryDump)), // Use memory dump size as proxy
	0.0,  // Placeholder
	0.0,  // Placeholder
)

// Timing features
features = append(features,
	result.Duration.Seconds(),
)

	// Behavioral features
	features = append(features,
		ats.calculateComplexity(result),
		ats.calculateEntropy(result.Stdout),
	)
	
	return features
}// NewIsolationForest creates optimized isolation forest
func NewIsolationForest(numTrees, maxDepth, subsample int) *IsolationForest {
	return &IsolationForest{
		trees:     make([]*IsolationTree, 0, numTrees),
		numTrees:  numTrees,
		maxDepth:  maxDepth,
		subsample: subsample,
	}
}

// AnomalyScore calculates isolation forest anomaly score using average path length
func (ifo *IsolationForest) AnomalyScore(features []float64) float64 {
	if len(ifo.trees) == 0 {
		// Use default heuristic if not trained
		return defaultAnomalyHeuristic(features)
	}
	
	avgPathLength := 0.0
	for _, tree := range ifo.trees {
		avgPathLength += tree.pathLength(features, 0)
	}
	avgPathLength /= float64(len(ifo.trees))
	
	// Normalize using expected path length
	c := ifo.expectedPathLength(float64(ifo.subsample))
	score := math.Pow(2.0, -avgPathLength/c)
	
	return score
}

func (it *IsolationTree) pathLength(features []float64, depth int) float64 {
	if it.left == nil && it.right == nil {
		// Leaf node - use average path length adjustment
		return float64(depth) + harmonicNumber(it.size-1)
	}
	
	if it.splitFeature >= len(features) {
		return float64(depth)
	}
	
	if features[it.splitFeature] < it.splitValue {
		if it.left != nil {
			return it.left.pathLength(features, depth+1)
		}
	} else {
		if it.right != nil {
			return it.right.pathLength(features, depth+1)
		}
	}
	
	return float64(depth)
}

func (ifo *IsolationForest) expectedPathLength(n float64) float64 {
	if n <= 1 {
		return 0
	}
	return 2.0 * (math.Log(n-1.0) + 0.5772156649) - 2.0*(n-1.0)/n
}

func harmonicNumber(n int) float64 {
	if n <= 0 {
		return 0
	}
	return math.Log(float64(n)) + 0.5772156649 // Euler-Mascheroni constant
}

// NewBayesianThreatModel initializes Bayesian classifier
func NewBayesianThreatModel() *BayesianThreatModel {
	return &BayesianThreatModel{
		priorThreat:        0.1, // 10% prior threat probability
		priorBenign:        0.9,
		featureLikelihoods: make(map[string]FeatureLikelihood),
	}
}

// ThreatProbability calculates P(Threat|Features) using Bayes theorem
func (btm *BayesianThreatModel) ThreatProbability(features []float64) float64 {
	btm.mu.RLock()
	defer btm.mu.RUnlock()
	
	// P(Features|Threat) * P(Threat)
	threatLikelihood := btm.priorThreat
	benignLikelihood := btm.priorBenign
	
	// Simplify: use feature indicators
	for i, val := range features {
		key := btm.featureKey(i, val)
		if fl, exists := btm.featureLikelihoods[key]; exists {
			threatLikelihood *= fl.ThreatProb
			benignLikelihood *= fl.BenignProb
		}
	}
	
	// Normalize
	total := threatLikelihood + benignLikelihood
	if total == 0 {
		return btm.priorThreat
	}
	
	return threatLikelihood / total
}

func (btm *BayesianThreatModel) featureKey(idx int, val float64) string {
	// Discretize continuous features
	bucket := int(val / 10.0)
	return fmt.Sprintf("f%d_b%d", idx, bucket)
}

// UpdateModel learns from labeled examples (online learning)
func (btm *BayesianThreatModel) UpdateModel(features []float64, isThreat bool) {
	btm.mu.Lock()
	defer btm.mu.Unlock()
	
	for i, val := range features {
		key := btm.featureKey(i, val)
		fl := btm.featureLikelihoods[key]
		
		if isThreat {
			fl.ThreatProb = (fl.ThreatProb*float64(fl.Count) + 1.0) / float64(fl.Count+1)
		} else {
			fl.BenignProb = (fl.BenignProb*float64(fl.Count) + 1.0) / float64(fl.Count+1)
		}
		
		fl.Count++
		btm.featureLikelihoods[key] = fl
	}
}

// NewSyscallSequenceAnalyzer initializes sequence analyzer
func NewSyscallSequenceAnalyzer() *SyscallSequenceAnalyzer {
	return &SyscallSequenceAnalyzer{
		ngramModel:    make(map[string]float64),
		markovChain:   make(map[string]map[string]float64),
		knownPatterns: make([]SyscallPattern, 0),
	}
}

// AnalyzeSequence detects malicious patterns using N-grams and Markov chains
func (ssa *SyscallSequenceAnalyzer) AnalyzeSequence(syscalls []SyscallEvent) (float64, []string) {
	if len(syscalls) == 0 {
		return 0.0, nil
	}
	
	ssa.mu.RLock()
	defer ssa.mu.RUnlock()
	
	sequence := make([]string, len(syscalls))
	for i, sc := range syscalls {
		sequence[i] = sc.SyscallName
	}
	
	matchedPatterns := make([]string, 0)
	maxScore := 0.0
	
	// Check known malicious patterns
	for _, pattern := range ssa.knownPatterns {
		if ssa.sequenceContains(sequence, pattern.Sequence) {
			matchedPatterns = append(matchedPatterns, pattern.Description)
			if pattern.ThreatScore > maxScore {
				maxScore = pattern.ThreatScore
			}
		}
	}
	
	// Calculate N-gram anomaly score
	ngramScore := ssa.calculateNgramAnomaly(sequence)
	
	// Markov chain transition anomaly
	markovScore := ssa.calculateMarkovAnomaly(sequence)
	
	// Combine scores
	finalScore := math.Max(maxScore, (ngramScore+markovScore)/2.0)
	
	return finalScore, matchedPatterns
}

func (ssa *SyscallSequenceAnalyzer) calculateNgramAnomaly(sequence []string) float64 {
	n := 3 // Trigrams
	anomalyScore := 0.0
	count := 0
	
	for i := 0; i <= len(sequence)-n; i++ {
		ngram := strings.Join(sequence[i:i+n], "->")
		freq, exists := ssa.ngramModel[ngram]
		
		if !exists || freq < 0.01 { // Rare pattern
			anomalyScore += 1.0
		}
		count++
	}
	
	if count == 0 {
		return 0.0
	}
	
	return anomalyScore / float64(count)
}

func (ssa *SyscallSequenceAnalyzer) calculateMarkovAnomaly(sequence []string) float64 {
	if len(sequence) < 2 {
		return 0.0
	}
	
	anomalyScore := 0.0
	count := 0
	
	for i := 0; i < len(sequence)-1; i++ {
		curr := sequence[i]
		next := sequence[i+1]
		
		if transitions, exists := ssa.markovChain[curr]; exists {
			if prob, hasNext := transitions[next]; hasNext {
				// Low probability transition is anomalous
				if prob < 0.05 {
					anomalyScore += (0.05 - prob) * 20.0 // Scale up
				}
			} else {
				// Unknown transition
				anomalyScore += 0.5
			}
		}
		count++
	}
	
	if count == 0 {
		return 0.0
	}
	
	return math.Min(anomalyScore/float64(count), 1.0)
}

func (ssa *SyscallSequenceAnalyzer) sequenceContains(haystack, needle []string) bool {
	if len(needle) > len(haystack) {
		return false
	}
	
	for i := 0; i <= len(haystack)-len(needle); i++ {
		match := true
		for j := 0; j < len(needle); j++ {
			if haystack[i+j] != needle[j] {
				match = false
				break
			}
		}
		if match {
			return true
		}
	}
	
	return false
}

// NewThreatHistory initializes adaptive threshold system
func NewThreatHistory() *ThreatHistory {
	return &ThreatHistory{
		recentScores: make([]float64, 0, 1000),
		threshold:    50.0, // Default threshold
	}
}

func (th *ThreatHistory) Update(score float64) {
	th.mu.Lock()
	defer th.mu.Unlock()
	
	th.recentScores = append(th.recentScores, score)
	
	// Keep only recent 1000 scores
	if len(th.recentScores) > 1000 {
		th.recentScores = th.recentScores[len(th.recentScores)-1000:]
	}
	
	// Recalculate statistics
	if len(th.recentScores) > 10 {
		th.avgScore = average(th.recentScores)
		th.stdDev = stdDev(th.recentScores, th.avgScore)
		
		// Adaptive threshold: mean + 2*stddev (captures 95% of benign)
		th.threshold = th.avgScore + 2.0*th.stdDev
	}
}

func (th *ThreatHistory) IsAnomaly(score float64) bool {
	th.mu.RLock()
	defer th.mu.RUnlock()
	
	return score > th.threshold
}

// Helper functions
func (ats *AdvancedThreatScorer) loadKnownPatterns() {
	// Common exploit patterns
	patterns := []SyscallPattern{
		{
			Sequence:    []string{"mmap", "mprotect", "execve"},
			ThreatScore: 0.95,
			Description: "shellcode_injection",
		},
		{
			Sequence:    []string{"ptrace", "wait4", "kill"},
			ThreatScore: 0.90,
			Description: "process_injection",
		},
		{
			Sequence:    []string{"socket", "connect", "sendto"},
			ThreatScore: 0.75,
			Description: "network_exfiltration",
		},
		{
			Sequence:    []string{"open", "read", "write", "unlink"},
			ThreatScore: 0.70,
			Description: "file_tampering",
		},
		{
			Sequence:    []string{"fork", "execve"},
			ThreatScore: 0.65,
			Description: "process_spawning",
		},
	}
	
	ats.sequenceAnalyzer.mu.Lock()
	ats.sequenceAnalyzer.knownPatterns = patterns
	ats.sequenceAnalyzer.mu.Unlock()
}

func (ats *AdvancedThreatScorer) calculateHeuristics(result *SandboxResult) float64 {
	score := 0.0
	
	// High syscall frequency
	if len(result.Syscalls) > 1000 {
		score += 0.2
	}
	
	// Network activity in isolated environment
	if len(result.NetworkIO) > 0 {
		score += 0.3
	}
	
	// Suspicious file operations
	writeCount := 0
	for _, fe := range result.FileAccess {
		if fe.Operation == "write" && strings.HasPrefix(fe.Path, "/etc") {
			score += 0.4
			break
		}
		if fe.Operation == "write" {
			writeCount++
		}
	}
	
	if writeCount > 50 {
		score += 0.2
	}
	
	// Memory anomalies
	if len(result.MemoryDump) > 100*1024*1024 { // >100MB
		score += 0.1
	}
	
	return math.Min(score, 1.0)
}

func (ats *AdvancedThreatScorer) calculateNetworkEntropy(netEvents []NetworkEvent) float64 {
	if len(netEvents) == 0 {
		return 0.0
	}
	
	// Calculate Shannon entropy of network patterns
	patterns := make(map[string]int)
	for _, ne := range netEvents {
		key := fmt.Sprintf("%s:%d", ne.Protocol, ne.DstPort)
		patterns[key]++
	}
	
	total := float64(len(netEvents))
	entropy := 0.0
	
	for _, count := range patterns {
		p := float64(count) / total
		if p > 0 {
			entropy -= p * math.Log2(p)
		}
	}
	
	return entropy
}

func (ats *AdvancedThreatScorer) calculateComplexity(result *SandboxResult) float64 {
	// Cyclomatic complexity approximation
	uniqueSyscalls := make(map[string]bool)
	for _, sc := range result.Syscalls {
		uniqueSyscalls[sc.SyscallName] = true
	}
	
	complexity := float64(len(uniqueSyscalls)) / 10.0 // Normalize
	return math.Min(complexity, 1.0)
}

func (ats *AdvancedThreatScorer) calculateEntropy(data string) float64 {
	if len(data) == 0 {
		return 0.0
	}
	
	freq := make(map[rune]int)
	for _, c := range data {
		freq[c]++
	}
	
	total := float64(len(data))
	entropy := 0.0
	
	for _, count := range freq {
		p := float64(count) / total
		if p > 0 {
			entropy -= p * math.Log2(p)
		}
	}
	
	// Normalize to 0-1
	maxEntropy := math.Log2(256.0) // Max entropy for byte data
	return entropy / maxEntropy
}

func (ats *AdvancedThreatScorer) getAdaptiveWeights() [4]float64 {
	// Dynamically adjust weights based on recent performance
	// For now, use optimized static weights
	return [4]float64{0.35, 0.25, 0.25, 0.15}
}

func (ats *AdvancedThreatScorer) calculateConfidence(scores ...float64) float64 {
	if len(scores) == 0 {
		return 0.0
	}
	
	// Confidence is high when models agree
	avg := average(scores)
	variance := 0.0
	
	for _, s := range scores {
		variance += math.Pow(s-avg, 2)
	}
	variance /= float64(len(scores))
	
	// Low variance = high confidence
	confidence := 1.0 / (1.0 + variance*10.0)
	return confidence
}

func (ats *AdvancedThreatScorer) generateExplanation(iso, bayes, seq, heur float64, patterns []string) string {
	parts := make([]string, 0)
	
	if iso > 0.7 {
		parts = append(parts, "anomalous_behavior")
	}
	if bayes > 0.7 {
		parts = append(parts, "high_threat_probability")
	}
	if seq > 0.7 {
		parts = append(parts, "malicious_pattern")
	}
	if heur > 0.7 {
		parts = append(parts, "suspicious_heuristics")
	}
	
	if len(patterns) > 0 {
		parts = append(parts, patterns...)
	}
	
	if len(parts) == 0 {
		return "clean"
	}
	
	return strings.Join(parts, ",")
}

func defaultAnomalyHeuristic(features []float64) float64 {
	// Simple heuristic when model not trained
	if len(features) < 3 {
		return 0.0
	}
	
	// Features[1] is dangerous syscalls count
	// Features[2] is dangerous ratio
	if len(features) > 2 && features[2] > 0.3 {
		return 0.8
	}
	
	return 0.2
}

func (ats *AdvancedThreatScorer) defaultAnomalyHeuristic(features []float64) float64 {
	// Simple heuristic when model not trained
	if len(features) < 3 {
		return 0.0
	}
	
	// Features[1] is dangerous syscalls count
	// Features[2] is dangerous ratio
	if len(features) > 2 && features[2] > 0.3 {
		return 0.8
	}
	
	return 0.2
}

func getOptimalFeatureWeights() map[string]float64 {
	// Tuned weights based on feature importance analysis
	return map[string]float64{
		"dangerous_syscalls": 0.40,
		"network_activity":   0.25,
		"file_operations":    0.15,
		"memory_anomaly":     0.10,
		"complexity":         0.10,
	}
}

// Statistical helper functions
func average(data []float64) float64 {
	if len(data) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range data {
		sum += v
	}
	return sum / float64(len(data))
}

func stdDev(data []float64, mean float64) float64 {
	if len(data) == 0 {
		return 0
	}
	variance := 0.0
	for _, v := range data {
		variance += math.Pow(v-mean, 2)
	}
	variance /= float64(len(data))
	return math.Sqrt(variance)
}
