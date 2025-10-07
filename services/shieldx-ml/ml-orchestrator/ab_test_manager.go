package main

import (
	"crypto/rand"
	"hash/fnv"
	"sync"
	"sync/atomic"
)

// ABTestManager manages A/B testing for ML models
type ABTestManager struct {
	mu            sync.RWMutex
	experiments   map[string]*Experiment
	activeTest    string
	trafficSplit  int // Percentage to variant B (0-100)
	variantACount atomic.Uint64
	variantBCount atomic.Uint64
}

type Experiment struct {
	ID             string            `json:"id"`
	Name           string            `json:"name"`
	Description    string            `json:"description"`
	VariantA       VariantConfig     `json:"variant_a"`
	VariantB       VariantConfig     `json:"variant_b"`
	TrafficSplit   int               `json:"traffic_split"` // % to B
	Status         string            `json:"status"`        // active, paused, completed
	Results        *ExperimentResult `json:"results,omitempty"`
	StickySession  bool              `json:"sticky_session"` // Same user always gets same variant
	sessionMapping map[string]string // sessionID -> variant
}

type VariantConfig struct {
	ModelVersion    string             `json:"model_version"`
	EnsembleWeight  float64            `json:"ensemble_weight"`
	Threshold       float64            `json:"threshold"`
	Hyperparameters map[string]float64 `json:"hyperparameters,omitempty"`
}

type ExperimentResult struct {
	VariantAStats VariantStats `json:"variant_a_stats"`
	VariantBStats VariantStats `json:"variant_b_stats"`
	Winner        string       `json:"winner,omitempty"`
	Confidence    float64      `json:"confidence,omitempty"`
	TotalSamples  int          `json:"total_samples"`
}

type VariantStats struct {
	Requests      int64   `json:"requests"`
	TruePositives int64   `json:"true_positives"`
	FalsePositives int64  `json:"false_positives"`
	TrueNegatives int64   `json:"true_negatives"`
	FalseNegatives int64  `json:"false_negatives"`
	AvgLatency    float64 `json:"avg_latency_ms"`
	Accuracy      float64 `json:"accuracy"`
	Precision     float64 `json:"precision"`
	Recall        float64 `json:"recall"`
	F1Score       float64 `json:"f1_score"`
}

func NewABTestManager() *ABTestManager {
	return &ABTestManager{
		experiments:  make(map[string]*Experiment),
		trafficSplit: 0, // Default: all traffic to A
	}
}

// CreateExperiment creates a new A/B test experiment
func (ab *ABTestManager) CreateExperiment(exp *Experiment) error {
	ab.mu.Lock()
	defer ab.mu.Unlock()

	if exp.ID == "" {
		exp.ID = generateExperimentID()
	}

	if exp.sessionMapping == nil {
		exp.sessionMapping = make(map[string]string)
	}

	exp.Status = "active"
	exp.Results = &ExperimentResult{}

	ab.experiments[exp.ID] = exp
	ab.activeTest = exp.ID
	ab.trafficSplit = exp.TrafficSplit

	return nil
}

// GetVariant determines which variant to use for a request
func (ab *ABTestManager) GetVariant(sessionID string) (string, *VariantConfig) {
	ab.mu.RLock()
	defer ab.mu.RUnlock()

	if ab.activeTest == "" {
		return "A", nil // Default to A if no active test
	}

	exp, exists := ab.experiments[ab.activeTest]
	if !exists || exp.Status != "active" {
		return "A", nil
	}

	// Check sticky session mapping
	if exp.StickySession && sessionID != "" {
		if variant, ok := exp.sessionMapping[sessionID]; ok {
			if variant == "B" {
				return "B", &exp.VariantB
			}
			return "A", &exp.VariantA
		}
	}

	// Determine variant based on traffic split
	variant := ab.selectVariant(sessionID, exp.TrafficSplit)

	// Store sticky mapping
	if exp.StickySession && sessionID != "" {
		exp.sessionMapping[sessionID] = variant
	}

	if variant == "B" {
		ab.variantBCount.Add(1)
		return "B", &exp.VariantB
	}

	ab.variantACount.Add(1)
	return "A", &exp.VariantA
}

// selectVariant uses consistent hashing for stable variant assignment
func (ab *ABTestManager) selectVariant(sessionID string, splitPercent int) string {
	if splitPercent <= 0 {
		return "A"
	}
	if splitPercent >= 100 {
		return "B"
	}

	// Use hash of sessionID for consistent assignment
	var hash uint32
	if sessionID != "" {
		h := fnv.New32a()
		h.Write([]byte(sessionID))
		hash = h.Sum32()
	} else {
		// Random assignment if no session
		var b [4]byte
		rand.Read(b[:])
		hash = uint32(b[0])<<24 | uint32(b[1])<<16 | uint32(b[2])<<8 | uint32(b[3])
	}

	// Map hash to 0-100 range
	bucket := hash % 100

	if int(bucket) < splitPercent {
		return "B"
	}
	return "A"
}

// RecordResult records the result of a variant execution
func (ab *ABTestManager) RecordResult(variant string, latencyMs float64, predicted, actual bool) {
	ab.mu.Lock()
	defer ab.mu.Unlock()

	if ab.activeTest == "" {
		return
	}

	exp, exists := ab.experiments[ab.activeTest]
	if !exists || exp.Status != "active" {
		return
	}

	var stats *VariantStats
	if variant == "B" {
		stats = &exp.Results.VariantBStats
	} else {
		stats = &exp.Results.VariantAStats
	}

	stats.Requests++

	// Update confusion matrix
	if predicted && actual {
		stats.TruePositives++
	} else if predicted && !actual {
		stats.FalsePositives++
	} else if !predicted && actual {
		stats.FalseNegatives++
	} else {
		stats.TrueNegatives++
	}

	// Update latency (running average)
	if stats.Requests == 1 {
		stats.AvgLatency = latencyMs
	} else {
		stats.AvgLatency = (stats.AvgLatency*float64(stats.Requests-1) + latencyMs) / float64(stats.Requests)
	}

	// Recalculate metrics
	ab.calculateMetrics(stats)
	exp.Results.TotalSamples = int(exp.Results.VariantAStats.Requests + exp.Results.VariantBStats.Requests)
}

// calculateMetrics calculates accuracy, precision, recall, F1
func (ab *ABTestManager) calculateMetrics(stats *VariantStats) {
	total := stats.TruePositives + stats.FalsePositives + stats.TrueNegatives + stats.FalseNegatives
	if total == 0 {
		return
	}

	// Accuracy
	stats.Accuracy = float64(stats.TruePositives+stats.TrueNegatives) / float64(total)

	// Precision
	if stats.TruePositives+stats.FalsePositives > 0 {
		stats.Precision = float64(stats.TruePositives) / float64(stats.TruePositives+stats.FalsePositives)
	}

	// Recall
	if stats.TruePositives+stats.FalseNegatives > 0 {
		stats.Recall = float64(stats.TruePositives) / float64(stats.TruePositives+stats.FalseNegatives)
	}

	// F1 Score
	if stats.Precision+stats.Recall > 0 {
		stats.F1Score = 2 * (stats.Precision * stats.Recall) / (stats.Precision + stats.Recall)
	}
}

// GetResults returns current experiment results
func (ab *ABTestManager) GetResults(experimentID string) (*ExperimentResult, error) {
	ab.mu.RLock()
	defer ab.mu.RUnlock()

	exp, exists := ab.experiments[experimentID]
	if !exists {
		return nil, nil
	}

	return exp.Results, nil
}

// DetermineWinner analyzes results and declares a winner
func (ab *ABTestManager) DetermineWinner(experimentID string, minSamples int) (string, float64, error) {
	ab.mu.Lock()
	defer ab.mu.Unlock()

	exp, exists := ab.experiments[experimentID]
	if !exists {
		return "", 0, nil
	}

	if exp.Results.TotalSamples < minSamples {
		return "", 0, nil // Not enough data
	}

	statsA := exp.Results.VariantAStats
	statsB := exp.Results.VariantBStats

	// Compare F1 scores (balanced metric)
	if statsA.F1Score > statsB.F1Score*1.02 { // 2% improvement threshold
		exp.Results.Winner = "A"
		exp.Results.Confidence = calculateConfidence(statsA, statsB)
		return "A", exp.Results.Confidence, nil
	} else if statsB.F1Score > statsA.F1Score*1.02 {
		exp.Results.Winner = "B"
		exp.Results.Confidence = calculateConfidence(statsB, statsA)
		return "B", exp.Results.Confidence, nil
	}

	// No clear winner
	exp.Results.Winner = "TIE"
	exp.Results.Confidence = 0.5
	return "TIE", 0.5, nil
}

// StopExperiment stops an active experiment
func (ab *ABTestManager) StopExperiment(experimentID string) error {
	ab.mu.Lock()
	defer ab.mu.Unlock()

	exp, exists := ab.experiments[experimentID]
	if !exists {
		return nil
	}

	exp.Status = "completed"

	if ab.activeTest == experimentID {
		ab.activeTest = ""
		ab.trafficSplit = 0
	}

	return nil
}

// GetActiveExperiment returns the currently active experiment
func (ab *ABTestManager) GetActiveExperiment() *Experiment {
	ab.mu.RLock()
	defer ab.mu.RUnlock()

	if ab.activeTest == "" {
		return nil
	}

	return ab.experiments[ab.activeTest]
}

// GetCounts returns variant selection counts
func (ab *ABTestManager) GetCounts() (uint64, uint64) {
	return ab.variantACount.Load(), ab.variantBCount.Load()
}

// calculateConfidence calculates statistical confidence (simplified)
func calculateConfidence(winner, loser VariantStats) float64 {
	// Simplified confidence based on sample size and delta
	sampleSize := float64(winner.Requests + loser.Requests)
	delta := winner.F1Score - loser.F1Score

	// More samples and larger delta = higher confidence
	confidence := 0.5 + (delta * 50) + (sampleSize / 10000 * 0.2)

	if confidence > 0.99 {
		confidence = 0.99
	}
	if confidence < 0.5 {
		confidence = 0.5
	}

	return confidence
}

// generateExperimentID generates a unique experiment ID
func generateExperimentID() string {
	b := make([]byte, 8)
	rand.Read(b)
	return fnvHash(b)
}

func fnvHash(data []byte) string {
	h := fnv.New64a()
	h.Write(data)
	return string(h.Sum(nil))
}
