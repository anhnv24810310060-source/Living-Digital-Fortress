package ml

import (
	"fmt"
	"sync"
)

// AnomalyModel interface that all anomaly detectors must implement
type AnomalyModel interface {
	Train(data [][]float64) error
	Detect(point []float64) (bool, float64)
	Algorithm() string
}

// EnsembleStrategy defines the ensemble combination method
type EnsembleStrategy string

const (
	// VotingMajority uses majority voting (most models agree)
	VotingMajority EnsembleStrategy = "voting_majority"

	// VotingWeighted uses weighted voting based on model weights
	VotingWeighted EnsembleStrategy = "voting_weighted"

	// VotingAverage averages the anomaly scores from all models
	VotingAverage EnsembleStrategy = "voting_average"

	// VotingMax uses the maximum anomaly score
	VotingMax EnsembleStrategy = "voting_max"

	// VotingMin uses the minimum anomaly score
	VotingMin EnsembleStrategy = "voting_min"
)

// EnsembleDetector combines multiple anomaly detection models
type EnsembleDetector struct {
	mu       sync.RWMutex
	models   []AnomalyModel
	weights  []float64
	strategy EnsembleStrategy
	trained  bool
}

// NewEnsembleDetector creates a new ensemble detector
func NewEnsembleDetector(strategy EnsembleStrategy) *EnsembleDetector {
	return &EnsembleDetector{
		models:   make([]AnomalyModel, 0),
		weights:  make([]float64, 0),
		strategy: strategy,
	}
}

// AddModel adds a model to the ensemble with optional weight
func (e *EnsembleDetector) AddModel(model AnomalyModel, weight float64) error {
	if model == nil {
		return fmt.Errorf("model cannot be nil")
	}
	if weight <= 0 {
		return fmt.Errorf("weight must be positive, got %f", weight)
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	e.models = append(e.models, model)
	e.weights = append(e.weights, weight)

	return nil
}

// Train trains all models in the ensemble
func (e *EnsembleDetector) Train(data [][]float64) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if len(e.models) == 0 {
		return fmt.Errorf("no models in ensemble")
	}

	// Train each model
	for i, model := range e.models {
		if err := model.Train(data); err != nil {
			return fmt.Errorf("failed to train model %d (%s): %w", i, model.Algorithm(), err)
		}
	}

	e.trained = true
	return nil
}

// Detect performs anomaly detection using the ensemble strategy
func (e *EnsembleDetector) Detect(point []float64) (bool, float64) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if !e.trained || len(e.models) == 0 {
		return false, 0.0
	}

	// Collect predictions from all models
	predictions := make([]bool, len(e.models))
	scores := make([]float64, len(e.models))

	for i, model := range e.models {
		predictions[i], scores[i] = model.Detect(point)
	}

	// Apply ensemble strategy
	switch e.strategy {
	case VotingMajority:
		return e.majorityVoting(predictions, scores)
	case VotingWeighted:
		return e.weightedVoting(predictions, scores)
	case VotingAverage:
		return e.averageVoting(predictions, scores)
	case VotingMax:
		return e.maxVoting(predictions, scores)
	case VotingMin:
		return e.minVoting(predictions, scores)
	default:
		return e.majorityVoting(predictions, scores)
	}
}

// Algorithm returns the algorithm name
func (e *EnsembleDetector) Algorithm() string {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if len(e.models) == 0 {
		return "ensemble-empty"
	}

	return fmt.Sprintf("ensemble-%s-%d-models", e.strategy, len(e.models))
}

// GetModels returns the list of models in the ensemble
func (e *EnsembleDetector) GetModels() []string {
	e.mu.RLock()
	defer e.mu.RUnlock()

	algorithms := make([]string, len(e.models))
	for i, model := range e.models {
		algorithms[i] = model.Algorithm()
	}
	return algorithms
}

// majorityVoting returns true if majority of models detect anomaly
func (e *EnsembleDetector) majorityVoting(predictions []bool, scores []float64) (bool, float64) {
	anomalyCount := 0
	totalScore := 0.0

	for i, pred := range predictions {
		if pred {
			anomalyCount++
		}
		totalScore += scores[i]
	}

	avgScore := totalScore / float64(len(scores))
	isAnomaly := anomalyCount > len(predictions)/2

	return isAnomaly, avgScore
}

// weightedVoting uses weighted voting based on model weights
func (e *EnsembleDetector) weightedVoting(predictions []bool, scores []float64) (bool, float64) {
	weightedAnomalyScore := 0.0
	weightedNormalScore := 0.0
	totalWeight := 0.0

	for i, pred := range predictions {
		weight := e.weights[i]
		totalWeight += weight

		if pred {
			weightedAnomalyScore += weight * scores[i]
		} else {
			weightedNormalScore += weight * (1.0 - scores[i])
		}
	}

	// Normalize
	avgScore := weightedAnomalyScore / totalWeight
	isAnomaly := weightedAnomalyScore > weightedNormalScore

	return isAnomaly, avgScore
}

// averageVoting averages the scores and uses threshold
func (e *EnsembleDetector) averageVoting(predictions []bool, scores []float64) (bool, float64) {
	totalScore := 0.0
	for _, score := range scores {
		totalScore += score
	}

	avgScore := totalScore / float64(len(scores))
	
	// Use threshold of 0.5 for average score
	isAnomaly := avgScore > 0.5

	return isAnomaly, avgScore
}

// maxVoting uses the maximum score among all models
func (e *EnsembleDetector) maxVoting(predictions []bool, scores []float64) (bool, float64) {
	maxScore := 0.0
	hasAnomaly := false

	for i, score := range scores {
		if score > maxScore {
			maxScore = score
		}
		if predictions[i] {
			hasAnomaly = true
		}
	}

	return hasAnomaly, maxScore
}

// minVoting uses the minimum score (most conservative)
func (e *EnsembleDetector) minVoting(predictions []bool, scores []float64) (bool, float64) {
	minScore := scores[0]
	allAnomaly := true

	for i, score := range scores {
		if score < minScore {
			minScore = score
		}
		if !predictions[i] {
			allAnomaly = false
		}
	}

	// Only flag as anomaly if all models agree
	return allAnomaly, minScore
}

// SetStrategy changes the ensemble strategy
func (e *EnsembleDetector) SetStrategy(strategy EnsembleStrategy) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.strategy = strategy
}

// GetStrategy returns the current ensemble strategy
func (e *EnsembleDetector) GetStrategy() EnsembleStrategy {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.strategy
}
