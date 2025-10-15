package ml

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"
)

// HyperparameterSpace defines the search space for a hyperparameter
type HyperparameterSpace struct {
	Name     string
	Type     string      // "int", "float", "categorical"
	Min      float64     // For int/float
	Max      float64     // For int/float
	Options  []string    // For categorical
	LogScale bool        // Use log scale for search
}

// HyperparameterConfig represents a set of hyperparameters
type HyperparameterConfig map[string]interface{}

// TrialResult represents the result of a hyperparameter trial
type TrialResult struct {
	Config HyperparameterConfig
	Score  float64 // Higher is better
	Error  error
}

// ObjectiveFunc is the function to optimize
type ObjectiveFunc func(config HyperparameterConfig) (float64, error)

// SearchStrategy defines the hyperparameter search strategy
type SearchStrategy string

const (
	RandomSearch   SearchStrategy = "random"
	GridSearch     SearchStrategy = "grid"
	BayesianSearch SearchStrategy = "bayesian"
)

// AutoMLEngine performs automated hyperparameter optimization
type AutoMLEngine struct {
	mu sync.RWMutex

	searchSpace []HyperparameterSpace
	strategy    SearchStrategy
	maxTrials   int
	timeout     time.Duration
	
	// Results tracking
	trials      []TrialResult
	bestConfig  HyperparameterConfig
	bestScore   float64
	
	// Bayesian optimization state
	observations []HyperparameterConfig
	scores       []float64
	
	// Parallel execution
	nWorkers int
}

// AutoMLConfig configures the AutoML engine
type AutoMLConfig struct {
	Strategy  SearchStrategy
	MaxTrials int
	Timeout   time.Duration
	NWorkers  int
}

// NewAutoMLEngine creates a new AutoML engine
func NewAutoMLEngine(searchSpace []HyperparameterSpace, config AutoMLConfig) *AutoMLEngine {
	if config.MaxTrials <= 0 {
		config.MaxTrials = 100
	}
	if config.Timeout <= 0 {
		config.Timeout = 1 * time.Hour
	}
	if config.NWorkers <= 0 {
		config.NWorkers = 4
	}
	if config.Strategy == "" {
		config.Strategy = RandomSearch
	}

	return &AutoMLEngine{
		searchSpace:  searchSpace,
		strategy:     config.Strategy,
		maxTrials:    config.MaxTrials,
		timeout:      config.Timeout,
		nWorkers:     config.NWorkers,
		trials:       make([]TrialResult, 0),
		bestScore:    math.Inf(-1),
		observations: make([]HyperparameterConfig, 0),
		scores:       make([]float64, 0),
	}
}

// Optimize runs the hyperparameter optimization
func (am *AutoMLEngine) Optimize(objective ObjectiveFunc) (*HyperparameterConfig, float64, error) {
	startTime := time.Now()
	
	switch am.strategy {
	case RandomSearch:
		return am.randomSearch(objective, startTime)
	case GridSearch:
		return am.gridSearch(objective, startTime)
	case BayesianSearch:
		return am.bayesianSearch(objective, startTime)
	default:
		return nil, 0, fmt.Errorf("unknown search strategy: %s", am.strategy)
	}
}

// randomSearch performs random hyperparameter search
func (am *AutoMLEngine) randomSearch(objective ObjectiveFunc, startTime time.Time) (*HyperparameterConfig, float64, error) {
	// Create work channel
	work := make(chan HyperparameterConfig, am.nWorkers)
	results := make(chan TrialResult, am.nWorkers)
	
	// Start workers
	var wg sync.WaitGroup
	for i := 0; i < am.nWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for config := range work {
				score, err := objective(config)
				results <- TrialResult{
					Config: config,
					Score:  score,
					Error:  err,
				}
			}
		}()
	}
	
	// Generate and send work
	go func() {
		for i := 0; i < am.maxTrials; i++ {
			if time.Since(startTime) > am.timeout {
				break
			}
			config := am.sampleRandomConfig()
			work <- config
		}
		close(work)
	}()
	
	// Collect results
	go func() {
		wg.Wait()
		close(results)
	}()
	
	// Process results
	for result := range results {
		am.recordTrial(result)
	}
	
	if len(am.trials) == 0 {
		return nil, 0, fmt.Errorf("no successful trials")
	}
	
	return &am.bestConfig, am.bestScore, nil
}

// gridSearch performs grid search over hyperparameters
func (am *AutoMLEngine) gridSearch(objective ObjectiveFunc, startTime time.Time) (*HyperparameterConfig, float64, error) {
	// Generate all grid combinations
	configs := am.generateGridConfigs()
	
	if len(configs) > am.maxTrials {
		configs = configs[:am.maxTrials]
	}
	
	// Parallel evaluation
	work := make(chan HyperparameterConfig, am.nWorkers)
	results := make(chan TrialResult, am.nWorkers)
	
	var wg sync.WaitGroup
	for i := 0; i < am.nWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for config := range work {
				score, err := objective(config)
				results <- TrialResult{
					Config: config,
					Score:  score,
					Error:  err,
				}
			}
		}()
	}
	
	// Send work
	go func() {
		for _, config := range configs {
			if time.Since(startTime) > am.timeout {
				break
			}
			work <- config
		}
		close(work)
	}()
	
	// Collect results
	go func() {
		wg.Wait()
		close(results)
	}()
	
	for result := range results {
		am.recordTrial(result)
	}
	
	if len(am.trials) == 0 {
		return nil, 0, fmt.Errorf("no successful trials")
	}
	
	return &am.bestConfig, am.bestScore, nil
}

// bayesianSearch performs Bayesian optimization (simplified)
func (am *AutoMLEngine) bayesianSearch(objective ObjectiveFunc, startTime time.Time) (*HyperparameterConfig, float64, error) {
	// Start with random exploration
	nRandom := min(10, am.maxTrials/4)
	
	for i := 0; i < nRandom; i++ {
		if time.Since(startTime) > am.timeout {
			break
		}
		
		config := am.sampleRandomConfig()
		score, err := objective(config)
		
		result := TrialResult{
			Config: config,
			Score:  score,
			Error:  err,
		}
		am.recordTrial(result)
		
		if err == nil {
			am.observations = append(am.observations, config)
			am.scores = append(am.scores, score)
		}
	}
	
	// Exploitation phase
	for i := nRandom; i < am.maxTrials; i++ {
		if time.Since(startTime) > am.timeout {
			break
		}
		
		// Sample next config using Expected Improvement
		config := am.sampleNextBayesian()
		score, err := objective(config)
		
		result := TrialResult{
			Config: config,
			Score:  score,
			Error:  err,
		}
		am.recordTrial(result)
		
		if err == nil {
			am.observations = append(am.observations, config)
			am.scores = append(am.scores, score)
		}
	}
	
	if len(am.trials) == 0 {
		return nil, 0, fmt.Errorf("no successful trials")
	}
	
	return &am.bestConfig, am.bestScore, nil
}

// sampleRandomConfig generates a random configuration
func (am *AutoMLEngine) sampleRandomConfig() HyperparameterConfig {
	config := make(HyperparameterConfig)
	
	for _, space := range am.searchSpace {
		switch space.Type {
		case "int":
			value := rand.Intn(int(space.Max-space.Min+1)) + int(space.Min)
			config[space.Name] = value
			
		case "float":
			var value float64
			if space.LogScale {
				logMin := math.Log(space.Min)
				logMax := math.Log(space.Max)
				logValue := logMin + rand.Float64()*(logMax-logMin)
				value = math.Exp(logValue)
			} else {
				value = space.Min + rand.Float64()*(space.Max-space.Min)
			}
			config[space.Name] = value
			
		case "categorical":
			idx := rand.Intn(len(space.Options))
			config[space.Name] = space.Options[idx]
		}
	}
	
	return config
}

// generateGridConfigs generates all grid search configurations
func (am *AutoMLEngine) generateGridConfigs() []HyperparameterConfig {
	// Simplified: generate 10 values per dimension
	gridSize := 10
	
	var generate func(idx int, current HyperparameterConfig) []HyperparameterConfig
	generate = func(idx int, current HyperparameterConfig) []HyperparameterConfig {
		if idx >= len(am.searchSpace) {
			result := make(HyperparameterConfig)
			for k, v := range current {
				result[k] = v
			}
			return []HyperparameterConfig{result}
		}
		
		space := am.searchSpace[idx]
		configs := make([]HyperparameterConfig, 0)
		
		switch space.Type {
		case "int":
			step := math.Max(1, (space.Max-space.Min)/float64(gridSize))
			for v := space.Min; v <= space.Max; v += step {
				current[space.Name] = int(v)
				configs = append(configs, generate(idx+1, current)...)
			}
			
		case "float":
			step := (space.Max - space.Min) / float64(gridSize)
			for v := space.Min; v <= space.Max; v += step {
				current[space.Name] = v
				configs = append(configs, generate(idx+1, current)...)
			}
			
		case "categorical":
			for _, option := range space.Options {
				current[space.Name] = option
				configs = append(configs, generate(idx+1, current)...)
			}
		}
		
		return configs
	}
	
	return generate(0, make(HyperparameterConfig))
}

// sampleNextBayesian samples next configuration using simplified acquisition function
func (am *AutoMLEngine) sampleNextBayesian() HyperparameterConfig {
	// Simplified: use Expected Improvement heuristic
	// Generate candidates and pick best based on distance to good observations
	
	nCandidates := 100
	bestEI := math.Inf(-1)
	var bestConfig HyperparameterConfig
	
	for i := 0; i < nCandidates; i++ {
		candidate := am.sampleRandomConfig()
		ei := am.expectedImprovement(candidate)
		
		if ei > bestEI {
			bestEI = ei
			bestConfig = candidate
		}
	}
	
	return bestConfig
}

// expectedImprovement computes simplified expected improvement
func (am *AutoMLEngine) expectedImprovement(config HyperparameterConfig) float64 {
	if len(am.observations) == 0 {
		return 1.0
	}
	
	// Find best score so far
	bestScore := math.Inf(-1)
	for _, score := range am.scores {
		if score > bestScore {
			bestScore = score
		}
	}
	
	// Compute distance to nearest good observation
	minDist := math.Inf(1)
	for i, obs := range am.observations {
		if am.scores[i] > bestScore*0.9 { // Consider top 10%
			dist := am.configDistance(config, obs)
			if dist < minDist {
				minDist = dist
			}
		}
	}
	
	// EI heuristic: balance between distance (exploration) and nearness to good points
	ei := 1.0 / (1.0 + minDist)
	return ei
}

// configDistance computes distance between configurations
func (am *AutoMLEngine) configDistance(c1, c2 HyperparameterConfig) float64 {
	dist := 0.0
	count := 0
	
	for _, space := range am.searchSpace {
		v1, ok1 := c1[space.Name]
		v2, ok2 := c2[space.Name]
		
		if !ok1 || !ok2 {
			continue
		}
		
		switch space.Type {
		case "int":
			d := math.Abs(float64(v1.(int) - v2.(int)))
			dist += d / (space.Max - space.Min)
			count++
			
		case "float":
			d := math.Abs(v1.(float64) - v2.(float64))
			dist += d / (space.Max - space.Min)
			count++
			
		case "categorical":
			if v1.(string) != v2.(string) {
				dist += 1.0
			}
			count++
		}
	}
	
	if count == 0 {
		return 0
	}
	
	return dist / float64(count)
}

// recordTrial records a trial result
func (am *AutoMLEngine) recordTrial(result TrialResult) {
	am.mu.Lock()
	defer am.mu.Unlock()
	
	am.trials = append(am.trials, result)
	
	if result.Error == nil && result.Score > am.bestScore {
		am.bestScore = result.Score
		am.bestConfig = result.Config
	}
}

// GetBestConfig returns the best configuration found
func (am *AutoMLEngine) GetBestConfig() (HyperparameterConfig, float64) {
	am.mu.RLock()
	defer am.mu.RUnlock()
	
	return am.bestConfig, am.bestScore
}

// GetTrials returns all trial results
func (am *AutoMLEngine) GetTrials() []TrialResult {
	am.mu.RLock()
	defer am.mu.RUnlock()
	
	trials := make([]TrialResult, len(am.trials))
	copy(trials, am.trials)
	return trials
}

// GetStats returns optimization statistics
func (am *AutoMLEngine) GetStats() map[string]interface{} {
	am.mu.RLock()
	defer am.mu.RUnlock()
	
	successful := 0
	for _, trial := range am.trials {
		if trial.Error == nil {
			successful++
		}
	}
	
	return map[string]interface{}{
		"total_trials":      len(am.trials),
		"successful_trials": successful,
		"best_score":        am.bestScore,
		"strategy":          am.strategy,
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
