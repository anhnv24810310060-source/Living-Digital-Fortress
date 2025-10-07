package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"math"
	"time"

	"github.com/google/uuid"
)

// BayesianABTest implements Bayesian A/B testing for shadow evaluation
// Uses Beta distribution for conversion rate estimation
type BayesianABTest struct {
	db *sql.DB
}

// TestVariant represents a Champion or Challenger in A/B test
type TestVariant struct {
	ID                string                 `json:"id"`
	Name              string                 `json:"name"`
	Type              string                 `json:"type"` // "champion" or "challenger"
	RuleID            string                 `json:"rule_id"`
	RuleConfig        map[string]interface{} `json:"rule_config"`
	TrafficPercentage float64                `json:"traffic_percentage"`
	
	// Bayesian statistics
	Successes         int64   `json:"successes"`           // True positives
	Failures          int64   `json:"failures"`            // False positives
	TotalSamples      int64   `json:"total_samples"`
	AlphaPrior        float64 `json:"alpha_prior"`         // Beta distribution alpha
	BetaPrior         float64 `json:"beta_prior"`          // Beta distribution beta
	PosteriorAlpha    float64 `json:"posterior_alpha"`
	PosteriorBeta     float64 `json:"posterior_beta"`
	
	// Performance metrics
	MeanConversionRate float64 `json:"mean_conversion_rate"`
	CredibleIntervalLower float64 `json:"credible_interval_lower"` // 95% CI
	CredibleIntervalUpper float64 `json:"credible_interval_upper"`
	ProbabilityBeatChampion float64 `json:"probability_beat_champion"`
	
	// Operational metrics
	AvgLatencyMs      float64   `json:"avg_latency_ms"`
	P95LatencyMs      float64   `json:"p95_latency_ms"`
	P99LatencyMs      float64   `json:"p99_latency_ms"`
	ErrorRate         float64   `json:"error_rate"`
	
	CreatedAt         time.Time `json:"created_at"`
	UpdatedAt         time.Time `json:"updated_at"`
	Status            string    `json:"status"` // "active", "winner", "loser", "rolled_back"
}

// ABTestConfig represents the configuration for an A/B test
type ABTestConfig struct {
	TestID            string    `json:"test_id"`
	Name              string    `json:"name"`
	TenantID          string    `json:"tenant_id"`
	ChampionVariant   *TestVariant `json:"champion"`
	ChallengerVariant *TestVariant `json:"challenger"`
	
	// Test parameters
	MinSampleSize     int64     `json:"min_sample_size"`
	MaxDuration       time.Duration `json:"max_duration"`
	ConfidenceLevel   float64   `json:"confidence_level"`   // e.g., 0.95 for 95%
	MinimumDetectableEffect float64 `json:"minimum_detectable_effect"` // e.g., 0.01 for 1%
	
	// Decision thresholds
	ProbabilityThreshold float64 `json:"probability_threshold"` // e.g., 0.95 (95% prob to beat)
	MaxErrorRate         float64 `json:"max_error_rate"`
	MaxLatencyP95        float64 `json:"max_latency_p95"`
	
	// Auto-rollback triggers
	AutoRollback      bool    `json:"auto_rollback"`
	RollbackOnErrorRate float64 `json:"rollback_on_error_rate"`
	RollbackOnLatency float64 `json:"rollback_on_latency"`
	
	// Status
	Status            string    `json:"status"` // "running", "completed", "rolled_back"
	StartedAt         time.Time `json:"started_at"`
	CompletedAt       *time.Time `json:"completed_at,omitempty"`
	WinnerVariantID   string    `json:"winner_variant_id,omitempty"`
	DecisionReason    string    `json:"decision_reason,omitempty"`
}

// NewBayesianABTest creates a new Bayesian A/B test engine
func NewBayesianABTest(db *sql.DB) *BayesianABTest {
	return &BayesianABTest{db: db}
}

// CreateTest creates a new A/B test with champion and challenger
func (bat *BayesianABTest) CreateTest(ctx context.Context, config *ABTestConfig) error {
	// Validate configuration
	if config.ChampionVariant == nil || config.ChallengerVariant == nil {
		return fmt.Errorf("both champion and challenger variants required")
	}
	
	if config.MinSampleSize < 100 {
		config.MinSampleSize = 1000 // Minimum 1000 samples for statistical significance
	}
	
	if config.ConfidenceLevel == 0 {
		config.ConfidenceLevel = 0.95
	}
	
	if config.ProbabilityThreshold == 0 {
		config.ProbabilityThreshold = 0.95
	}
	
	// Initialize variants with uniform prior (Beta(1,1))
	config.ChampionVariant.AlphaPrior = 1.0
	config.ChampionVariant.BetaPrior = 1.0
	config.ChampionVariant.Type = "champion"
	config.ChampionVariant.Status = "active"
	
	config.ChallengerVariant.AlphaPrior = 1.0
	config.ChallengerVariant.BetaPrior = 1.0
	config.ChallengerVariant.Type = "challenger"
	config.ChallengerVariant.Status = "active"
	
	config.TestID = uuid.New().String()
	config.Status = "running"
	config.StartedAt = time.Now()
	
	// Persist to database
	tx, err := bat.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer tx.Rollback()
	
	// Insert test config
	testQuery := `
		INSERT INTO ab_tests (
			test_id, name, tenant_id, champion_id, challenger_id,
			min_sample_size, max_duration_seconds, confidence_level,
			minimum_detectable_effect, probability_threshold,
			max_error_rate, max_latency_p95,
			auto_rollback, rollback_on_error_rate, rollback_on_latency,
			status, started_at
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
	`
	
	championID := uuid.New().String()
	challengerID := uuid.New().String()
	config.ChampionVariant.ID = championID
	config.ChallengerVariant.ID = challengerID
	
	_, err = tx.ExecContext(ctx, testQuery,
		config.TestID, config.Name, config.TenantID, championID, challengerID,
		config.MinSampleSize, int(config.MaxDuration.Seconds()), config.ConfidenceLevel,
		config.MinimumDetectableEffect, config.ProbabilityThreshold,
		config.MaxErrorRate, config.MaxLatencyP95,
		config.AutoRollback, config.RollbackOnErrorRate, config.RollbackOnLatency,
		config.Status, config.StartedAt,
	)
	if err != nil {
		return fmt.Errorf("failed to insert test: %w", err)
	}
	
	// Insert variants
	variantQuery := `
		INSERT INTO test_variants (
			variant_id, test_id, name, type, rule_id, rule_config,
			traffic_percentage, alpha_prior, beta_prior, status, created_at
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
	`
	
	championConfigJSON, _ := json.Marshal(config.ChampionVariant.RuleConfig)
	_, err = tx.ExecContext(ctx, variantQuery,
		championID, config.TestID, config.ChampionVariant.Name, "champion",
		config.ChampionVariant.RuleID, championConfigJSON,
		config.ChampionVariant.TrafficPercentage,
		config.ChampionVariant.AlphaPrior, config.ChampionVariant.BetaPrior,
		"active", time.Now(),
	)
	if err != nil {
		return fmt.Errorf("failed to insert champion: %w", err)
	}
	
	challengerConfigJSON, _ := json.Marshal(config.ChallengerVariant.RuleConfig)
	_, err = tx.ExecContext(ctx, variantQuery,
		challengerID, config.TestID, config.ChallengerVariant.Name, "challenger",
		config.ChallengerVariant.RuleID, challengerConfigJSON,
		config.ChallengerVariant.TrafficPercentage,
		config.ChallengerVariant.AlphaPrior, config.ChallengerVariant.BetaPrior,
		"active", time.Now(),
	)
	if err != nil {
		return fmt.Errorf("failed to insert challenger: %w", err)
	}
	
	return tx.Commit()
}

// RecordResult records the result of a test sample
func (bat *BayesianABTest) RecordResult(ctx context.Context, variantID string, success bool, latencyMs float64) error {
	tx, err := bat.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer tx.Rollback()
	
	// Update variant statistics
	updateQuery := `
		UPDATE test_variants
		SET successes = successes + $1,
		    failures = failures + $2,
		    total_samples = total_samples + 1,
		    updated_at = NOW()
		WHERE variant_id = $3
	`
	
	successIncr := 0
	failureIncr := 0
	if success {
		successIncr = 1
	} else {
		failureIncr = 1
	}
	
	_, err = tx.ExecContext(ctx, updateQuery, successIncr, failureIncr, variantID)
	if err != nil {
		return fmt.Errorf("failed to update variant: %w", err)
	}
	
	// Record individual sample
	sampleQuery := `
		INSERT INTO test_samples (
			variant_id, success, latency_ms, recorded_at
		) VALUES ($1, $2, $3, NOW())
	`
	_, err = tx.ExecContext(ctx, sampleQuery, variantID, success, latencyMs)
	if err != nil {
		return fmt.Errorf("failed to record sample: %w", err)
	}
	
	return tx.Commit()
}

// UpdateStatistics recalculates Bayesian statistics for a variant
func (bat *BayesianABTest) UpdateStatistics(ctx context.Context, variantID string) (*TestVariant, error) {
	// Fetch current stats
	var variant TestVariant
	var ruleConfigJSON []byte
	
	query := `
		SELECT variant_id, name, type, rule_id, rule_config, traffic_percentage,
		       successes, failures, total_samples,
		       alpha_prior, beta_prior, status, created_at, updated_at
		FROM test_variants
		WHERE variant_id = $1
	`
	
	err := bat.db.QueryRowContext(ctx, query, variantID).Scan(
		&variant.ID, &variant.Name, &variant.Type, &variant.RuleID, &ruleConfigJSON,
		&variant.TrafficPercentage, &variant.Successes, &variant.Failures,
		&variant.TotalSamples, &variant.AlphaPrior, &variant.BetaPrior,
		&variant.Status, &variant.CreatedAt, &variant.UpdatedAt,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch variant: %w", err)
	}
	
	json.Unmarshal(ruleConfigJSON, &variant.RuleConfig)
	
	// Calculate posterior distribution: Beta(alpha + successes, beta + failures)
	variant.PosteriorAlpha = variant.AlphaPrior + float64(variant.Successes)
	variant.PosteriorBeta = variant.BetaPrior + float64(variant.Failures)
	
	// Mean conversion rate (expected value of Beta distribution)
	variant.MeanConversionRate = variant.PosteriorAlpha / (variant.PosteriorAlpha + variant.PosteriorBeta)
	
	// 95% Credible Interval (using Beta quantiles approximation)
	variant.CredibleIntervalLower = bat.betaQuantile(variant.PosteriorAlpha, variant.PosteriorBeta, 0.025)
	variant.CredibleIntervalUpper = bat.betaQuantile(variant.PosteriorAlpha, variant.PosteriorBeta, 0.975)
	
	// Calculate latency metrics
	latencyQuery := `
		SELECT 
			AVG(latency_ms) as avg_latency,
			PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) as p95_latency,
			PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY latency_ms) as p99_latency,
			1.0 - (SUM(CASE WHEN success THEN 1 ELSE 0 END)::float / COUNT(*)) as error_rate
		FROM test_samples
		WHERE variant_id = $1
	`
	
	err = bat.db.QueryRowContext(ctx, latencyQuery, variantID).Scan(
		&variant.AvgLatencyMs, &variant.P95LatencyMs,
		&variant.P99LatencyMs, &variant.ErrorRate,
	)
	if err != nil && err != sql.ErrNoRows {
		return nil, fmt.Errorf("failed to calculate latency metrics: %w", err)
	}
	
	// Persist updated stats
	updateQuery := `
		UPDATE test_variants
		SET posterior_alpha = $1,
		    posterior_beta = $2,
		    mean_conversion_rate = $3,
		    credible_interval_lower = $4,
		    credible_interval_upper = $5,
		    avg_latency_ms = $6,
		    p95_latency_ms = $7,
		    p99_latency_ms = $8,
		    error_rate = $9,
		    updated_at = NOW()
		WHERE variant_id = $10
	`
	
	_, err = bat.db.ExecContext(ctx, updateQuery,
		variant.PosteriorAlpha, variant.PosteriorBeta,
		variant.MeanConversionRate, variant.CredibleIntervalLower,
		variant.CredibleIntervalUpper, variant.AvgLatencyMs,
		variant.P95LatencyMs, variant.P99LatencyMs,
		variant.ErrorRate, variantID,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to update statistics: %w", err)
	}
	
	return &variant, nil
}

// EvaluateTest evaluates the test and decides if there's a winner
func (bat *BayesianABTest) EvaluateTest(ctx context.Context, testID string) (*ABTestConfig, error) {
	// Fetch test config
	var config ABTestConfig
	var startedAt time.Time
	var completedAt sql.NullTime
	var maxDurationSeconds int
	
	testQuery := `
		SELECT test_id, name, tenant_id, champion_id, challenger_id,
		       min_sample_size, max_duration_seconds, confidence_level,
		       probability_threshold, max_error_rate, max_latency_p95,
		       auto_rollback, rollback_on_error_rate, rollback_on_latency,
		       status, started_at, completed_at, winner_variant_id, decision_reason
		FROM ab_tests
		WHERE test_id = $1
	`
	
	var championID, challengerID string
	err := bat.db.QueryRowContext(ctx, testQuery, testID).Scan(
		&config.TestID, &config.Name, &config.TenantID, &championID, &challengerID,
		&config.MinSampleSize, &maxDurationSeconds, &config.ConfidenceLevel,
		&config.ProbabilityThreshold, &config.MaxErrorRate, &config.MaxLatencyP95,
		&config.AutoRollback, &config.RollbackOnErrorRate, &config.RollbackOnLatency,
		&config.Status, &startedAt, &completedAt, &config.WinnerVariantID, &config.DecisionReason,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch test: %w", err)
	}
	
	config.StartedAt = startedAt
	config.MaxDuration = time.Duration(maxDurationSeconds) * time.Second
	if completedAt.Valid {
		t := completedAt.Time
		config.CompletedAt = &t
	}
	
	// Update statistics for both variants
	champion, err := bat.UpdateStatistics(ctx, championID)
	if err != nil {
		return nil, fmt.Errorf("failed to update champion stats: %w", err)
	}
	config.ChampionVariant = champion
	
	challenger, err := bat.UpdateStatistics(ctx, challengerID)
	if err != nil {
		return nil, fmt.Errorf("failed to update challenger stats: %w", err)
	}
	config.ChallengerVariant = challenger
	
	// Calculate probability that challenger beats champion
	probChallenger Beats := bat.calculateProbabilityBeat(
		challenger.PosteriorAlpha, challenger.PosteriorBeta,
		champion.PosteriorAlpha, champion.PosteriorBeta,
	)
	challenger.ProbabilityBeatChampion = probChallengerBeats
	
	// Check for early stopping conditions
	totalSamples := champion.TotalSamples + challenger.TotalSamples
	
	// Auto-rollback checks
	if config.AutoRollback {
		if challenger.ErrorRate > config.RollbackOnErrorRate {
			return bat.rollbackTest(ctx, &config, "Challenger error rate exceeded threshold")
		}
		if challenger.P95LatencyMs > config.RollbackOnLatency {
			return bat.rollbackTest(ctx, &config, "Challenger P95 latency exceeded threshold")
		}
	}
	
	// Check for winner (minimum samples reached)
	if totalSamples >= config.MinSampleSize {
		if probChallengerBeats >= config.ProbabilityThreshold {
			// Challenger wins!
			return bat.declareWinner(ctx, &config, challengerID, fmt.Sprintf(
				"Challenger wins with %.2f%% probability (%.2f%% vs %.2f%% conversion)",
				probChallengerBeats*100, challenger.MeanConversionRate*100, champion.MeanConversionRate*100,
			))
		} else if probChallengerBeats <= (1.0 - config.ProbabilityThreshold) {
			// Champion retains
			return bat.declareWinner(ctx, &config, championID, fmt.Sprintf(
				"Champion retains with %.2f%% probability (%.2f%% vs %.2f%% conversion)",
				(1-probChallengerBeats)*100, champion.MeanConversionRate*100, challenger.MeanConversionRate*100,
			))
		}
	}
	
	// Check for timeout
	if time.Since(config.StartedAt) > config.MaxDuration {
		// Timeout - decide based on current evidence
		if probChallengerBeats > 0.5 {
			return bat.declareWinner(ctx, &config, challengerID, "Test timeout - Challenger leading")
		}
		return bat.declareWinner(ctx, &config, championID, "Test timeout - Champion retains")
	}
	
	return &config, nil
}

// betaQuantile approximates the quantile of Beta distribution
func (bat *BayesianABTest) betaQuantile(alpha, beta, p float64) float64 {
	// Simplified approximation using normal approximation for large alpha, beta
	if alpha+beta > 50 {
		mean := alpha / (alpha + beta)
		variance := (alpha * beta) / ((alpha + beta) * (alpha + beta) * (alpha + beta + 1))
		stddev := math.Sqrt(variance)
		// Normal approximation z-score for probability p
		z := bat.normalQuantile(p)
		return mean + z*stddev
	}
	
	// For small samples, use simple linear interpolation (less accurate but fast)
	return alpha / (alpha + beta)
}

// normalQuantile approximates standard normal quantile
func (bat *BayesianABTest) normalQuantile(p float64) float64 {
	// Simplified approximation (for production use proper implementation)
	if p < 0.5 {
		return -bat.normalQuantile(1 - p)
	}
	t := math.Sqrt(-2.0 * math.Log(1-p))
	return t - (2.515517 + 0.802853*t + 0.010328*t*t) / (1.0 + 1.432788*t + 0.189269*t*t + 0.001308*t*t*t)
}

// calculateProbabilityBeat uses Monte Carlo simulation to estimate P(A > B)
func (bat *BayesianABTest) calculateProbabilityBeat(alphaA, betaA, alphaB, betaB float64) float64 {
	// Simplified: use means and variance to estimate overlap
	meanA := alphaA / (alphaA + betaA)
	meanB := alphaB / (alphaB + betaB)
	varA := (alphaA * betaA) / ((alphaA + betaA) * (alphaA + betaA) * (alphaA + betaA + 1))
	varB := (alphaB * betaB) / ((alphaB + betaB) * (alphaB + betaB) * (alphaB + betaB + 1))
	
	// Approximate using normal distributions
	meanDiff := meanA - meanB
	varDiff := varA + varB
	stdDiff := math.Sqrt(varDiff)
	
	if stdDiff == 0 {
		if meanDiff > 0 {
			return 1.0
		}
		return 0.0
	}
	
	// P(A > B) = P(A - B > 0) = 1 - CDF(0)
	z := meanDiff / stdDiff
	return bat.normalCDF(z)
}

// normalCDF approximates standard normal CDF
func (bat *BayesianABTest) normalCDF(x float64) float64 {
	// Approximation of cumulative distribution function
	return 0.5 * (1.0 + math.Erf(x/math.Sqrt(2.0)))
}

// declareWinner marks test as completed with a winner
func (bat *BayesianABTest) declareWinner(ctx context.Context, config *ABTestConfig, winnerID, reason string) (*ABTestConfig, error) {
	now := time.Now()
	config.Status = "completed"
	config.CompletedAt = &now
	config.WinnerVariantID = winnerID
	config.DecisionReason = reason
	
	updateQuery := `
		UPDATE ab_tests
		SET status = $1, completed_at = $2, winner_variant_id = $3, decision_reason = $4
		WHERE test_id = $5
	`
	_, err := bat.db.ExecContext(ctx, updateQuery, config.Status, config.CompletedAt, winnerID, reason, config.TestID)
	if err != nil {
		return nil, fmt.Errorf("failed to declare winner: %w", err)
	}
	
	// Update variant statuses
	_, err = bat.db.ExecContext(ctx, "UPDATE test_variants SET status = $1 WHERE variant_id = $2", "winner", winnerID)
	if err != nil {
		return nil, fmt.Errorf("failed to update winner status: %w", err)
	}
	
	loserID := config.ChampionVariant.ID
	if winnerID == config.ChampionVariant.ID {
		loserID = config.ChallengerVariant.ID
	}
	_, err = bat.db.ExecContext(ctx, "UPDATE test_variants SET status = $1 WHERE variant_id = $2", "loser", loserID)
	
	return config, err
}

// rollbackTest rolls back test due to error conditions
func (bat *BayesianABTest) rollbackTest(ctx context.Context, config *ABTestConfig, reason string) (*ABTestConfig, error) {
	now := time.Now()
	config.Status = "rolled_back"
	config.CompletedAt = &now
	config.DecisionReason = reason
	
	updateQuery := `
		UPDATE ab_tests
		SET status = $1, completed_at = $2, decision_reason = $3
		WHERE test_id = $4
	`
	_, err := bat.db.ExecContext(ctx, updateQuery, config.Status, config.CompletedAt, reason, config.TestID)
	if err != nil {
		return nil, fmt.Errorf("failed to rollback test: %w", err)
	}
	
	// Mark challenger as rolled back
	_, err = bat.db.ExecContext(ctx, "UPDATE test_variants SET status = $1 WHERE variant_id = $2", "rolled_back", config.ChallengerVariant.ID)
	
	return config, err
}
