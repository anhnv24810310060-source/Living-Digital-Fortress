package shadow
package shadow

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"time"
)

// BayesianEngine implements Thompson Sampling for A/B testing
// This provides better statistical power than traditional A/B tests
// and automatically allocates more traffic to better-performing variants
type BayesianEngine struct {
	db *sql.DB
}

type Variant struct {
	ID           string  `json:"id"`
	Name         string  `json:"name"`
	Config       string  `json:"config"`
	Alpha        float64 `json:"alpha"`        // Success count + prior
	Beta         float64 `json:"beta"`         // Failure count + prior
	Probability  float64 `json:"probability"`  // Probability of being best
	SampleCount  int64   `json:"sample_count"`
	SuccessRate  float64 `json:"success_rate"`
}

type ABTest struct {
	ID              string     `json:"id"`
	Name            string     `json:"name"`
	Description     string     `json:"description"`
	ControlVariant  string     `json:"control_variant"`
	Variants        []Variant  `json:"variants"`
	Status          string     `json:"status"`
	ConfidenceLevel float64    `json:"confidence_level"`
	MinSampleSize   int64      `json:"min_sample_size"`
	CreatedAt       time.Time  `json:"created_at"`
}

type TestResult struct {
	VariantID string `json:"variant_id"`
	Success   bool   `json:"success"`
	Latency   int64  `json:"latency_ms"`
	Metadata  string `json:"metadata,omitempty"`
}

// NewBayesianEngine creates Bayesian A/B testing engine
func NewBayesianEngine(db *sql.DB) *BayesianEngine {
	return &BayesianEngine{db: db}
}

// SelectVariant uses Thompson Sampling to select best variant
// This balances exploration vs exploitation automatically
func (be *BayesianEngine) SelectVariant(ctx context.Context, testID string) (*Variant, error) {
	// Get all variants for this test
	rows, err := be.db.QueryContext(ctx, `
		SELECT id, name, config, alpha, beta, sample_count
		FROM ab_test_variants
		WHERE test_id = $1 AND enabled = true
		ORDER BY id
	`, testID)
	
	if err != nil {
		return nil, fmt.Errorf("failed to query variants: %w", err)
	}
	defer rows.Close()

	var variants []Variant
	for rows.Next() {
		var v Variant
		err := rows.Scan(&v.ID, &v.Name, &v.Config, &v.Alpha, &v.Beta, &v.SampleCount)
		if err != nil {
			return nil, fmt.Errorf("failed to scan variant: %w", err)
		}
		variants = append(variants, v)
	}

	if len(variants) == 0 {
		return nil, fmt.Errorf("no enabled variants found for test %s", testID)
	}

	// Thompson Sampling: sample from Beta distribution for each variant
	bestVariant := 0
	maxSample := 0.0
	
	for i, v := range variants {
		// Sample from Beta(alpha, beta) distribution
		sample := sampleBeta(v.Alpha, v.Beta)
		if sample > maxSample {
			maxSample = sample
			bestVariant = i
		}
	}

	return &variants[bestVariant], nil
}

// RecordResult updates variant statistics using Bayesian inference
func (be *BayesianEngine) RecordResult(ctx context.Context, result TestResult) error {
	tx, err := be.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer tx.Rollback()

	// Update variant statistics
	increment := 1.0
	alphaInc := 0.0
	betaInc := 0.0
	
	if result.Success {
		alphaInc = increment
	} else {
		betaInc = increment
	}

	_, err = tx.ExecContext(ctx, `
		UPDATE ab_test_variants
		SET alpha = alpha + $1,
		    beta = beta + $2,
		    sample_count = sample_count + 1,
		    updated_at = NOW()
		WHERE id = $3
	`, alphaInc, betaInc, result.VariantID)

	if err != nil {
		return fmt.Errorf("failed to update variant: %w", err)
	}

	// Record detailed result
	_, err = tx.ExecContext(ctx, `
		INSERT INTO ab_test_results (
			id, variant_id, success, latency_ms, metadata, created_at
		) VALUES (gen_random_uuid(), $1, $2, $3, $4, NOW())
	`, result.VariantID, result.Success, result.Latency, result.Metadata)

	if err != nil {
		return fmt.Errorf("failed to record result: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit: %w", err)
	}

	return nil
}

// AnalyzeTest calculates probability of each variant being best
// Uses Monte Carlo simulation with 10,000 samples
func (be *BayesianEngine) AnalyzeTest(ctx context.Context, testID string) (*ABTest, error) {
	// Get test info
	var test ABTest
	err := be.db.QueryRowContext(ctx, `
		SELECT id, name, description, control_variant, confidence_level, min_sample_size, created_at
		FROM ab_tests
		WHERE id = $1
	`, testID).Scan(&test.ID, &test.Name, &test.Description, &test.ControlVariant, 
		&test.ConfidenceLevel, &test.MinSampleSize, &test.CreatedAt)

	if err != nil {
		return nil, fmt.Errorf("test not found: %w", err)
	}

	// Get variants
	rows, err := be.db.QueryContext(ctx, `
		SELECT id, name, config, alpha, beta, sample_count
		FROM ab_test_variants
		WHERE test_id = $1
		ORDER BY id
	`, testID)

	if err != nil {
		return nil, fmt.Errorf("failed to query variants: %w", err)
	}
	defer rows.Close()

	var variants []Variant
	for rows.Next() {
		var v Variant
		err := rows.Scan(&v.ID, &v.Name, &v.Config, &v.Alpha, &v.Beta, &v.SampleCount)
		if err != nil {
			return nil, fmt.Errorf("failed to scan variant: %w", err)
		}
		
		if v.SampleCount > 0 {
			v.SuccessRate = v.Alpha / (v.Alpha + v.Beta)
		}
		
		variants = append(variants, v)
	}

	if len(variants) == 0 {
		return nil, fmt.Errorf("no variants found")
	}

	// Monte Carlo simulation to calculate probability of being best
	numSimulations := 10000
	winCount := make([]int, len(variants))

	for i := 0; i < numSimulations; i++ {
		maxSample := -1.0
		bestIdx := 0

		for j, v := range variants {
			sample := sampleBeta(v.Alpha, v.Beta)
			if sample > maxSample {
				maxSample = sample
				bestIdx = j
			}
		}

		winCount[bestIdx]++
	}

	// Calculate probabilities
	for i := range variants {
		variants[i].Probability = float64(winCount[i]) / float64(numSimulations)
	}

	test.Variants = variants
	test.Status = be.determineTestStatus(variants, test.ConfidenceLevel, test.MinSampleSize)

	return &test, nil
}

// determineTestStatus decides if test has conclusive results
func (be *BayesianEngine) determineTestStatus(variants []Variant, confidenceLevel float64, minSampleSize int64) string {
	// Check if minimum sample size reached
	totalSamples := int64(0)
	for _, v := range variants {
		totalSamples += v.SampleCount
	}

	if totalSamples < minSampleSize {
		return "collecting_data"
	}

	// Check if any variant has probability > confidence level
	for _, v := range variants {
		if v.Probability >= confidenceLevel {
			return "conclusive"
		}
	}

	return "inconclusive"
}

// SafeDeployVariant deploys winning variant after validation
func (be *BayesianEngine) SafeDeployVariant(ctx context.Context, testID, variantID string) error {
	// Get test analysis
	test, err := be.AnalyzeTest(ctx, testID)
	if err != nil {
		return fmt.Errorf("failed to analyze test: %w", err)
	}

	// Safety checks before deployment
	if test.Status != "conclusive" {
		return fmt.Errorf("test not conclusive: status=%s", test.Status)
	}

	// Find winning variant
	var winner *Variant
	maxProb := 0.0
	for i, v := range test.Variants {
		if v.Probability > maxProb {
			maxProb = v.Probability
			winner = &test.Variants[i]
		}
	}

	if winner == nil || winner.ID != variantID {
		return fmt.Errorf("variant %s is not the winner (probability=%.2f%%)", variantID, winner.Probability*100)
	}

	// Additional safety check: ensure winner is significantly better than control
	var controlVariant *Variant
	for i, v := range test.Variants {
		if v.ID == test.ControlVariant {
			controlVariant = &test.Variants[i]
			break
		}
	}

	if controlVariant != nil {
		// Check if winner success rate is at least 5% better than control
		improvement := (winner.SuccessRate - controlVariant.SuccessRate) / controlVariant.SuccessRate
		if improvement < 0.05 {
			return fmt.Errorf("insufficient improvement over control: %.2f%%", improvement*100)
		}
	}

	// Deploy variant (P0 requirement: mark as deployed, don't modify production directly)
	_, err = be.db.ExecContext(ctx, `
		UPDATE ab_test_variants
		SET deployed = true, deployed_at = NOW()
		WHERE id = $1
	`, variantID)

	if err != nil {
		return fmt.Errorf("failed to deploy variant: %w", err)
	}

	// Mark test as completed
	_, err = be.db.ExecContext(ctx, `
		UPDATE ab_tests
		SET status = 'completed', completed_at = NOW(), winner_variant = $1
		WHERE id = $2
	`, variantID, testID)

	if err != nil {
		return fmt.Errorf("failed to complete test: %w", err)
	}

	log.Printf("[shadow] Safely deployed variant %s for test %s (probability=%.2f%%, improvement=%.2f%%)",
		variantID, testID, winner.Probability*100,
		(winner.SuccessRate-controlVariant.SuccessRate)/controlVariant.SuccessRate*100)

	return nil
}

// sampleBeta generates sample from Beta distribution
// Using Gamma distribution ratio method for Beta sampling
func sampleBeta(alpha, beta float64) float64 {
	if alpha <= 0 || beta <= 0 {
		return 0.5 // fallback to uniform
	}

	// Beta(α, β) = Gamma(α) / (Gamma(α) + Gamma(β))
	x := sampleGamma(alpha, 1.0)
	y := sampleGamma(beta, 1.0)
	
	if x+y == 0 {
		return 0.5
	}
	
	return x / (x + y)
}

// sampleGamma generates sample from Gamma distribution
// Using Marsaglia and Tsang method for shape >= 1
func sampleGamma(shape, scale float64) float64 {
	if shape < 1 {
		// Use transformation: Gamma(α) = Gamma(α+1) * U^(1/α)
		return sampleGamma(shape+1, scale) * math.Pow(randomUniform(), 1.0/shape)
	}

	d := shape - 1.0/3.0
	c := 1.0 / math.Sqrt(9.0*d)

	for {
		var x, v float64
		for {
			x = randomNormal()
			v = 1.0 + c*x
			if v > 0 {
				break
			}
		}

		v = v * v * v
		u := randomUniform()
		x2 := x * x

		if u < 1.0-0.0331*x2*x2 {
			return scale * d * v
		}

		if math.Log(u) < 0.5*x2+d*(1.0-v+math.Log(v)) {
			return scale * d * v
		}
	}
}

// randomUniform generates uniform random number in [0, 1)
func randomUniform() float64 {
	// Use time-based pseudo-random (fast but not cryptographically secure)
	t := time.Now().UnixNano()
	return float64(t%1000000) / 1000000.0
}

// randomNormal generates standard normal random variable
// Using Box-Muller transform
func randomNormal() float64 {
	u1 := randomUniform()
	u2 := randomUniform()
	
	// Box-Muller transform
	return math.Sqrt(-2.0*math.Log(u1)) * math.Cos(2.0*math.Pi*u2)
}

// ExportMetrics exports test metrics for monitoring
func (be *BayesianEngine) ExportMetrics(ctx context.Context, testID string) (map[string]interface{}, error) {
	test, err := be.AnalyzeTest(ctx, testID)
	if err != nil {
		return nil, err
	}

	metrics := map[string]interface{}{
		"test_id":          test.ID,
		"status":           test.Status,
		"total_samples":    int64(0),
		"variants":         []map[string]interface{}{},
		"recommendation":   "",
	}

	totalSamples := int64(0)
	var bestVariant *Variant
	maxProb := 0.0

	for _, v := range test.Variants {
		totalSamples += v.SampleCount
		
		variantMetrics := map[string]interface{}{
			"id":            v.ID,
			"name":          v.Name,
			"success_rate":  v.SuccessRate,
			"probability":   v.Probability,
			"sample_count":  v.SampleCount,
			"alpha":         v.Alpha,
			"beta":          v.Beta,
		}
		metrics["variants"] = append(metrics["variants"].([]map[string]interface{}), variantMetrics)

		if v.Probability > maxProb {
			maxProb = v.Probability
			bestVariant = &v
		}
	}

	metrics["total_samples"] = totalSamples

	if bestVariant != nil {
		if test.Status == "conclusive" {
			metrics["recommendation"] = fmt.Sprintf("Deploy variant '%s' (%.1f%% probability of being best)", 
				bestVariant.Name, bestVariant.Probability*100)
		} else {
			metrics["recommendation"] = fmt.Sprintf("Continue testing - leading variant '%s' at %.1f%% probability",
				bestVariant.Name, bestVariant.Probability*100)
		}
	}

	return metrics, nil
}
