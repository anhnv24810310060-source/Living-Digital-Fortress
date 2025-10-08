package orchestrator

import (
	"context"
	"sync"
	"time"
)

type DecoyAutoscaler struct {
	threatScorer  *ThreatScorer
	decoyManager  DecoyManager
	policy        ScalingPolicy
	mu            sync.RWMutex
	currentScale  int
	lastScaleTime time.Time
}

type ThreatScorer struct {
	anomalyWeight   float64
	volumeWeight    float64
	diversityWeight float64
	historyWeight   float64
}

type ScalingPolicy struct {
	MinDecoys          int
	MaxDecoys          int
	ScaleUpThreshold   float64
	ScaleDownThreshold float64
	CooldownPeriod     time.Duration
}

type DecoyManager interface {
	GetActiveCount() int
	ScaleTo(ctx context.Context, target int) error
}

func NewDecoyAutoscaler(dm DecoyManager, policy ScalingPolicy) *DecoyAutoscaler {
	return &DecoyAutoscaler{
		threatScorer: &ThreatScorer{
			anomalyWeight:   0.4,
			volumeWeight:    0.3,
			diversityWeight: 0.2,
			historyWeight:   0.1,
		},
		decoyManager: dm,
		policy:       policy,
		currentScale: policy.MinDecoys,
	}
}

func (da *DecoyAutoscaler) Scale(ctx context.Context) error {
	da.mu.Lock()
	defer da.mu.Unlock()

	if time.Since(da.lastScaleTime) < da.policy.CooldownPeriod {
		return nil
	}

	score := da.threatScorer.GetCurrentScore()
	current := da.decoyManager.GetActiveCount()

	var target int
	switch {
	case score > da.policy.ScaleUpThreshold:
		target = min(current*2, da.policy.MaxDecoys)
	case score < da.policy.ScaleDownThreshold:
		target = max(current/2, da.policy.MinDecoys)
	default:
		return nil
	}

	if err := da.decoyManager.ScaleTo(ctx, target); err != nil {
		return err
	}

	da.lastScaleTime = time.Now()
	return nil
}

func (ts *ThreatScorer) GetCurrentScore() float64 {
	// Simplified threat calculation
	return 0.5 // Placeholder - would integrate with real telemetry
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
