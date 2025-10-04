package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"sync"
	"time"
)

// AdvancedShadowEngine provides optimized rule evaluation with parallel processing
type AdvancedShadowEngine struct {
	workerPool      *WorkerPool
	evaluationCache *EvaluationCache
	metricsAgg      *MetricsAggregator
	mu              sync.RWMutex
}

// WorkerPool manages concurrent evaluation workers with bounded concurrency
type WorkerPool struct {
	workerCount int
	taskQueue   chan *EvalTask
	resultQueue chan *EvalResult
	wg          sync.WaitGroup
	ctx         context.Context
	cancel      context.CancelFunc
}

// EvalTask represents a single rule evaluation job
type EvalTask struct {
	ID         string
	RuleID     string
	RuleConfig map[string]interface{}
	Samples    []TrafficSample
	StartTime  time.Time
}

// EvalResult contains evaluation outcome
type EvalResult struct {
	TaskID          string
	TP              int
	FP              int
	TN              int
	FN              int
	Precision       float64
	Recall          float64
	F1Score         float64
	ExecutionTime   time.Duration
	Recommendations []string
	Error           error
}

// EvaluationCache stores recent evaluation results to avoid redundant computation
type EvaluationCache struct {
	cache map[string]*CachedEval
	mu    sync.RWMutex
	ttl   time.Duration
}

type CachedEval struct {
	Result    *EvalResult
	ExpiresAt time.Time
}

// MetricsAggregator collects and analyzes evaluation metrics
type MetricsAggregator struct {
	totalEvals      int64
	totalDuration   time.Duration
	avgPrecision    float64
	avgRecall       float64
	avgF1           float64
	rulePerformance map[string]*RuleMetrics
	mu              sync.RWMutex
}

type RuleMetrics struct {
	EvalCount     int64
	AvgPrecision  float64
	AvgRecall     float64
	AvgF1         float64
	AvgExecTime   time.Duration
	LastEvalTime  time.Time
}

// NewAdvancedShadowEngine creates an optimized shadow evaluation engine
func NewAdvancedShadowEngine(workerCount int) *AdvancedShadowEngine {
	ctx, cancel := context.WithCancel(context.Background())

	engine := &AdvancedShadowEngine{
		workerPool: &WorkerPool{
			workerCount: workerCount,
			taskQueue:   make(chan *EvalTask, workerCount*2),
			resultQueue: make(chan *EvalResult, workerCount*2),
			ctx:         ctx,
			cancel:      cancel,
		},
		evaluationCache: &EvaluationCache{
			cache: make(map[string]*CachedEval),
			ttl:   15 * time.Minute,
		},
		metricsAgg: &MetricsAggregator{
			rulePerformance: make(map[string]*RuleMetrics),
		},
	}

	// Start worker pool
	engine.workerPool.Start()

	// Start cache cleanup
	go engine.evaluationCache.CleanupExpired()

	return engine
}

// WorkerPool methods
func (wp *WorkerPool) Start() {
	for i := 0; i < wp.workerCount; i++ {
		wp.wg.Add(1)
		go wp.worker(i)
	}
}

func (wp *WorkerPool) worker(id int) {
	defer wp.wg.Done()

	for {
		select {
		case task := <-wp.taskQueue:
			if task == nil {
				return
			}
			result := wp.evaluateTask(task)
			wp.resultQueue <- result
		case <-wp.ctx.Done():
			return
		}
	}
}

func (wp *WorkerPool) evaluateTask(task *EvalTask) *EvalResult {
	start := time.Now()
	result := &EvalResult{
		TaskID:        task.ID,
		Recommendations: []string{},
	}

	// Parallel evaluation using goroutines for each sample batch
	batchSize := 100
	numBatches := (len(task.Samples) + batchSize - 1) / batchSize
	
	type batchResult struct {
		tp, fp, tn, fn int
	}
	
	batchResults := make(chan batchResult, numBatches)
	var batchWg sync.WaitGroup

	for i := 0; i < numBatches; i++ {
		start := i * batchSize
		end := min(start+batchSize, len(task.Samples))
		batch := task.Samples[start:end]

		batchWg.Add(1)
		go func(samples []TrafficSample) {
			defer batchWg.Done()
			
			var tp, fp, tn, fn int
			for _, sample := range samples {
				predicted := evaluateRuleOnSample(task.RuleConfig, &sample)
				actual := sample.IsAttack

				if predicted && actual {
					tp++
				} else if predicted && !actual {
					fp++
				} else if !predicted && !actual {
					tn++
				} else {
					fn++
				}
			}

			batchResults <- batchResult{tp, fp, tn, fn}
		}(batch)
	}

	// Wait for all batches and aggregate results
	go func() {
		batchWg.Wait()
		close(batchResults)
	}()

	for br := range batchResults {
		result.TP += br.tp
		result.FP += br.fp
		result.TN += br.tn
		result.FN += br.fn
	}

	// Calculate metrics
	result.Precision = calculatePrecision(result.TP, result.FP)
	result.Recall = calculateRecall(result.TP, result.FN)
	result.F1Score = calculateF1(result.Precision, result.Recall)
	result.ExecutionTime = time.Since(start)

	// Generate recommendations
	result.Recommendations = generateRecommendations(result)

	return result
}

func (wp *WorkerPool) SubmitTask(task *EvalTask) error {
	select {
	case wp.taskQueue <- task:
		return nil
	case <-wp.ctx.Done():
		return fmt.Errorf("worker pool shut down")
	default:
		return fmt.Errorf("task queue full")
	}
}

func (wp *WorkerPool) GetResult() *EvalResult {
	select {
	case result := <-wp.resultQueue:
		return result
	case <-wp.ctx.Done():
		return nil
	}
}

func (wp *WorkerPool) Shutdown() {
	wp.cancel()
	close(wp.taskQueue)
	wp.wg.Wait()
	close(wp.resultQueue)
}

// EvaluationCache methods
func (ec *EvaluationCache) Get(key string) (*EvalResult, bool) {
	ec.mu.RLock()
	defer ec.mu.RUnlock()

	cached, exists := ec.cache[key]
	if !exists {
		return nil, false
	}

	if time.Now().After(cached.ExpiresAt) {
		return nil, false
	}

	return cached.Result, true
}

func (ec *EvaluationCache) Set(key string, result *EvalResult) {
	ec.mu.Lock()
	defer ec.mu.Unlock()

	ec.cache[key] = &CachedEval{
		Result:    result,
		ExpiresAt: time.Now().Add(ec.ttl),
	}
}

func (ec *EvaluationCache) CleanupExpired() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		ec.mu.Lock()
		now := time.Now()
		for key, cached := range ec.cache {
			if now.After(cached.ExpiresAt) {
				delete(ec.cache, key)
			}
		}
		ec.mu.Unlock()
	}
}

// MetricsAggregator methods
func (ma *MetricsAggregator) RecordEvaluation(ruleID string, result *EvalResult) {
	ma.mu.Lock()
	defer ma.mu.Unlock()

	ma.totalEvals++
	ma.totalDuration += result.ExecutionTime

	// Update global averages (exponential moving average)
	alpha := 0.2
	ma.avgPrecision = alpha*result.Precision + (1-alpha)*ma.avgPrecision
	ma.avgRecall = alpha*result.Recall + (1-alpha)*ma.avgRecall
	ma.avgF1 = alpha*result.F1Score + (1-alpha)*ma.avgF1

	// Update rule-specific metrics
	if _, exists := ma.rulePerformance[ruleID]; !exists {
		ma.rulePerformance[ruleID] = &RuleMetrics{}
	}

	rm := ma.rulePerformance[ruleID]
	rm.EvalCount++
	rm.AvgPrecision = alpha*result.Precision + (1-alpha)*rm.AvgPrecision
	rm.AvgRecall = alpha*result.Recall + (1-alpha)*rm.AvgRecall
	rm.AvgF1 = alpha*result.F1Score + (1-alpha)*rm.AvgF1
	rm.AvgExecTime = time.Duration(alpha*float64(result.ExecutionTime) + (1-alpha)*float64(rm.AvgExecTime))
	rm.LastEvalTime = time.Now()
}

func (ma *MetricsAggregator) GetGlobalMetrics() map[string]interface{} {
	ma.mu.RLock()
	defer ma.mu.RUnlock()

	avgExecTime := time.Duration(0)
	if ma.totalEvals > 0 {
		avgExecTime = ma.totalDuration / time.Duration(ma.totalEvals)
	}

	return map[string]interface{}{
		"total_evaluations": ma.totalEvals,
		"avg_precision":     ma.avgPrecision,
		"avg_recall":        ma.avgRecall,
		"avg_f1_score":      ma.avgF1,
		"avg_exec_time_ms":  avgExecTime.Milliseconds(),
	}
}

func (ma *MetricsAggregator) GetRuleMetrics(ruleID string) map[string]interface{} {
	ma.mu.RLock()
	defer ma.mu.RUnlock()

	rm, exists := ma.rulePerformance[ruleID]
	if !exists {
		return nil
	}

	return map[string]interface{}{
		"eval_count":        rm.EvalCount,
		"avg_precision":     rm.AvgPrecision,
		"avg_recall":        rm.AvgRecall,
		"avg_f1_score":      rm.AvgF1,
		"avg_exec_time_ms":  rm.AvgExecTime.Milliseconds(),
		"last_eval_time":    rm.LastEvalTime,
	}
}

// Helper functions
func evaluateRuleOnSample(ruleConfig map[string]interface{}, sample *TrafficSample) bool {
	// Fast path: Check rule type
	ruleType, _ := ruleConfig["type"].(string)

	switch ruleType {
	case "rate_limit":
		threshold, _ := ruleConfig["threshold"].(float64)
		// Simplified: check if sample exceeds rate limit
		return checkRateLimit(sample, int(threshold))

	case "signature":
		pattern, _ := ruleConfig["pattern"].(string)
		return checkSignature(sample, pattern)

	case "anomaly":
		// Use ML-based anomaly detection
		return checkAnomaly(sample, ruleConfig)

	case "blacklist":
		blacklist, _ := ruleConfig["ips"].([]interface{})
		return checkBlacklist(sample, blacklist)

	default:
		// Default deny for unknown rule types
		return false
	}
}

func checkRateLimit(sample *TrafficSample, threshold int) bool {
	// Simplified: assume we have rate counter
	// In production, this would query Redis/time-series DB
	return false
}

func checkSignature(sample *TrafficSample, pattern string) bool {
	// Fast string matching using Boyer-Moore or similar
	// For now, simple substring check
	if sample.Payload != "" && contains(sample.Payload, pattern) {
		return true
	}
	if sample.UserAgent != "" && contains(sample.UserAgent, pattern) {
		return true
	}
	return false
}

func checkAnomaly(sample *TrafficSample, config map[string]interface{}) bool {
	// Placeholder for ML-based detection
	// In production, call ML service
	return false
}

func checkBlacklist(sample *TrafficSample, blacklist []interface{}) bool {
	for _, ip := range blacklist {
		if ipStr, ok := ip.(string); ok && ipStr == sample.SourceIP {
			return true
		}
	}
	return false
}

func calculatePrecision(tp, fp int) float64 {
	if tp+fp == 0 {
		return 0.0
	}
	return float64(tp) / float64(tp+fp)
}

func calculateRecall(tp, fn int) float64 {
	if tp+fn == 0 {
		return 0.0
	}
	return float64(tp) / float64(tp+fn)
}

func calculateF1(precision, recall float64) float64 {
	if precision+recall == 0 {
		return 0.0
	}
	return 2 * (precision * recall) / (precision + recall)
}

func generateRecommendations(result *EvalResult) []string {
	recommendations := []string{}

	// High false positive rate
	if result.FP > result.TP && result.FP > 100 {
		recommendations = append(recommendations, "High false positive rate detected. Consider relaxing rule thresholds.")
	}

	// Low recall
	if result.Recall < 0.7 {
		recommendations = append(recommendations, "Low recall (<70%). Rule may miss real attacks. Consider broadening detection criteria.")
	}

	// Low precision
	if result.Precision < 0.8 {
		recommendations = append(recommendations, "Low precision (<80%). Rule may trigger too many false alarms.")
	}

	// Good performance
	if result.F1Score > 0.9 {
		recommendations = append(recommendations, "Excellent performance (F1>0.9). Rule is production-ready.")
	} else if result.F1Score > 0.8 {
		recommendations = append(recommendations, "Good performance (F1>0.8). Minor tuning recommended before full deployment.")
	} else {
		recommendations = append(recommendations, "Suboptimal performance (F1<0.8). Significant tuning required before production.")
	}

	return recommendations
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && indexOf(s, substr) >= 0
}

func indexOf(s, substr string) int {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return i
		}
	}
	return -1
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// SerializeMetrics returns JSON-encoded metrics
func (ma *MetricsAggregator) SerializeMetrics() ([]byte, error) {
	ma.mu.RLock()
	defer ma.mu.RUnlock()

	data := map[string]interface{}{
		"global": ma.GetGlobalMetrics(),
		"rules":  ma.rulePerformance,
	}

	return json.Marshal(data)
}

// GetTopPerformingRules returns rules with highest F1 scores
func (ma *MetricsAggregator) GetTopPerformingRules(limit int) []string {
	ma.mu.RLock()
	defer ma.mu.RUnlock()

	type rulePair struct {
		id string
		f1 float64
	}

	rules := make([]rulePair, 0, len(ma.rulePerformance))
	for id, metrics := range ma.rulePerformance {
		rules = append(rules, rulePair{id, metrics.AvgF1})
	}

	// Simple bubble sort (good enough for small datasets)
	for i := 0; i < len(rules)-1; i++ {
		for j := 0; j < len(rules)-i-1; j++ {
			if rules[j].f1 < rules[j+1].f1 {
				rules[j], rules[j+1] = rules[j+1], rules[j]
			}
		}
	}

	result := make([]string, 0, limit)
	for i := 0; i < min(limit, len(rules)); i++ {
		result = append(result, rules[i].id)
	}

	return result
}

// AdvancedShadowEngine high-level methods
func (ase *AdvancedShadowEngine) EvaluateRule(ruleID string, ruleConfig map[string]interface{}, samples []TrafficSample) (*EvalResult, error) {
	// Check cache first
	cacheKey := fmt.Sprintf("%s:%d", ruleID, len(samples))
	if cached, found := ase.evaluationCache.Get(cacheKey); found {
		log.Printf("[shadow-engine] Cache hit for rule %s", ruleID)
		return cached, nil
	}

	// Submit task to worker pool
	task := &EvalTask{
		ID:         fmt.Sprintf("eval-%d", time.Now().UnixNano()),
		RuleID:     ruleID,
		RuleConfig: ruleConfig,
		Samples:    samples,
		StartTime:  time.Now(),
	}

	if err := ase.workerPool.SubmitTask(task); err != nil {
		return nil, fmt.Errorf("failed to submit task: %w", err)
	}

	// Wait for result
	result := ase.workerPool.GetResult()
	if result == nil {
		return nil, fmt.Errorf("evaluation failed or timed out")
	}

	// Cache result
	ase.evaluationCache.Set(cacheKey, result)

	// Record metrics
	ase.metricsAgg.RecordEvaluation(ruleID, result)

	return result, nil
}

func (ase *AdvancedShadowEngine) GetMetrics() map[string]interface{} {
	return ase.metricsAgg.GetGlobalMetrics()
}

func (ase *AdvancedShadowEngine) Shutdown() {
	ase.workerPool.Shutdown()
}
