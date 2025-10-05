package guardian

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"shieldx/pkg/metrics"
	"shieldx/pkg/sandbox"
)

// EnhancedGuardianAPI provides Phase 2 P0 production-ready threat detection API
// Features:
// - Advanced multi-layer threat detection (Transformer + eBPF + Memory Forensics)
// - Automated incident response workflows
// - Real-time behavioral analysis
// - Zero-trust security validation

type EnhancedGuardianAPI struct {
	// Core detector
	detector *AdvancedThreatDetector
	
	// Sandbox execution engine
	sandboxRunner sandbox.Runner
	
	// Job management
	jobs      map[string]*ExecutionJob
	jobsMu    sync.RWMutex
	jobIDCounter uint64
	
	// Metrics
	totalExecutions *metrics.Counter
	blockedExecutions *metrics.Counter
	avgThreatScore  *metrics.Gauge
	detectionLatency *metrics.Gauge
	
	// Configuration
	config APIConfig
}

// APIConfig defines API operational parameters
type APIConfig struct {
	MaxConcurrentExecutions int
	JobTTL                  time.Duration
	MaxPayloadSize          int
	AutoQuarantine          bool
	EnableIncidentResponse  bool
	NotificationWebhook     string
}

// ExecutionJob tracks sandbox execution lifecycle
type ExecutionJob struct {
	ID              string                       `json:"id"`
	Status          string                       `json:"status"` // queued, running, analyzing, done, blocked, error
	CreatedAt       time.Time                    `json:"created_at"`
	CompletedAt     time.Time                    `json:"completed_at,omitempty"`
	
	// Input
	Payload         string                       `json:"payload,omitempty"`
	TenantID        string                       `json:"tenant_id"`
	
	// Sandbox results
	SandboxResult   *sandbox.SandboxResult       `json:"sandbox_result,omitempty"`
	
	// Threat analysis
	ThreatAnalysis  *ThreatDetectionResult `json:"threat_analysis,omitempty"`
	
	// Decision
	Action          string                       `json:"action"` // ALLOW, BLOCK, QUARANTINE
	Reason          string                       `json:"reason"`
	
	// Errors
	Error           string                       `json:"error,omitempty"`
	
	// Performance
	TotalDuration   time.Duration                `json:"total_duration"`
}

// DefaultAPIConfig returns production configuration
func DefaultAPIConfig() APIConfig {
	return APIConfig{
		MaxConcurrentExecutions: 32,
		JobTTL:                  10 * time.Minute,
		MaxPayloadSize:          64 * 1024, // 64KB
		AutoQuarantine:          true,
		EnableIncidentResponse:  true,
		NotificationWebhook:     os.Getenv("GUARDIAN_WEBHOOK_URL"),
	}
}

// NewEnhancedGuardianAPI creates production-ready API instance
func NewEnhancedGuardianAPI() (*EnhancedGuardianAPI, error) {
	// Initialize advanced threat detector
	detectorConfig := DefaultDetectorConfig()
	detector, err := NewAdvancedThreatDetector(detectorConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create detector: %w", err)
	}
	
	// Initialize sandbox runner
	sandboxRunner := sandbox.NewFromEnv()
	
	api := &EnhancedGuardianAPI{
		detector:      detector,
		sandboxRunner: sandboxRunner,
		jobs:          make(map[string]*ExecutionJob),
		config:        DefaultAPIConfig(),
		
		totalExecutions:   metrics.NewCounter("guardian_total_executions", "Total executions"),
		blockedExecutions: metrics.NewCounter("guardian_blocked_executions", "Blocked executions"),
		avgThreatScore:    metrics.NewGauge("guardian_avg_threat_score", "Average threat score"),
		detectionLatency:  metrics.NewGauge("guardian_detection_latency_ms", "Detection latency"),
	}
	
	// Start cleanup goroutine
	go api.jobCleanupWorker()
	
	return api, nil
}

// HandleExecuteEnhanced processes execution request with advanced threat detection
// POST /guardian/execute/enhanced
// P0 Requirements:
// - MUST NOT execute untrusted code outside sandbox
// - MUST apply multi-layer threat detection
// - MUST provide actionable results within 100ms
func (api *EnhancedGuardianAPI) HandleExecuteEnhanced(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	
	// Parse request
	var req struct {
		Payload  string `json:"payload"`
		TenantID string `json:"tenant_id"`
		Async    bool   `json:"async"` // If true, return immediately
	}
	
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, int64(api.config.MaxPayloadSize))).Decode(&req); err != nil {
		http.Error(w, "invalid request", http.StatusBadRequest)
		return
	}
	
	if len(req.Payload) == 0 {
		http.Error(w, "payload required", http.StatusBadRequest)
		return
	}
	
	// Create job
	api.jobsMu.Lock()
	api.jobIDCounter++
	jobID := fmt.Sprintf("job-%d", api.jobIDCounter)
	job := &ExecutionJob{
		ID:        jobID,
		Status:    "queued",
		CreatedAt: time.Now(),
		Payload:   req.Payload,
		TenantID:  req.TenantID,
	}
	api.jobs[jobID] = job
	api.jobsMu.Unlock()
	
	api.totalExecutions.Add(1)
	
	// Execute asynchronously
	go api.executeWithThreatDetection(job)
	
	// Return job ID immediately
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"job_id": jobID,
		"status": job.Status,
	})
}

// executeWithThreatDetection performs sandbox execution + threat analysis
func (api *EnhancedGuardianAPI) executeWithThreatDetection(job *ExecutionJob) {
	startTime := time.Now()
	job.Status = "running"
	
	// Phase 1: Sandbox execution (30s timeout)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	
	sandboxResult, err := api.executeSandboxed(ctx, job.Payload)
	if err != nil {
		job.Status = "error"
		job.Error = err.Error()
		job.CompletedAt = time.Now()
		job.TotalDuration = time.Since(startTime)
		return
	}
	
	job.SandboxResult = sandboxResult
	job.Status = "analyzing"
	
	// Phase 2: Advanced threat detection
	threatCtx, threatCancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer threatCancel()
	
	threatResult, err := api.detector.DetectThreats(threatCtx, sandboxResult)
	if err != nil {
		log.Printf("[guardian] threat detection error: %v", err)
		// Fallback to basic scoring
		threatResult = &ThreatDetectionResult{
			ThreatScore:  sandboxResult.ThreatScore,
			RiskLevel:    "UNKNOWN",
			Confidence:   0.5,
			Explanation:  "Fallback scoring due to detection error",
		}
	}
	
	job.ThreatAnalysis = threatResult
	
	// Update metrics
	api.avgThreatScore.Set(uint64(threatResult.ThreatScore))
	api.detectionLatency.Set(uint64(threatResult.DetectionLatency.Milliseconds()))
	
	// Phase 3: Decision making
	job.Action, job.Reason = api.makeDecision(threatResult)
	
	if job.Action == "BLOCK" || job.Action == "QUARANTINE" {
		api.blockedExecutions.Add(1)
		
		// Automated incident response
		if api.config.EnableIncidentResponse {
			go api.triggerIncidentResponse(job)
		}
	}
	
	job.Status = "done"
	job.CompletedAt = time.Now()
	job.TotalDuration = time.Since(startTime)
	
	log.Printf("[guardian] job %s completed: threat=%.1f risk=%s action=%s duration=%s",
		job.ID, threatResult.ThreatScore, threatResult.RiskLevel, job.Action, job.TotalDuration)
}

// executeSandboxed runs payload in secure sandbox
func (api *EnhancedGuardianAPI) executeSandboxed(ctx context.Context, payload string) (*sandbox.SandboxResult, error) {
	// Type assertion to get SandboxResult from different runner types
	switch runner := api.sandboxRunner.(type) {
	case interface{ Run(context.Context, string) (*sandbox.SandboxResult, error) }:
		return runner.Run(ctx, payload)
	case interface{ Run(context.Context, string) (string, error) }:
		// Fallback for basic runners
		stdout, err := runner.Run(ctx, payload)
		if err != nil {
			return nil, err
		}
		return &sandbox.SandboxResult{
			Stdout:      stdout,
			ThreatScore: 0,
			Duration:    0,
		}, nil
	default:
		return nil, fmt.Errorf("unsupported sandbox runner type")
	}
}

// makeDecision determines action based on threat analysis
func (api *EnhancedGuardianAPI) makeDecision(threat *ThreatDetectionResult) (string, string) {
	switch threat.RiskLevel {
	case "CRITICAL":
		return "BLOCK", fmt.Sprintf("Critical threat detected (score: %.1f): %s", threat.ThreatScore, threat.Explanation)
	case "HIGH":
		if api.config.AutoQuarantine {
			return "QUARANTINE", fmt.Sprintf("High-risk execution quarantined (score: %.1f)", threat.ThreatScore)
		}
		return "BLOCK", fmt.Sprintf("High-risk execution blocked (score: %.1f)", threat.ThreatScore)
	case "MEDIUM":
		return "ALLOW", fmt.Sprintf("Medium-risk allowed with monitoring (score: %.1f)", threat.ThreatScore)
	case "LOW":
		return "ALLOW", fmt.Sprintf("Low-risk execution (score: %.1f)", threat.ThreatScore)
	default:
		return "ALLOW", "Safe execution"
	}
}

// triggerIncidentResponse initiates automated response workflow
func (api *EnhancedGuardianAPI) triggerIncidentResponse(job *ExecutionJob) {
	log.Printf("[guardian] INCIDENT RESPONSE triggered for job %s", job.ID)
	
	// Send notification to webhook
	if api.config.NotificationWebhook != "" {
		incident := map[string]interface{}{
			"job_id":       job.ID,
			"tenant_id":    job.TenantID,
			"threat_score": job.ThreatAnalysis.ThreatScore,
			"risk_level":   job.ThreatAnalysis.RiskLevel,
			"action":       job.Action,
			"explanation":  job.ThreatAnalysis.Explanation,
			"timestamp":    time.Now(),
		}
		
		api.sendWebhookNotification(api.config.NotificationWebhook, incident)
	}
	
	// Additional response actions could include:
	// - IP blocking
	// - Account suspension
	// - Evidence collection
	// - Alert escalation
}

// sendWebhookNotification sends incident data to external webhook
func (api *EnhancedGuardianAPI) sendWebhookNotification(url string, data map[string]interface{}) {
	jsonData, err := json.Marshal(data)
	if err != nil {
		log.Printf("[guardian] webhook marshal error: %v", err)
		return
	}
	
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, strings.NewReader(string(jsonData)))
	if err != nil {
		log.Printf("[guardian] webhook request error: %v", err)
		return
	}
	
	req.Header.Set("Content-Type", "application/json")
	
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		log.Printf("[guardian] webhook send error: %v", err)
		return
	}
	defer resp.Body.Close()
	
	if resp.StatusCode >= 300 {
		log.Printf("[guardian] webhook returned status %d", resp.StatusCode)
	}
}

// HandleGetJobStatus returns enhanced job status
// GET /guardian/job/{id}
func (api *EnhancedGuardianAPI) HandleGetJobStatus(w http.ResponseWriter, r *http.Request) {
	jobID := strings.TrimPrefix(r.URL.Path, "/guardian/job/")
	
	api.jobsMu.RLock()
	job, exists := api.jobs[jobID]
	api.jobsMu.RUnlock()
	
	if !exists {
		http.Error(w, "job not found", http.StatusNotFound)
		return
	}
	
	// Return sanitized job info (no raw payload in response)
	sanitized := map[string]interface{}{
		"job_id":       job.ID,
		"status":       job.Status,
		"created_at":   job.CreatedAt,
		"completed_at": job.CompletedAt,
		"action":       job.Action,
		"reason":       job.Reason,
	}
	
	if job.ThreatAnalysis != nil {
		sanitized["threat_analysis"] = map[string]interface{}{
			"threat_score":      job.ThreatAnalysis.ThreatScore,
			"risk_level":        job.ThreatAnalysis.RiskLevel,
			"confidence":        job.ThreatAnalysis.Confidence,
			"explanation":       job.ThreatAnalysis.Explanation,
			"detected_patterns": job.ThreatAnalysis.DetectedPatterns,
			"recommended_action": job.ThreatAnalysis.RecommendedAction,
		}
	}
	
	if job.Error != "" {
		sanitized["error"] = job.Error
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(sanitized)
}

// HandleGetDetectorMetrics returns detector performance metrics
// GET /guardian/metrics/detector
func (api *EnhancedGuardianAPI) HandleGetDetectorMetrics(w http.ResponseWriter, r *http.Request) {
	metrics := api.detector.GetMetrics()
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(metrics)
}

// jobCleanupWorker periodically removes old jobs
func (api *EnhancedGuardianAPI) jobCleanupWorker() {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()
	
	for range ticker.C {
		now := time.Now()
		removed := 0
		
		api.jobsMu.Lock()
		for id, job := range api.jobs {
			if job.Status == "done" || job.Status == "error" {
				if now.Sub(job.CompletedAt) > api.config.JobTTL {
					delete(api.jobs, id)
					removed++
				}
			}
		}
		api.jobsMu.Unlock()
		
		if removed > 0 {
			log.Printf("[guardian] cleaned up %d old jobs", removed)
		}
	}
}

// RegisterRoutes registers all API routes
func (api *EnhancedGuardianAPI) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/guardian/execute/enhanced", api.HandleExecuteEnhanced)
	mux.HandleFunc("/guardian/job/", api.HandleGetJobStatus)
	mux.HandleFunc("/guardian/metrics/detector", api.HandleGetDetectorMetrics)
}

// GetMetrics returns API-level metrics
func (api *EnhancedGuardianAPI) GetMetrics() map[string]interface{} {
	api.jobsMu.RLock()
	activeJobs := 0
	for _, job := range api.jobs {
		if job.Status == "running" || job.Status == "analyzing" {
			activeJobs++
		}
	}
	totalJobs := len(api.jobs)
	api.jobsMu.RUnlock()
	
	return map[string]interface{}{
		"total_jobs":         totalJobs,
		"active_jobs":        activeJobs,
		"total_executions":   api.totalExecutions.Value(),
		"blocked_executions": api.blockedExecutions.Value(),
	}
}

// Helper to parse int from env
func getEnvInt(key string, defaultVal int) int {
	if val := os.Getenv(key); val != "" {
		if i, err := strconv.Atoi(val); err == nil {
			return i
		}
	}
	return defaultVal
}
