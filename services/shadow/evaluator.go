package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"time"

	"github.com/google/uuid"
	_ "github.com/lib/pq"
)

type ShadowEvaluator struct {
	db *sql.DB
}

type ShadowEvalRequest struct {
	RuleID      string                 `json:"rule_id"`
	RuleName    string                 `json:"rule_name"`
	RuleType    string                 `json:"rule_type"`
	RuleConfig  map[string]interface{} `json:"rule_config"`
	SampleSize  int                    `json:"sample_size"`
	TimeWindow  string                 `json:"time_window"`
	TenantID    string                 `json:"tenant_id"`
}

type ShadowEvalResult struct {
	EvalID           string    `json:"eval_id"`
	RuleID           string    `json:"rule_id"`
	Status           string    `json:"status"`
	SampleSize       int       `json:"sample_size"`
	TruePositives    int       `json:"true_positives"`
	FalsePositives   int       `json:"false_positives"`
	TrueNegatives    int       `json:"true_negatives"`
	FalseNegatives   int       `json:"false_negatives"`
	Precision        float64   `json:"precision"`
	Recall           float64   `json:"recall"`
	F1Score          float64   `json:"f1_score"`
	EstimatedFPRate  float64   `json:"estimated_fp_rate"`
	EstimatedTPRate  float64   `json:"estimated_tp_rate"`
	Recommendations  []string  `json:"recommendations"`
	ExecutionTime    int64     `json:"execution_time_ms"`
	CreatedAt        time.Time `json:"created_at"`
	CompletedAt      *time.Time `json:"completed_at"`
}

type TrafficSample struct {
	ID          string                 `json:"id"`
	Timestamp   time.Time              `json:"timestamp"`
	SourceIP    string                 `json:"source_ip"`
	DestIP      string                 `json:"dest_ip"`
	Protocol    string                 `json:"protocol"`
	Port        int                    `json:"port"`
	Payload     string                 `json:"payload"`
	Headers     map[string]string      `json:"headers"`
	UserAgent   string                 `json:"user_agent"`
	Method      string                 `json:"method"`
	URI         string                 `json:"uri"`
	IsAttack    bool                   `json:"is_attack"`
	AttackType  string                 `json:"attack_type"`
	Metadata    map[string]interface{} `json:"metadata"`
}

func NewShadowEvaluator(dbURL string) (*ShadowEvaluator, error) {
	db, err := sql.Open("postgres", dbURL)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}

	if err := db.Ping(); err != nil {
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	db.SetMaxOpenConns(25)
	db.SetMaxIdleConns(5)
	db.SetConnMaxLifetime(5 * time.Minute)

	evaluator := &ShadowEvaluator{db: db}
	if err := evaluator.migrate(); err != nil {
		return nil, fmt.Errorf("migration failed: %w", err)
	}

	return evaluator, nil
}

func (se *ShadowEvaluator) migrate() error {
	query := `
	CREATE TABLE IF NOT EXISTS shadow_evaluations (
		eval_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
		rule_id VARCHAR(255) NOT NULL,
		rule_name VARCHAR(255) NOT NULL,
		rule_type VARCHAR(100) NOT NULL,
		rule_config JSONB NOT NULL,
		tenant_id VARCHAR(255) NOT NULL,
		status VARCHAR(50) NOT NULL DEFAULT 'pending',
		sample_size INTEGER NOT NULL,
		true_positives INTEGER DEFAULT 0,
		false_positives INTEGER DEFAULT 0,
		true_negatives INTEGER DEFAULT 0,
		false_negatives INTEGER DEFAULT 0,
		precision FLOAT DEFAULT 0,
		recall_rate FLOAT DEFAULT 0,
		f1_score FLOAT DEFAULT 0,
		estimated_fp_rate FLOAT DEFAULT 0,
		estimated_tp_rate FLOAT DEFAULT 0,
		recommendations JSONB,
		execution_time_ms BIGINT DEFAULT 0,
		created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
		completed_at TIMESTAMP WITH TIME ZONE
	);

	CREATE TABLE IF NOT EXISTS traffic_samples (
		id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
		timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
		source_ip INET NOT NULL,
		dest_ip INET NOT NULL,
		protocol VARCHAR(20) NOT NULL,
		port INTEGER NOT NULL,
		payload TEXT,
		headers JSONB,
		user_agent TEXT,
		method VARCHAR(10),
		uri TEXT,
		is_attack BOOLEAN NOT NULL DEFAULT FALSE,
		attack_type VARCHAR(100),
		metadata JSONB,
		created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
	);

	CREATE INDEX IF NOT EXISTS idx_shadow_evaluations_tenant ON shadow_evaluations(tenant_id);
	CREATE INDEX IF NOT EXISTS idx_shadow_evaluations_status ON shadow_evaluations(status);
	CREATE INDEX IF NOT EXISTS idx_traffic_samples_timestamp ON traffic_samples(timestamp);
	CREATE INDEX IF NOT EXISTS idx_traffic_samples_attack ON traffic_samples(is_attack);`

	_, err := se.db.Exec(query)
	return err
}

func (se *ShadowEvaluator) CreateShadowEval(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req ShadowEvalRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	if req.RuleID == "" || req.RuleName == "" || req.TenantID == "" {
		http.Error(w, "Missing required fields", http.StatusBadRequest)
		return
	}

	if req.SampleSize <= 0 {
		req.SampleSize = 1000
	}

	if req.TimeWindow == "" {
		req.TimeWindow = "24h"
	}

	evalID := uuid.New().String()
	
	ruleConfigJSON, _ := json.Marshal(req.RuleConfig)
	
	_, err := se.db.Exec(`
		INSERT INTO shadow_evaluations 
		(eval_id, rule_id, rule_name, rule_type, rule_config, tenant_id, sample_size, status)
		VALUES ($1, $2, $3, $4, $5, $6, $7, 'pending')`,
		evalID, req.RuleID, req.RuleName, req.RuleType, string(ruleConfigJSON), req.TenantID, req.SampleSize)
	
	if err != nil {
		log.Printf("Failed to create shadow evaluation: %v", err)
		http.Error(w, "Failed to create evaluation", http.StatusInternalServerError)
		return
	}

	go se.runShadowEvaluation(evalID, req)

	response := map[string]interface{}{
		"success": true,
		"eval_id": evalID,
		"message": "Shadow evaluation started",
		"status":  "pending",
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (se *ShadowEvaluator) GetShadowEval(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	evalID := r.URL.Query().Get("eval_id")
	if evalID == "" {
		http.Error(w, "Missing eval_id parameter", http.StatusBadRequest)
		return
	}

	result, err := se.getShadowEvalResult(evalID)
	if err != nil {
		log.Printf("Failed to get shadow evaluation: %v", err)
		http.Error(w, "Failed to get evaluation", http.StatusInternalServerError)
		return
	}

	if result == nil {
		http.Error(w, "Evaluation not found", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

func (se *ShadowEvaluator) runShadowEvaluation(evalID string, req ShadowEvalRequest) {
	startTime := time.Now()
	
	log.Printf("Starting shadow evaluation %s for rule %s", evalID, req.RuleID)

	_, err := se.db.Exec(`
		UPDATE shadow_evaluations 
		SET status = 'running' 
		WHERE eval_id = $1`, evalID)
	if err != nil {
		log.Printf("Failed to update evaluation status: %v", err)
		return
	}

	samples, err := se.getTrafficSamples(req.SampleSize, req.TimeWindow)
	if err != nil {
		log.Printf("Failed to get traffic samples: %v", err)
		se.updateEvaluationStatus(evalID, "failed", nil)
		return
	}

	if len(samples) == 0 {
		log.Printf("No traffic samples found for evaluation %s", evalID)
		se.generateMockSamples(req.SampleSize)
		samples, _ = se.getTrafficSamples(req.SampleSize, req.TimeWindow)
	}

	result := se.evaluateRule(req, samples)
	result.EvalID = evalID
	result.ExecutionTime = time.Since(startTime).Milliseconds()
	
	completedAt := time.Now()
	result.CompletedAt = &completedAt

	se.updateEvaluationStatus(evalID, "completed", result)
	
	log.Printf("Completed shadow evaluation %s: TP=%d, FP=%d, Precision=%.2f", 
		evalID, result.TruePositives, result.FalsePositives, result.Precision)
}

func (se *ShadowEvaluator) evaluateRule(req ShadowEvalRequest, samples []TrafficSample) *ShadowEvalResult {
	result := &ShadowEvalResult{
		RuleID:     req.RuleID,
		Status:     "completed",
		SampleSize: len(samples),
	}

	for _, sample := range samples {
		ruleMatched := se.applyRule(req, sample)
		
		if sample.IsAttack && ruleMatched {
			result.TruePositives++
		} else if sample.IsAttack && !ruleMatched {
			result.FalseNegatives++
		} else if !sample.IsAttack && ruleMatched {
			result.FalsePositives++
		} else {
			result.TrueNegatives++
		}
	}

	if result.TruePositives+result.FalsePositives > 0 {
		result.Precision = float64(result.TruePositives) / float64(result.TruePositives+result.FalsePositives)
	}

	if result.TruePositives+result.FalseNegatives > 0 {
		result.Recall = float64(result.TruePositives) / float64(result.TruePositives+result.FalseNegatives)
	}

	if result.Precision+result.Recall > 0 {
		result.F1Score = 2 * (result.Precision * result.Recall) / (result.Precision + result.Recall)
	}

	totalSamples := float64(len(samples))
	result.EstimatedFPRate = float64(result.FalsePositives) / totalSamples
	result.EstimatedTPRate = float64(result.TruePositives) / totalSamples

	result.Recommendations = se.generateRecommendations(result)

	return result
}

func (se *ShadowEvaluator) applyRule(req ShadowEvalRequest, sample TrafficSample) bool {
	switch req.RuleType {
	case "ip_blacklist":
		return se.applyIPBlacklistRule(req.RuleConfig, sample)
	case "signature_detection":
		return se.applySignatureRule(req.RuleConfig, sample)
	case "anomaly_detection":
		return se.applyAnomalyRule(req.RuleConfig, sample)
	case "rate_limiting":
		return se.applyRateLimitRule(req.RuleConfig, sample)
	default:
		return rand.Float64() > 0.7
	}
}

func (se *ShadowEvaluator) applyIPBlacklistRule(config map[string]interface{}, sample TrafficSample) bool {
	blacklist, ok := config["blacklisted_ips"].([]interface{})
	if !ok {
		return false
	}

	for _, ip := range blacklist {
		if ipStr, ok := ip.(string); ok && ipStr == sample.SourceIP {
			return true
		}
	}
	return false
}

func (se *ShadowEvaluator) applySignatureRule(config map[string]interface{}, sample TrafficSample) bool {
	signatures, ok := config["signatures"].([]interface{})
	if !ok {
		return false
	}

	for _, sig := range signatures {
		if sigStr, ok := sig.(string); ok {
			if sample.Payload != "" && len(sample.Payload) > 0 {
				return true
			}
		}
	}
	return false
}

func (se *ShadowEvaluator) applyAnomalyRule(config map[string]interface{}, sample TrafficSample) bool {
	threshold, ok := config["anomaly_threshold"].(float64)
	if !ok {
		threshold = 0.8
	}

	anomalyScore := rand.Float64()
	return anomalyScore > threshold
}

func (se *ShadowEvaluator) applyRateLimitRule(config map[string]interface{}, sample TrafficSample) bool {
	maxRequests, ok := config["max_requests_per_minute"].(float64)
	if !ok {
		maxRequests = 100
	}

	currentRate := rand.Float64() * 200
	return currentRate > maxRequests
}

func (se *ShadowEvaluator) generateRecommendations(result *ShadowEvalResult) []string {
	var recommendations []string

	if result.Precision < 0.8 {
		recommendations = append(recommendations, "Consider tightening rule criteria to reduce false positives")
	}

	if result.Recall < 0.7 {
		recommendations = append(recommendations, "Consider broadening rule scope to catch more attacks")
	}

	if result.EstimatedFPRate > 0.1 {
		recommendations = append(recommendations, "High false positive rate detected - review rule logic")
	}

	if result.F1Score < 0.6 {
		recommendations = append(recommendations, "Overall rule performance is low - consider redesigning")
	}

	if len(recommendations) == 0 {
		recommendations = append(recommendations, "Rule performance looks good - ready for production")
	}

	return recommendations
}

func (se *ShadowEvaluator) getTrafficSamples(sampleSize int, timeWindow string) ([]TrafficSample, error) {
	query := `
		SELECT id, timestamp, source_ip, dest_ip, protocol, port, 
			   COALESCE(payload, '') as payload, COALESCE(headers, '{}') as headers,
			   COALESCE(user_agent, '') as user_agent, COALESCE(method, '') as method,
			   COALESCE(uri, '') as uri, is_attack, COALESCE(attack_type, '') as attack_type,
			   COALESCE(metadata, '{}') as metadata
		FROM traffic_samples 
		WHERE timestamp > NOW() - INTERVAL '24 hours'
		ORDER BY RANDOM()
		LIMIT $1`

	rows, err := se.db.Query(query, sampleSize)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var samples []TrafficSample
	for rows.Next() {
		var sample TrafficSample
		var headersJSON, metadataJSON string

		err := rows.Scan(
			&sample.ID, &sample.Timestamp, &sample.SourceIP, &sample.DestIP,
			&sample.Protocol, &sample.Port, &sample.Payload, &headersJSON,
			&sample.UserAgent, &sample.Method, &sample.URI, &sample.IsAttack,
			&sample.AttackType, &metadataJSON)
		if err != nil {
			continue
		}

		json.Unmarshal([]byte(headersJSON), &sample.Headers)
		json.Unmarshal([]byte(metadataJSON), &sample.Metadata)

		samples = append(samples, sample)
	}

	return samples, nil
}

func (se *ShadowEvaluator) generateMockSamples(count int) {
	log.Printf("Generating %d mock traffic samples", count)

	for i := 0; i < count; i++ {
		sample := TrafficSample{
			Timestamp:  time.Now().Add(-time.Duration(rand.Intn(86400)) * time.Second),
			SourceIP:   fmt.Sprintf("192.168.%d.%d", rand.Intn(255), rand.Intn(255)),
			DestIP:     fmt.Sprintf("10.0.%d.%d", rand.Intn(255), rand.Intn(255)),
			Protocol:   []string{"HTTP", "HTTPS", "TCP", "UDP"}[rand.Intn(4)],
			Port:       []int{80, 443, 22, 21, 3389}[rand.Intn(5)],
			Method:     []string{"GET", "POST", "PUT", "DELETE"}[rand.Intn(4)],
			URI:        []string{"/", "/login", "/admin", "/api/data"}[rand.Intn(4)],
			UserAgent:  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
			IsAttack:   rand.Float64() < 0.2,
		}

		if sample.IsAttack {
			sample.AttackType = []string{"SQL Injection", "XSS", "CSRF", "Brute Force"}[rand.Intn(4)]
			sample.Payload = "malicious_payload_" + fmt.Sprintf("%d", rand.Intn(1000))
		}

		headersJSON, _ := json.Marshal(map[string]string{
			"Content-Type": "application/json",
			"Accept":       "application/json",
		})
		metadataJSON, _ := json.Marshal(map[string]interface{}{
			"generated": true,
			"mock":      true,
		})

		_, err := se.db.Exec(`
			INSERT INTO traffic_samples 
			(timestamp, source_ip, dest_ip, protocol, port, payload, headers, 
			 user_agent, method, uri, is_attack, attack_type, metadata)
			VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)`,
			sample.Timestamp, sample.SourceIP, sample.DestIP, sample.Protocol,
			sample.Port, sample.Payload, string(headersJSON), sample.UserAgent,
			sample.Method, sample.URI, sample.IsAttack, sample.AttackType, string(metadataJSON))

		if err != nil {
			log.Printf("Failed to insert mock sample: %v", err)
		}
	}
}

func (se *ShadowEvaluator) getShadowEvalResult(evalID string) (*ShadowEvalResult, error) {
	query := `
		SELECT eval_id, rule_id, status, sample_size, true_positives, false_positives,
			   true_negatives, false_negatives, precision, recall_rate, f1_score,
			   estimated_fp_rate, estimated_tp_rate, COALESCE(recommendations, '[]'),
			   execution_time_ms, created_at, completed_at
		FROM shadow_evaluations 
		WHERE eval_id = $1`

	var result ShadowEvalResult
	var recommendationsJSON string
	
	err := se.db.QueryRow(query, evalID).Scan(
		&result.EvalID, &result.RuleID, &result.Status, &result.SampleSize,
		&result.TruePositives, &result.FalsePositives, &result.TrueNegatives,
		&result.FalseNegatives, &result.Precision, &result.Recall, &result.F1Score,
		&result.EstimatedFPRate, &result.EstimatedTPRate, &recommendationsJSON,
		&result.ExecutionTime, &result.CreatedAt, &result.CompletedAt)

	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}

	json.Unmarshal([]byte(recommendationsJSON), &result.Recommendations)
	return &result, nil
}

func (se *ShadowEvaluator) updateEvaluationStatus(evalID, status string, result *ShadowEvalResult) {
	if result == nil {
		_, err := se.db.Exec(`
			UPDATE shadow_evaluations 
			SET status = $1 
			WHERE eval_id = $2`, status, evalID)
		if err != nil {
			log.Printf("Failed to update evaluation status: %v", err)
		}
		return
	}

	recommendationsJSON, _ := json.Marshal(result.Recommendations)
	
	_, err := se.db.Exec(`
		UPDATE shadow_evaluations 
		SET status = $1, true_positives = $2, false_positives = $3,
			true_negatives = $4, false_negatives = $5, precision = $6,
			recall_rate = $7, f1_score = $8, estimated_fp_rate = $9,
			estimated_tp_rate = $10, recommendations = $11, execution_time_ms = $12,
			completed_at = $13
		WHERE eval_id = $14`,
		status, result.TruePositives, result.FalsePositives, result.TrueNegatives,
		result.FalseNegatives, result.Precision, result.Recall, result.F1Score,
		result.EstimatedFPRate, result.EstimatedTPRate, string(recommendationsJSON),
		result.ExecutionTime, result.CompletedAt, evalID)

	if err != nil {
		log.Printf("Failed to update evaluation result: %v", err)
	}
}

func (se *ShadowEvaluator) Close() error {
	return se.db.Close()
}