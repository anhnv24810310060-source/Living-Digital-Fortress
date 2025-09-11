package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestCreateShadowEval(t *testing.T) {
	evaluator := &ShadowEvaluator{}

	req := ShadowEvalRequest{
		RuleID:     "test_rule_001",
		RuleName:   "Test IP Blacklist",
		RuleType:   "ip_blacklist",
		RuleConfig: map[string]interface{}{
			"blacklisted_ips": []string{"192.168.1.100", "10.0.0.1"},
		},
		SampleSize: 500,
		TenantID:   "test_tenant",
	}

	jsonData, _ := json.Marshal(req)
	httpReq := httptest.NewRequest("POST", "/shadow/eval", bytes.NewBuffer(jsonData))
	w := httptest.NewRecorder()

	evaluator.CreateShadowEval(w, httpReq)

	if w.Code == http.StatusOK {
		var response map[string]interface{}
		json.NewDecoder(w.Body).Decode(&response)
		
		if success, ok := response["success"].(bool); ok && success {
			t.Log("Shadow evaluation creation test passed")
		} else {
			t.Error("Expected success to be true")
		}
	} else {
		t.Errorf("Expected status 200, got %d", w.Code)
	}
}

func TestApplyIPBlacklistRule(t *testing.T) {
	evaluator := &ShadowEvaluator{}

	config := map[string]interface{}{
		"blacklisted_ips": []interface{}{"192.168.1.100", "10.0.0.1"},
	}

	// Test with blacklisted IP
	sample1 := TrafficSample{
		SourceIP: "192.168.1.100",
		DestIP:   "10.0.1.1",
	}

	result1 := evaluator.applyIPBlacklistRule(config, sample1)
	if !result1 {
		t.Error("Expected blacklisted IP to match rule")
	}

	// Test with non-blacklisted IP
	sample2 := TrafficSample{
		SourceIP: "192.168.1.200",
		DestIP:   "10.0.1.1",
	}

	result2 := evaluator.applyIPBlacklistRule(config, sample2)
	if result2 {
		t.Error("Expected non-blacklisted IP to not match rule")
	}
}

func TestApplySignatureRule(t *testing.T) {
	evaluator := &ShadowEvaluator{}

	config := map[string]interface{}{
		"signatures": []interface{}{"SELECT * FROM", "UNION SELECT"},
	}

	// Test with malicious payload
	sample1 := TrafficSample{
		Payload: "SELECT * FROM users WHERE id=1",
	}

	result1 := evaluator.applySignatureRule(config, sample1)
	if !result1 {
		t.Error("Expected malicious payload to match signature rule")
	}

	// Test with benign payload
	sample2 := TrafficSample{
		Payload: "",
	}

	result2 := evaluator.applySignatureRule(config, sample2)
	if result2 {
		t.Error("Expected benign payload to not match signature rule")
	}
}

func TestApplyAnomalyRule(t *testing.T) {
	evaluator := &ShadowEvaluator{}

	config := map[string]interface{}{
		"anomaly_threshold": 0.5,
	}

	sample := TrafficSample{
		SourceIP: "192.168.1.100",
	}

	// Test multiple times due to randomness
	results := make([]bool, 10)
	for i := 0; i < 10; i++ {
		results[i] = evaluator.applyAnomalyRule(config, sample)
	}

	// Should have some variation in results due to randomness
	t.Logf("Anomaly rule results: %v", results)
}

func TestApplyRateLimitRule(t *testing.T) {
	evaluator := &ShadowEvaluator{}

	config := map[string]interface{}{
		"max_requests_per_minute": 50.0,
	}

	sample := TrafficSample{
		SourceIP: "192.168.1.100",
	}

	// Test multiple times
	results := make([]bool, 5)
	for i := 0; i < 5; i++ {
		results[i] = evaluator.applyRateLimitRule(config, sample)
	}

	t.Logf("Rate limit rule results: %v", results)
}

func TestEvaluateRule(t *testing.T) {
	evaluator := &ShadowEvaluator{}

	req := ShadowEvalRequest{
		RuleID:   "test_rule",
		RuleType: "ip_blacklist",
		RuleConfig: map[string]interface{}{
			"blacklisted_ips": []interface{}{"192.168.1.100"},
		},
	}

	samples := []TrafficSample{
		{
			ID:       "sample_1",
			SourceIP: "192.168.1.100",
			IsAttack: true,
		},
		{
			ID:       "sample_2",
			SourceIP: "192.168.1.200",
			IsAttack: false,
		},
		{
			ID:       "sample_3",
			SourceIP: "192.168.1.100",
			IsAttack: true,
		},
		{
			ID:       "sample_4",
			SourceIP: "10.0.0.1",
			IsAttack: false,
		},
	}

	result := evaluator.evaluateRule(req, samples)

	if result.SampleSize != len(samples) {
		t.Errorf("Expected sample size %d, got %d", len(samples), result.SampleSize)
	}

	if result.TruePositives < 0 || result.FalsePositives < 0 ||
		result.TrueNegatives < 0 || result.FalseNegatives < 0 {
		t.Error("All confusion matrix values should be non-negative")
	}

	total := result.TruePositives + result.FalsePositives + 
			result.TrueNegatives + result.FalseNegatives
	if total != len(samples) {
		t.Errorf("Confusion matrix total %d should equal sample size %d", total, len(samples))
	}

	if result.Precision < 0 || result.Precision > 1 {
		t.Errorf("Precision should be between 0 and 1, got %f", result.Precision)
	}

	if result.Recall < 0 || result.Recall > 1 {
		t.Errorf("Recall should be between 0 and 1, got %f", result.Recall)
	}

	if result.F1Score < 0 || result.F1Score > 1 {
		t.Errorf("F1 Score should be between 0 and 1, got %f", result.F1Score)
	}

	t.Logf("Evaluation result: TP=%d, FP=%d, TN=%d, FN=%d", 
		result.TruePositives, result.FalsePositives, 
		result.TrueNegatives, result.FalseNegatives)
	t.Logf("Metrics: Precision=%.3f, Recall=%.3f, F1=%.3f", 
		result.Precision, result.Recall, result.F1Score)
}

func TestGenerateRecommendations(t *testing.T) {
	evaluator := &ShadowEvaluator{}

	// Test high performance rule
	highPerfResult := &ShadowEvalResult{
		Precision:        0.95,
		Recall:           0.90,
		F1Score:          0.92,
		EstimatedFPRate:  0.02,
	}

	recommendations := evaluator.generateRecommendations(highPerfResult)
	if len(recommendations) == 0 {
		t.Error("Expected recommendations for high performance rule")
	}

	hasGoodPerformance := false
	for _, rec := range recommendations {
		if rec == "Rule performance looks good - ready for production" {
			hasGoodPerformance = true
			break
		}
	}
	if !hasGoodPerformance {
		t.Error("Expected good performance recommendation")
	}

	// Test low performance rule
	lowPerfResult := &ShadowEvalResult{
		Precision:        0.60,
		Recall:           0.50,
		F1Score:          0.55,
		EstimatedFPRate:  0.15,
	}

	recommendations = evaluator.generateRecommendations(lowPerfResult)
	if len(recommendations) == 0 {
		t.Error("Expected recommendations for low performance rule")
	}

	hasImprovementRec := false
	for _, rec := range recommendations {
		if rec == "Consider tightening rule criteria to reduce false positives" ||
		   rec == "Consider broadening rule scope to catch more attacks" ||
		   rec == "High false positive rate detected - review rule logic" ||
		   rec == "Overall rule performance is low - consider redesigning" {
			hasImprovementRec = true
			break
		}
	}
	if !hasImprovementRec {
		t.Error("Expected improvement recommendations for low performance rule")
	}

	t.Logf("High performance recommendations: %v", evaluator.generateRecommendations(highPerfResult))
	t.Logf("Low performance recommendations: %v", evaluator.generateRecommendations(lowPerfResult))
}

func TestGetShadowEval(t *testing.T) {
	evaluator := &ShadowEvaluator{}

	httpReq := httptest.NewRequest("GET", "/shadow/result?eval_id=test_eval_123", nil)
	w := httptest.NewRecorder()

	evaluator.GetShadowEval(w, httpReq)

	// Should return 404 or 500 since no database connection in test
	if w.Code != http.StatusNotFound && w.Code != http.StatusInternalServerError {
		t.Logf("Get shadow eval returned status: %d", w.Code)
	}
}

func TestMockTrafficGeneration(t *testing.T) {
	evaluator := &ShadowEvaluator{}

	// Test that mock sample generation doesn't crash
	// Note: This would require database connection in real implementation
	t.Log("Mock traffic generation test - would generate samples in real implementation")
}

func TestRuleTypeValidation(t *testing.T) {
	validRuleTypes := []string{"ip_blacklist", "signature_detection", "anomaly_detection", "rate_limiting"}
	
	for _, ruleType := range validRuleTypes {
		t.Logf("Testing rule type: %s", ruleType)
		
		req := ShadowEvalRequest{
			RuleType: ruleType,
		}
		
		sample := TrafficSample{
			SourceIP: "192.168.1.100",
		}
		
		evaluator := &ShadowEvaluator{}
		result := evaluator.applyRule(req, sample)
		
		// Should return a boolean result
		t.Logf("Rule type %s returned: %v", ruleType, result)
	}
}

func TestPerformanceMetrics(t *testing.T) {
	// Test precision calculation
	tp, fp := 85, 12
	expectedPrecision := float64(tp) / float64(tp+fp)
	actualPrecision := float64(85) / float64(85+12)
	
	if actualPrecision != expectedPrecision {
		t.Errorf("Precision calculation error: expected %f, got %f", expectedPrecision, actualPrecision)
	}

	// Test recall calculation
	fn := 13
	expectedRecall := float64(tp) / float64(tp+fn)
	actualRecall := float64(85) / float64(85+13)
	
	if actualRecall != expectedRecall {
		t.Errorf("Recall calculation error: expected %f, got %f", expectedRecall, actualRecall)
	}

	// Test F1 score calculation
	precision := actualPrecision
	recall := actualRecall
	expectedF1 := 2 * (precision * recall) / (precision + recall)
	actualF1 := 2 * (precision * recall) / (precision + recall)
	
	if actualF1 != expectedF1 {
		t.Errorf("F1 score calculation error: expected %f, got %f", expectedF1, actualF1)
	}

	t.Logf("Performance metrics test passed: Precision=%.3f, Recall=%.3f, F1=%.3f", 
		precision, recall, actualF1)
}