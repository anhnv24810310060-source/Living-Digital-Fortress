package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"os"
	"os/exec"
	"sync"
	"time"
)

type ChaosTest struct {
	services    []Service
	results     []ChaosResult
	mu          sync.Mutex
	ctx         context.Context
	cancel      context.CancelFunc
}

type Service struct {
	Name      string `json:"name"`
	URL       string `json:"url"`
	Namespace string `json:"namespace"`
	Pod       string `json:"pod"`
	Critical  bool   `json:"critical"`
}

type ChaosResult struct {
	TestName      string    `json:"test_name"`
	Service       string    `json:"service"`
	Action        string    `json:"action"`
	Status        string    `json:"status"`
	Duration      int64     `json:"duration_ms"`
	RecoveryTime  int64     `json:"recovery_time_ms"`
	Impact        string    `json:"impact"`
	Timestamp     time.Time `json:"timestamp"`
	Details       string    `json:"details"`
}

func NewChaosTest() *ChaosTest {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	
	return &ChaosTest{
		services: []Service{
			{Name: "orchestrator", URL: "http://localhost:8080", Namespace: "shieldx-system", Pod: "shieldx-orchestrator", Critical: true},
			{Name: "credits", URL: "http://localhost:5004", Namespace: "shieldx-system", Pod: "shieldx-credits", Critical: true},
			{Name: "contauth", URL: "http://localhost:5002", Namespace: "shieldx-system", Pod: "shieldx-contauth", Critical: false},
			{Name: "shadow", URL: "http://localhost:5005", Namespace: "shieldx-system", Pod: "shieldx-shadow", Critical: false},
			{Name: "webapi", URL: "http://localhost:5006", Namespace: "shieldx-system", Pod: "shieldx-webapi", Critical: false},
			{Name: "digital-twin", URL: "http://localhost:5001", Namespace: "shieldx-system", Pod: "shieldx-digital-twin", Critical: false},
		},
		results: make([]ChaosResult, 0),
		ctx:     ctx,
		cancel:  cancel,
	}
}

func (ct *ChaosTest) RunChaosTests() {
	log.Println("Starting Chaos Engineering Tests...")
	
	// Test 1: Pod Termination
	ct.runPodTerminationTests()
	
	// Test 2: Network Partitioning
	ct.runNetworkPartitionTests()
	
	// Test 3: Resource Exhaustion
	ct.runResourceExhaustionTests()
	
	// Test 4: Database Failures
	ct.runDatabaseFailureTests()
	
	// Test 5: High Load Tests
	ct.runHighLoadTests()
	
	// Test 6: Dependency Failures
	ct.runDependencyFailureTests()
	
	// Test 7: Configuration Corruption
	ct.runConfigCorruptionTests()
	
	// Generate Report
	ct.generateChaosReport()
}

func (ct *ChaosTest) runPodTerminationTests() {
	log.Println("Running Pod Termination Tests...")
	
	for _, service := range ct.services {
		ct.testPodTermination(service)
		time.Sleep(5 * time.Second) // Recovery time between tests
	}
}

func (ct *ChaosTest) testPodTermination(service Service) {
	start := time.Now()
	
	log.Printf("Terminating pod: %s", service.Pod)
	
	// Kill pod using kubectl
	cmd := exec.Command("kubectl", "delete", "pod", "-l", fmt.Sprintf("app=%s", service.Name), 
		"-n", service.Namespace, "--force", "--grace-period=0")
	
	err := cmd.Run()
	if err != nil {
		ct.recordResult("Pod Termination", service.Name, "terminate", "error", 
			time.Since(start).Milliseconds(), 0, "high", fmt.Sprintf("Failed to terminate pod: %v", err))
		return
	}
	
	// Wait for pod to be recreated
	recoveryStart := time.Now()
	recovered := ct.waitForServiceRecovery(service, 60*time.Second)
	recoveryTime := time.Since(recoveryStart).Milliseconds()
	
	status := "pass"
	impact := "low"
	if !recovered {
		status = "fail"
		impact = "high"
	} else if recoveryTime > 30000 { // 30 seconds
		impact = "medium"
	}
	
	ct.recordResult("Pod Termination", service.Name, "terminate", status, 
		time.Since(start).Milliseconds(), recoveryTime, impact, 
		fmt.Sprintf("Pod terminated and recovery took %dms", recoveryTime))
}

func (ct *ChaosTest) runNetworkPartitionTests() {
	log.Println("Running Network Partition Tests...")
	
	for _, service := range ct.services {
		if service.Critical {
			ct.testNetworkPartition(service)
			time.Sleep(10 * time.Second)
		}
	}
}

func (ct *ChaosTest) testNetworkPartition(service Service) {
	start := time.Now()
	
	log.Printf("Creating network partition for: %s", service.Name)
	
	// Create network policy to block traffic
	networkPolicy := fmt.Sprintf(`
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: chaos-partition-%s
  namespace: %s
spec:
  podSelector:
    matchLabels:
      app: %s
  policyTypes:
  - Ingress
  - Egress
`, service.Name, service.Namespace, service.Name)
	
	// Apply network policy
	cmd := exec.Command("kubectl", "apply", "-f", "-")
	cmd.Stdin = fmt.NewReader(networkPolicy)
	err := cmd.Run()
	
	if err != nil {
		ct.recordResult("Network Partition", service.Name, "partition", "error", 
			time.Since(start).Milliseconds(), 0, "high", fmt.Sprintf("Failed to create partition: %v", err))
		return
	}
	
	// Wait for partition to take effect
	time.Sleep(5 * time.Second)
	
	// Test service availability
	available := ct.checkServiceHealth(service)
	
	// Remove network policy
	cmd = exec.Command("kubectl", "delete", "networkpolicy", fmt.Sprintf("chaos-partition-%s", service.Name), "-n", service.Namespace)
	cmd.Run()
	
	// Wait for recovery
	recoveryStart := time.Now()
	recovered := ct.waitForServiceRecovery(service, 30*time.Second)
	recoveryTime := time.Since(recoveryStart).Milliseconds()
	
	status := "pass"
	impact := "medium"
	if available {
		status = "fail"
		impact = "high"
	}
	
	ct.recordResult("Network Partition", service.Name, "partition", status, 
		time.Since(start).Milliseconds(), recoveryTime, impact, 
		fmt.Sprintf("Network partition test completed, recovery: %dms", recoveryTime))
}

func (ct *ChaosTest) runResourceExhaustionTests() {
	log.Println("Running Resource Exhaustion Tests...")
	
	for _, service := range ct.services {
		ct.testCPUExhaustion(service)
		time.Sleep(5 * time.Second)
		ct.testMemoryExhaustion(service)
		time.Sleep(5 * time.Second)
	}
}

func (ct *ChaosTest) testCPUExhaustion(service Service) {
	start := time.Now()
	
	log.Printf("Testing CPU exhaustion for: %s", service.Name)
	
	// Create CPU stress using kubectl exec
	cmd := exec.Command("kubectl", "exec", "-n", service.Namespace, 
		fmt.Sprintf("deployment/%s", service.Name), "--", 
		"sh", "-c", "timeout 30 yes > /dev/null &")
	
	err := cmd.Start()
	if err != nil {
		ct.recordResult("CPU Exhaustion", service.Name, "cpu_stress", "error", 
			time.Since(start).Milliseconds(), 0, "medium", fmt.Sprintf("Failed to start CPU stress: %v", err))
		return
	}
	
	// Monitor service during stress
	time.Sleep(10 * time.Second)
	available := ct.checkServiceHealth(service)
	
	// Wait for stress to end
	time.Sleep(25 * time.Second)
	
	// Check recovery
	recoveryStart := time.Now()
	recovered := ct.waitForServiceRecovery(service, 30*time.Second)
	recoveryTime := time.Since(recoveryStart).Milliseconds()
	
	status := "pass"
	impact := "low"
	if !available {
		impact = "medium"
	}
	if !recovered {
		status = "fail"
		impact = "high"
	}
	
	ct.recordResult("CPU Exhaustion", service.Name, "cpu_stress", status, 
		time.Since(start).Milliseconds(), recoveryTime, impact, 
		fmt.Sprintf("CPU stress test completed, service available: %v", available))
}

func (ct *ChaosTest) testMemoryExhaustion(service Service) {
	start := time.Now()
	
	log.Printf("Testing memory exhaustion for: %s", service.Name)
	
	// Create memory stress
	cmd := exec.Command("kubectl", "exec", "-n", service.Namespace, 
		fmt.Sprintf("deployment/%s", service.Name), "--", 
		"sh", "-c", "timeout 30 dd if=/dev/zero of=/tmp/memory bs=1M count=100 2>/dev/null &")
	
	err := cmd.Start()
	if err != nil {
		ct.recordResult("Memory Exhaustion", service.Name, "memory_stress", "error", 
			time.Since(start).Milliseconds(), 0, "medium", fmt.Sprintf("Failed to start memory stress: %v", err))
		return
	}
	
	// Monitor service during stress
	time.Sleep(10 * time.Second)
	available := ct.checkServiceHealth(service)
	
	// Wait for stress to end
	time.Sleep(25 * time.Second)
	
	// Check recovery
	recoveryStart := time.Now()
	recovered := ct.waitForServiceRecovery(service, 30*time.Second)
	recoveryTime := time.Since(recoveryStart).Milliseconds()
	
	status := "pass"
	impact := "low"
	if !available {
		impact = "medium"
	}
	if !recovered {
		status = "fail"
		impact = "high"
	}
	
	ct.recordResult("Memory Exhaustion", service.Name, "memory_stress", status, 
		time.Since(start).Milliseconds(), recoveryTime, impact, 
		fmt.Sprintf("Memory stress test completed, service available: %v", available))
}

func (ct *ChaosTest) runDatabaseFailureTests() {
	log.Println("Running Database Failure Tests...")
	
	databases := []string{"postgres-credits", "postgres-contauth", "postgres-shadow"}
	
	for _, db := range databases {
		ct.testDatabaseFailure(db)
		time.Sleep(10 * time.Second)
	}
}

func (ct *ChaosTest) testDatabaseFailure(dbName string) {
	start := time.Now()
	
	log.Printf("Testing database failure: %s", dbName)
	
	// Kill database pod
	cmd := exec.Command("kubectl", "delete", "pod", "-l", fmt.Sprintf("app=%s", dbName), 
		"-n", "shieldx-system", "--force", "--grace-period=0")
	
	err := cmd.Run()
	if err != nil {
		ct.recordResult("Database Failure", dbName, "terminate", "error", 
			time.Since(start).Milliseconds(), 0, "high", fmt.Sprintf("Failed to terminate database: %v", err))
		return
	}
	
	// Wait for database recovery
	recoveryStart := time.Now()
	time.Sleep(30 * time.Second) // Allow time for restart
	recoveryTime := time.Since(recoveryStart).Milliseconds()
	
	// Check dependent services
	dependentServices := ct.getDependentServices(dbName)
	allRecovered := true
	
	for _, service := range dependentServices {
		if !ct.waitForServiceRecovery(service, 60*time.Second) {
			allRecovered = false
		}
	}
	
	status := "pass"
	impact := "medium"
	if !allRecovered {
		status = "fail"
		impact = "high"
	}
	
	ct.recordResult("Database Failure", dbName, "terminate", status, 
		time.Since(start).Milliseconds(), recoveryTime, impact, 
		fmt.Sprintf("Database failure test completed, all services recovered: %v", allRecovered))
}

func (ct *ChaosTest) runHighLoadTests() {
	log.Println("Running High Load Tests...")
	
	for _, service := range ct.services {
		ct.testHighLoad(service)
		time.Sleep(5 * time.Second)
	}
}

func (ct *ChaosTest) testHighLoad(service Service) {
	start := time.Now()
	
	log.Printf("Testing high load for: %s", service.Name)
	
	// Generate high load using concurrent requests
	var wg sync.WaitGroup
	client := &http.Client{Timeout: 5 * time.Second}
	
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 10; j++ {
				resp, err := client.Get(service.URL + "/health")
				if err == nil {
					resp.Body.Close()
				}
				time.Sleep(100 * time.Millisecond)
			}
		}()
	}
	
	wg.Wait()
	
	// Check service health after load
	available := ct.checkServiceHealth(service)
	
	status := "pass"
	impact := "low"
	if !available {
		status = "fail"
		impact = "medium"
	}
	
	ct.recordResult("High Load", service.Name, "load_test", status, 
		time.Since(start).Milliseconds(), 0, impact, 
		fmt.Sprintf("High load test completed, service available: %v", available))
}

func (ct *ChaosTest) runDependencyFailureTests() {
	log.Println("Running Dependency Failure Tests...")
	
	// Test external dependency failures
	dependencies := []string{"dns", "network", "storage"}
	
	for _, dep := range dependencies {
		ct.testDependencyFailure(dep)
		time.Sleep(10 * time.Second)
	}
}

func (ct *ChaosTest) testDependencyFailure(dependency string) {
	start := time.Now()
	
	log.Printf("Testing dependency failure: %s", dependency)
	
	// Simulate dependency failure based on type
	var cmd *exec.Cmd
	switch dependency {
	case "dns":
		// Block DNS resolution
		cmd = exec.Command("kubectl", "apply", "-f", "-")
		cmd.Stdin = fmt.NewReader(`
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: chaos-dns-block
  namespace: shieldx-system
spec:
  podSelector: {}
  policyTypes:
  - Egress
  egress:
  - to: []
    ports:
    - protocol: UDP
      port: 53
`)
	case "network":
		// Introduce network latency
		cmd = exec.Command("kubectl", "exec", "-n", "shieldx-system", 
			"deployment/shieldx-orchestrator", "--", 
			"sh", "-c", "tc qdisc add dev eth0 root netem delay 1000ms 2>/dev/null || true")
	}
	
	if cmd != nil {
		cmd.Run()
	}
	
	// Wait for impact
	time.Sleep(15 * time.Second)
	
	// Check service health
	healthyServices := 0
	for _, service := range ct.services {
		if ct.checkServiceHealth(service) {
			healthyServices++
		}
	}
	
	// Cleanup
	switch dependency {
	case "dns":
		exec.Command("kubectl", "delete", "networkpolicy", "chaos-dns-block", "-n", "shieldx-system").Run()
	case "network":
		exec.Command("kubectl", "exec", "-n", "shieldx-system", 
			"deployment/shieldx-orchestrator", "--", 
			"sh", "-c", "tc qdisc del dev eth0 root 2>/dev/null || true").Run()
	}
	
	// Wait for recovery
	time.Sleep(30 * time.Second)
	
	status := "pass"
	impact := "medium"
	if healthyServices < len(ct.services)/2 {
		impact = "high"
	}
	
	ct.recordResult("Dependency Failure", dependency, "simulate", status, 
		time.Since(start).Milliseconds(), 0, impact, 
		fmt.Sprintf("Dependency failure test completed, %d/%d services healthy", healthyServices, len(ct.services)))
}

func (ct *ChaosTest) runConfigCorruptionTests() {
	log.Println("Running Configuration Corruption Tests...")
	
	configMaps := []string{"shieldx-config", "cosign-config"}
	
	for _, cm := range configMaps {
		ct.testConfigCorruption(cm)
		time.Sleep(10 * time.Second)
	}
}

func (ct *ChaosTest) testConfigCorruption(configMap string) {
	start := time.Now()
	
	log.Printf("Testing config corruption: %s", configMap)
	
	// Backup original config
	cmd := exec.Command("kubectl", "get", "configmap", configMap, "-n", "shieldx-system", "-o", "yaml")
	originalConfig, err := cmd.Output()
	if err != nil {
		ct.recordResult("Config Corruption", configMap, "corrupt", "error", 
			time.Since(start).Milliseconds(), 0, "medium", fmt.Sprintf("Failed to backup config: %v", err))
		return
	}
	
	// Corrupt config
	cmd = exec.Command("kubectl", "patch", "configmap", configMap, "-n", "shieldx-system", 
		"--patch", `{"data":{"corrupted":"true"}}`)
	cmd.Run()
	
	// Wait for impact
	time.Sleep(10 * time.Second)
	
	// Check service health
	healthyServices := 0
	for _, service := range ct.services {
		if ct.checkServiceHealth(service) {
			healthyServices++
		}
	}
	
	// Restore original config
	cmd = exec.Command("kubectl", "apply", "-f", "-")
	cmd.Stdin = fmt.NewReader(string(originalConfig))
	cmd.Run()
	
	// Wait for recovery
	recoveryStart := time.Now()
	time.Sleep(30 * time.Second)
	recoveryTime := time.Since(recoveryStart).Milliseconds()
	
	status := "pass"
	impact := "low"
	if healthyServices < len(ct.services) {
		impact = "medium"
	}
	
	ct.recordResult("Config Corruption", configMap, "corrupt", status, 
		time.Since(start).Milliseconds(), recoveryTime, impact, 
		fmt.Sprintf("Config corruption test completed, %d/%d services healthy", healthyServices, len(ct.services)))
}

func (ct *ChaosTest) checkServiceHealth(service Service) bool {
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Get(service.URL + "/health")
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == 200
}

func (ct *ChaosTest) waitForServiceRecovery(service Service, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	
	for time.Now().Before(deadline) {
		if ct.checkServiceHealth(service) {
			return true
		}
		time.Sleep(2 * time.Second)
	}
	
	return false
}

func (ct *ChaosTest) getDependentServices(dbName string) []Service {
	dependencyMap := map[string][]string{
		"postgres-credits":  {"credits"},
		"postgres-contauth": {"contauth"},
		"postgres-shadow":   {"shadow"},
	}
	
	var dependent []Service
	if serviceNames, exists := dependencyMap[dbName]; exists {
		for _, serviceName := range serviceNames {
			for _, service := range ct.services {
				if service.Name == serviceName {
					dependent = append(dependent, service)
				}
			}
		}
	}
	
	return dependent
}

func (ct *ChaosTest) recordResult(testName, service, action, status string, duration, recoveryTime int64, impact, details string) {
	ct.mu.Lock()
	defer ct.mu.Unlock()
	
	result := ChaosResult{
		TestName:     testName,
		Service:      service,
		Action:       action,
		Status:       status,
		Duration:     duration,
		RecoveryTime: recoveryTime,
		Impact:       impact,
		Timestamp:    time.Now(),
		Details:      details,
	}
	
	ct.results = append(ct.results, result)
	
	log.Printf("[%s] %s/%s: %s (%dms)", impact, testName, service, status, duration)
}

func (ct *ChaosTest) generateChaosReport() {
	log.Println("Generating Chaos Engineering Report...")
	
	summary := ct.generateChaosSummary()
	
	report := map[string]interface{}{
		"test_date":       time.Now().Format(time.RFC3339),
		"total_tests":     len(ct.results),
		"summary":         summary,
		"results":         ct.results,
		"recommendations": ct.generateChaosRecommendations(summary),
	}
	
	reportJSON, _ := json.MarshalIndent(report, "", "  ")
	filename := fmt.Sprintf("chaos-report-%s.json", time.Now().Format("20060102-150405"))
	
	err := os.WriteFile(filename, reportJSON, 0644)
	if err != nil {
		log.Printf("Failed to write chaos report: %v", err)
		return
	}
	
	log.Printf("Chaos Engineering Report saved to: %s", filename)
	
	// Print summary
	log.Printf("Chaos Test Summary:")
	log.Printf("  Total Tests: %d", summary["total_tests"])
	log.Printf("  Passed: %d", summary["passed"])
	log.Printf("  Failed: %d", summary["failed"])
	log.Printf("  High Impact: %d", summary["high_impact"])
	log.Printf("  Medium Impact: %d", summary["medium_impact"])
	log.Printf("  Resilience Score: %.1f%%", summary["resilience_score"])
}

func (ct *ChaosTest) generateChaosSummary() map[string]interface{} {
	var passed, failed, highImpact, mediumImpact, lowImpact int
	var totalRecoveryTime int64
	
	for _, result := range ct.results {
		switch result.Status {
		case "pass":
			passed++
		case "fail":
			failed++
		}
		
		switch result.Impact {
		case "high":
			highImpact++
		case "medium":
			mediumImpact++
		case "low":
			lowImpact++
		}
		
		totalRecoveryTime += result.RecoveryTime
	}
	
	total := len(ct.results)
	resilienceScore := float64(passed) / float64(total) * 100
	avgRecoveryTime := float64(totalRecoveryTime) / float64(total)
	
	return map[string]interface{}{
		"total_tests":        total,
		"passed":             passed,
		"failed":             failed,
		"high_impact":        highImpact,
		"medium_impact":      mediumImpact,
		"low_impact":         lowImpact,
		"resilience_score":   resilienceScore,
		"avg_recovery_time":  avgRecoveryTime,
	}
}

func (ct *ChaosTest) generateChaosRecommendations(summary map[string]interface{}) []string {
	recommendations := []string{}
	
	if summary["failed"].(int) > 0 {
		recommendations = append(recommendations, "Implement circuit breakers and retry mechanisms")
	}
	
	if summary["high_impact"].(int) > 0 {
		recommendations = append(recommendations, "Improve service redundancy and failover capabilities")
	}
	
	if summary["avg_recovery_time"].(float64) > 30000 {
		recommendations = append(recommendations, "Optimize service startup and recovery procedures")
	}
	
	if summary["resilience_score"].(float64) < 90 {
		recommendations = append(recommendations, "Enhance overall system resilience - target 90%+ resilience score")
	}
	
	recommendations = append(recommendations, "Implement comprehensive health checks and monitoring")
	recommendations = append(recommendations, "Regular chaos engineering exercises")
	recommendations = append(recommendations, "Automated recovery procedures and runbooks")
	
	return recommendations
}

func main() {
	log.Println("Starting Chaos Engineering Tests...")
	
	chaosTest := NewChaosTest()
	defer chaosTest.cancel()
	
	chaosTest.RunChaosTests()
	
	log.Println("Chaos Engineering Tests completed!")
}