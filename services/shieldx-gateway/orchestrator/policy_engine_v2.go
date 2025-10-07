// Package main - Production Policy Engine với versioning và rollback
// Implements zero-downtime policy updates with atomic versioning
package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"shieldx/pkg/ledger"
	"shieldx/pkg/policy"
)

// PolicyEngineV2 manages versioned policies with rollback capability
type PolicyEngineV2 struct {
	mu sync.RWMutex

	// Versioned policies
	current    atomic.Value // *VersionedPolicy
	history    []*VersionedPolicy
	maxHistory int

	// Hot reload
	watchPath   string
	watchTicker *time.Ticker
	stopWatch   chan struct{}

	// Rollback tracking
	rollbackStack []uint64 // Version IDs for rollback
	maxRollbacks  int

	// A/B Testing support
	abTestEnabled atomic.Bool
	testPolicy    atomic.Value // *VersionedPolicy
	testTraffic   float64      // Percentage of traffic to test policy

	// Metrics
	evaluations       atomic.Uint64
	hotReloads        atomic.Uint64
	rollbacks         atomic.Uint64
	abTestEvaluations atomic.Uint64
	versionMismatches atomic.Uint64

	// Policy validation
	validator *PolicyValidator
}

// VersionedPolicy wraps policy with version metadata
type VersionedPolicy struct {
	Version   uint64
	Policy    policy.Config
	Hash      string
	CreatedAt time.Time
	CreatedBy string
	Checksum  string
	Metadata  map[string]string
}

// PolicyValidator validates policy before activation
type PolicyValidator struct {
	// Validation rules
	maxRuleCount      int
	maxComplexity     int
	forbiddenPatterns []string
	requireApproval   bool
}

// NewPolicyEngineV2 creates advanced policy engine
func NewPolicyEngineV2(watchPath string) *PolicyEngineV2 {
	pe := &PolicyEngineV2{
		maxHistory:    20,
		watchPath:     watchPath,
		stopWatch:     make(chan struct{}),
		rollbackStack: make([]uint64, 0, 10),
		maxRollbacks:  10,
		testTraffic:   0.1, // 10% default
		validator:     NewPolicyValidator(),
	}

	// Initialize with default policy
	defaultPolicy := &VersionedPolicy{
		Version:   1,
		Policy:    policy.Config{AllowAll: false},
		Hash:      pe.calculateHash(policy.Config{AllowAll: false}),
		CreatedAt: time.Now(),
		CreatedBy: "system",
	}
	pe.current.Store(defaultPolicy)
	pe.history = append(pe.history, defaultPolicy)

	return pe
}

// Start begins hot reload watching
func (pe *PolicyEngineV2) Start() error {
	if pe.watchPath == "" {
		return fmt.Errorf("watch path not configured")
	}

	// Start file watcher
	pe.watchTicker = time.NewTicker(3 * time.Second)
	go pe.watchLoop()

	log.Printf("[policy-v2] Started hot reload watcher on %s", pe.watchPath)
	return nil
}

// Stop stops the policy engine
func (pe *PolicyEngineV2) Stop() {
	if pe.watchTicker != nil {
		pe.watchTicker.Stop()
	}
	close(pe.stopWatch)
}

// LoadPolicy loads and activates a new policy version
func (pe *PolicyEngineV2) LoadPolicy(pol policy.Config, createdBy string) error {
	// Validate policy
	if err := pe.validator.Validate(pol); err != nil {
		return fmt.Errorf("policy validation failed: %w", err)
	}

	// Create versioned policy
	pe.mu.Lock()
	currentVer := pe.getCurrentVersion()
	newVersion := &VersionedPolicy{
		Version:   currentVer + 1,
		Policy:    pol,
		Hash:      pe.calculateHash(pol),
		CreatedAt: time.Now(),
		CreatedBy: createdBy,
		Checksum:  pe.calculateChecksum(pol),
	}
	pe.mu.Unlock()

	// Atomic activation
	return pe.activatePolicy(newVersion)
}

// activatePolicy atomically switches to new policy version
func (pe *PolicyEngineV2) activatePolicy(newVer *VersionedPolicy) error {
	pe.mu.Lock()
	defer pe.mu.Unlock()

	// Store previous version for rollback
	if current := pe.getCurrent(); current != nil {
		pe.rollbackStack = append(pe.rollbackStack, current.Version)
		if len(pe.rollbackStack) > pe.maxRollbacks {
			pe.rollbackStack = pe.rollbackStack[1:]
		}
	}

	// Atomic swap
	pe.current.Store(newVer)

	// Add to history
	pe.history = append(pe.history, newVer)
	if len(pe.history) > pe.maxHistory {
		pe.history = pe.history[1:]
	}

	pe.hotReloads.Add(1)

	// Audit log
	_ = ledger.AppendJSONLine(ledgerPath, serviceName, "policy.activated", map[string]any{
		"version":    newVer.Version,
		"hash":       newVer.Hash,
		"created_by": newVer.CreatedBy,
		"timestamp":  newVer.CreatedAt,
	})

	log.Printf("[policy-v2] Activated policy version %d (hash: %s)", newVer.Version, newVer.Hash[:8])

	return nil
}

// Rollback reverts to previous policy version
func (pe *PolicyEngineV2) Rollback() error {
	pe.mu.Lock()
	defer pe.mu.Unlock()

	if len(pe.rollbackStack) == 0 {
		return fmt.Errorf("no version available for rollback")
	}

	// Pop last version from stack
	targetVersion := pe.rollbackStack[len(pe.rollbackStack)-1]
	pe.rollbackStack = pe.rollbackStack[:len(pe.rollbackStack)-1]

	// Find target version in history
	var targetPolicy *VersionedPolicy
	for _, ver := range pe.history {
		if ver.Version == targetVersion {
			targetPolicy = ver
			break
		}
	}

	if targetPolicy == nil {
		return fmt.Errorf("target version %d not found in history", targetVersion)
	}

	// Atomic swap back
	pe.current.Store(targetPolicy)
	pe.rollbacks.Add(1)

	// Audit log
	_ = ledger.AppendJSONLine(ledgerPath, serviceName, "policy.rollback", map[string]any{
		"version":   targetPolicy.Version,
		"hash":      targetPolicy.Hash,
		"timestamp": time.Now(),
	})

	log.Printf("[policy-v2] Rolled back to policy version %d", targetPolicy.Version)

	return nil
}

// Evaluate evaluates request against current policy
func (pe *PolicyEngineV2) Evaluate(tenant, scope, path string, isABTest bool) (policy.Action, uint64, error) {
	pe.evaluations.Add(1)

	// Select policy (A/B test or current)
	var pol *VersionedPolicy

	if isABTest && pe.abTestEnabled.Load() {
		// Use test policy for A/B testing
		if testPol := pe.getTestPolicy(); testPol != nil {
			pol = testPol
			pe.abTestEvaluations.Add(1)
		} else {
			pol = pe.getCurrent()
		}
	} else {
		pol = pe.getCurrent()
	}

	if pol == nil {
		return policy.ActionDeny, 0, fmt.Errorf("no active policy")
	}

	// Evaluate against policy
	action := policy.Evaluate(pol.Policy, tenant, scope, path)

	return action, pol.Version, nil
}

// EnableABTest enables A/B testing with a new test policy
func (pe *PolicyEngineV2) EnableABTest(testPol policy.Config, trafficPct float64) error {
	// Validate test policy
	if err := pe.validator.Validate(testPol); err != nil {
		return fmt.Errorf("test policy validation failed: %w", err)
	}

	if trafficPct < 0.01 || trafficPct > 0.5 {
		return fmt.Errorf("traffic percentage must be between 1%% and 50%%")
	}

	pe.mu.Lock()
	currentVer := pe.getCurrentVersion()
	testVersion := &VersionedPolicy{
		Version:   currentVer + 1,
		Policy:    testPol,
		Hash:      pe.calculateHash(testPol),
		CreatedAt: time.Now(),
		CreatedBy: "ab-test",
		Metadata:  map[string]string{"type": "ab-test"},
	}
	pe.mu.Unlock()

	pe.testPolicy.Store(testVersion)
	pe.testTraffic = trafficPct
	pe.abTestEnabled.Store(true)

	log.Printf("[policy-v2] A/B test enabled: %.1f%% traffic to test policy v%d", trafficPct*100, testVersion.Version)

	return nil
}

// DisableABTest disables A/B testing
func (pe *PolicyEngineV2) DisableABTest() {
	pe.abTestEnabled.Store(false)
	pe.testPolicy.Store((*VersionedPolicy)(nil))
	log.Printf("[policy-v2] A/B test disabled")
}

// PromoteTestPolicy promotes test policy to current (after successful A/B test)
func (pe *PolicyEngineV2) PromoteTestPolicy() error {
	if !pe.abTestEnabled.Load() {
		return fmt.Errorf("no A/B test active")
	}

	testPol := pe.getTestPolicy()
	if testPol == nil {
		return fmt.Errorf("test policy not found")
	}

	// Activate test policy as current
	if err := pe.activatePolicy(testPol); err != nil {
		return err
	}

	// Disable A/B test
	pe.DisableABTest()

	log.Printf("[policy-v2] Test policy v%d promoted to current", testPol.Version)

	return nil
}

// watchLoop monitors policy file for changes
func (pe *PolicyEngineV2) watchLoop() {
	var lastModTime int64

	for {
		select {
		case <-pe.stopWatch:
			return
		case <-pe.watchTicker.C:
			fi, err := os.Stat(pe.watchPath)
			if err != nil {
				continue
			}

			modTime := fi.ModTime().UnixNano()
			if modTime != lastModTime && lastModTime != 0 {
				// File changed, reload
				if pol, err := policy.Load(pe.watchPath); err == nil {
					if err := pe.LoadPolicy(pol, "hot-reload"); err != nil {
						log.Printf("[policy-v2] Hot reload failed: %v", err)
					}
				}
			}
			lastModTime = modTime
		}
	}
}

// getCurrent returns current active policy
func (pe *PolicyEngineV2) getCurrent() *VersionedPolicy {
	if v := pe.current.Load(); v != nil {
		return v.(*VersionedPolicy)
	}
	return nil
}

// getTestPolicy returns test policy for A/B testing
func (pe *PolicyEngineV2) getTestPolicy() *VersionedPolicy {
	if v := pe.testPolicy.Load(); v != nil {
		return v.(*VersionedPolicy)
	}
	return nil
}

// getCurrentVersion returns current policy version number
func (pe *PolicyEngineV2) getCurrentVersion() uint64 {
	if current := pe.getCurrent(); current != nil {
		return current.Version
	}
	return 0
}

// calculateHash calculates SHA256 hash of policy
func (pe *PolicyEngineV2) calculateHash(pol policy.Config) string {
	data, _ := json.Marshal(pol)
	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:])
}

// calculateChecksum calculates checksum for integrity verification
func (pe *PolicyEngineV2) calculateChecksum(pol policy.Config) string {
	// Simple CRC32-like checksum
	data, _ := json.Marshal(pol)
	sum := uint32(0)
	for _, b := range data {
		sum = sum*31 + uint32(b)
	}
	return fmt.Sprintf("%08x", sum)
}

// GetVersionHistory returns policy version history
func (pe *PolicyEngineV2) GetVersionHistory() []map[string]interface{} {
	pe.mu.RLock()
	defer pe.mu.RUnlock()

	history := make([]map[string]interface{}, len(pe.history))
	for i, ver := range pe.history {
		history[i] = map[string]interface{}{
			"version":    ver.Version,
			"hash":       ver.Hash,
			"created_at": ver.CreatedAt,
			"created_by": ver.CreatedBy,
			"checksum":   ver.Checksum,
		}
	}
	return history
}

// Metrics returns policy engine metrics
func (pe *PolicyEngineV2) Metrics() map[string]interface{} {
	current := pe.getCurrent()
	var currentVer uint64
	var currentHash string
	if current != nil {
		currentVer = current.Version
		currentHash = current.Hash
	}

	return map[string]interface{}{
		"current_version":     currentVer,
		"current_hash":        currentHash,
		"evaluations":         pe.evaluations.Load(),
		"hot_reloads":         pe.hotReloads.Load(),
		"rollbacks":           pe.rollbacks.Load(),
		"ab_test_enabled":     pe.abTestEnabled.Load(),
		"ab_test_evaluations": pe.abTestEvaluations.Load(),
		"version_mismatches":  pe.versionMismatches.Load(),
		"history_size":        len(pe.history),
		"rollback_stack_size": len(pe.rollbackStack),
	}
}

// ---------- Policy Validator ----------

// NewPolicyValidator creates a policy validator with production rules
func NewPolicyValidator() *PolicyValidator {
	return &PolicyValidator{
		maxRuleCount:  1000,
		maxComplexity: 100,
		forbiddenPatterns: []string{
			"eval(",
			"exec(",
			"__import__",
			"subprocess",
		},
	}
}

// Validate validates policy before activation
func (pv *PolicyValidator) Validate(pol policy.Config) error {
	// Check rule count
	totalRules := len(pol.Allowed) + len(pol.Advanced)
	if totalRules > pv.maxRuleCount {
		return fmt.Errorf("rule count %d exceeds maximum %d", totalRules, pv.maxRuleCount)
	}

	// Check complexity
	complexity := pv.calculateComplexity(pol)
	if complexity > pv.maxComplexity {
		return fmt.Errorf("policy complexity %d exceeds maximum %d", complexity, pv.maxComplexity)
	}

	// Check for forbidden patterns in rules
	for _, rule := range pol.Advanced {
		for _, pattern := range pv.forbiddenPatterns {
			if contains := pv.containsPattern(rule, pattern); contains {
				return fmt.Errorf("forbidden pattern '%s' found in rule", pattern)
			}
		}
	}

	return nil
}

// calculateComplexity estimates policy evaluation complexity
func (pv *PolicyValidator) calculateComplexity(pol policy.Config) int {
	complexity := 0

	// Simple rules: O(n)
	complexity += len(pol.Allowed)

	// Advanced rules: O(n * m) where m is average conditions
	for _, rule := range pol.Advanced {
		ruleComplexity := 1
		if rule.Tenant != "" { // correct field names from policy.AdvancedRule
			ruleComplexity++
		}
		if len(rule.Scopes) > 0 {
			ruleComplexity += len(rule.Scopes)
		}
		if rule.PathPrefix != "" {
			// single prefix string
			ruleComplexity++
		}
		complexity += ruleComplexity
	}

	return complexity
}

// containsPattern checks if rule contains forbidden pattern
func (pv *PolicyValidator) containsPattern(rule policy.AdvancedRule, pattern string) bool {
	if pattern == "" {
		return false
	}
	if rule.Tenant != "" && matchSubstring(rule.Tenant, pattern) {
		return true
	}
	for _, sc := range rule.Scopes {
		if matchSubstring(sc, pattern) {
			return true
		}
	}
	if rule.PathPrefix != "" && matchSubstring(rule.PathPrefix, pattern) {
		return true
	}
	return false
}

func matchSubstring(s, substr string) bool {
	if len(substr) > len(s) {
		return false
	}
	for i := 0; i <= len(s)-len(substr); i++ {
		match := true
		for j := 0; j < len(substr); j++ {
			if s[i+j] != substr[j] {
				match = false
				break
			}
		}
		if match {
			return true
		}
	}
	return false
}

// ---------- HTTP Handlers for Policy Management ----------

// handlePolicyV2Status returns current policy status
func handlePolicyV2Status(w http.ResponseWriter, r *http.Request, pe *PolicyEngineV2) {
	current := pe.getCurrent()
	if current == nil {
		http.Error(w, "no active policy", 500)
		return
	}

	resp := map[string]interface{}{
		"version":        current.Version,
		"hash":           current.Hash,
		"created_at":     current.CreatedAt,
		"created_by":     current.CreatedBy,
		"checksum":       current.Checksum,
		"ab_test_active": pe.abTestEnabled.Load(),
		"metrics":        pe.Metrics(),
	}

	writeJSON(w, 200, resp)
}

// handlePolicyV2History returns policy version history
func handlePolicyV2History(w http.ResponseWriter, r *http.Request, pe *PolicyEngineV2) {
	history := pe.GetVersionHistory()
	writeJSON(w, 200, map[string]interface{}{
		"history": history,
	})
}

// handlePolicyV2Rollback rolls back to previous version
func handlePolicyV2Rollback(w http.ResponseWriter, r *http.Request, pe *PolicyEngineV2) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", 405)
		return
	}

	if err := pe.Rollback(); err != nil {
		http.Error(w, err.Error(), 400)
		return
	}

	writeJSON(w, 200, map[string]string{"status": "rolled back"})
}
