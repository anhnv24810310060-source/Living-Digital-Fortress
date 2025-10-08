// Package policy - Dynamic Policy Compilation Engine
// Supports hot-reloading, versioning, A/B testing, and rollback
package policy

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"sync/atomic"
	"time"
)

// DynamicEngine provides real-time policy updates without restart
type DynamicEngine struct {
	// Policy storage
	policies    sync.Map // version -> *CompiledPolicy
	currentVer  uint64   // Atomic current version
	rollbackVer uint64   // Atomic rollback version

	// A/B testing
	abTests map[string]*ABTest
	abMu    sync.RWMutex

	// Compilation cache
	compileCache sync.Map // source_hash -> *CompiledPolicy

	// Watchers for policy changes
	watchers   []chan PolicyEvent
	watchersMu sync.RWMutex

	// Metrics
	evaluations  uint64
	hotReloads   uint64
	rollbacks    uint64
	compilations uint64
	cacheHits    uint64

	ctx    context.Context
	cancel context.CancelFunc
}

// CompiledPolicy represents a compiled and optimized policy
type CompiledPolicy struct {
	Version  uint64
	Source   string
	Compiled time.Time
	Hash     string

	// Fast lookup structures
	tenantIndex map[string]*TenantRules
	pathTrie    *PathTrie

	// Advanced rules
	abacRules []*ABACRule
	riskRules []*RiskRule

	// Metadata
	metadata PolicyMetadata
}

// TenantRules contains tenant-specific policies
type TenantRules struct {
	Tenant    string
	Allow     []string // Allowed scopes
	Deny      []string // Denied scopes
	RiskLevel RiskLevel
}

// PathTrie provides efficient path prefix matching
type PathTrie struct {
	root *TrieNode
}

type TrieNode struct {
	children map[string]*TrieNode
	action   Action
	terminal bool
}

// ABACRule represents Attribute-Based Access Control rule
type ABACRule struct {
	ID         string
	Priority   int
	Conditions []Condition
	Action     Action
	Score      float64 // Risk score adjustment
}

// Condition represents a single ABAC condition
type Condition struct {
	Attribute string // e.g., "user.role", "resource.sensitivity", "time.hour"
	Operator  string // e.g., "eq", "in", "gt", "lt", "contains"
	Value     interface{}
}

// RiskRule evaluates risk score based on context
type RiskRule struct {
	ID        string
	Factors   []RiskFactor
	Threshold float64
	Action    Action
}

type RiskFactor struct {
	Name   string
	Weight float64
	Eval   func(ctx *EvalContext) float64
}

// RiskLevel defines risk tiers
type RiskLevel int

const (
	RiskLevelLow RiskLevel = iota
	RiskLevelMedium
	RiskLevelHigh
	RiskLevelCritical
)

// PolicyMetadata contains policy metadata
type PolicyMetadata struct {
	Author      string
	Description string
	ValidFrom   time.Time
	ValidUntil  time.Time
	Tags        []string
}

// ABTest represents an A/B test configuration
type ABTest struct {
	Name       string
	VersionA   uint64  // Control
	VersionB   uint64  // Treatment
	SplitRatio float64 // 0.0 - 1.0 (ratio for B)
	Active     bool
	Metrics    ABTestMetrics
}

type ABTestMetrics struct {
	RequestsA   uint64
	RequestsB   uint64
	AllowsA     uint64
	AllowsB     uint64
	DeniesA     uint64
	DeniesB     uint64
	AvgLatencyA time.Duration
	AvgLatencyB time.Duration
}

// PolicyEvent represents a policy change event
type PolicyEvent struct {
	Type      EventType
	Version   uint64
	Timestamp time.Time
	Details   string
}

type EventType int

const (
	EventCompiled EventType = iota
	EventActivated
	EventRolledBack
	EventABTestStarted
	EventABTestEnded
)

// EvalContext provides context for policy evaluation
type EvalContext struct {
	Tenant    string
	Scope     string
	Path      string
	IP        string
	UserID    string
	Timestamp time.Time

	// ABAC attributes
	UserAttrs     map[string]interface{}
	ResourceAttrs map[string]interface{}
	EnvAttrs      map[string]interface{}

	// Risk factors
	ReputationScore float64
	AnomalyScore    float64
	GeoRisk         float64
}

// NewDynamicEngine creates a new dynamic policy engine
func NewDynamicEngine() *DynamicEngine {
	ctx, cancel := context.WithCancel(context.Background())

	return &DynamicEngine{
		abTests: make(map[string]*ABTest),
		ctx:     ctx,
		cancel:  cancel,
	}
}

// CompileAndLoad compiles policy source and loads it
func (de *DynamicEngine) CompileAndLoad(source string, metadata PolicyMetadata) (uint64, error) {
	// Check compilation cache
	hash := hashSource(source)
	if cached, ok := de.compileCache.Load(hash); ok {
		atomic.AddUint64(&de.cacheHits, 1)
		compiled := cached.(*CompiledPolicy)
		version := atomic.AddUint64(&de.currentVer, 1)
		compiled.Version = version
		de.policies.Store(version, compiled)
		de.notifyWatchers(PolicyEvent{Type: EventActivated, Version: version, Timestamp: time.Now()})
		return version, nil
	}

	// Compile policy
	compiled, err := de.compile(source, metadata)
	if err != nil {
		return 0, fmt.Errorf("compile: %w", err)
	}

	atomic.AddUint64(&de.compilations, 1)

	// Cache compiled policy
	de.compileCache.Store(hash, compiled)

	// Assign version and store
	version := atomic.AddUint64(&de.currentVer, 1)
	compiled.Version = version
	compiled.Hash = hash

	de.policies.Store(version, compiled)

	// Notify watchers
	de.notifyWatchers(PolicyEvent{
		Type:      EventCompiled,
		Version:   version,
		Timestamp: time.Now(),
		Details:   "Policy compiled and activated",
	})

	atomic.AddUint64(&de.hotReloads, 1)

	log.Printf("[policy] hot-reload: version=%d hash=%s", version, hash[:8])

	return version, nil
}

// compile compiles policy source into optimized structures
func (de *DynamicEngine) compile(source string, metadata PolicyMetadata) (*CompiledPolicy, error) {
	// Parse source (JSON for now, could support Rego/CEL/custom DSL)
	var rawPolicy struct {
		Tenants []struct {
			Name      string   `json:"name"`
			Allow     []string `json:"allow"`
			Deny      []string `json:"deny"`
			RiskLevel string   `json:"riskLevel"`
		} `json:"tenants"`

		Paths []struct {
			Pattern string `json:"pattern"`
			Action  string `json:"action"`
		} `json:"paths"`

		ABACRules []struct {
			ID         string                   `json:"id"`
			Priority   int                      `json:"priority"`
			Conditions []map[string]interface{} `json:"conditions"`
			Action     string                   `json:"action"`
		} `json:"abacRules"`
	}

	if err := json.Unmarshal([]byte(source), &rawPolicy); err != nil {
		return nil, fmt.Errorf("parse: %w", err)
	}

	compiled := &CompiledPolicy{
		Compiled:    time.Now(),
		Source:      source,
		tenantIndex: make(map[string]*TenantRules),
		pathTrie:    NewPathTrie(),
		metadata:    metadata,
	}

	// Build tenant index
	for _, t := range rawPolicy.Tenants {
		compiled.tenantIndex[t.Name] = &TenantRules{
			Tenant:    t.Name,
			Allow:     t.Allow,
			Deny:      t.Deny,
			RiskLevel: parseRiskLevel(t.RiskLevel),
		}
	}

	// Build path trie
	for _, p := range rawPolicy.Paths {
		compiled.pathTrie.Insert(p.Pattern, parseAction(p.Action))
	}

	// Compile ABAC rules
	for _, r := range rawPolicy.ABACRules {
		rule := &ABACRule{
			ID:       r.ID,
			Priority: r.Priority,
			Action:   parseAction(r.Action),
		}

		for _, cond := range r.Conditions {
			rule.Conditions = append(rule.Conditions, Condition{
				Attribute: cond["attribute"].(string),
				Operator:  cond["operator"].(string),
				Value:     cond["value"],
			})
		}

		compiled.abacRules = append(compiled.abacRules, rule)
	}

	return compiled, nil
}

// Evaluate evaluates policy for given context
func (de *DynamicEngine) Evaluate(ctx *EvalContext) (*Decision, error) {
	atomic.AddUint64(&de.evaluations, 1)

	// Get current policy version
	version := atomic.LoadUint64(&de.currentVer)

	// Check A/B test assignment
	if abVersion := de.getABTestVersion(ctx, version); abVersion != version {
		version = abVersion
	}

	// Load policy
	val, ok := de.policies.Load(version)
	if !ok {
		return nil, fmt.Errorf("policy version %d not found", version)
	}

	policy := val.(*CompiledPolicy)

	// Evaluate in priority order:
	// 1. ABAC rules (highest priority)
	// 2. Tenant-specific rules
	// 3. Path-based rules
	// 4. Risk-based evaluation

	decision := &Decision{
		Version:   version,
		Timestamp: time.Now(),
	}

	// ABAC evaluation
	if action, matched := de.evaluateABAC(policy, ctx); matched {
		decision.Action = action
		decision.Reason = "abac_rule"
		return decision, nil
	}

	// Tenant rules
	if rules, ok := policy.tenantIndex[ctx.Tenant]; ok {
		// Check deny list first
		for _, denyScope := range rules.Deny {
			if matchScope(denyScope, ctx.Scope) {
				decision.Action = ActionDeny
				decision.Reason = "tenant_deny"
				return decision, nil
			}
		}

		// Check allow list
		for _, allowScope := range rules.Allow {
			if matchScope(allowScope, ctx.Scope) {
				decision.Action = ActionAllow
				decision.RiskScore = float64(rules.RiskLevel) / 10.0
				decision.Reason = "tenant_allow"
				return decision, nil
			}
		}
	}

	// Path matching
	if action := policy.pathTrie.Match(ctx.Path); action != "" {
		decision.Action = action
		decision.Reason = "path_match"
		return decision, nil
	}

	// Risk-based evaluation
	riskScore := de.calculateRiskScore(ctx)
	decision.RiskScore = riskScore

	if riskScore > 0.8 {
		decision.Action = ActionDeny
		decision.Reason = "high_risk"
	} else if riskScore > 0.5 {
		decision.Action = ActionTarpit
		decision.Reason = "medium_risk"
	} else {
		decision.Action = ActionAllow
		decision.Reason = "low_risk"
	}

	return decision, nil
}

// evaluateABAC evaluates ABAC rules
func (de *DynamicEngine) evaluateABAC(policy *CompiledPolicy, ctx *EvalContext) (Action, bool) {
	// Sort by priority (higher first)
	for _, rule := range policy.abacRules {
		if de.matchABACConditions(rule.Conditions, ctx) {
			return rule.Action, true
		}
	}
	return "", false
}

// matchABACConditions checks if all conditions match
func (de *DynamicEngine) matchABACConditions(conditions []Condition, ctx *EvalContext) bool {
	for _, cond := range conditions {
		if !de.evaluateCondition(cond, ctx) {
			return false
		}
	}
	return true
}

// evaluateCondition evaluates a single ABAC condition
func (de *DynamicEngine) evaluateCondition(cond Condition, ctx *EvalContext) bool {
	// Extract attribute value from context
	value := de.getAttribute(cond.Attribute, ctx)

	// Apply operator
	switch cond.Operator {
	case "eq":
		return value == cond.Value
	case "ne":
		return value != cond.Value
	case "in":
		if arr, ok := cond.Value.([]interface{}); ok {
			for _, v := range arr {
				if v == value {
					return true
				}
			}
		}
		return false
	case "gt":
		if v1, ok := value.(float64); ok {
			if v2, ok := cond.Value.(float64); ok {
				return v1 > v2
			}
		}
		return false
	case "lt":
		if v1, ok := value.(float64); ok {
			if v2, ok := cond.Value.(float64); ok {
				return v1 < v2
			}
		}
		return false
	default:
		return false
	}
}

// getAttribute extracts attribute from context using dot notation
func (de *DynamicEngine) getAttribute(attr string, ctx *EvalContext) interface{} {
	// Simple implementation (could be enhanced with gjson for complex paths)
	switch attr {
	case "user.id":
		return ctx.UserID
	case "resource.path":
		return ctx.Path
	case "time.hour":
		return ctx.Timestamp.Hour()
	case "env.risk_score":
		return ctx.ReputationScore
	default:
		// Check custom attributes
		if val, ok := ctx.UserAttrs[attr]; ok {
			return val
		}
		if val, ok := ctx.ResourceAttrs[attr]; ok {
			return val
		}
		if val, ok := ctx.EnvAttrs[attr]; ok {
			return val
		}
		return nil
	}
}

// calculateRiskScore computes overall risk score
func (de *DynamicEngine) calculateRiskScore(ctx *EvalContext) float64 {
	// Weighted combination of risk factors
	score := 0.0
	score += ctx.ReputationScore * 0.4
	score += ctx.AnomalyScore * 0.3
	score += ctx.GeoRisk * 0.3

	// Clamp to [0, 1]
	if score < 0 {
		score = 0
	}
	if score > 1 {
		score = 1
	}

	return score
}

// Rollback rolls back to previous policy version
func (de *DynamicEngine) Rollback() error {
	current := atomic.LoadUint64(&de.currentVer)
	if current <= 1 {
		return fmt.Errorf("no previous version to rollback")
	}

	previous := current - 1

	// Check if previous version exists
	if _, ok := de.policies.Load(previous); !ok {
		return fmt.Errorf("previous version %d not found", previous)
	}

	// Perform rollback
	atomic.StoreUint64(&de.currentVer, previous)
	atomic.StoreUint64(&de.rollbackVer, current)
	atomic.AddUint64(&de.rollbacks, 1)

	de.notifyWatchers(PolicyEvent{
		Type:      EventRolledBack,
		Version:   previous,
		Timestamp: time.Now(),
		Details:   fmt.Sprintf("Rolled back from v%d to v%d", current, previous),
	})

	log.Printf("[policy] rollback: v%d -> v%d", current, previous)

	return nil
}

// StartABTest starts an A/B test between two policy versions
func (de *DynamicEngine) StartABTest(name string, versionA, versionB uint64, splitRatio float64) error {
	de.abMu.Lock()
	defer de.abMu.Unlock()

	de.abTests[name] = &ABTest{
		Name:       name,
		VersionA:   versionA,
		VersionB:   versionB,
		SplitRatio: splitRatio,
		Active:     true,
	}

	de.notifyWatchers(PolicyEvent{
		Type:      EventABTestStarted,
		Timestamp: time.Now(),
		Details:   fmt.Sprintf("A/B test '%s': v%d vs v%d (split=%.0f%%)", name, versionA, versionB, splitRatio*100),
	})

	return nil
}

// getABTestVersion returns version based on A/B test assignment
func (de *DynamicEngine) getABTestVersion(ctx *EvalContext, defaultVer uint64) uint64 {
	de.abMu.RLock()
	defer de.abMu.RUnlock()

	for _, test := range de.abTests {
		if !test.Active {
			continue
		}

		// Consistent hashing for stable assignment
		hash := hashSource(ctx.UserID + ctx.IP)
		hashVal := int(hash[0])
		assignToB := (hashVal % 100) < int(test.SplitRatio*100)

		if assignToB {
			atomic.AddUint64(&test.Metrics.RequestsB, 1)
			return test.VersionB
		} else {
			atomic.AddUint64(&test.Metrics.RequestsA, 1)
			return test.VersionA
		}
	}

	return defaultVer
}

// Watch registers a watcher for policy events
func (de *DynamicEngine) Watch() <-chan PolicyEvent {
	ch := make(chan PolicyEvent, 10)

	de.watchersMu.Lock()
	de.watchers = append(de.watchers, ch)
	de.watchersMu.Unlock()

	return ch
}

// notifyWatchers sends event to all watchers
func (de *DynamicEngine) notifyWatchers(event PolicyEvent) {
	de.watchersMu.RLock()
	defer de.watchersMu.RUnlock()

	for _, ch := range de.watchers {
		select {
		case ch <- event:
		default:
			// Channel full, skip
		}
	}
}

// Metrics returns engine metrics
func (de *DynamicEngine) Metrics() map[string]interface{} {
	var policyCount int
	de.policies.Range(func(_, _ interface{}) bool {
		policyCount++
		return true
	})

	return map[string]interface{}{
		"evaluations_total":  atomic.LoadUint64(&de.evaluations),
		"hot_reloads_total":  atomic.LoadUint64(&de.hotReloads),
		"rollbacks_total":    atomic.LoadUint64(&de.rollbacks),
		"compilations_total": atomic.LoadUint64(&de.compilations),
		"cache_hits_total":   atomic.LoadUint64(&de.cacheHits),
		"current_version":    atomic.LoadUint64(&de.currentVer),
		"policies_stored":    policyCount,
	}
}

// Close stops the engine
func (de *DynamicEngine) Close() {
	de.cancel()
}

// ---------- Helper Functions ----------

func hashSource(source string) string {
	h := sha256.Sum256([]byte(source))
	return hex.EncodeToString(h[:])
}

func parseRiskLevel(s string) RiskLevel {
	switch s {
	case "low":
		return RiskLevelLow
	case "medium":
		return RiskLevelMedium
	case "high":
		return RiskLevelHigh
	case "critical":
		return RiskLevelCritical
	default:
		return RiskLevelMedium
	}
}

func parseAction(s string) Action {
	return Action(s)
}

func matchScope(pattern, scope string) bool {
	// Simple wildcard matching (* means any)
	if pattern == "*" {
		return true
	}
	return pattern == scope
}

// ---------- Path Trie Implementation ----------

func NewPathTrie() *PathTrie {
	return &PathTrie{
		root: &TrieNode{
			children: make(map[string]*TrieNode),
		},
	}
}

func (pt *PathTrie) Insert(path string, action Action) {
	node := pt.root
	parts := splitPath(path)

	for _, part := range parts {
		if _, ok := node.children[part]; !ok {
			node.children[part] = &TrieNode{
				children: make(map[string]*TrieNode),
			}
		}
		node = node.children[part]
	}

	node.action = action
	node.terminal = true
}

func (pt *PathTrie) Match(path string) Action {
	node := pt.root
	parts := splitPath(path)

	var lastAction Action
	for _, part := range parts {
		child, ok := node.children[part]
		if !ok {
			// Check wildcard
			child, ok = node.children["*"]
			if !ok {
				break
			}
		}

		if child.terminal {
			lastAction = child.action
		}

		node = child
	}

	return lastAction
}

func splitPath(path string) []string {
	// Simple split by / (could be enhanced)
	var parts []string
	var current string
	for _, ch := range path {
		if ch == '/' {
			if current != "" {
				parts = append(parts, current)
				current = ""
			}
		} else {
			current += string(ch)
		}
	}
	if current != "" {
		parts = append(parts, current)
	}
	return parts
}

// Decision represents a policy decision
type Decision struct {
	Action    Action
	Version   uint64
	Timestamp time.Time
	Reason    string
	RiskScore float64
}
