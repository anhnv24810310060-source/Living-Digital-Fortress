package security

// Package graphql implements GraphQL-specific security controls

import (
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
)

// SecurityMiddleware implements GraphQL security controls
type SecurityMiddleware struct {
	config         *Config
	queryCache     sync.Map // hash -> *QueryAnalysis
	blockedQueries sync.Map // hash -> reason
	mu             sync.RWMutex

	// Metrics
	queriesBlocked       atomic.Uint64
	depthViolations      atomic.Uint64
	complexityViolations atomic.Uint64
}

// Config holds GraphQL security configuration
type Config struct {
	// Depth limiting
	MaxDepth        int
	MaxDepthEnabled bool

	// Complexity analysis
	MaxComplexity     int
	ComplexityEnabled bool
	DefaultFieldCost  int
	ConnectionCost    int // cost per connection/list field

	// Query whitelisting
	WhitelistEnabled bool
	AllowedQueries   map[string]bool // query hash -> allowed

	// Introspection control
	DisableIntrospection bool

	// Batch query limits
	MaxBatchSize int

	// Alias limits (to prevent alias-based DoS)
	MaxAliasCount int

	// Directive limits
	AllowedDirectives map[string]bool
}

// QueryAnalysis holds analysis results for a GraphQL query
type QueryAnalysis struct {
	Hash             string
	Depth            int
	Complexity       int
	AliasCount       int
	FieldCount       int
	HasIntrospection bool
	Directives       []string
	Errors           []string
}

// NewSecurityMiddleware creates a new GraphQL security middleware
func NewSecurityMiddleware(cfg *Config) *SecurityMiddleware {
	if cfg == nil {
		cfg = DefaultConfig()
	}
	return &SecurityMiddleware{
		config: cfg,
	}
}

// DefaultConfig returns default GraphQL security configuration
func DefaultConfig() *Config {
	return &Config{
		MaxDepth:             10,
		MaxDepthEnabled:      true,
		MaxComplexity:        1000,
		ComplexityEnabled:    true,
		DefaultFieldCost:     1,
		ConnectionCost:       10,
		WhitelistEnabled:     false,
		AllowedQueries:       make(map[string]bool),
		DisableIntrospection: true,
		MaxBatchSize:         10,
		MaxAliasCount:        15,
		AllowedDirectives: map[string]bool{
			"include":    true,
			"skip":       true,
			"deprecated": true,
		},
	}
}

// ValidateQuery validates a GraphQL query against security policies
func (sm *SecurityMiddleware) ValidateQuery(query string) (*QueryAnalysis, error) {
	// Check cache first
	hash := hashQuery(query)
	if cached, ok := sm.queryCache.Load(hash); ok {
		return cached.(*QueryAnalysis), nil
	}

	// Check if query is blocked
	if reason, ok := sm.blockedQueries.Load(hash); ok {
		sm.queriesBlocked.Add(1)
		return nil, fmt.Errorf("query blocked: %v", reason)
	}

	// Analyze query
	analysis := &QueryAnalysis{
		Hash: hash,
	}

	// Parse and analyze (simplified - production would use actual GraphQL parser)
	analysis.Depth = sm.calculateDepth(query)
	analysis.Complexity = sm.calculateComplexity(query)
	analysis.AliasCount = sm.countAliases(query)
	analysis.FieldCount = sm.countFields(query)
	analysis.HasIntrospection = sm.hasIntrospection(query)
	analysis.Directives = sm.extractDirectives(query)

	// Apply depth limit
	if sm.config.MaxDepthEnabled && analysis.Depth > sm.config.MaxDepth {
		sm.depthViolations.Add(1)
		analysis.Errors = append(analysis.Errors, fmt.Sprintf("depth %d exceeds limit %d", analysis.Depth, sm.config.MaxDepth))
		return analysis, errors.New("depth limit exceeded")
	}

	// Apply complexity limit
	if sm.config.ComplexityEnabled && analysis.Complexity > sm.config.MaxComplexity {
		sm.complexityViolations.Add(1)
		analysis.Errors = append(analysis.Errors, fmt.Sprintf("complexity %d exceeds limit %d", analysis.Complexity, sm.config.MaxComplexity))
		return analysis, errors.New("complexity limit exceeded")
	}

	// Check introspection
	if sm.config.DisableIntrospection && analysis.HasIntrospection {
		analysis.Errors = append(analysis.Errors, "introspection disabled in production")
		return analysis, errors.New("introspection disabled")
	}

	// Check alias count
	if analysis.AliasCount > sm.config.MaxAliasCount {
		analysis.Errors = append(analysis.Errors, fmt.Sprintf("alias count %d exceeds limit %d", analysis.AliasCount, sm.config.MaxAliasCount))
		return analysis, errors.New("too many aliases")
	}

	// Check directives
	for _, directive := range analysis.Directives {
		if _, allowed := sm.config.AllowedDirectives[directive]; !allowed {
			analysis.Errors = append(analysis.Errors, fmt.Sprintf("directive @%s not allowed", directive))
			return analysis, fmt.Errorf("directive @%s not allowed", directive)
		}
	}

	// Check whitelist (if enabled)
	if sm.config.WhitelistEnabled {
		if _, ok := sm.config.AllowedQueries[hash]; !ok {
			analysis.Errors = append(analysis.Errors, "query not in whitelist")
			return analysis, errors.New("query not whitelisted")
		}
	}

	// Cache successful analysis
	sm.queryCache.Store(hash, analysis)

	return analysis, nil
}

// calculateDepth calculates the nesting depth of a GraphQL query
func (sm *SecurityMiddleware) calculateDepth(query string) int {
	// Simplified depth calculation by counting nested braces
	maxDepth := 0
	currentDepth := 0

	for _, ch := range query {
		if ch == '{' {
			currentDepth++
			if currentDepth > maxDepth {
				maxDepth = currentDepth
			}
		} else if ch == '}' {
			currentDepth--
		}
	}

	return maxDepth
}

// calculateComplexity calculates query complexity using cost-based scoring
func (sm *SecurityMiddleware) calculateComplexity(query string) int {
	// Simplified complexity: field count * cost + connection multipliers
	fieldCount := sm.countFields(query)
	connectionCount := sm.countConnections(query)

	complexity := fieldCount*sm.config.DefaultFieldCost +
		connectionCount*sm.config.ConnectionCost

	return complexity
}

// countFields counts the number of fields in the query
func (sm *SecurityMiddleware) countFields(query string) int {
	// Simplified: count words that are likely field names
	// Production: use actual GraphQL parser
	count := 0
	lines := strings.Split(query, "\n")
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed != "" && !strings.HasPrefix(trimmed, "#") {
			// Rough heuristic: each non-empty, non-comment line is a field
			count++
		}
	}
	return count
}

// countAliases counts the number of aliases in the query
func (sm *SecurityMiddleware) countAliases(query string) int {
	// Simplified: count occurrences of ':'
	// Production: use actual GraphQL parser to identify aliases
	return strings.Count(query, ":")
}

// countConnections counts connection/list fields
func (sm *SecurityMiddleware) countConnections(query string) int {
	// Simplified: look for common connection patterns
	// Production: analyze schema and identify list fields
	count := 0
	patterns := []string{"Connection", "edges", "nodes", "items"}
	for _, pattern := range patterns {
		count += strings.Count(query, pattern)
	}
	return count
}

// hasIntrospection checks if query contains introspection
func (sm *SecurityMiddleware) hasIntrospection(query string) bool {
	introspectionKeywords := []string{
		"__schema",
		"__type",
		"__typename",
		"__Field",
		"__EnumValue",
		"__InputValue",
		"__Directive",
	}

	lowerQuery := strings.ToLower(query)
	for _, keyword := range introspectionKeywords {
		if strings.Contains(lowerQuery, strings.ToLower(keyword)) {
			return true
		}
	}

	return false
}

// extractDirectives extracts directive names from query
func (sm *SecurityMiddleware) extractDirectives(query string) []string {
	directives := []string{}

	// Simple pattern matching for @directive
	tokens := strings.Split(query, "@")
	for i := 1; i < len(tokens); i++ {
		// Get directive name (word after @)
		parts := strings.Fields(tokens[i])
		if len(parts) > 0 {
			name := strings.TrimFunc(parts[0], func(r rune) bool {
				return r == '(' || r == ')'
			})
			directives = append(directives, name)
		}
	}

	return directives
}

// AddToWhitelist adds a query to the whitelist
func (sm *SecurityMiddleware) AddToWhitelist(query string) {
	hash := hashQuery(query)
	sm.config.AllowedQueries[hash] = true
}

// BlockQuery permanently blocks a query
func (sm *SecurityMiddleware) BlockQuery(query string, reason string) {
	hash := hashQuery(query)
	sm.blockedQueries.Store(hash, reason)
}

// hashQuery creates a hash of the query for caching/comparison
func hashQuery(query string) string {
	// Normalize query: remove whitespace and comments
	normalized := strings.Join(strings.Fields(query), " ")
	normalized = removeComments(normalized)

	// Simple hash (production: use crypto/sha256)
	return fmt.Sprintf("%x", len(normalized))
}

// removeComments removes GraphQL comments from query
func removeComments(query string) string {
	lines := strings.Split(query, "\n")
	filtered := []string{}
	for _, line := range lines {
		if !strings.HasPrefix(strings.TrimSpace(line), "#") {
			filtered = append(filtered, line)
		}
	}
	return strings.Join(filtered, "\n")
}

// Stats returns middleware statistics
func (sm *SecurityMiddleware) Stats() map[string]interface{} {
	cacheSize := 0
	sm.queryCache.Range(func(k, v interface{}) bool {
		cacheSize++
		return true
	})

	blockedSize := 0
	sm.blockedQueries.Range(func(k, v interface{}) bool {
		blockedSize++
		return true
	})

	return map[string]interface{}{
		"queriesBlocked":       sm.queriesBlocked.Load(),
		"depthViolations":      sm.depthViolations.Load(),
		"complexityViolations": sm.complexityViolations.Load(),
		"cacheSize":            cacheSize,
		"blockedSize":          blockedSize,
	}
}

// ValidateBatch validates a batch of GraphQL queries
func (sm *SecurityMiddleware) ValidateBatch(queries []string) ([]*QueryAnalysis, error) {
	if len(queries) > sm.config.MaxBatchSize {
		return nil, fmt.Errorf("batch size %d exceeds limit %d", len(queries), sm.config.MaxBatchSize)
	}

	results := make([]*QueryAnalysis, len(queries))
	for i, query := range queries {
		analysis, err := sm.ValidateQuery(query)
		if err != nil {
			return nil, fmt.Errorf("query %d validation failed: %w", i, err)
		}
		results[i] = analysis
	}

	return results, nil
}

// GenerateQueryID generates a persistent ID for a query (for whitelisting)
func GenerateQueryID(operation string, query string) string {
	return fmt.Sprintf("%s:%s", operation, hashQuery(query))
}

// CostEstimator estimates the cost of executing a query (for budgeting)
type CostEstimator struct {
	FieldCosts           map[string]int // fieldName -> cost
	DefaultCost          int
	ConnectionMultiplier int
}

// NewCostEstimator creates a new cost estimator
func NewCostEstimator() *CostEstimator {
	return &CostEstimator{
		FieldCosts: map[string]int{
			"user":     10,
			"users":    50,
			"post":     5,
			"posts":    25,
			"comment":  2,
			"comments": 10,
			"search":   100, // expensive
		},
		DefaultCost:          1,
		ConnectionMultiplier: 10,
	}
}

// EstimateCost estimates query execution cost
func (ce *CostEstimator) EstimateCost(analysis *QueryAnalysis) int {
	// Simplified cost model
	baseCost := analysis.FieldCount * ce.DefaultCost
	complexityPenalty := analysis.Complexity
	depthPenalty := analysis.Depth * 10

	totalCost := baseCost + complexityPenalty + depthPenalty
	return totalCost
}
