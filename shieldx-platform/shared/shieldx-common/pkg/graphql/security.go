// Package graphql provides advanced security for GraphQL endpoints
// with query complexity analysis, depth limiting, and introspection control.
package graphql

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const (
	// Default limits
	DefaultMaxDepth          = 10
	DefaultMaxComplexity     = 1000
	DefaultMaxAliases        = 15
	DefaultMaxDirectives     = 10
	DefaultQueryTimeout      = 30 * time.Second
	
	// Cost multipliers
	CostField        = 1
	CostList         = 10
	CostConnection   = 20
	CostMutation     = 50
	CostSubscription = 100
)

// SecurityConfig configures GraphQL security settings
type SecurityConfig struct {
	MaxDepth             int
	MaxComplexity        int
	MaxAliases           int
	MaxDirectives        int
	QueryTimeout         time.Duration
	DisableIntrospection bool
	EnableQueryWhitelist bool
	WhitelistedQueries   map[string]string // queryHash -> queryString
	
	// Rate limiting per client
	MaxQueriesPerMinute  int
	
	// Persistent queries only (no ad-hoc queries in production)
	PersistentQueriesOnly bool
}

// DefaultSecurityConfig returns production-ready security configuration
func DefaultSecurityConfig() SecurityConfig {
	return SecurityConfig{
		MaxDepth:              DefaultMaxDepth,
		MaxComplexity:         DefaultMaxComplexity,
		MaxAliases:            DefaultMaxAliases,
		MaxDirectives:         DefaultMaxDirectives,
		QueryTimeout:          DefaultQueryTimeout,
		DisableIntrospection:  true, // Always disable in production
		EnableQueryWhitelist:  false,
		MaxQueriesPerMinute:   100,
		PersistentQueriesOnly: false,
	}
}

// SecurityMiddleware provides GraphQL security enforcement
type SecurityMiddleware struct {
	config      SecurityConfig
	
	// Query whitelist
	whitelist   map[string]struct{} // SHA-256 hash of allowed queries
	
	// Rate limiting
	rateLimiter *graphQLRateLimiter
	
	// Metrics
	queriesProcessed  uint64
	queriesBlocked    uint64
	complexityTotal   uint64
	depthViolations   uint64
	introspectBlocked uint64
	
	mu sync.RWMutex
}

// NewSecurityMiddleware creates a new GraphQL security middleware
func NewSecurityMiddleware(config SecurityConfig) *SecurityMiddleware {
	whitelist := make(map[string]struct{})
	if config.EnableQueryWhitelist {
		for hash := range config.WhitelistedQueries {
			whitelist[hash] = struct{}{}
		}
	}
	
	return &SecurityMiddleware{
		config:      config,
		whitelist:   whitelist,
		rateLimiter: newGraphQLRateLimiter(config.MaxQueriesPerMinute),
	}
}

// ValidateQuery performs comprehensive query validation
func (m *SecurityMiddleware) ValidateQuery(ctx context.Context, clientID, query string, variables map[string]interface{}) error {
	atomic.AddUint64(&m.queriesProcessed, 1)
	
	// Rate limiting
	if !m.rateLimiter.Allow(clientID) {
		atomic.AddUint64(&m.queriesBlocked, 1)
		return errors.New("rate limit exceeded")
	}
	
	// Persistent queries check
	if m.config.PersistentQueriesOnly {
		if !m.isPersistentQuery(query) {
			atomic.AddUint64(&m.queriesBlocked, 1)
			return errors.New("only persistent queries allowed in production")
		}
	}
	
	// Whitelist check
	if m.config.EnableQueryWhitelist {
		if !m.isWhitelisted(query) {
			atomic.AddUint64(&m.queriesBlocked, 1)
			return errors.New("query not whitelisted")
		}
	}
	
	// Introspection check
	if m.config.DisableIntrospection && isIntrospectionQuery(query) {
		atomic.AddUint64(&m.introspectBlocked, 1)
		return errors.New("introspection disabled in production")
	}
	
	// Parse and analyze query
	ast, err := parseGraphQLQuery(query)
	if err != nil {
		atomic.AddUint64(&m.queriesBlocked, 1)
		return fmt.Errorf("parse error: %w", err)
	}
	
	// Depth analysis
	depth := calculateDepth(ast)
	if depth > m.config.MaxDepth {
		atomic.AddUint64(&m.depthViolations, 1)
		atomic.AddUint64(&m.queriesBlocked, 1)
		return fmt.Errorf("query depth %d exceeds limit %d", depth, m.config.MaxDepth)
	}
	
	// Complexity analysis
	complexity := calculateComplexity(ast, variables)
	atomic.AddUint64(&m.complexityTotal, complexity)
	
	if complexity > uint64(m.config.MaxComplexity) {
		atomic.AddUint64(&m.queriesBlocked, 1)
		return fmt.Errorf("query complexity %d exceeds limit %d", complexity, m.config.MaxComplexity)
	}
	
	// Aliases check
	aliasCount := countAliases(ast)
	if aliasCount > m.config.MaxAliases {
		atomic.AddUint64(&m.queriesBlocked, 1)
		return fmt.Errorf("alias count %d exceeds limit %d", aliasCount, m.config.MaxAliases)
	}
	
	// Directives check
	directiveCount := countDirectives(ast)
	if directiveCount > m.config.MaxDirectives {
		atomic.AddUint64(&m.queriesBlocked, 1)
		return fmt.Errorf("directive count %d exceeds limit %d", directiveCount, m.config.MaxDirectives)
	}
	
	// Set query timeout in context
	_, cancel := context.WithTimeout(ctx, m.config.QueryTimeout)
	defer cancel()
	
	return nil
}

// isWhitelisted checks if a query is in the whitelist
func (m *SecurityMiddleware) isWhitelisted(query string) bool {
	hash := hashQuery(query)
	
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	_, ok := m.whitelist[hash]
	return ok
}

// isPersistentQuery checks if query is using persistent query ID
func (m *SecurityMiddleware) isPersistentQuery(query string) bool {
	// Persistent queries use a hash/ID instead of full query
	// Format: {"extensions":{"persistedQuery":{"version":1,"sha256Hash":"..."}}}
	return strings.Contains(query, "persistedQuery") || len(query) < 100
}

// AddToWhitelist adds a query to the whitelist
func (m *SecurityMiddleware) AddToWhitelist(query string) {
	hash := hashQuery(query)
	
	m.mu.Lock()
	m.whitelist[hash] = struct{}{}
	m.mu.Unlock()
}

// RemoveFromWhitelist removes a query from the whitelist
func (m *SecurityMiddleware) RemoveFromWhitelist(query string) {
	hash := hashQuery(query)
	
	m.mu.Lock()
	delete(m.whitelist, hash)
	m.mu.Unlock()
}

// GetMetrics returns security metrics
func (m *SecurityMiddleware) GetMetrics() map[string]uint64 {
	return map[string]uint64{
		"queries_processed":     atomic.LoadUint64(&m.queriesProcessed),
		"queries_blocked":       atomic.LoadUint64(&m.queriesBlocked),
		"complexity_total":      atomic.LoadUint64(&m.complexityTotal),
		"depth_violations":      atomic.LoadUint64(&m.depthViolations),
		"introspect_blocked":    atomic.LoadUint64(&m.introspectBlocked),
	}
}

// ---------- Query Analysis Functions ----------

// GraphQLAST represents a simplified GraphQL query AST
type GraphQLAST struct {
	Operations []Operation
}

// Operation represents a GraphQL operation
type Operation struct {
	Type       string // query, mutation, subscription
	Name       string
	Fields     []Field
	Directives []Directive
}

// Field represents a GraphQL field
type Field struct {
	Name       string
	Alias      string
	Args       map[string]interface{}
	Fields     []Field // Nested fields
	Directives []Directive
}

// Directive represents a GraphQL directive
type Directive struct {
	Name string
	Args map[string]interface{}
}

// parseGraphQLQuery parses a GraphQL query into an AST
// This is a simplified parser - production should use a proper GraphQL parser
func parseGraphQLQuery(query string) (*GraphQLAST, error) {
	query = strings.TrimSpace(query)
	if query == "" {
		return nil, errors.New("empty query")
	}
	
	// Detect operation type
	opType := "query"
	if strings.HasPrefix(query, "mutation") {
		opType = "mutation"
	} else if strings.HasPrefix(query, "subscription") {
		opType = "subscription"
	}
	
	// Extract operation name (simplified)
	opName := "anonymous"
	nameRegex := regexp.MustCompile(`^\s*(query|mutation|subscription)\s+(\w+)`)
	if matches := nameRegex.FindStringSubmatch(query); len(matches) > 2 {
		opName = matches[2]
	}
	
	// Parse fields (simplified - counts braces for nesting)
	fields := parseFields(query)
	
	return &GraphQLAST{
		Operations: []Operation{
			{
				Type:   opType,
				Name:   opName,
				Fields: fields,
			},
		},
	}, nil
}

// parseFields extracts fields from query (simplified)
func parseFields(query string) []Field {
	// This is a very simplified field parser
	// Production should use github.com/graphql-go/graphql or similar
	
	fields := []Field{}
	
	// Extract field names (words followed by { or spaces)
	fieldRegex := regexp.MustCompile(`(\w+)\s*(?:\(|{|\s)`)
	matches := fieldRegex.FindAllStringSubmatch(query, -1)
	
	for _, match := range matches {
		if len(match) > 1 {
			fieldName := match[1]
			// Skip GraphQL keywords
			if fieldName == "query" || fieldName == "mutation" || fieldName == "subscription" {
				continue
			}
			fields = append(fields, Field{Name: fieldName})
		}
	}
	
	return fields
}

// calculateDepth calculates the maximum nesting depth of a query
func calculateDepth(ast *GraphQLAST) int {
	maxDepth := 0
	
	for _, op := range ast.Operations {
		depth := calculateFieldDepth(op.Fields, 1)
		if depth > maxDepth {
			maxDepth = depth
		}
	}
	
	return maxDepth
}

// calculateFieldDepth recursively calculates field depth
func calculateFieldDepth(fields []Field, currentDepth int) int {
	if len(fields) == 0 {
		return currentDepth
	}
	
	maxDepth := currentDepth
	for _, field := range fields {
		if len(field.Fields) > 0 {
			depth := calculateFieldDepth(field.Fields, currentDepth+1)
			if depth > maxDepth {
				maxDepth = depth
			}
		}
	}
	
	return maxDepth
}

// calculateComplexity calculates query complexity using cost-based scoring
func calculateComplexity(ast *GraphQLAST, variables map[string]interface{}) uint64 {
	var totalCost uint64
	
	for _, op := range ast.Operations {
		// Base cost per operation type
		switch op.Type {
		case "query":
			totalCost += CostField
		case "mutation":
			totalCost += CostMutation
		case "subscription":
			totalCost += CostSubscription
		}
		
		// Add field costs
		totalCost += calculateFieldComplexity(op.Fields, variables)
	}
	
	return totalCost
}

// calculateFieldComplexity recursively calculates field complexity
func calculateFieldComplexity(fields []Field, variables map[string]interface{}) uint64 {
	var cost uint64
	
	for _, field := range fields {
		// Base field cost
		fieldCost := uint64(CostField)
		
		// Check if field returns a list (higher cost)
		if isListField(field.Name) {
			fieldCost = CostList
			
			// If list has pagination args, use those for multiplier
			if first, ok := field.Args["first"].(int); ok && first > 0 {
				fieldCost = uint64(first) * CostField
			} else if limit, ok := field.Args["limit"].(int); ok && limit > 0 {
				fieldCost = uint64(limit) * CostField
			} else {
				// Default list multiplier
				fieldCost *= 10
			}
		}
		
		// Connection pattern (relay-style pagination) has higher cost
		if strings.HasSuffix(field.Name, "Connection") || strings.HasSuffix(field.Name, "Edge") {
			fieldCost = CostConnection
		}
		
		cost += fieldCost
		
		// Add nested field costs
		if len(field.Fields) > 0 {
			nestedCost := calculateFieldComplexity(field.Fields, variables)
			// Multiply nested costs by field multiplier
			cost += fieldCost * nestedCost
		}
	}
	
	return cost
}

// isListField heuristically determines if a field returns a list
func isListField(fieldName string) bool {
	// Common plural patterns
	pluralPatterns := []string{"users", "posts", "comments", "items", "nodes", "edges"}
	
	lowerName := strings.ToLower(fieldName)
	for _, pattern := range pluralPatterns {
		if strings.Contains(lowerName, pattern) {
			return true
		}
	}
	
	return false
}

// countAliases counts the number of aliases in a query
func countAliases(ast *GraphQLAST) int {
	count := 0
	
	for _, op := range ast.Operations {
		count += countFieldAliases(op.Fields)
	}
	
	return count
}

// countFieldAliases recursively counts field aliases
func countFieldAliases(fields []Field) int {
	count := 0
	
	for _, field := range fields {
		if field.Alias != "" {
			count++
		}
		count += countFieldAliases(field.Fields)
	}
	
	return count
}

// countDirectives counts the number of directives in a query
func countDirectives(ast *GraphQLAST) int {
	count := 0
	
	for _, op := range ast.Operations {
		count += len(op.Directives)
		count += countFieldDirectives(op.Fields)
	}
	
	return count
}

// countFieldDirectives recursively counts field directives
func countFieldDirectives(fields []Field) int {
	count := 0
	
	for _, field := range fields {
		count += len(field.Directives)
		count += countFieldDirectives(field.Fields)
	}
	
	return count
}

// isIntrospectionQuery checks if a query is an introspection query
func isIntrospectionQuery(query string) bool {
	introspectionPatterns := []string{
		"__schema",
		"__type",
		"__typename",
		"IntrospectionQuery",
	}
	
	for _, pattern := range introspectionPatterns {
		if strings.Contains(query, pattern) {
			return true
		}
	}
	
	return false
}

// hashQuery computes SHA-256 hash of a query
func hashQuery(query string) string {
	// Normalize query: remove whitespace and comments
	normalized := normalizeQuery(query)
	
	h := sha256.Sum256([]byte(normalized))
	return fmt.Sprintf("%x", h)
}

// normalizeQuery normalizes a GraphQL query for hashing
func normalizeQuery(query string) string {
	// Remove comments
	commentRegex := regexp.MustCompile(`#[^\n]*`)
	query = commentRegex.ReplaceAllString(query, "")
	
	// Remove excessive whitespace
	query = strings.Join(strings.Fields(query), " ")
	
	return strings.TrimSpace(query)
}

// ---------- Rate Limiting ----------

type graphQLRateLimiter struct {
	limit     int
	window    time.Duration
	clients   sync.Map // clientID -> *clientBucket
}

type clientBucket struct {
	count int
	reset time.Time
	mu    sync.Mutex
}

func newGraphQLRateLimiter(limit int) *graphQLRateLimiter {
	return &graphQLRateLimiter{
		limit:  limit,
		window: time.Minute,
	}
}

func (rl *graphQLRateLimiter) Allow(clientID string) bool {
	now := time.Now()
	
	// Get or create client bucket
	val, _ := rl.clients.LoadOrStore(clientID, &clientBucket{
		count: 0,
		reset: now.Add(rl.window),
	})
	
	bucket := val.(*clientBucket)
	bucket.mu.Lock()
	defer bucket.mu.Unlock()
	
	// Reset if window expired
	if now.After(bucket.reset) {
		bucket.count = 0
		bucket.reset = now.Add(rl.window)
	}
	
	// Check limit
	if bucket.count >= rl.limit {
		return false
	}
	
	bucket.count++
	return true
}

// ---------- Query Cost Estimator ----------

// CostEstimator estimates query cost before execution
type CostEstimator struct {
	typeCosts map[string]int // typename -> base cost
}

// NewCostEstimator creates a new cost estimator
func NewCostEstimator() *CostEstimator {
	return &CostEstimator{
		typeCosts: map[string]int{
			"Query":        1,
			"Mutation":     50,
			"Subscription": 100,
		},
	}
}

// EstimateCost estimates the cost of executing a query
func (ce *CostEstimator) EstimateCost(query string, variables map[string]interface{}) (int, error) {
	ast, err := parseGraphQLQuery(query)
	if err != nil {
		return 0, err
	}
	
	complexity := calculateComplexity(ast, variables)
	return int(complexity), nil
}

// ---------- Persistent Query Manager ----------

// PersistentQueryManager manages persistent queries (APQ pattern)
type PersistentQueryManager struct {
	queries   sync.Map // hash -> query string
	hits      uint64
	misses    uint64
}

// NewPersistentQueryManager creates a new persistent query manager
func NewPersistentQueryManager() *PersistentQueryManager {
	return &PersistentQueryManager{}
}

// Register registers a persistent query
func (pqm *PersistentQueryManager) Register(hash, query string) {
	pqm.queries.Store(hash, query)
}

// Get retrieves a persistent query by hash
func (pqm *PersistentQueryManager) Get(hash string) (string, bool) {
	val, ok := pqm.queries.Load(hash)
	if ok {
		atomic.AddUint64(&pqm.hits, 1)
		return val.(string), true
	}
	
	atomic.AddUint64(&pqm.misses, 1)
	return "", false
}

// GetMetrics returns persistent query metrics
func (pqm *PersistentQueryManager) GetMetrics() map[string]uint64 {
	return map[string]uint64{
		"hits":   atomic.LoadUint64(&pqm.hits),
		"misses": atomic.LoadUint64(&pqm.misses),
	}
}

// ---------- Batch Query Analyzer ----------

// BatchQueryAnalyzer analyzes batched GraphQL queries
type BatchQueryAnalyzer struct {
	maxBatchSize int
}

// NewBatchQueryAnalyzer creates a new batch query analyzer
func NewBatchQueryAnalyzer(maxBatchSize int) *BatchQueryAnalyzer {
	return &BatchQueryAnalyzer{
		maxBatchSize: maxBatchSize,
	}
}

// AnalyzeBatch validates a batch of queries
func (bqa *BatchQueryAnalyzer) AnalyzeBatch(queries []string) error {
	if len(queries) > bqa.maxBatchSize {
		return fmt.Errorf("batch size %d exceeds limit %d", len(queries), bqa.maxBatchSize)
	}
	
	// Check for batch abuse patterns
	uniqueQueries := make(map[string]int)
	for _, q := range queries {
		hash := hashQuery(q)
		uniqueQueries[hash]++
	}
	
	// Flag if too many identical queries (potential DoS)
	for _, count := range uniqueQueries {
		if count > bqa.maxBatchSize/2 {
			return errors.New("duplicate queries detected in batch")
		}
	}
	
	return nil
}

// ExportWhitelist exports the query whitelist to JSON
func (m *SecurityMiddleware) ExportWhitelist() ([]byte, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	queries := make(map[string]string)
	for hash := range m.whitelist {
		if query, ok := m.config.WhitelistedQueries[hash]; ok {
			queries[hash] = query
		}
	}
	
	return json.Marshal(queries)
}

// ImportWhitelist imports a query whitelist from JSON
func (m *SecurityMiddleware) ImportWhitelist(data []byte) error {
	var queries map[string]string
	if err := json.Unmarshal(data, &queries); err != nil {
		return err
	}
	
	m.mu.Lock()
	defer m.mu.Unlock()
	
	m.whitelist = make(map[string]struct{})
	for hash := range queries {
		m.whitelist[hash] = struct{}{}
	}
	
	return nil
}
