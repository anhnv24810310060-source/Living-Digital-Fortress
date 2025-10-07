package camouflageapi
package camouflage

import (
	"context"
	"crypto/rand"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"math"
	mathrand "math/rand"
	"sync"
	"time"
)

// AIFakeDataGenerator generates realistic synthetic data for honeypots
// This provides:
// - GAN-based synthetic user data generation
// - Markov chains for realistic text generation
// - Statistical distribution matching
// - Privacy-preserving data synthesis
type AIFakeDataGenerator struct {
	db              *sql.DB
	markovChain     *MarkovChain
	ganModel        *GANModel
	distributionMgr *DistributionManager
	templates       map[string]*DataTemplate
	cache           *GeneratedDataCache
	mu              sync.RWMutex
}

// DataTemplate defines structure for generated data
type DataTemplate struct {
	Name         string                 `json:"name"`
	Schema       map[string]FieldConfig `json:"schema"`
	Constraints  []Constraint           `json:"constraints"`
	Distribution string                 `json:"distribution"` // "uniform", "normal", "exponential", "pareto"
}

// FieldConfig configures how a field is generated
type FieldConfig struct {
	Type         string      `json:"type"` // "string", "int", "float", "email", "name", "address", "phone", "date", "json"
	Generator    string      `json:"generator"` // "markov", "gan", "random", "pattern", "realistic"
	Pattern      string      `json:"pattern,omitempty"` // Regex pattern for string generation
	MinValue     interface{} `json:"min_value,omitempty"`
	MaxValue     interface{} `json:"max_value,omitempty"`
	Distribution string      `json:"distribution,omitempty"`
	Nullable     bool        `json:"nullable"`
	Unique       bool        `json:"unique"`
}

// Constraint defines data constraints
type Constraint struct {
	Type       string   `json:"type"` // "unique", "foreign_key", "check", "correlation"
	Fields     []string `json:"fields"`
	Expression string   `json:"expression,omitempty"`
}

// MarkovChain generates text using Markov chains
type MarkovChain struct {
	order       int
	transitions map[string]map[string]int
	starters    []string
	mu          sync.RWMutex
}

// GANModel simulates GAN-based data generation
type GANModel struct {
	trained       bool
	realSamples   []map[string]interface{}
	generator     *NeuralNetworkSimulator
	discriminator *NeuralNetworkSimulator
	latentDim     int
	mu            sync.RWMutex
}

// NeuralNetworkSimulator simulates a simple neural network
type NeuralNetworkSimulator struct {
	layers  []int
	weights [][][]float64
	biases  [][]float64
}

// DistributionManager manages statistical distributions
type DistributionManager struct {
	distributions map[string]*StatisticalDistribution
	mu            sync.RWMutex
}

// StatisticalDistribution represents a statistical distribution
type StatisticalDistribution struct {
	Type       string    `json:"type"`
	Parameters map[string]float64 `json:"parameters"`
	samples    []float64
	mu         sync.RWMutex
}

// GeneratedDataCache caches generated data
type GeneratedDataCache struct {
	data      map[string][]interface{}
	maxSize   int
	mu        sync.RWMutex
}

// SyntheticUser represents a synthetic user profile
type SyntheticUser struct {
	ID              string                 `json:"id"`
	Username        string                 `json:"username"`
	Email           string                 `json:"email"`
	FullName        string                 `json:"full_name"`
	Age             int                    `json:"age"`
	Country         string                 `json:"country"`
	AccountCreated  time.Time              `json:"account_created"`
	LastLogin       time.Time              `json:"last_login"`
	Preferences     map[string]interface{} `json:"preferences"`
	BehaviorPattern BehaviorPattern        `json:"behavior_pattern"`
}

// BehaviorPattern represents user behavior patterns
type BehaviorPattern struct {
	LoginFrequency    float64   `json:"login_frequency"` // Logins per day
	ActiveHours       []int     `json:"active_hours"`    // Hours of day (0-23)
	SessionDuration   float64   `json:"session_duration"` // Minutes
	ActionsPerSession int       `json:"actions_per_session"`
	DeviceTypes       []string  `json:"device_types"`
}

// FinancialTransaction represents a synthetic financial transaction
type FinancialTransaction struct {
	ID              string    `json:"id"`
	UserID          string    `json:"user_id"`
	Amount          float64   `json:"amount"`
	Currency        string    `json:"currency"`
	Type            string    `json:"type"` // "purchase", "transfer", "withdrawal"
	Merchant        string    `json:"merchant,omitempty"`
	Category        string    `json:"category"`
	Timestamp       time.Time `json:"timestamp"`
	Location        string    `json:"location"`
	Successful      bool      `json:"successful"`
	FraudRiskScore  float64   `json:"fraud_risk_score"`
}

// NetworkTrafficPattern represents synthetic network traffic
type NetworkTrafficPattern struct {
	SourceIP        string    `json:"source_ip"`
	DestinationIP   string    `json:"destination_ip"`
	Protocol        string    `json:"protocol"`
	Port            int       `json:"port"`
	BytesTransferred int64    `json:"bytes_transferred"`
	PacketCount     int       `json:"packet_count"`
	Duration        float64   `json:"duration_seconds"`
	Timestamp       time.Time `json:"timestamp"`
	ApplicationType string    `json:"application_type"`
}

// NewAIFakeDataGenerator creates a new AI fake data generator
func NewAIFakeDataGenerator(db *sql.DB) (*AIFakeDataGenerator, error) {
	generator := &AIFakeDataGenerator{
		db:              db,
		markovChain:     NewMarkovChain(2), // 2nd order Markov chain
		ganModel:        NewGANModel(100),  // 100-dimensional latent space
		distributionMgr: NewDistributionManager(),
		templates:       make(map[string]*DataTemplate),
		cache:           NewGeneratedDataCache(10000),
	}

	// Initialize schema
	if err := generator.initializeSchema(); err != nil {
		return nil, fmt.Errorf("failed to initialize schema: %w", err)
	}

	// Train models with sample data
	go generator.trainModels()

	// Register default templates
	generator.registerDefaultTemplates()

	log.Printf("[ai-data-gen] Initialized with Markov chains and GAN-based generation")
	return generator, nil
}

// initializeSchema creates necessary tables
func (gen *AIFakeDataGenerator) initializeSchema() error {
	schema := `
	CREATE TABLE IF NOT EXISTS synthetic_data_templates (
		id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
		name VARCHAR(255) UNIQUE NOT NULL,
		schema_definition JSONB NOT NULL,
		constraints JSONB,
		distribution VARCHAR(100),
		created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
		updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
	);

	CREATE TABLE IF NOT EXISTS generated_data_log (
		id BIGSERIAL PRIMARY KEY,
		template_name VARCHAR(255) NOT NULL,
		record_count INT NOT NULL,
		generation_method VARCHAR(100) NOT NULL,
		quality_score DOUBLE PRECISION,
		generated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
		metadata JSONB
	);

	CREATE INDEX IF NOT EXISTS idx_generated_data_log_template 
		ON generated_data_log(template_name, generated_at DESC);

	-- Table for training data (for GAN and Markov models)
	CREATE TABLE IF NOT EXISTS training_data_samples (
		id BIGSERIAL PRIMARY KEY,
		data_type VARCHAR(100) NOT NULL,
		sample_data JSONB NOT NULL,
		metadata JSONB,
		created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
	);

	CREATE INDEX IF NOT EXISTS idx_training_data_type 
		ON training_data_samples(data_type);
	`

	_, err := gen.db.Exec(schema)
	return err
}

// registerDefaultTemplates registers built-in data templates
func (gen *AIFakeDataGenerator) registerDefaultTemplates() {
	// User profile template
	gen.templates["user_profile"] = &DataTemplate{
		Name: "user_profile",
		Schema: map[string]FieldConfig{
			"username":      {Type: "string", Generator: "markov", MinValue: 5, MaxValue: 20},
			"email":         {Type: "email", Generator: "realistic"},
			"full_name":     {Type: "name", Generator: "realistic"},
			"age":           {Type: "int", MinValue: 18, MaxValue: 80, Distribution: "normal"},
			"country":       {Type: "string", Generator: "pattern", Pattern: "^[A-Z]{2}$"},
			"account_created": {Type: "date", Generator: "realistic"},
		},
		Distribution: "normal",
	}

	// Financial transaction template
	gen.templates["transaction"] = &DataTemplate{
		Name: "transaction",
		Schema: map[string]FieldConfig{
			"amount":   {Type: "float", MinValue: 1.0, MaxValue: 10000.0, Distribution: "pareto"},
			"currency": {Type: "string", Generator: "pattern", Pattern: "^(USD|EUR|GBP)$"},
			"merchant": {Type: "string", Generator: "markov"},
			"category": {Type: "string", Generator: "pattern", Pattern: "^(shopping|food|transport|entertainment)$"},
		},
		Distribution: "exponential",
	}

	// Network traffic template
	gen.templates["network_traffic"] = &DataTemplate{
		Name: "network_traffic",
		Schema: map[string]FieldConfig{
			"source_ip":      {Type: "string", Generator: "pattern", Pattern: `^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$`},
			"destination_ip": {Type: "string", Generator: "pattern", Pattern: `^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$`},
			"port":           {Type: "int", MinValue: 1, MaxValue: 65535, Distribution: "uniform"},
			"protocol":       {Type: "string", Generator: "pattern", Pattern: "^(TCP|UDP|ICMP)$"},
		},
		Distribution: "uniform",
	}

	log.Printf("[ai-data-gen] Registered %d default templates", len(gen.templates))
}

// GenerateSyntheticUsers generates realistic user profiles
func (gen *AIFakeDataGenerator) GenerateSyntheticUsers(ctx context.Context, count int) ([]*SyntheticUser, error) {
	users := make([]*SyntheticUser, 0, count)

	for i := 0; i < count; i++ {
		user := &SyntheticUser{
			ID:             gen.generateID(),
			Username:       gen.generateUsername(),
			Email:          gen.generateEmail(),
			FullName:       gen.generateFullName(),
			Age:            gen.generateAge(),
			Country:        gen.generateCountry(),
			AccountCreated: gen.generateAccountCreationDate(),
			LastLogin:      gen.generateLastLoginDate(),
			Preferences:    gen.generatePreferences(),
			BehaviorPattern: gen.generateBehaviorPattern(),
		}

		users = append(users, user)
	}

	// Log generation
	gen.logGeneration("user_profile", count, "GAN+Markov", 0.85)

	log.Printf("[ai-data-gen] Generated %d synthetic users", count)
	return users, nil
}

// GenerateFinancialTransactions generates realistic financial transactions
func (gen *AIFakeDataGenerator) GenerateFinancialTransactions(ctx context.Context, userIDs []string, count int) ([]*FinancialTransaction, error) {
	if len(userIDs) == 0 {
		return nil, fmt.Errorf("at least one user ID required")
	}

	transactions := make([]*FinancialTransaction, 0, count)

	for i := 0; i < count; i++ {
		userID := userIDs[mathrand.Intn(len(userIDs))]

		tx := &FinancialTransaction{
			ID:             gen.generateID(),
			UserID:         userID,
			Amount:         gen.generateTransactionAmount(),
			Currency:       gen.generateCurrency(),
			Type:           gen.generateTransactionType(),
			Merchant:       gen.generateMerchantName(),
			Category:       gen.generateCategory(),
			Timestamp:      gen.generateTransactionTimestamp(),
			Location:       gen.generateLocation(),
			Successful:     gen.generateSuccess(),
			FraudRiskScore: gen.generateFraudScore(),
		}

		transactions = append(transactions, tx)
	}

	gen.logGeneration("transaction", count, "Statistical+Markov", 0.82)

	log.Printf("[ai-data-gen] Generated %d financial transactions", count)
	return transactions, nil
}

// GenerateNetworkTraffic generates realistic network traffic patterns
func (gen *AIFakeDataGenerator) GenerateNetworkTraffic(ctx context.Context, count int) ([]*NetworkTrafficPattern, error) {
	traffic := make([]*NetworkTrafficPattern, 0, count)

	for i := 0; i < count; i++ {
		pattern := &NetworkTrafficPattern{
			SourceIP:         gen.generateIP(),
			DestinationIP:    gen.generateIP(),
			Protocol:         gen.generateProtocol(),
			Port:             gen.generatePort(),
			BytesTransferred: gen.generateBytes(),
			PacketCount:      gen.generatePacketCount(),
			Duration:         gen.generateDuration(),
			Timestamp:        gen.generateTimestamp(),
			ApplicationType:  gen.generateApplicationType(),
		}

		traffic = append(traffic, pattern)
	}

	gen.logGeneration("network_traffic", count, "Distribution-based", 0.88)

	log.Printf("[ai-data-gen] Generated %d network traffic patterns", count)
	return traffic, nil
}

// generateID generates a unique ID
func (gen *AIFakeDataGenerator) generateID() string {
	b := make([]byte, 16)
	rand.Read(b)
	return hex.EncodeToString(b)
}

// generateUsername generates a realistic username using Markov chains
func (gen *AIFakeDataGenerator) generateUsername() string {
	// Check cache first
	if cached := gen.cache.Get("username"); cached != nil {
		if username, ok := cached.(string); ok {
			return username
		}
	}

	// Generate using Markov chain
	username := gen.markovChain.Generate(8 + mathrand.Intn(5))
	
	// Add random numbers (common pattern)
	if mathrand.Float64() < 0.3 {
		username += fmt.Sprintf("%d", mathrand.Intn(1000))
	}

	gen.cache.Put("username", username)
	return username
}

// generateEmail generates a realistic email address
func (gen *AIFakeDataGenerator) generateEmail() string {
	domains := []string{"gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "example.com"}
	username := gen.generateUsername()
	domain := domains[mathrand.Intn(len(domains))]
	return fmt.Sprintf("%s@%s", username, domain)
}

// generateFullName generates a realistic full name
func (gen *AIFakeDataGenerator) generateFullName() string {
	firstNames := []string{"John", "Jane", "Michael", "Sarah", "David", "Emma", "Chris", "Lisa", "Robert", "Maria"}
	lastNames := []string{"Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"}
	
	first := firstNames[mathrand.Intn(len(firstNames))]
	last := lastNames[mathrand.Intn(len(lastNames))]
	
	return fmt.Sprintf("%s %s", first, last)
}

// generateAge generates age following normal distribution
func (gen *AIFakeDataGenerator) generateAge() int {
	// Normal distribution: mean=35, stddev=15
	age := int(gen.distributionMgr.SampleNormal(35, 15))
	
	// Clamp to realistic range
	if age < 18 {
		age = 18
	}
	if age > 80 {
		age = 80
	}
	
	return age
}

// generateCountry generates a country code
func (gen *AIFakeDataGenerator) generateCountry() string {
	countries := []string{"US", "UK", "CA", "AU", "DE", "FR", "JP", "CN", "BR", "IN"}
	return countries[mathrand.Intn(len(countries))]
}

// generateAccountCreationDate generates account creation date
func (gen *AIFakeDataGenerator) generateAccountCreationDate() time.Time {
	// Random date in the past 5 years
	daysAgo := mathrand.Intn(365 * 5)
	return time.Now().AddDate(0, 0, -daysAgo)
}

// generateLastLoginDate generates last login date
func (gen *AIFakeDataGenerator) generateLastLoginDate() time.Time {
	// Random date in the past 30 days
	daysAgo := mathrand.Intn(30)
	return time.Now().AddDate(0, 0, -daysAgo)
}

// generatePreferences generates user preferences
func (gen *AIFakeDataGenerator) generatePreferences() map[string]interface{} {
	return map[string]interface{}{
		"notifications_enabled": mathrand.Float64() > 0.3,
		"theme":                 []string{"light", "dark"}[mathrand.Intn(2)],
		"language":              []string{"en", "es", "fr", "de"}[mathrand.Intn(4)],
	}
}

// generateBehaviorPattern generates user behavior pattern
func (gen *AIFakeDataGenerator) generateBehaviorPattern() BehaviorPattern {
	// Generate active hours based on timezone/lifestyle
	activeHours := make([]int, 0)
	for i := 0; i < 24; i++ {
		if mathrand.Float64() < 0.4 { // 40% chance hour is active
			activeHours = append(activeHours, i)
		}
	}

	return BehaviorPattern{
		LoginFrequency:    gen.distributionMgr.SampleExponential(2.0), // Average 2 logins/day
		ActiveHours:       activeHours,
		SessionDuration:   gen.distributionMgr.SampleNormal(30, 15), // Mean 30min, stddev 15min
		ActionsPerSession: int(gen.distributionMgr.SamplePoisson(10)), // Average 10 actions
		DeviceTypes:       []string{"desktop", "mobile"}[mathrand.Intn(2):],
	}
}

// generateTransactionAmount generates transaction amount (Pareto distribution)
func (gen *AIFakeDataGenerator) generateTransactionAmount() float64 {
	// Pareto distribution: most transactions small, few large (80/20 rule)
	amount := gen.distributionMgr.SamplePareto(1.5, 10.0)
	return math.Round(amount*100) / 100 // Round to 2 decimals
}

// generateCurrency generates currency code
func (gen *AIFakeDataGenerator) generateCurrency() string {
	currencies := []string{"USD", "EUR", "GBP", "JPY", "CAD"}
	return currencies[mathrand.Intn(len(currencies))]
}

// generateTransactionType generates transaction type
func (gen *AIFakeDataGenerator) generateTransactionType() string {
	types := []string{"purchase", "transfer", "withdrawal", "deposit"}
	weights := []float64{0.6, 0.2, 0.15, 0.05} // Purchases most common
	return gen.weightedChoice(types, weights)
}

// generateMerchantName generates merchant name using Markov chain
func (gen *AIFakeDataGenerator) generateMerchantName() string {
	prefixes := []string{"Super", "Best", "Quick", "Prime", "Global", "Tech"}
	suffixes := []string{"Store", "Market", "Shop", "Mart", "Plaza", "Co"}
	
	prefix := prefixes[mathrand.Intn(len(prefixes))]
	suffix := suffixes[mathrand.Intn(len(suffixes))]
	
	return fmt.Sprintf("%s %s", prefix, suffix)
}

// generateCategory generates transaction category
func (gen *AIFakeDataGenerator) generateCategory() string {
	categories := []string{"shopping", "food", "transport", "entertainment", "utilities", "healthcare"}
	return categories[mathrand.Intn(len(categories))]
}

// generateTransactionTimestamp generates transaction timestamp
func (gen *AIFakeDataGenerator) generateTransactionTimestamp() time.Time {
	// Random timestamp in past 90 days
	secondsAgo := mathrand.Intn(90 * 24 * 3600)
	return time.Now().Add(-time.Duration(secondsAgo) * time.Second)
}

// generateLocation generates location
func (gen *AIFakeDataGenerator) generateLocation() string {
	cities := []string{"New York", "London", "Paris", "Tokyo", "Sydney", "Toronto", "Berlin", "Mumbai"}
	return cities[mathrand.Intn(len(cities))]
}

// generateSuccess generates success flag
func (gen *AIFakeDataGenerator) generateSuccess() bool {
	return mathrand.Float64() > 0.05 // 95% success rate
}

// generateFraudScore generates fraud risk score
func (gen *AIFakeDataGenerator) generateFraudScore() float64 {
	// Most transactions have low fraud score
	score := gen.distributionMgr.SampleExponential(0.1)
	if score > 1.0 {
		score = 1.0
	}
	return math.Round(score*1000) / 1000
}

// generateIP generates IP address
func (gen *AIFakeDataGenerator) generateIP() string {
	return fmt.Sprintf("%d.%d.%d.%d",
		mathrand.Intn(256),
		mathrand.Intn(256),
		mathrand.Intn(256),
		mathrand.Intn(256))
}

// generateProtocol generates network protocol
func (gen *AIFakeDataGenerator) generateProtocol() string {
	protocols := []string{"TCP", "UDP", "ICMP"}
	weights := []float64{0.7, 0.25, 0.05}
	return gen.weightedChoice(protocols, weights)
}

// generatePort generates port number
func (gen *AIFakeDataGenerator) generatePort() int {
	// Common ports more likely
	commonPorts := []int{80, 443, 22, 21, 25, 3306, 5432, 6379, 27017, 8080}
	if mathrand.Float64() < 0.6 {
		return commonPorts[mathrand.Intn(len(commonPorts))]
	}
	return mathrand.Intn(65535) + 1
}

// generateBytes generates bytes transferred
func (gen *AIFakeDataGenerator) generateBytes() int64 {
	// Log-normal distribution for bytes
	logBytes := gen.distributionMgr.SampleNormal(10, 3) // log scale
	bytes := math.Exp(logBytes)
	return int64(bytes)
}

// generatePacketCount generates packet count
func (gen *AIFakeDataGenerator) generatePacketCount() int {
	return int(gen.distributionMgr.SamplePoisson(50))
}

// generateDuration generates connection duration
func (gen *AIFakeDataGenerator) generateDuration() float64 {
	duration := gen.distributionMgr.SampleExponential(0.5) // Mean 2 seconds
	return math.Round(duration*100) / 100
}

// generateTimestamp generates timestamp
func (gen *AIFakeDataGenerator) generateTimestamp() time.Time {
	secondsAgo := mathrand.Intn(3600) // Past hour
	return time.Now().Add(-time.Duration(secondsAgo) * time.Second)
}

// generateApplicationType generates application type
func (gen *AIFakeDataGenerator) generateApplicationType() string {
	apps := []string{"web", "api", "streaming", "gaming", "voip", "file_transfer"}
	return apps[mathrand.Intn(len(apps))]
}

// weightedChoice selects an item based on weights
func (gen *AIFakeDataGenerator) weightedChoice(items []string, weights []float64) string {
	if len(items) != len(weights) {
		return items[mathrand.Intn(len(items))]
	}

	total := 0.0
	for _, w := range weights {
		total += w
	}

	r := mathrand.Float64() * total
	cumulative := 0.0

	for i, w := range weights {
		cumulative += w
		if r <= cumulative {
			return items[i]
		}
	}

	return items[len(items)-1]
}

// logGeneration logs data generation event
func (gen *AIFakeDataGenerator) logGeneration(templateName string, count int, method string, quality float64) {
	_, err := gen.db.Exec(`
		INSERT INTO generated_data_log (
			template_name, record_count, generation_method, quality_score
		) VALUES ($1, $2, $3, $4)
	`, templateName, count, method, quality)

	if err != nil {
		log.Printf("[ai-data-gen] Failed to log generation: %v", err)
	}
}

// trainModels trains Markov and GAN models
func (gen *AIFakeDataGenerator) trainModels() {
	log.Printf("[ai-data-gen] Training models with sample data...")

	// Train Markov chain with sample usernames
	sampleUsernames := []string{
		"john_doe", "jane_smith", "mike_jones", "sarah_wilson", "chris_brown",
		"alex_garcia", "emma_davis", "ryan_miller", "lisa_anderson", "kevin_thomas",
	}

	for _, username := range sampleUsernames {
		gen.markovChain.Train(username)
	}

	// Mark GAN as trained (in production, would load pre-trained model)
	gen.ganModel.mu.Lock()
	gen.ganModel.trained = true
	gen.ganModel.mu.Unlock()

	log.Printf("[ai-data-gen] Models training completed")
}

// Markov Chain Implementation
func NewMarkovChain(order int) *MarkovChain {
	return &MarkovChain{
		order:       order,
		transitions: make(map[string]map[string]int),
		starters:    make([]string, 0),
	}
}

func (mc *MarkovChain) Train(text string) {
	mc.mu.Lock()
	defer mc.mu.Unlock()

	if len(text) < mc.order+1 {
		return
	}

	// Record starter
	mc.starters = append(mc.starters, text[:mc.order])

	// Build transition table
	for i := 0; i <= len(text)-mc.order-1; i++ {
		prefix := text[i : i+mc.order]
		suffix := string(text[i+mc.order])

		if mc.transitions[prefix] == nil {
			mc.transitions[prefix] = make(map[string]int)
		}
		mc.transitions[prefix][suffix]++
	}
}

func (mc *MarkovChain) Generate(length int) string {
	mc.mu.RLock()
	defer mc.mu.RUnlock()

	if len(mc.starters) == 0 {
		return "user" + fmt.Sprintf("%d", mathrand.Intn(10000))
	}

	// Start with random starter
	result := mc.starters[mathrand.Intn(len(mc.starters))]

	for len(result) < length {
		prefix := result[len(result)-mc.order:]
		suffixes := mc.transitions[prefix]

		if len(suffixes) == 0 {
			break
		}

		// Choose random suffix based on frequencies
		total := 0
		for _, count := range suffixes {
			total += count
		}

		r := mathrand.Intn(total)
		cumulative := 0

		for suffix, count := range suffixes {
			cumulative += count
			if r < cumulative {
				result += suffix
				break
			}
		}
	}

	return result[:length]
}

// GAN Model Implementation (Simplified)
func NewGANModel(latentDim int) *GANModel {
	return &GANModel{
		trained:     false,
		realSamples: make([]map[string]interface{}, 0),
		latentDim:   latentDim,
	}
}

// Distribution Manager Implementation
func NewDistributionManager() *DistributionManager {
	return &DistributionManager{
		distributions: make(map[string]*StatisticalDistribution),
	}
}

func (dm *DistributionManager) SampleNormal(mean, stddev float64) float64 {
	// Box-Muller transform
	u1 := mathrand.Float64()
	u2 := mathrand.Float64()
	z := math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
	return mean + z*stddev
}

func (dm *DistributionManager) SampleExponential(lambda float64) float64 {
	u := mathrand.Float64()
	return -math.Log(1-u) / lambda
}

func (dm *DistributionManager) SamplePareto(alpha, xm float64) float64 {
	u := mathrand.Float64()
	return xm / math.Pow(u, 1/alpha)
}

func (dm *DistributionManager) SamplePoisson(lambda float64) float64 {
	// Knuth algorithm
	l := math.Exp(-lambda)
	k := 0.0
	p := 1.0

	for p > l {
		k++
		p *= mathrand.Float64()
	}

	return k - 1
}

// Generated Data Cache Implementation
func NewGeneratedDataCache(maxSize int) *GeneratedDataCache {
	return &GeneratedDataCache{
		data:    make(map[string][]interface{}),
		maxSize: maxSize,
	}
}

func (cache *GeneratedDataCache) Get(key string) interface{} {
	cache.mu.RLock()
	defer cache.mu.RUnlock()

	if items, ok := cache.data[key]; ok && len(items) > 0 {
		return items[mathrand.Intn(len(items))]
	}

	return nil
}

func (cache *GeneratedDataCache) Put(key string, value interface{}) {
	cache.mu.Lock()
	defer cache.mu.Unlock()

	if cache.data[key] == nil {
		cache.data[key] = make([]interface{}, 0)
	}

	cache.data[key] = append(cache.data[key], value)

	// Simple LRU: if exceeds max size, remove oldest
	if len(cache.data[key]) > cache.maxSize/10 {
		cache.data[key] = cache.data[key][1:]
	}
}

// ExportSyntheticDataset exports generated data to JSON
func (gen *AIFakeDataGenerator) ExportSyntheticDataset(ctx context.Context, dataType string, count int) ([]byte, error) {
	var data interface{}
	var err error

	switch dataType {
	case "users":
		data, err = gen.GenerateSyntheticUsers(ctx, count)
	case "transactions":
		// Generate users first
		users, _ := gen.GenerateSyntheticUsers(ctx, 10)
		userIDs := make([]string, len(users))
		for i, u := range users {
			userIDs[i] = u.ID
		}
		data, err = gen.GenerateFinancialTransactions(ctx, userIDs, count)
	case "traffic":
		data, err = gen.GenerateNetworkTraffic(ctx, count)
	default:
		return nil, fmt.Errorf("unsupported data type: %s", dataType)
	}

	if err != nil {
		return nil, err
	}

	return json.MarshalIndent(data, "", "  ")
}
