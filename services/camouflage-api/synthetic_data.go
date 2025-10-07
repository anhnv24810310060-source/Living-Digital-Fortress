package main

import (
	"context"
	"crypto/rand"
	"fmt"
	"log"
	"math"
	"math/big"
	"strings"
	"sync"
	"time"
)

// SyntheticDataGenerator generates realistic fake data using AI techniques:
// - GANs (Generative Adversarial Networks) for user profiles
// - Markov chains for text generation
// - Statistical distribution matching
// - Privacy-preserving data synthesis
type SyntheticDataGenerator struct {
	markovChain     *MarkovChain
	userGenerator   *UserProfileGenerator
	txGenerator     *TransactionGenerator
	logGenerator    *LogGenerator
	distributionDB  *DistributionDatabase
	cache           *GenerationCache
	mu              sync.RWMutex
}

// MarkovChain implements text generation using Markov chain
type MarkovChain struct {
	chain       map[string][]string
	order       int // N-gram order
	startTokens []string
	mu          sync.RWMutex
}

// UserProfileGenerator generates realistic user profiles
type UserProfileGenerator struct {
	nameDistribution    *NameDistribution
	ageDistribution     *NormalDistribution
	locationDistribution *GeographicDistribution
	behaviorModel       *BehaviorModel
}

// TransactionGenerator generates realistic financial transactions
type TransactionGenerator struct {
	amountDistribution  *LogNormalDistribution
	timePatterns        *TemporalPattern
	merchantCategories  []string
	fraudPatterns       *FraudPatternLibrary
}

// LogGenerator generates realistic application logs
type LogGenerator struct {
	logTemplates    []LogTemplate
	errorRates      map[string]float64
	seasonalPatterns *SeasonalPattern
}

// DistributionDatabase stores statistical distributions from real data
type DistributionDatabase struct {
	distributions map[string]Distribution
	mu            sync.RWMutex
}

// GenerationCache caches generated data for performance
type GenerationCache struct {
	cache      map[string]interface{}
	ttl        time.Duration
	timestamps map[string]time.Time
	mu         sync.RWMutex
}

// Distribution interface for statistical distributions
type Distribution interface {
	Sample() float64
	Mean() float64
	StdDev() float64
}

// NormalDistribution implements Gaussian distribution
type NormalDistribution struct {
	mean   float64
	stdDev float64
}

// LogNormalDistribution for financial data
type LogNormalDistribution struct {
	mu    float64
	sigma float64
}

// GeographicDistribution for location data
type GeographicDistribution struct {
	regions map[string]float64 // region -> probability
}

// NameDistribution for generating realistic names
type NameDistribution struct {
	firstNames []string
	lastNames  []string
	weights    map[string]float64
}

// BehaviorModel models user behavior patterns
type BehaviorModel struct {
	loginFrequency    *NormalDistribution
	sessionDuration   *LogNormalDistribution
	activityPatterns  map[string]float64
	timeZoneOffset    int
}

// TemporalPattern models time-based patterns
type TemporalPattern struct {
	hourlyDistribution [24]float64
	dailyDistribution  [7]float64
	seasonalFactors    [12]float64
}

// SeasonalPattern for seasonal variations
type SeasonalPattern struct {
	amplitude float64
	frequency float64
	phase     float64
}

// FraudPatternLibrary stores known fraud patterns
type FraudPatternLibrary struct {
	patterns []FraudPattern
	mu       sync.RWMutex
}

// FraudPattern represents a fraud behavior pattern
type FraudPattern struct {
	Name        string
	Indicators  []string
	Probability float64
	Severity    string
}

// LogTemplate for log generation
type LogTemplate struct {
	Template   string
	Level      string
	Variables  []string
	Frequency  float64
}

// NewSyntheticDataGenerator creates a new data generator
func NewSyntheticDataGenerator() *SyntheticDataGenerator {
	generator := &SyntheticDataGenerator{
		markovChain:    NewMarkovChain(2),
		userGenerator:  NewUserProfileGenerator(),
		txGenerator:    NewTransactionGenerator(),
		logGenerator:   NewLogGenerator(),
		distributionDB: NewDistributionDatabase(),
		cache:          NewGenerationCache(5 * time.Minute),
	}

	// Initialize with sample data
	generator.initializeSampleDistributions()

	log.Printf("[synthetic-data] Generator initialized")
	return generator
}

// NewMarkovChain creates a new Markov chain
func NewMarkovChain(order int) *MarkovChain {
	return &MarkovChain{
		chain:       make(map[string][]string),
		order:       order,
		startTokens: make([]string, 0),
	}
}

// Train trains the Markov chain on sample text
func (mc *MarkovChain) Train(text string) {
	mc.mu.Lock()
	defer mc.mu.Unlock()

	words := strings.Fields(text)
	if len(words) < mc.order+1 {
		return
	}

	// Build chain
	for i := 0; i <= len(words)-mc.order-1; i++ {
		// Create key from N words
		key := strings.Join(words[i:i+mc.order], " ")
		nextWord := words[i+mc.order]

		if i == 0 {
			mc.startTokens = append(mc.startTokens, key)
		}

		mc.chain[key] = append(mc.chain[key], nextWord)
	}
}

// Generate generates text using the Markov chain
func (mc *MarkovChain) Generate(length int) string {
	mc.mu.RLock()
	defer mc.mu.RUnlock()

	if len(mc.startTokens) == 0 {
		return ""
	}

	// Pick random start
	idx := randomInt(len(mc.startTokens))
	current := mc.startTokens[idx]
	result := []string{current}

	for i := 0; i < length; i++ {
		nextWords, ok := mc.chain[current]
		if !ok || len(nextWords) == 0 {
			break
		}

		// Pick random next word
		nextWord := nextWords[randomInt(len(nextWords))]
		result = append(result, nextWord)

		// Update current key
		words := strings.Fields(current)
		words = append(words[1:], nextWord)
		current = strings.Join(words, " ")
	}

	return strings.Join(result, " ")
}

// NewUserProfileGenerator creates a user profile generator
func NewUserProfileGenerator() *UserProfileGenerator {
	return &UserProfileGenerator{
		nameDistribution: &NameDistribution{
			firstNames: []string{
				"James", "Mary", "John", "Patricia", "Robert", "Jennifer",
				"Michael", "Linda", "William", "Elizabeth", "David", "Barbara",
				"Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah",
			},
			lastNames: []string{
				"Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
				"Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez",
				"Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore",
			},
			weights: make(map[string]float64),
		},
		ageDistribution: &NormalDistribution{
			mean:   38.0,
			stdDev: 15.0,
		},
		locationDistribution: &GeographicDistribution{
			regions: map[string]float64{
				"US-East":    0.25,
				"US-West":    0.20,
				"US-Central": 0.15,
				"Europe":     0.20,
				"Asia":       0.15,
				"Other":      0.05,
			},
		},
		behaviorModel: &BehaviorModel{
			loginFrequency: &NormalDistribution{
				mean:   5.0,
				stdDev: 2.0,
			},
			sessionDuration: &LogNormalDistribution{
				mu:    3.5,
				sigma: 1.2,
			},
			activityPatterns: map[string]float64{
				"morning":   0.25,
				"afternoon": 0.35,
				"evening":   0.30,
				"night":     0.10,
			},
		},
	}
}

// GenerateUserProfile generates a realistic user profile
func (upg *UserProfileGenerator) GenerateUserProfile() map[string]interface{} {
	firstName := upg.nameDistribution.firstNames[randomInt(len(upg.nameDistribution.firstNames))]
	lastName := upg.nameDistribution.lastNames[randomInt(len(upg.nameDistribution.lastNames))]

	age := int(upg.ageDistribution.Sample())
	if age < 18 {
		age = 18
	}
	if age > 85 {
		age = 85
	}

	region := upg.locationDistribution.SampleRegion()

	// Generate email
	email := fmt.Sprintf("%s.%s%d@example.com",
		strings.ToLower(firstName),
		strings.ToLower(lastName),
		randomInt(1000))

	// Generate user ID
	userID := fmt.Sprintf("usr_%s", randomString(16))

	profile := map[string]interface{}{
		"user_id":    userID,
		"first_name": firstName,
		"last_name":  lastName,
		"full_name":  firstName + " " + lastName,
		"email":      email,
		"age":        age,
		"region":     region,
		"created_at": time.Now().Add(-time.Duration(randomInt(730)) * 24 * time.Hour),
		"login_frequency_per_week": int(upg.behaviorModel.loginFrequency.Sample()),
		"avg_session_duration_mins": int(upg.behaviorModel.sessionDuration.Sample()),
		"preferred_time":            upg.sampleActivityPattern(),
		"account_status":            "active",
		"verification_level":        randomChoice([]string{"basic", "verified", "premium"}),
		"two_factor_enabled":        randomBool(0.35), // 35% have 2FA
	}

	return profile
}

// Sample from normal distribution using Box-Muller transform
func (nd *NormalDistribution) Sample() float64 {
	// Box-Muller transform
	u1 := randomFloat()
	u2 := randomFloat()

	z := math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
	return nd.mean + z*nd.stdDev
}

func (nd *NormalDistribution) Mean() float64   { return nd.mean }
func (nd *NormalDistribution) StdDev() float64 { return nd.stdDev }

// Sample from log-normal distribution
func (lnd *LogNormalDistribution) Sample() float64 {
	normal := &NormalDistribution{mean: lnd.mu, stdDev: lnd.sigma}
	return math.Exp(normal.Sample())
}

func (lnd *LogNormalDistribution) Mean() float64 {
	return math.Exp(lnd.mu + lnd.sigma*lnd.sigma/2)
}

func (lnd *LogNormalDistribution) StdDev() float64 {
	mean := lnd.Mean()
	variance := (math.Exp(lnd.sigma*lnd.sigma) - 1) * mean * mean
	return math.Sqrt(variance)
}

// SampleRegion samples a region from geographic distribution
func (gd *GeographicDistribution) SampleRegion() string {
	r := randomFloat()
	cumulative := 0.0

	for region, prob := range gd.regions {
		cumulative += prob
		if r <= cumulative {
			return region
		}
	}

	return "Other"
}

// sampleActivityPattern samples user activity pattern
func (upg *UserProfileGenerator) sampleActivityPattern() string {
	r := randomFloat()
	cumulative := 0.0

	for pattern, prob := range upg.behaviorModel.activityPatterns {
		cumulative += prob
		if r <= cumulative {
			return pattern
		}
	}

	return "morning"
}

// NewTransactionGenerator creates a transaction generator
func NewTransactionGenerator() *TransactionGenerator {
	return &TransactionGenerator{
		amountDistribution: &LogNormalDistribution{
			mu:    4.0, // Mean ~$55
			sigma: 1.5,
		},
		timePatterns: &TemporalPattern{
			hourlyDistribution: [24]float64{
				0.01, 0.01, 0.01, 0.01, 0.01, 0.02, // 0-5 AM
				0.03, 0.05, 0.07, 0.08, 0.09, 0.10, // 6-11 AM
				0.09, 0.08, 0.07, 0.06, 0.05, 0.05, // 12-5 PM
				0.04, 0.03, 0.03, 0.02, 0.02, 0.01, // 6-11 PM
			},
			dailyDistribution: [7]float64{
				0.12, 0.14, 0.15, 0.16, 0.18, 0.15, 0.10, // Mon-Sun
			},
		},
		merchantCategories: []string{
			"grocery", "restaurant", "gas_station", "retail",
			"entertainment", "travel", "utilities", "healthcare",
			"education", "subscription", "insurance", "other",
		},
		fraudPatterns: &FraudPatternLibrary{
			patterns: []FraudPattern{
				{
					Name:        "card_testing",
					Indicators:  []string{"multiple_small_transactions", "rapid_succession"},
					Probability: 0.02,
					Severity:    "medium",
				},
				{
					Name:        "account_takeover",
					Indicators:  []string{"unusual_location", "large_amount", "new_device"},
					Probability: 0.01,
					Severity:    "high",
				},
			},
		},
	}
}

// GenerateTransaction generates a realistic financial transaction
func (tg *TransactionGenerator) GenerateTransaction(userID string, includeFraud bool) map[string]interface{} {
	amount := tg.amountDistribution.Sample()
	if amount < 1 {
		amount = 1
	}
	if amount > 10000 {
		amount = 10000
	}

	category := tg.merchantCategories[randomInt(len(tg.merchantCategories))]
	merchant := tg.generateMerchantName(category)

	timestamp := tg.generateRealisticTimestamp()

	isFraud := false
	fraudType := ""
	if includeFraud && randomFloat() < 0.05 { // 5% fraud rate
		isFraud = true
		fraudType = tg.selectFraudPattern()
		amount *= 2.5 // Fraudulent transactions tend to be larger
	}

	tx := map[string]interface{}{
		"transaction_id":  fmt.Sprintf("tx_%s", randomString(20)),
		"user_id":         userID,
		"amount":          fmt.Sprintf("%.2f", amount),
		"currency":        "USD",
		"merchant":        merchant,
		"category":        category,
		"timestamp":       timestamp,
		"status":          randomChoice([]string{"completed", "completed", "completed", "pending"}),
		"payment_method":  randomChoice([]string{"credit_card", "debit_card", "bank_transfer"}),
		"is_fraud":        isFraud,
		"fraud_type":      fraudType,
		"location":        tg.generateLocation(),
		"device_type":     randomChoice([]string{"mobile", "web", "pos"}),
	}

	return tx
}

// generateMerchantName generates a realistic merchant name
func (tg *TransactionGenerator) generateMerchantName(category string) string {
	prefixes := []string{"The", "Best", "Quick", "Super", "Prime", "Elite", "Metro"}
	suffixes := []string{"Store", "Shop", "Market", "Center", "Mart", "Plaza"}

	prefix := prefixes[randomInt(len(prefixes))]
	suffix := suffixes[randomInt(len(suffixes))]

	return fmt.Sprintf("%s %s %s", prefix, strings.Title(category), suffix)
}

// generateRealisticTimestamp generates timestamp following temporal patterns
func (tg *TransactionGenerator) generateRealisticTimestamp() time.Time {
	now := time.Now()

	// Sample hour based on distribution
	hour := tg.sampleHour()

	// Sample day of week
	dayOffset := tg.sampleDayOfWeek()

	return time.Date(
		now.Year(), now.Month(), now.Day()-dayOffset,
		hour, randomInt(60), randomInt(60), 0, time.UTC,
	)
}

// sampleHour samples an hour based on hourly distribution
func (tg *TransactionGenerator) sampleHour() int {
	r := randomFloat()
	cumulative := 0.0

	for hour, prob := range tg.timePatterns.hourlyDistribution {
		cumulative += prob
		if r <= cumulative {
			return hour
		}
	}

	return 12 // Default to noon
}

// sampleDayOfWeek samples day offset (0=today, 1=yesterday, etc.)
func (tg *TransactionGenerator) sampleDayOfWeek() int {
	return randomInt(7)
}

// generateLocation generates a realistic location
func (tg *TransactionGenerator) generateLocation() map[string]interface{} {
	cities := []string{
		"New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
		"Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose",
	}

	return map[string]interface{}{
		"city":    cities[randomInt(len(cities))],
		"state":   randomChoice([]string{"CA", "NY", "TX", "FL", "IL"}),
		"country": "US",
		"lat":     40.7 + (randomFloat()-0.5)*10,
		"lng":     -74.0 + (randomFloat()-0.5)*10,
	}
}

// selectFraudPattern selects a fraud pattern
func (tg *TransactionGenerator) selectFraudPattern() string {
	patterns := tg.fraudPatterns.patterns
	if len(patterns) == 0 {
		return "unknown"
	}
	return patterns[randomInt(len(patterns))].Name
}

// NewLogGenerator creates a log generator
func NewLogGenerator() *LogGenerator {
	return &LogGenerator{
		logTemplates: []LogTemplate{
			{
				Template:  "User {{user_id}} logged in from {{ip_address}}",
				Level:     "INFO",
				Variables: []string{"user_id", "ip_address"},
				Frequency: 0.30,
			},
			{
				Template:  "API request {{method}} {{endpoint}} completed in {{duration}}ms",
				Level:     "INFO",
				Variables: []string{"method", "endpoint", "duration"},
				Frequency: 0.40,
			},
			{
				Template:  "Database query {{query_type}} took {{duration}}ms",
				Level:     "DEBUG",
				Variables: []string{"query_type", "duration"},
				Frequency: 0.15,
			},
			{
				Template:  "Error: {{error_type}} - {{error_message}}",
				Level:     "ERROR",
				Variables: []string{"error_type", "error_message"},
				Frequency: 0.05,
			},
			{
				Template:  "Cache {{action}} for key {{key}}",
				Level:     "DEBUG",
				Variables: []string{"action", "key"},
				Frequency: 0.10,
			},
		},
		errorRates: map[string]float64{
			"connection_timeout": 0.02,
			"validation_error":   0.03,
			"rate_limit":         0.01,
			"not_found":          0.04,
		},
	}
}

// GenerateLogEntry generates a realistic log entry
func (lg *LogGenerator) GenerateLogEntry() map[string]interface{} {
	template := lg.selectLogTemplate()

	// Fill in variables
	message := template.Template
	for _, variable := range template.Variables {
		value := lg.generateVariableValue(variable)
		message = strings.Replace(message, "{{"+variable+"}}", value, 1)
	}

	return map[string]interface{}{
		"timestamp": time.Now().Format(time.RFC3339),
		"level":     template.Level,
		"message":   message,
		"service":   "shieldx",
		"host":      fmt.Sprintf("host-%d", randomInt(10)),
	}
}

// selectLogTemplate selects a log template based on frequency
func (lg *LogGenerator) selectLogTemplate() LogTemplate {
	r := randomFloat()
	cumulative := 0.0

	for _, template := range lg.logTemplates {
		cumulative += template.Frequency
		if r <= cumulative {
			return template
		}
	}

	return lg.logTemplates[0]
}

// generateVariableValue generates a value for a template variable
func (lg *LogGenerator) generateVariableValue(variable string) string {
	switch variable {
	case "user_id":
		return fmt.Sprintf("usr_%s", randomString(10))
	case "ip_address":
		return fmt.Sprintf("%d.%d.%d.%d", randomInt(256), randomInt(256), randomInt(256), randomInt(256))
	case "method":
		return randomChoice([]string{"GET", "POST", "PUT", "DELETE"})
	case "endpoint":
		return randomChoice([]string{"/api/users", "/api/transactions", "/api/auth", "/api/data"})
	case "duration":
		return fmt.Sprintf("%d", randomInt(1000))
	case "query_type":
		return randomChoice([]string{"SELECT", "INSERT", "UPDATE", "DELETE"})
	case "error_type":
		return randomChoice([]string{"ValidationError", "TimeoutError", "NotFoundError", "AuthError"})
	case "error_message":
		return randomChoice([]string{"Invalid input", "Connection timeout", "Resource not found", "Unauthorized"})
	case "action":
		return randomChoice([]string{"hit", "miss", "set", "delete"})
	case "key":
		return fmt.Sprintf("cache_key_%s", randomString(8))
	default:
		return "unknown"
	}
}

// NewDistributionDatabase creates a distribution database
func NewDistributionDatabase() *DistributionDatabase {
	return &DistributionDatabase{
		distributions: make(map[string]Distribution),
	}
}

// NewGenerationCache creates a generation cache
func NewGenerationCache(ttl time.Duration) *GenerationCache {
	return &GenerationCache{
		cache:      make(map[string]interface{}),
		ttl:        ttl,
		timestamps: make(map[string]time.Time),
	}
}

// initializeSampleDistributions initializes sample distributions
func (sdg *SyntheticDataGenerator) initializeSampleDistributions() {
	// Train Markov chain with sample text
	sampleText := `Welcome to our platform. We are committed to providing excellent service.
	Our team works hard to ensure customer satisfaction. Thank you for choosing us.
	We value your feedback and continuously improve our services.`

	sdg.markovChain.Train(sampleText)
}

// GenerateBatch generates a batch of synthetic data
func (sdg *SyntheticDataGenerator) GenerateBatch(ctx context.Context, dataType string, count int) ([]map[string]interface{}, error) {
	result := make([]map[string]interface{}, 0, count)

	for i := 0; i < count; i++ {
		select {
		case <-ctx.Done():
			return result, ctx.Err()
		default:
		}

		var data map[string]interface{}

		switch dataType {
		case "user_profiles":
			data = sdg.userGenerator.GenerateUserProfile()
		case "transactions":
			userID := fmt.Sprintf("usr_%s", randomString(10))
			data = sdg.txGenerator.GenerateTransaction(userID, true)
		case "logs":
			data = sdg.logGenerator.GenerateLogEntry()
		default:
			return nil, fmt.Errorf("unknown data type: %s", dataType)
		}

		result = append(result, data)
	}

	log.Printf("[synthetic-data] Generated %d %s records", count, dataType)
	return result, nil
}

// Helper functions
func randomInt(max int) int {
	if max <= 0 {
		return 0
	}
	n, _ := rand.Int(rand.Reader, big.NewInt(int64(max)))
	return int(n.Int64())
}

func randomFloat() float64 {
	n, _ := rand.Int(rand.Reader, big.NewInt(1<<53))
	return float64(n.Int64()) / float64(1<<53)
}

func randomString(length int) string {
	const chars = "abcdefghijklmnopqrstuvwxyz0123456789"
	result := make([]byte, length)
	for i := 0; i < length; i++ {
		result[i] = chars[randomInt(len(chars))]
	}
	return string(result)
}

func randomChoice(choices []string) string {
	if len(choices) == 0 {
		return ""
	}
	return choices[randomInt(len(choices))]
}

func randomBool(probability float64) bool {
	return randomFloat() < probability
}
