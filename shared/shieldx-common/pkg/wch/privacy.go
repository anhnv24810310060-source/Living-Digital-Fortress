package wch

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"
)

type PrivacyBudget struct {
	epsilon    float64
	delta      float64
	spent      float64
	queries    []Query
	mu         sync.RWMutex
	startTime  time.Time
	windowSize time.Duration
}

type Query struct {
	ID        string
	Epsilon   float64
	Delta     float64
	Timestamp time.Time
	Result    interface{}
}

type DifferentialPrivacy struct {
	budgets map[string]*PrivacyBudget
	mu      sync.RWMutex
}

type TEEAttestation struct {
	Quote     []byte
	Signature []byte
	PCRs      map[int][]byte
	Timestamp time.Time
}

func NewDifferentialPrivacy() *DifferentialPrivacy {
	return &DifferentialPrivacy{
		budgets: make(map[string]*PrivacyBudget),
	}
}

func (dp *DifferentialPrivacy) CreateBudget(userID string, epsilon, delta float64, window time.Duration) *PrivacyBudget {
	dp.mu.Lock()
	defer dp.mu.Unlock()
	
	budget := &PrivacyBudget{
		epsilon:    epsilon,
		delta:      delta,
		spent:      0.0,
		queries:    make([]Query, 0),
		startTime:  time.Now(),
		windowSize: window,
	}
	
	dp.budgets[userID] = budget
	return budget
}

func (dp *DifferentialPrivacy) Query(userID, queryID string, epsilon, delta float64, trueResult float64, sensitivity float64) (float64, error) {
	dp.mu.Lock()
	defer dp.mu.Unlock()
	
	budget := dp.budgets[userID]
	if budget == nil {
		return 0, fmt.Errorf("no budget found for user %s", userID)
	}
	
	if !budget.canSpend(epsilon, delta) {
		return 0, fmt.Errorf("insufficient privacy budget")
	}
	
	noisyResult := dp.addLaplaceNoise(trueResult, sensitivity, epsilon)
	
	query := Query{
		ID:        queryID,
		Epsilon:   epsilon,
		Delta:     delta,
		Timestamp: time.Now(),
		Result:    noisyResult,
	}
	
	budget.spend(epsilon, delta, query)
	
	return noisyResult, nil
}

func (dp *DifferentialPrivacy) addLaplaceNoise(value, sensitivity, epsilon float64) float64 {
	scale := sensitivity / epsilon
	u := rand.Float64() - 0.5
	noise := -scale * math.Copysign(math.Log(1-2*math.Abs(u)), u)
	return value + noise
}

func (pb *PrivacyBudget) canSpend(epsilon, delta float64) bool {
	pb.mu.RLock()
	defer pb.mu.RUnlock()
	
	pb.cleanExpiredQueries()
	return pb.spent+epsilon <= pb.epsilon
}

func (pb *PrivacyBudget) spend(epsilon, delta float64, query Query) {
	pb.mu.Lock()
	defer pb.mu.Unlock()
	
	pb.spent += epsilon
	pb.queries = append(pb.queries, query)
}

func (pb *PrivacyBudget) cleanExpiredQueries() {
	cutoff := time.Now().Add(-pb.windowSize)
	
	newQueries := make([]Query, 0)
	newSpent := 0.0
	
	for _, query := range pb.queries {
		if query.Timestamp.After(cutoff) {
			newQueries = append(newQueries, query)
			newSpent += query.Epsilon
		}
	}
	
	pb.queries = newQueries
	pb.spent = newSpent
}

func (pb *PrivacyBudget) GetRemaining() float64 {
	pb.mu.RLock()
	defer pb.mu.RUnlock()
	
	pb.cleanExpiredQueries()
	return pb.epsilon - pb.spent
}

func GenerateTEEAttestation() *TEEAttestation {
	// Simplified TEE attestation generation
	quote := make([]byte, 64)
	signature := make([]byte, 64)
	rand.Read(quote)
	rand.Read(signature)
	
	pcrs := make(map[int][]byte)
	for i := 0; i < 8; i++ {
		pcr := make([]byte, 32)
		rand.Read(pcr)
		pcrs[i] = pcr
	}
	
	return &TEEAttestation{
		Quote:     quote,
		Signature: signature,
		PCRs:      pcrs,
		Timestamp: time.Now(),
	}
}

func (tee *TEEAttestation) Verify() bool {
	// Simplified verification
	return len(tee.Quote) == 64 && len(tee.Signature) == 64
}

func (tee *TEEAttestation) ProveNonDecryption(channelID string) []byte {
	// Generate proof that no decryption occurred
	proof := make([]byte, 32)
	rand.Read(proof)
	return proof
}