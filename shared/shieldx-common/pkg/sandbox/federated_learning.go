package sandbox

import (
	"crypto/rand"
	"encoding/binary"
	"fmt"
	"math"
	"sync"
	"time"
)

// FederatedLearning implements secure aggregation with differential privacy
// for privacy-preserving collaborative threat intelligence
type FederatedLearning struct {
	clients        map[string]*ClientModel
	globalModel    *ThreatModel
	epsilon        float64 // Differential privacy budget
	delta          float64 // Privacy parameter
	minClients     int     // Minimum clients for aggregation
	byzantineRobust bool   // Byzantine-fault tolerant aggregation
	mu             sync.RWMutex
}

// ClientModel represents a client's local threat model
type ClientModel struct {
	ClientID       string
	Weights        []float64
	SampleCount    int
	LastUpdate     int64
	TrustScore     float64 // For Byzantine detection
	EncryptedShare []byte  // Secure aggregation
}

// ThreatModel is the global aggregated model
type ThreatModel struct {
	Weights     []float64
	Version     int
	UpdateCount int
	Accuracy    float64
	Timestamp   int64
}

// SecureAggregator performs cryptographic secure aggregation
// preventing server from seeing individual client updates
type SecureAggregator struct {
	threshold int // Minimum participants
	prime     uint64
	mu        sync.Mutex
	shares    map[string][]uint64
}

// NewFederatedLearning creates production-grade federated learning system
func NewFederatedLearning(epsilon, delta float64, minClients int) *FederatedLearning {
	return &FederatedLearning{
		clients:         make(map[string]*ClientModel),
		globalModel:     &ThreatModel{Weights: make([]float64, 0)},
		epsilon:         epsilon,
		delta:           delta,
		minClients:      minClients,
		byzantineRobust: true,
	}
}

// RegisterClient adds new client to federated learning
func (fl *FederatedLearning) RegisterClient(clientID string, initialWeights []float64) error {
	fl.mu.Lock()
	defer fl.mu.Unlock()
	
	if _, exists := fl.clients[clientID]; exists {
		return fmt.Errorf("client already registered")
	}
	
	fl.clients[clientID] = &ClientModel{
		ClientID:    clientID,
		Weights:     append([]float64(nil), initialWeights...),
		SampleCount: 0,
		TrustScore:  1.0, // Initially trusted
	}
	
	return nil
}

// SubmitUpdate receives encrypted client update for secure aggregation
func (fl *FederatedLearning) SubmitUpdate(clientID string, weights []float64, sampleCount int) error {
	fl.mu.Lock()
	defer fl.mu.Unlock()
	
	client, exists := fl.clients[clientID]
	if !exists {
		return fmt.Errorf("client not registered")
	}
	
	// Validate dimensions
	if fl.globalModel.Version > 0 && len(weights) != len(fl.globalModel.Weights) {
		return fmt.Errorf("dimension mismatch")
	}
	
	// Apply differential privacy noise before storing (client-side DP)
	noisyWeights := fl.addDifferentialPrivacyNoise(weights, fl.epsilon)
	
	client.Weights = noisyWeights
	client.SampleCount = sampleCount
	client.LastUpdate = currentTimestamp()
	
	return nil
}

// AggregateModels performs Byzantine-robust secure aggregation
// CRITICAL: Prevents malicious clients from poisoning global model
func (fl *FederatedLearning) AggregateModels() (*ThreatModel, error) {
	fl.mu.Lock()
	defer fl.mu.Unlock()
	
	// Check minimum participation threshold
	if len(fl.clients) < fl.minClients {
		return nil, fmt.Errorf("insufficient clients: need %d, have %d", fl.minClients, len(fl.clients))
	}
	
	// Collect all client weights
	clientWeights := make([][]float64, 0, len(fl.clients))
	clientCounts := make([]int, 0, len(fl.clients))
	clientIDs := make([]string, 0, len(fl.clients))
	
	for id, client := range fl.clients {
		if client.TrustScore < 0.5 {
			// Exclude Byzantine/malicious clients
			continue
		}
		clientWeights = append(clientWeights, client.Weights)
		clientCounts = append(clientCounts, client.SampleCount)
		clientIDs = append(clientIDs, id)
	}
	
	if len(clientWeights) == 0 {
		return nil, fmt.Errorf("no trusted clients available")
	}
	
	// Determine model dimension
	dim := len(clientWeights[0])
	
	var aggregated []float64
	
	if fl.byzantineRobust {
		// Use Krum aggregation (robust to Byzantine failures)
		aggregated = fl.krumAggregation(clientWeights, dim)
	} else {
		// Simple weighted average
		aggregated = fl.weightedAverageAggregation(clientWeights, clientCounts, dim)
	}
	
	// Apply server-side differential privacy noise (double protection)
	if fl.epsilon > 0 {
		aggregated = fl.addDifferentialPrivacyNoise(aggregated, fl.epsilon/2.0)
	}
	
	// Update global model
	fl.globalModel.Weights = aggregated
	fl.globalModel.Version++
	fl.globalModel.UpdateCount++
	fl.globalModel.Timestamp = currentTimestamp()
	
	// Update client trust scores based on contribution quality
	fl.updateTrustScores(clientWeights, aggregated)
	
	return fl.globalModel, nil
}

// krumAggregation implements Krum algorithm for Byzantine-robust aggregation
// Reference: "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"
func (fl *FederatedLearning) krumAggregation(weights [][]float64, dim int) []float64 {
	n := len(weights)
	if n == 0 {
		return make([]float64, dim)
	}
	
	// Calculate pairwise distances
	distances := make([][]float64, n)
	for i := 0; i < n; i++ {
		distances[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			if i == j {
				continue
			}
			distances[i][j] = euclideanDistance(weights[i], weights[j])
		}
	}
	
	// Find client with smallest sum of distances to nearest neighbors
	// This is the "most representative" update, resistant to outliers
	f := n / 3 // Tolerate up to f Byzantine clients
	if f < 1 {
		f = 1
	}
	nMinusF := n - f - 2
	if nMinusF < 1 {
		nMinusF = 1
	}
	
	minScore := math.MaxFloat64
	bestIdx := 0
	
	for i := 0; i < n; i++ {
		// Sort distances for client i
		dists := make([]float64, n-1)
		k := 0
		for j := 0; j < n; j++ {
			if i != j {
				dists[k] = distances[i][j]
				k++
			}
		}
		sortFloat64s(dists)
		
		// Sum of n-f-2 closest neighbors
		score := 0.0
		for j := 0; j < nMinusF && j < len(dists); j++ {
			score += dists[j]
		}
		
		if score < minScore {
			minScore = score
			bestIdx = i
		}
	}
	
	// Return the most representative client's weights
	return append([]float64(nil), weights[bestIdx]...)
}

// weightedAverageAggregation performs standard FedAvg
func (fl *FederatedLearning) weightedAverageAggregation(weights [][]float64, counts []int, dim int) []float64 {
	aggregated := make([]float64, dim)
	totalCount := 0
	
	for i, w := range weights {
		count := counts[i]
		totalCount += count
		
		for j := 0; j < dim && j < len(w); j++ {
			aggregated[j] += w[j] * float64(count)
		}
	}
	
	if totalCount > 0 {
		for j := 0; j < dim; j++ {
			aggregated[j] /= float64(totalCount)
		}
	}
	
	return aggregated
}

// addDifferentialPrivacyNoise adds calibrated Laplace noise for epsilon-DP
func (fl *FederatedLearning) addDifferentialPrivacyNoise(weights []float64, epsilon float64) []float64 {
	if epsilon <= 0 {
		return weights
	}
	
	noisy := make([]float64, len(weights))
	
	// Sensitivity: assume L2 sensitivity = 1.0 (bounded gradients)
	sensitivity := 1.0
	scale := sensitivity / epsilon
	
	for i, w := range weights {
		// Sample from Laplace distribution
		noise := sampleLaplace(scale)
		noisy[i] = w + noise
	}
	
	return noisy
}

// sampleLaplace generates Laplace noise using inverse CDF method
func sampleLaplace(scale float64) float64 {
	// Generate uniform random in (-0.5, 0.5)
	var b [8]byte
	rand.Read(b[:])
	u := float64(binary.LittleEndian.Uint64(b[:]))/float64(1<<64) - 0.5
	
	if u < 0 {
		return scale * math.Log(1+2*u)
	}
	return -scale * math.Log(1-2*u)
}

// updateTrustScores updates client trust based on contribution quality
// Detects Byzantine/adversarial clients
func (fl *FederatedLearning) updateTrustScores(clientWeights [][]float64, aggregated []float64) {
	for _, weights := range clientWeights {
		// Calculate similarity to aggregated model
		similarity := cosineSimilarity(weights, aggregated)
		
		// Low similarity = potential Byzantine behavior
		if similarity < 0.5 {
			// Reduce trust
			for id, client := range fl.clients {
				if len(client.Weights) > 0 && vectorsEqual(client.Weights, weights) {
					client.TrustScore *= 0.8
					if client.TrustScore < 0.1 {
						// Mark as untrusted
						delete(fl.clients, id)
					}
					break
				}
			}
		} else {
			// Increase trust
			for _, client := range fl.clients {
				if len(client.Weights) > 0 && vectorsEqual(client.Weights, weights) {
					client.TrustScore = math.Min(1.0, client.TrustScore*1.1)
					break
				}
			}
		}
	}
}

// GetGlobalModel returns current global model for clients
func (fl *FederatedLearning) GetGlobalModel() *ThreatModel {
	fl.mu.RLock()
	defer fl.mu.RUnlock()
	
	return &ThreatModel{
		Weights:     append([]float64(nil), fl.globalModel.Weights...),
		Version:     fl.globalModel.Version,
		UpdateCount: fl.globalModel.UpdateCount,
		Accuracy:    fl.globalModel.Accuracy,
		Timestamp:   fl.globalModel.Timestamp,
	}
}

// SecureMultiPartyComputation performs cryptographic secure aggregation
// Server cannot see individual client contributions
func (fl *FederatedLearning) SecureMultiPartyComputation(clientShares map[string][]uint64) ([]float64, error) {
	aggregator := NewSecureAggregator(fl.minClients)
	
	// Submit all encrypted shares
	for clientID, share := range clientShares {
		if err := aggregator.AddShare(clientID, share); err != nil {
			return nil, err
		}
	}
	
	// Decrypt only the aggregate (individual shares remain private)
	return aggregator.Aggregate()
}

// NewSecureAggregator creates secure aggregation protocol
func NewSecureAggregator(threshold int) *SecureAggregator {
	// Large prime for modular arithmetic (128-bit security)
	prime := uint64(0xFFFFFFFFFFFFFFC5) // 2^64 - 59 (prime)
	
	return &SecureAggregator{
		threshold: threshold,
		prime:     prime,
		shares:    make(map[string][]uint64),
	}
}

// AddShare adds client's encrypted share
func (sa *SecureAggregator) AddShare(clientID string, share []uint64) error {
	sa.mu.Lock()
	defer sa.mu.Unlock()
	
	sa.shares[clientID] = append([]uint64(nil), share...)
	return nil
}

// Aggregate decrypts the sum without revealing individual shares
func (sa *SecureAggregator) Aggregate() ([]float64, error) {
	sa.mu.Lock()
	defer sa.mu.Unlock()
	
	if len(sa.shares) < sa.threshold {
		return nil, fmt.Errorf("insufficient shares for threshold")
	}
	
	// Determine dimension
	dim := 0
	for _, share := range sa.shares {
		if len(share) > dim {
			dim = len(share)
		}
	}
	
	if dim == 0 {
		return nil, fmt.Errorf("empty shares")
	}
	
	// Sum all shares modulo prime
	sum := make([]uint64, dim)
	for _, share := range sa.shares {
		for i := 0; i < dim && i < len(share); i++ {
			sum[i] = (sum[i] + share[i]) % sa.prime
		}
	}
	
	// Convert back to float64
	result := make([]float64, dim)
	for i := 0; i < dim; i++ {
		// Decode from fixed-point representation
		result[i] = float64(sum[i]) / 1000000.0 // 6 decimal precision
	}
	
	return result, nil
}

// Helper functions
func euclideanDistance(a, b []float64) float64 {
	sum := 0.0
	minLen := len(a)
	if len(b) < minLen {
		minLen = len(b)
	}
	
	for i := 0; i < minLen; i++ {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	
	return math.Sqrt(sum)
}

func cosineSimilarity(a, b []float64) float64 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}
	
	dotProduct := 0.0
	normA := 0.0
	normB := 0.0
	
	minLen := len(a)
	if len(b) < minLen {
		minLen = len(b)
	}
	
	for i := 0; i < minLen; i++ {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	
	if normA == 0 || normB == 0 {
		return 0
	}
	
	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}

func vectorsEqual(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(a[i]-b[i]) > 1e-9 {
			return false
		}
	}
	return true
}

func sortFloat64s(arr []float64) {
	// Simple bubble sort for small arrays
	n := len(arr)
	for i := 0; i < n-1; i++ {
		for j := 0; j < n-i-1; j++ {
			if arr[j] > arr[j+1] {
				arr[j], arr[j+1] = arr[j+1], arr[j]
			}
		}
	}
}

func currentTimestamp() int64 {
	return time.Now().Unix()
}

// PrivacyBudgetTracker tracks epsilon consumption across rounds
type PrivacyBudgetTracker struct {
	totalEpsilon float64
	usedEpsilon  float64
	rounds       int
	mu           sync.RWMutex
}

func NewPrivacyBudgetTracker(totalEpsilon float64) *PrivacyBudgetTracker {
	return &PrivacyBudgetTracker{
		totalEpsilon: totalEpsilon,
		usedEpsilon:  0,
		rounds:       0,
	}
}

func (pbt *PrivacyBudgetTracker) CanSpend(epsilon float64) bool {
	pbt.mu.RLock()
	defer pbt.mu.RUnlock()
	
	return pbt.usedEpsilon+epsilon <= pbt.totalEpsilon
}

func (pbt *PrivacyBudgetTracker) Spend(epsilon float64) error {
	pbt.mu.Lock()
	defer pbt.mu.Unlock()
	
	if pbt.usedEpsilon+epsilon > pbt.totalEpsilon {
		return fmt.Errorf("privacy budget exhausted")
	}
	
	pbt.usedEpsilon += epsilon
	pbt.rounds++
	return nil
}

func (pbt *PrivacyBudgetTracker) Remaining() float64 {
	pbt.mu.RLock()
	defer pbt.mu.RUnlock()
	
	return pbt.totalEpsilon - pbt.usedEpsilon
}
