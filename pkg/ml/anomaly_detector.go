package ml

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"
)

type AnomalyDetector struct {
	model     *IsolationForest
	scaler    *StandardScaler
	isTrained bool
	threshold float64
	features  []string
	mu        sync.RWMutex

	// Online learning
	buffer      [][]float64
	bufferSize  int
	retrainFreq time.Duration
	lastRetrain time.Time
}

type TelemetryEvent struct {
	Timestamp   time.Time      `json:"timestamp"`
	Source      string         `json:"source"`     // ingress/decoy/sandbox
	EventType   string         `json:"event_type"` // connection/exploit/anomaly
	TenantID    string         `json:"tenant_id"`
	SessionID   string         `json:"session_id"`
	Metadata    map[string]any `json:"metadata"`
	Features    []float64      `json:"features"` // ML feature vector
	ThreatScore float64        `json:"threat_score"`
}

type AnomalyResult struct {
	IsAnomaly         bool               `json:"is_anomaly"`
	Score             float64            `json:"score"`
	Confidence        float64            `json:"confidence"`
	Explanation       string             `json:"explanation"`
	FeatureImportance map[string]float64 `json:"feature_importance"`
}

type IsolationForest struct {
	trees      []*IsolationTree
	numTrees   int
	sampleSize int
	maxDepth   int
}

type IsolationTree struct {
	root *TreeNode
}

type TreeNode struct {
	splitFeature int
	splitValue   float64
	left         *TreeNode
	right        *TreeNode
	depth        int
	size         int
}

type StandardScaler struct {
	mean []float64
	std  []float64
}

func NewAnomalyDetector(threshold float64, bufferSize int, retrainFreq time.Duration) *AnomalyDetector {
	return &AnomalyDetector{
		model:       NewIsolationForest(100, 256),
		scaler:      NewStandardScaler(),
		threshold:   threshold,
		bufferSize:  bufferSize,
		retrainFreq: retrainFreq,
		buffer:      make([][]float64, 0, bufferSize),
		features: []string{
			"packet_size_mean", "packet_size_std", "packet_count",
			"connection_duration", "bytes_per_second", "packets_per_second",
			"tcp_flags_entropy", "payload_entropy", "inter_arrival_time_mean",
			"inter_arrival_time_std", "port_diversity", "ip_diversity",
			"syscall_count", "dangerous_syscall_ratio", "memory_allocations",
			"file_operations", "network_operations", "process_spawns",
		},
	}
}

func (ad *AnomalyDetector) Train(ctx context.Context, events []TelemetryEvent) error {
	ad.mu.Lock()
	defer ad.mu.Unlock()

	if len(events) == 0 {
		return fmt.Errorf("no training data provided")
	}

	// Extract features
	features := make([][]float64, len(events))
	for i, event := range events {
		features[i] = event.Features
	}

	// Fit scaler
	if err := ad.scaler.Fit(features); err != nil {
		return fmt.Errorf("failed to fit scaler: %w", err)
	}

	// Scale features
	scaledFeatures, err := ad.scaler.Transform(features)
	if err != nil {
		return fmt.Errorf("failed to scale features: %w", err)
	}

	// Train isolation forest
	if err := ad.model.Fit(scaledFeatures); err != nil {
		return fmt.Errorf("failed to train model: %w", err)
	}

	ad.isTrained = true
	ad.lastRetrain = time.Now()

	return nil
}

func (ad *AnomalyDetector) Predict(event TelemetryEvent) (*AnomalyResult, error) {
	ad.mu.RLock()
	defer ad.mu.RUnlock()

	if !ad.isTrained {
		return &AnomalyResult{
			IsAnomaly:   false,
			Score:       0.0,
			Confidence:  0.0,
			Explanation: "Model not trained",
		}, nil
	}

	// Scale features
	scaledFeatures, err := ad.scaler.Transform([][]float64{event.Features})
	if err != nil {
		return nil, fmt.Errorf("failed to scale features: %w", err)
	}

	// Predict
	score := ad.model.DecisionFunction(scaledFeatures[0])
	isAnomaly := score < ad.threshold
	confidence := math.Abs(score - ad.threshold)

	// Calculate feature importance
	importance := ad.calculateFeatureImportance(event.Features)

	result := &AnomalyResult{
		IsAnomaly:         isAnomaly,
		Score:             score,
		Confidence:        confidence,
		FeatureImportance: importance,
	}

	if isAnomaly {
		result.Explanation = ad.generateExplanation(event.Features, importance)
	}

	// Add to buffer for online learning
	go ad.addToBuffer(event.Features)

	return result, nil
}

func (ad *AnomalyDetector) addToBuffer(features []float64) {
	ad.mu.Lock()
	defer ad.mu.Unlock()

	if len(ad.buffer) >= ad.bufferSize {
		// Remove oldest sample
		ad.buffer = ad.buffer[1:]
	}

	// Add new sample
	featuresCopy := make([]float64, len(features))
	copy(featuresCopy, features)
	ad.buffer = append(ad.buffer, featuresCopy)

	// Check if retrain is needed
	if time.Since(ad.lastRetrain) > ad.retrainFreq && len(ad.buffer) >= ad.bufferSize/2 {
		go ad.onlineRetrain()
	}
}

func (ad *AnomalyDetector) onlineRetrain() {
	ad.mu.Lock()
	buffer := make([][]float64, len(ad.buffer))
	copy(buffer, ad.buffer)
	ad.mu.Unlock()

	// Retrain with buffered data
	if err := ad.scaler.Fit(buffer); err != nil {
		return
	}

	scaledBuffer, err := ad.scaler.Transform(buffer)
	if err != nil {
		return
	}

	if err := ad.model.Fit(scaledBuffer); err != nil {
		return
	}

	ad.mu.Lock()
	ad.lastRetrain = time.Now()
	ad.mu.Unlock()
}

func (ad *AnomalyDetector) calculateFeatureImportance(features []float64) map[string]float64 {
	importance := make(map[string]float64)

	// Simple feature importance based on deviation from mean
	for i, feature := range features {
		if i < len(ad.features) && i < len(ad.scaler.mean) {
			deviation := math.Abs(feature - ad.scaler.mean[i])
			if ad.scaler.std[i] > 0 {
				importance[ad.features[i]] = deviation / ad.scaler.std[i]
			}
		}
	}

	return importance
}

func (ad *AnomalyDetector) generateExplanation(features []float64, importance map[string]float64) string {
	// Find top 3 most important features
	type featureScore struct {
		name  string
		score float64
	}

	var scores []featureScore
	for name, score := range importance {
		scores = append(scores, featureScore{name, score})
	}

	// Sort by score descending
	for i := 0; i < len(scores)-1; i++ {
		for j := i + 1; j < len(scores); j++ {
			if scores[j].score > scores[i].score {
				scores[i], scores[j] = scores[j], scores[i]
			}
		}
	}

	explanation := "Anomaly detected due to unusual: "
	for i := 0; i < min(3, len(scores)); i++ {
		if i > 0 {
			explanation += ", "
		}
		explanation += scores[i].name
	}

	return explanation
}

// Isolation Forest Implementation
func NewIsolationForest(numTrees, sampleSize int) *IsolationForest {
	return &IsolationForest{
		numTrees:   numTrees,
		sampleSize: sampleSize,
		maxDepth:   int(math.Ceil(math.Log2(float64(sampleSize)))),
	}
}

func (iforest *IsolationForest) Fit(data [][]float64) error {
	iforest.trees = make([]*IsolationTree, iforest.numTrees)

	for i := 0; i < iforest.numTrees; i++ {
		// Sample data
		sample := sampleData(data, iforest.sampleSize)

		// Build tree
		tree := &IsolationTree{}
		tree.root = buildTree(sample, 0, iforest.maxDepth)
		iforest.trees[i] = tree
	}

	return nil
}

func (iforest *IsolationForest) DecisionFunction(sample []float64) float64 {
	if len(iforest.trees) == 0 {
		return 0.0
	}

	pathLengths := make([]float64, len(iforest.trees))
	for i, tree := range iforest.trees {
		pathLengths[i] = pathLength(tree.root, sample, 0)
	}

	// Average path length
	avgPathLength := 0.0
	for _, length := range pathLengths {
		avgPathLength += length
	}
	avgPathLength /= float64(len(pathLengths))

	// Anomaly score (lower = more anomalous)
	c := averagePathLength(iforest.sampleSize)
	return math.Pow(2, -avgPathLength/c)
}

func buildTree(data [][]float64, depth, maxDepth int) *TreeNode {
	if len(data) <= 1 || depth >= maxDepth {
		return &TreeNode{
			depth: depth,
			size:  len(data),
		}
	}

	// Random feature and split value
	featureIdx := randInt(len(data[0]))
	minVal, maxVal := getFeatureRange(data, featureIdx)

	if minVal == maxVal {
		return &TreeNode{
			depth: depth,
			size:  len(data),
		}
	}

	splitValue := minVal + rand()*(maxVal-minVal)

	// Split data
	var leftData, rightData [][]float64
	for _, sample := range data {
		if sample[featureIdx] < splitValue {
			leftData = append(leftData, sample)
		} else {
			rightData = append(rightData, sample)
		}
	}

	node := &TreeNode{
		splitFeature: featureIdx,
		splitValue:   splitValue,
		depth:        depth,
		size:         len(data),
	}

	node.left = buildTree(leftData, depth+1, maxDepth)
	node.right = buildTree(rightData, depth+1, maxDepth)

	return node
}

func pathLength(node *TreeNode, sample []float64, currentDepth int) float64 {
	if node.left == nil && node.right == nil {
		// Leaf node
		return float64(currentDepth) + averagePathLength(node.size)
	}

	if sample[node.splitFeature] < node.splitValue {
		return pathLength(node.left, sample, currentDepth+1)
	}
	return pathLength(node.right, sample, currentDepth+1)
}

// Standard Scaler Implementation
func NewStandardScaler() *StandardScaler {
	return &StandardScaler{}
}

func (scaler *StandardScaler) Fit(data [][]float64) error {
	if len(data) == 0 {
		return fmt.Errorf("no data provided")
	}

	numFeatures := len(data[0])
	scaler.mean = make([]float64, numFeatures)
	scaler.std = make([]float64, numFeatures)

	// Calculate mean
	for _, sample := range data {
		for i, value := range sample {
			scaler.mean[i] += value
		}
	}

	for i := range scaler.mean {
		scaler.mean[i] /= float64(len(data))
	}

	// Calculate standard deviation
	for _, sample := range data {
		for i, value := range sample {
			diff := value - scaler.mean[i]
			scaler.std[i] += diff * diff
		}
	}

	for i := range scaler.std {
		scaler.std[i] = math.Sqrt(scaler.std[i] / float64(len(data)))
		if scaler.std[i] == 0 {
			scaler.std[i] = 1.0 // Avoid division by zero
		}
	}

	return nil
}

func (scaler *StandardScaler) Transform(data [][]float64) ([][]float64, error) {
	if len(scaler.mean) == 0 {
		return nil, fmt.Errorf("scaler not fitted")
	}

	result := make([][]float64, len(data))
	for i, sample := range data {
		scaled := make([]float64, len(sample))
		for j, value := range sample {
			if j < len(scaler.mean) {
				scaled[j] = (value - scaler.mean[j]) / scaler.std[j]
			} else {
				scaled[j] = value
			}
		}
		result[i] = scaled
	}

	return result, nil
}

// Helper functions
func sampleData(data [][]float64, sampleSize int) [][]float64 {
	if len(data) <= sampleSize {
		return data
	}

	sample := make([][]float64, sampleSize)
	for i := 0; i < sampleSize; i++ {
		idx := randInt(len(data))
		sample[i] = data[idx]
	}

	return sample
}

func getFeatureRange(data [][]float64, featureIdx int) (float64, float64) {
	if len(data) == 0 {
		return 0, 0
	}

	min := data[0][featureIdx]
	max := data[0][featureIdx]

	for _, sample := range data {
		if sample[featureIdx] < min {
			min = sample[featureIdx]
		}
		if sample[featureIdx] > max {
			max = sample[featureIdx]
		}
	}

	return min, max
}

func averagePathLength(n int) float64 {
	if n <= 1 {
		return 0
	}
	return 2.0*(math.Log(float64(n-1))+0.5772156649) - 2.0*float64(n-1)/float64(n)
}

func randInt(max int) int {
	return int(rand() * float64(max))
}

func rand() float64 {
	return float64(time.Now().UnixNano()%1000000) / 1000000.0
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
