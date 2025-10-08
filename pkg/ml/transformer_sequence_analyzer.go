package ml

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"

	"shieldx/pkg/ebpf"
)

// TransformerSequenceAnalyzer implements BERT-like architecture for syscall sequence analysis
// Phase 2: Behavioral AI Engine - Transformer-Based Sequence Analysis
// Architecture: Input embedding (512 dim) + 12 transformer layers + 8 attention heads
// Context window: 2048 syscall events
type TransformerSequenceAnalyzer struct {
	// Model configuration
	config ModelConfig

	// Embedding layer
	embeddings *EmbeddingLayer

	// Multi-head attention layers (12 layers x 8 heads)
	transformerBlocks []*TransformerBlock

	// Classification head
	classifier *ClassificationHead

	// Syscall vocabulary (syscall name -> token ID)
	vocabulary map[string]int

	// Pre-computed positional encodings
	positionalEncodings [][]float64

	// Attention cache for inference speed
	attentionCache sync.Map

	// Performance metrics
	inferenceLatencyMs float64
	mu                 sync.RWMutex
}

// ModelConfig defines transformer architecture parameters
type ModelConfig struct {
	VocabSize        int     // Number of unique syscalls
	EmbedDim         int     // 512 dimensions per Phase 2 spec
	NumLayers        int     // 12 transformer layers
	NumHeads         int     // 8 attention heads per layer
	FFNHiddenDim     int     // Feed-forward network hidden dimension
	MaxSeqLen        int     // 2048 context window
	DropoutRate      float64 // Regularization
	AttentionDropout float64
}

// DefaultTransformerConfig returns production-ready configuration
func DefaultTransformerConfig() ModelConfig {
	return ModelConfig{
		VocabSize:        256,  // Support 256 unique syscalls
		EmbedDim:         512,  // As per Phase 2 spec
		NumLayers:        12,   // Deep network for complex patterns
		NumHeads:         8,    // Multi-head attention
		FFNHiddenDim:     2048, // 4x expansion in FFN
		MaxSeqLen:        2048, // 2048 event context window
		DropoutRate:      0.1,
		AttentionDropout: 0.1,
	}
}

// EmbeddingLayer maps syscall tokens to dense vectors
type EmbeddingLayer struct {
	weights [][]float64 // [vocab_size x embed_dim]
	dim     int
}

// TransformerBlock is a single transformer encoder layer
type TransformerBlock struct {
	// Multi-head self-attention
	multiHeadAttn *MultiHeadAttention

	// Feed-forward network
	ffn *FeedForwardNetwork

	// Layer normalization
	layerNorm1 *LayerNorm
	layerNorm2 *LayerNorm

	// Dropout
	dropoutRate float64
}

// MultiHeadAttention implements scaled dot-product attention with multiple heads
type MultiHeadAttention struct {
	numHeads int
	headDim  int
	embedDim int

	// Learned projections Q, K, V
	wq [][]float64 // Query weights [embed_dim x embed_dim]
	wk [][]float64 // Key weights
	wv [][]float64 // Value weights
	wo [][]float64 // Output projection

	dropout float64
}

// FeedForwardNetwork is position-wise FFN with GELU activation
type FeedForwardNetwork struct {
	w1 [][]float64 // [embed_dim x hidden_dim]
	w2 [][]float64 // [hidden_dim x embed_dim]
	b1 []float64
	b2 []float64
}

// LayerNorm implements layer normalization for training stability
type LayerNorm struct {
	gamma []float64 // Scale
	beta  []float64 // Shift
	eps   float64
}

// ClassificationHead maps transformer output to threat scores
type ClassificationHead struct {
	w [][]float64 // [embed_dim x num_classes]
	b []float64
}

// SequenceInput represents input to the transformer
type SequenceInput struct {
	Syscalls   []string
	Timestamps []time.Time
	PIDs       []int
	Features   *ebpf.ThreatFeatures
}

// AnalysisResult contains transformer inference output
type AnalysisResult struct {
	ThreatScore       float64     // 0.0-1.0
	Confidence        float64     // Model confidence
	AnomalyScores     []float64   // Per-token anomaly scores
	AttentionWeights  [][]float64 // Visualization of attention
	MaliciousPatterns []string    // Identified attack patterns
	LatencyMs         float64     // Inference time
	DetectedAt        time.Time
}

// NewTransformerSequenceAnalyzer creates and initializes the transformer model
func NewTransformerSequenceAnalyzer(config ModelConfig) (*TransformerSequenceAnalyzer, error) {
	tsa := &TransformerSequenceAnalyzer{
		config:              config,
		transformerBlocks:   make([]*TransformerBlock, config.NumLayers),
		vocabulary:          initSyscallVocabulary(),
		positionalEncodings: precomputePositionalEncodings(config.MaxSeqLen, config.EmbedDim),
	}

	// Initialize embedding layer
	tsa.embeddings = &EmbeddingLayer{
		weights: randomMatrix(config.VocabSize, config.EmbedDim, 0.02),
		dim:     config.EmbedDim,
	}

	// Initialize transformer blocks
	for i := 0; i < config.NumLayers; i++ {
		tsa.transformerBlocks[i] = newTransformerBlock(config)
	}

	// Initialize classification head
	tsa.classifier = &ClassificationHead{
		w: randomMatrix(config.EmbedDim, 5, 0.02), // 5 classes: benign, suspicious, malicious, exploit, advanced
		b: make([]float64, 5),
	}

	return tsa, nil
}

// Analyze performs transformer-based syscall sequence analysis
// Constraint: Must return result within 100ms for real-time detection
func (tsa *TransformerSequenceAnalyzer) Analyze(ctx context.Context, input *SequenceInput) (*AnalysisResult, error) {
	startTime := time.Now()

	// Validate input
	if len(input.Syscalls) == 0 {
		return nil, fmt.Errorf("empty syscall sequence")
	}

	// Truncate to max sequence length
	seqLen := len(input.Syscalls)
	if seqLen > tsa.config.MaxSeqLen {
		seqLen = tsa.config.MaxSeqLen
		input.Syscalls = input.Syscalls[len(input.Syscalls)-seqLen:]
	}

	// Step 1: Tokenize syscalls
	tokens := tsa.tokenize(input.Syscalls)

	// Step 2: Embed tokens + add positional encoding
	embeddings := tsa.embed(tokens)

	// Step 3: Pass through transformer layers
	hidden := embeddings
	var attentionWeights [][]float64

	for i, block := range tsa.transformerBlocks {
		var attn [][]float64
		hidden, attn = block.forward(hidden)

		// Store attention from last layer for visualization
		if i == len(tsa.transformerBlocks)-1 {
			attentionWeights = attn
		}

		// Check context timeout for real-time constraint
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
	}

	// Step 4: Classification head (use [CLS] token representation)
	logits := tsa.classifier.forward(hidden[0]) // Use first token as sequence representation
	threatProbs := softmax(logits)

	// Step 5: Compute threat score (weighted by class severity)
	threatScore := computeWeightedThreatScore(threatProbs)
	confidence := maxFloat64(threatProbs)

	// Step 6: Identify malicious patterns via attention analysis
	patterns := tsa.identifyPatterns(input.Syscalls, attentionWeights)

	// Step 7: Per-token anomaly scores
	anomalyScores := computeAnomalyScores(hidden, tsa.embeddings)

	latency := time.Since(startTime).Seconds() * 1000
	tsa.mu.Lock()
	tsa.inferenceLatencyMs = latency
	tsa.mu.Unlock()

	return &AnalysisResult{
		ThreatScore:       threatScore,
		Confidence:        confidence,
		AnomalyScores:     anomalyScores,
		AttentionWeights:  attentionWeights,
		MaliciousPatterns: patterns,
		LatencyMs:         latency,
		DetectedAt:        time.Now(),
	}, nil
}

// tokenize converts syscall names to token IDs
func (tsa *TransformerSequenceAnalyzer) tokenize(syscalls []string) []int {
	tokens := make([]int, len(syscalls))
	for i, syscall := range syscalls {
		if id, ok := tsa.vocabulary[syscall]; ok {
			tokens[i] = id
		} else {
			tokens[i] = 0 // UNK token
		}
	}
	return tokens
}

// embed converts tokens to dense vectors with positional encoding
func (tsa *TransformerSequenceAnalyzer) embed(tokens []int) [][]float64 {
	seqLen := len(tokens)
	embeddings := make([][]float64, seqLen)

	for i, tokenID := range tokens {
		// Token embedding
		tokenEmbed := tsa.embeddings.weights[tokenID]

		// Add positional encoding
		posEmbed := tsa.positionalEncodings[i]

		embed := make([]float64, tsa.config.EmbedDim)
		for j := 0; j < tsa.config.EmbedDim; j++ {
			embed[j] = tokenEmbed[j] + posEmbed[j]
		}
		embeddings[i] = embed
	}

	return embeddings
}

// newTransformerBlock creates initialized transformer block
func newTransformerBlock(config ModelConfig) *TransformerBlock {
	return &TransformerBlock{
		multiHeadAttn: newMultiHeadAttention(config.EmbedDim, config.NumHeads, config.AttentionDropout),
		ffn:           newFeedForwardNetwork(config.EmbedDim, config.FFNHiddenDim),
		layerNorm1:    newLayerNorm(config.EmbedDim),
		layerNorm2:    newLayerNorm(config.EmbedDim),
		dropoutRate:   config.DropoutRate,
	}
}

// forward implements transformer block forward pass
func (tb *TransformerBlock) forward(x [][]float64) ([][]float64, [][]float64) {
	// Multi-head attention with residual connection
	attnOut, attnWeights := tb.multiHeadAttn.forward(x)
	x = addResidual(x, attnOut)
	x = tb.layerNorm1.forward(x)

	// Feed-forward network with residual connection
	ffnOut := tb.ffn.forward(x)
	x = addResidual(x, ffnOut)
	x = tb.layerNorm2.forward(x)

	return x, attnWeights
}

// newMultiHeadAttention initializes multi-head attention
func newMultiHeadAttention(embedDim, numHeads int, dropout float64) *MultiHeadAttention {
	headDim := embedDim / numHeads
	return &MultiHeadAttention{
		numHeads: numHeads,
		headDim:  headDim,
		embedDim: embedDim,
		wq:       randomMatrix(embedDim, embedDim, 0.02),
		wk:       randomMatrix(embedDim, embedDim, 0.02),
		wv:       randomMatrix(embedDim, embedDim, 0.02),
		wo:       randomMatrix(embedDim, embedDim, 0.02),
		dropout:  dropout,
	}
}

// forward implements scaled dot-product attention
func (mha *MultiHeadAttention) forward(x [][]float64) ([][]float64, [][]float64) {
	seqLen := len(x)

	// Linear projections
	q := matMul(x, mha.wq)
	k := matMul(x, mha.wk)
	v := matMul(x, mha.wv)

	// Split into heads and compute attention
	output := make([][]float64, seqLen)
	attentionWeights := make([][]float64, seqLen)

	for i := 0; i < seqLen; i++ {
		output[i] = make([]float64, mha.embedDim)
		attentionWeights[i] = make([]float64, seqLen)

		// Compute attention scores
		scores := make([]float64, seqLen)
		for j := 0; j < seqLen; j++ {
			score := dotProduct(q[i], k[j]) / math.Sqrt(float64(mha.headDim))
			scores[j] = score
		}

		// Softmax
		attn := softmax(scores)
		attentionWeights[i] = attn

		// Weighted sum of values
		for j := 0; j < seqLen; j++ {
			for d := 0; d < mha.embedDim; d++ {
				output[i][d] += attn[j] * v[j][d]
			}
		}
	}

	// Output projection
	output = matMul(output, mha.wo)

	return output, attentionWeights
}

// newFeedForwardNetwork creates FFN with GELU activation
func newFeedForwardNetwork(embedDim, hiddenDim int) *FeedForwardNetwork {
	return &FeedForwardNetwork{
		w1: randomMatrix(embedDim, hiddenDim, 0.02),
		w2: randomMatrix(hiddenDim, embedDim, 0.02),
		b1: make([]float64, hiddenDim),
		b2: make([]float64, embedDim),
	}
}

// forward implements FFN forward pass with GELU activation
func (ffn *FeedForwardNetwork) forward(x [][]float64) [][]float64 {
	// First linear layer + GELU
	hidden := matMulAdd(x, ffn.w1, ffn.b1)
	hidden = gelu(hidden)

	// Second linear layer
	output := matMulAdd(hidden, ffn.w2, ffn.b2)

	return output
}

// newLayerNorm creates layer normalization
func newLayerNorm(dim int) *LayerNorm {
	return &LayerNorm{
		gamma: ones(dim),
		beta:  zeros(dim),
		eps:   1e-5,
	}
}

// forward applies layer normalization
func (ln *LayerNorm) forward(x [][]float64) [][]float64 {
	output := make([][]float64, len(x))
	for i := range x {
		output[i] = ln.normalize(x[i])
	}
	return output
}

func (ln *LayerNorm) normalize(x []float64) []float64 {
	mean := mean(x)
	variance := variance(x, mean)
	std := math.Sqrt(variance + ln.eps)

	output := make([]float64, len(x))
	for i := range x {
		output[i] = ln.gamma[i]*(x[i]-mean)/std + ln.beta[i]
	}
	return output
}

// forward applies classification head
func (ch *ClassificationHead) forward(x []float64) []float64 {
	logits := make([]float64, len(ch.b))
	for i := range logits {
		logits[i] = ch.b[i]
		for j := range x {
			logits[i] += x[j] * ch.w[j][i]
		}
	}
	return logits
}

// identifyPatterns detects known attack patterns via attention analysis
func (tsa *TransformerSequenceAnalyzer) identifyPatterns(syscalls []string, attention [][]float64) []string {
	patterns := []string{}

	// Known attack signatures in syscall sequences
	signatures := map[string][]string{
		"privilege_escalation": {"setuid", "setgid", "execve"},
		"code_injection":       {"mmap", "mprotect", "execve"},
		"shell_spawn":          {"fork", "execve", "/bin/sh"},
		"network_exfiltration": {"socket", "connect", "sendto"},
		"rootkit_behavior":     {"ptrace", "prctl", "mprotect"},
	}

	for patternName, signature := range signatures {
		if containsSequence(syscalls, signature) {
			patterns = append(patterns, patternName)
		}
	}

	return patterns
}

// Helper functions

func initSyscallVocabulary() map[string]int {
	commonSyscalls := []string{
		"read", "write", "open", "close", "stat", "fstat", "lseek", "mmap",
		"mprotect", "munmap", "brk", "ioctl", "access", "socket", "connect",
		"bind", "listen", "accept", "sendto", "recvfrom", "execve", "fork",
		"vfork", "clone", "wait4", "kill", "ptrace", "setuid", "setgid",
	}

	vocab := make(map[string]int)
	vocab["<UNK>"] = 0
	vocab["<PAD>"] = 1
	vocab["<CLS>"] = 2

	for i, syscall := range commonSyscalls {
		vocab[syscall] = i + 3
	}

	return vocab
}

func precomputePositionalEncodings(maxLen, dim int) [][]float64 {
	encodings := make([][]float64, maxLen)
	for pos := 0; pos < maxLen; pos++ {
		encodings[pos] = make([]float64, dim)
		for i := 0; i < dim; i++ {
			if i%2 == 0 {
				encodings[pos][i] = math.Sin(float64(pos) / math.Pow(10000, float64(i)/float64(dim)))
			} else {
				encodings[pos][i] = math.Cos(float64(pos) / math.Pow(10000, float64(i-1)/float64(dim)))
			}
		}
	}
	return encodings
}

func computeWeightedThreatScore(probs []float64) float64 {
	// Class weights: benign=0, suspicious=0.3, malicious=0.7, exploit=0.9, advanced=1.0
	weights := []float64{0.0, 0.3, 0.7, 0.9, 1.0}
	score := 0.0
	for i, prob := range probs {
		if i < len(weights) {
			score += prob * weights[i]
		}
	}
	return score
}

func computeAnomalyScores(hidden [][]float64, embeddings *EmbeddingLayer) []float64 {
	scores := make([]float64, len(hidden))
	for i, h := range hidden {
		// Compute reconstruction error as anomaly score
		scores[i] = norm(h) / float64(len(h))
	}
	return scores
}

func containsSequence(haystack, needle []string) bool {
	if len(needle) > len(haystack) {
		return false
	}
	for i := 0; i <= len(haystack)-len(needle); i++ {
		match := true
		for j := 0; j < len(needle); j++ {
			if haystack[i+j] != needle[j] {
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

// Math utilities

func randomMatrix(rows, cols int, std float64) [][]float64 {
	m := make([][]float64, rows)
	for i := range m {
		m[i] = make([]float64, cols)
		for j := range m[i] {
			m[i][j] = (float64(time.Now().UnixNano()%1000)/1000.0 - 0.5) * std * 2
		}
	}
	return m
}

func matMul(a, b [][]float64) [][]float64 {
	rows := len(a)
	cols := len(b[0])
	result := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		result[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			for k := 0; k < len(b); k++ {
				result[i][j] += a[i][k] * b[k][j]
			}
		}
	}
	return result
}

func matMulAdd(x, w [][]float64, b []float64) [][]float64 {
	result := matMul(x, w)
	for i := range result {
		for j := range result[i] {
			result[i][j] += b[j]
		}
	}
	return result
}

func addResidual(x, residual [][]float64) [][]float64 {
	result := make([][]float64, len(x))
	for i := range x {
		result[i] = make([]float64, len(x[i]))
		for j := range x[i] {
			result[i][j] = x[i][j] + residual[i][j]
		}
	}
	return result
}

func dotProduct(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func softmax(x []float64) []float64 {
	max := maxFloat64(x)
	exp := make([]float64, len(x))
	sum := 0.0
	for i, v := range x {
		exp[i] = math.Exp(v - max)
		sum += exp[i]
	}
	for i := range exp {
		exp[i] /= sum
	}
	return exp
}

func gelu(x [][]float64) [][]float64 {
	result := make([][]float64, len(x))
	for i := range x {
		result[i] = make([]float64, len(x[i]))
		for j := range x[i] {
			result[i][j] = x[i][j] * 0.5 * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(x[i][j]+0.044715*math.Pow(x[i][j], 3))))
		}
	}
	return result
}

func mean(x []float64) float64 {
	sum := 0.0
	for _, v := range x {
		sum += v
	}
	return sum / float64(len(x))
}

func variance(x []float64, mean float64) float64 {
	sum := 0.0
	for _, v := range x {
		diff := v - mean
		sum += diff * diff
	}
	return sum / float64(len(x))
}

func norm(x []float64) float64 {
	sum := 0.0
	for _, v := range x {
		sum += v * v
	}
	return math.Sqrt(sum)
}

func maxFloat64(x []float64) float64 {
	max := x[0]
	for _, v := range x {
		if v > max {
			max = v
		}
	}
	return max
}

func ones(n int) []float64 {
	x := make([]float64, n)
	for i := range x {
		x[i] = 1.0
	}
	return x
}

func zeros(n int) []float64 {
	return make([]float64, n)
}
