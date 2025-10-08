package verifier

import (
	"context"
	"crypto/ed25519"
	"fmt"
	"sync"
	"time"
)

type VerifierNode struct {
	ID         string    `json:"id"`
	PublicKey  []byte    `json:"public_key"`
	Endpoint   string    `json:"endpoint"`
	Stake      uint64    `json:"stake"`
	Reputation float64   `json:"reputation"`
	LastSeen   time.Time `json:"last_seen"`
}

type ValidationRequest struct {
	PackHash    string            `json:"pack_hash"`
	Metadata    map[string]string `json:"metadata"`
	Timestamp   time.Time         `json:"timestamp"`
	RequesterID string            `json:"requester_id"`
}

type ValidationResult struct {
	NodeID    string    `json:"node_id"`
	PackHash  string    `json:"pack_hash"`
	Valid     bool      `json:"valid"`
	Signature []byte    `json:"signature"`
	Timestamp time.Time `json:"timestamp"`
}

type ConsensusResult struct {
	PackHash   string              `json:"pack_hash"`
	Valid      bool                `json:"valid"`
	Confidence float64             `json:"confidence"`
	Results    []*ValidationResult `json:"results"`
	Timestamp  time.Time           `json:"timestamp"`
}

type Pool struct {
	nodes     map[string]*VerifierNode
	mu        sync.RWMutex
	minNodes  int
	consensus float64
}

func NewPool(minNodes int, consensusThreshold float64) *Pool {
	return &Pool{
		nodes:     make(map[string]*VerifierNode),
		minNodes:  minNodes,
		consensus: consensusThreshold,
	}
}

func (p *Pool) AddNode(node *VerifierNode) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if len(node.PublicKey) != ed25519.PublicKeySize {
		return fmt.Errorf("invalid public key size")
	}

	p.nodes[node.ID] = node
	return nil
}

func (p *Pool) ValidatePack(ctx context.Context, req *ValidationRequest) (*ConsensusResult, error) {
	p.mu.RLock()
	activeNodes := p.getActiveNodes()
	p.mu.RUnlock()

	if len(activeNodes) < p.minNodes {
		return nil, fmt.Errorf("insufficient active nodes: %d < %d", len(activeNodes), p.minNodes)
	}

	results := make(chan *ValidationResult, len(activeNodes))

	for _, node := range activeNodes {
		go p.requestValidation(ctx, node, req, results)
	}

	return p.collectConsensus(ctx, results, len(activeNodes), req.PackHash)
}

func (p *Pool) requestValidation(ctx context.Context, node *VerifierNode, req *ValidationRequest, results chan<- *ValidationResult) {
	select {
	case <-ctx.Done():
		return
	case <-time.After(5 * time.Second):
		result := &ValidationResult{
			NodeID:    node.ID,
			PackHash:  req.PackHash,
			Valid:     true,
			Timestamp: time.Now(),
		}
		results <- result
	}
}

func (p *Pool) collectConsensus(ctx context.Context, results <-chan *ValidationResult, nodeCount int, packHash string) (*ConsensusResult, error) {
	var validationResults []*ValidationResult
	validCount := 0

	timeout := time.After(10 * time.Second)

	for i := 0; i < nodeCount; i++ {
		select {
		case result := <-results:
			validationResults = append(validationResults, result)
			if result.Valid {
				validCount++
			}
		case <-timeout:
			break
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}

	if len(validationResults) == 0 {
		return nil, fmt.Errorf("no validation results received")
	}

	confidence := float64(validCount) / float64(len(validationResults))
	isValid := confidence >= p.consensus

	return &ConsensusResult{
		PackHash:   packHash,
		Valid:      isValid,
		Confidence: confidence,
		Results:    validationResults,
		Timestamp:  time.Now(),
	}, nil
}

func (p *Pool) getActiveNodes() []*VerifierNode {
	var active []*VerifierNode
	cutoff := time.Now().Add(-5 * time.Minute)

	for _, node := range p.nodes {
		if node.LastSeen.After(cutoff) {
			active = append(active, node)
		}
	}
	return active
}

func (p *Pool) UpdateReputation(nodeID string, delta float64) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if node, exists := p.nodes[nodeID]; exists {
		node.Reputation += delta
		if node.Reputation < 0 {
			node.Reputation = 0
		}
		if node.Reputation > 1.0 {
			node.Reputation = 1.0
		}
	}
}
