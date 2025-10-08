package verifier

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"time"
)

type OnChainAnchor struct {
	BlockHash   string    `json:"block_hash"`
	TxHash      string    `json:"tx_hash"`
	MerkleRoot  string    `json:"merkle_root"`
	Timestamp   time.Time `json:"timestamp"`
	BlockHeight uint64    `json:"block_height"`
}

type ReputationScore struct {
	NodeID           string    `json:"node_id"`
	Score            float64   `json:"score"`
	ValidationsCount uint64    `json:"validations_count"`
	SuccessRate      float64   `json:"success_rate"`
	StakeAmount      uint64    `json:"stake_amount"`
	LastUpdate       time.Time `json:"last_update"`
}

type OnChainManager struct {
	chainID      string
	contractAddr string
	anchors      map[string]*OnChainAnchor
	reputations  map[string]*ReputationScore
}

func NewOnChainManager(chainID, contractAddr string) *OnChainManager {
	return &OnChainManager{
		chainID:      chainID,
		contractAddr: contractAddr,
		anchors:      make(map[string]*OnChainAnchor),
		reputations:  make(map[string]*ReputationScore),
	}
}

func (ocm *OnChainManager) AnchorMerkleRoot(merkleRoot string) (*OnChainAnchor, error) {
	// Mock blockchain interaction
	blockHash := ocm.generateBlockHash()
	txHash := ocm.generateTxHash(merkleRoot)

	anchor := &OnChainAnchor{
		BlockHash:   blockHash,
		TxHash:      txHash,
		MerkleRoot:  merkleRoot,
		Timestamp:   time.Now(),
		BlockHeight: uint64(time.Now().Unix()), // Mock block height
	}

	ocm.anchors[merkleRoot] = anchor
	return anchor, nil
}

func (ocm *OnChainManager) VerifyAnchor(merkleRoot string) (*OnChainAnchor, bool) {
	anchor, exists := ocm.anchors[merkleRoot]
	return anchor, exists
}

func (ocm *OnChainManager) UpdateReputation(nodeID string, validationSuccess bool, stakeAmount uint64) {
	rep, exists := ocm.reputations[nodeID]
	if !exists {
		rep = &ReputationScore{
			NodeID:      nodeID,
			Score:       0.5, // Start with neutral score
			StakeAmount: stakeAmount,
		}
		ocm.reputations[nodeID] = rep
	}

	rep.ValidationsCount++

	if validationSuccess {
		rep.Score += 0.01 // Small positive increment
	} else {
		rep.Score -= 0.05 // Larger negative penalty
	}

	// Clamp score between 0 and 1
	if rep.Score < 0 {
		rep.Score = 0
	}
	if rep.Score > 1 {
		rep.Score = 1
	}

	// Calculate success rate
	if rep.ValidationsCount > 0 {
		successCount := float64(rep.ValidationsCount)
		if !validationSuccess {
			successCount--
		}
		rep.SuccessRate = successCount / float64(rep.ValidationsCount)
	}

	rep.LastUpdate = time.Now()
}

func (ocm *OnChainManager) GetReputation(nodeID string) (*ReputationScore, bool) {
	rep, exists := ocm.reputations[nodeID]
	return rep, exists
}

func (ocm *OnChainManager) GetTopVerifiers(limit int) []*ReputationScore {
	var verifiers []*ReputationScore
	for _, rep := range ocm.reputations {
		verifiers = append(verifiers, rep)
	}

	// Simple bubble sort by score (for small datasets)
	for i := 0; i < len(verifiers)-1; i++ {
		for j := 0; j < len(verifiers)-i-1; j++ {
			if verifiers[j].Score < verifiers[j+1].Score {
				verifiers[j], verifiers[j+1] = verifiers[j+1], verifiers[j]
			}
		}
	}

	if len(verifiers) > limit {
		verifiers = verifiers[:limit]
	}

	return verifiers
}

func (ocm *OnChainManager) generateBlockHash() string {
	data := fmt.Sprintf("block_%d", time.Now().UnixNano())
	hash := sha256.Sum256([]byte(data))
	return hex.EncodeToString(hash[:])
}

func (ocm *OnChainManager) generateTxHash(merkleRoot string) string {
	data := fmt.Sprintf("tx_%s_%d", merkleRoot, time.Now().UnixNano())
	hash := sha256.Sum256([]byte(data))
	return hex.EncodeToString(hash[:])
}

type AnchorEvent struct {
	Type       string         `json:"type"`
	MerkleRoot string         `json:"merkle_root"`
	Anchor     *OnChainAnchor `json:"anchor"`
	Timestamp  time.Time      `json:"timestamp"`
}

func (ocm *OnChainManager) ExportAnchors() ([]byte, error) {
	var events []AnchorEvent
	for root, anchor := range ocm.anchors {
		events = append(events, AnchorEvent{
			Type:       "anchor_created",
			MerkleRoot: root,
			Anchor:     anchor,
			Timestamp:  anchor.Timestamp,
		})
	}
	return json.Marshal(events)
}
