package wch

import (
	"crypto/rand"
	"crypto/sha256"
	"fmt"
	"math/big"
	"sync"
	"time"
)

type ZKRateLimit struct {
	groups     map[string]*Group
	nullifiers map[string]time.Time
	mu         sync.RWMutex
	window     time.Duration
}

type Group struct {
	ID        string
	Members   []*big.Int
	Root      *big.Int
	RateLimit int
}

type ZKProof struct {
	Nullifier *big.Int
	Proof     []byte
	Signal    []byte
	GroupID   string
}

type ObliviousRelay struct {
	relays map[string]*RelayNode
	mu     sync.RWMutex
}

type RelayNode struct {
	ID       string
	PublicKey []byte
	Endpoint string
	Load     int
}

type PQKeyExchange struct {
	kyberPriv []byte
	kyberPub  []byte
	x25519Priv []byte
	x25519Pub  []byte
}

func NewZKRateLimit(window time.Duration) *ZKRateLimit {
	return &ZKRateLimit{
		groups:     make(map[string]*Group),
		nullifiers: make(map[string]time.Time),
		window:     window,
	}
}

func (zk *ZKRateLimit) CreateGroup(groupID string, rateLimit int) *Group {
	zk.mu.Lock()
	defer zk.mu.Unlock()
	
	group := &Group{
		ID:        groupID,
		Members:   make([]*big.Int, 0),
		RateLimit: rateLimit,
	}
	
	zk.groups[groupID] = group
	return group
}

func (zk *ZKRateLimit) AddMember(groupID string, identity *big.Int) error {
	zk.mu.Lock()
	defer zk.mu.Unlock()
	
	group := zk.groups[groupID]
	if group == nil {
		return fmt.Errorf("group not found")
	}
	
	group.Members = append(group.Members, identity)
	group.Root = zk.calculateRoot(group.Members)
	return nil
}

func (zk *ZKRateLimit) GenerateProof(groupID string, identity *big.Int, signal []byte) (*ZKProof, error) {
	group := zk.groups[groupID]
	if group == nil {
		return nil, fmt.Errorf("group not found")
	}
	
	nullifier := zk.generateNullifier(identity, signal)
	proof := zk.generateProof(group, identity, nullifier)
	
	return &ZKProof{
		Nullifier: nullifier,
		Proof:     proof,
		Signal:    signal,
		GroupID:   groupID,
	}, nil
}

func (zk *ZKRateLimit) VerifyProof(proof *ZKProof) bool {
	zk.mu.Lock()
	defer zk.mu.Unlock()
	
	nullifierStr := proof.Nullifier.String()
	if lastUsed, exists := zk.nullifiers[nullifierStr]; exists {
		if time.Since(lastUsed) < zk.window {
			return false
		}
	}
	
	group := zk.groups[proof.GroupID]
	if group == nil || !zk.verifyProof(group, proof) {
		return false
	}
	
	zk.nullifiers[nullifierStr] = time.Now()
	go zk.cleanNullifiers()
	
	return true
}

func (zk *ZKRateLimit) generateNullifier(identity *big.Int, signal []byte) *big.Int {
	h := sha256.New()
	h.Write(identity.Bytes())
	h.Write(signal)
	hash := h.Sum(nil)
	
	nullifier := new(big.Int)
	nullifier.SetBytes(hash)
	return nullifier
}

func (zk *ZKRateLimit) calculateRoot(members []*big.Int) *big.Int {
	if len(members) == 0 {
		return big.NewInt(0)
	}
	
	h := sha256.New()
	for _, member := range members {
		h.Write(member.Bytes())
	}
	
	root := new(big.Int)
	root.SetBytes(h.Sum(nil))
	return root
}

func (zk *ZKRateLimit) generateProof(group *Group, identity *big.Int, nullifier *big.Int) []byte {
	h := sha256.New()
	h.Write(group.Root.Bytes())
	h.Write(identity.Bytes())
	h.Write(nullifier.Bytes())
	
	nonce := make([]byte, 32)
	rand.Read(nonce)
	h.Write(nonce)
	
	return h.Sum(nil)
}

func (zk *ZKRateLimit) verifyProof(group *Group, proof *ZKProof) bool {
	for _, member := range group.Members {
		expectedNullifier := zk.generateNullifier(member, proof.Signal)
		if expectedNullifier.Cmp(proof.Nullifier) == 0 {
			return true
		}
	}
	return false
}

func (zk *ZKRateLimit) cleanNullifiers() {
	zk.mu.Lock()
	defer zk.mu.Unlock()
	
	cutoff := time.Now().Add(-zk.window)
	for nullifier, timestamp := range zk.nullifiers {
		if timestamp.Before(cutoff) {
			delete(zk.nullifiers, nullifier)
		}
	}
}

func NewObliviousRelay() *ObliviousRelay {
	return &ObliviousRelay{
		relays: make(map[string]*RelayNode),
	}
}

func (or *ObliviousRelay) AddRelay(id, endpoint string, pubKey []byte) {
	or.mu.Lock()
	defer or.mu.Unlock()
	
	or.relays[id] = &RelayNode{
		ID:       id,
		PublicKey: pubKey,
		Endpoint: endpoint,
		Load:     0,
	}
}

func (or *ObliviousRelay) SelectRelay() *RelayNode {
	or.mu.RLock()
	defer or.mu.RUnlock()
	
	var bestRelay *RelayNode
	minLoad := int(^uint(0) >> 1)
	
	for _, relay := range or.relays {
		if relay.Load < minLoad {
			minLoad = relay.Load
			bestRelay = relay
		}
	}
	
	if bestRelay != nil {
		bestRelay.Load++
	}
	
	return bestRelay
}

func NewPQKeyExchange() (*PQKeyExchange, error) {
	pq := &PQKeyExchange{}
	
	// Generate Kyber keys (simplified)
	pq.kyberPriv = make([]byte, 32)
	pq.kyberPub = make([]byte, 32)
	rand.Read(pq.kyberPriv)
	rand.Read(pq.kyberPub)
	
	// Generate X25519 keys (simplified)
	pq.x25519Priv = make([]byte, 32)
	pq.x25519Pub = make([]byte, 32)
	rand.Read(pq.x25519Priv)
	rand.Read(pq.x25519Pub)
	
	return pq, nil
}

func (pq *PQKeyExchange) GetPublicKeys() ([]byte, []byte) {
	return pq.kyberPub, pq.x25519Pub
}

func (pq *PQKeyExchange) DeriveSharedSecret(kyberCiphertext, x25519PeerPub []byte) []byte {
	// Simplified hybrid key derivation
	h := sha256.New()
	h.Write(pq.kyberPriv)
	h.Write(kyberCiphertext)
	h.Write(pq.x25519Priv)
	h.Write(x25519PeerPub)
	
	return h.Sum(nil)
}