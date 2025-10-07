package honeypotdeception

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"
)

type DeceptionGraph struct {
	Nodes  map[string]*DeceptionNode `json:"nodes"`
	Bandit *MultiArmedBandit         `json:"bandit"`
	mu     sync.RWMutex
}

type DeceptionNode struct {
	ID           string                 `json:"id"`
	Type         string                 `json:"type"`
	Config       map[string]interface{} `json:"config"`
	Effectiveness float64               `json:"effectiveness"`
	Visits       int                    `json:"visits"`
	LastUsed     time.Time              `json:"last_used"`
}

type MultiArmedBandit struct {
	Arms      map[string]*BanditArm `json:"arms"`
	Algorithm string                `json:"algorithm"`
	Epsilon   float64               `json:"epsilon"`
}

type BanditArm struct {
	ID     string  `json:"id"`
	Pulls  int     `json:"pulls"`
	Reward float64 `json:"reward"`
	Value  float64 `json:"value"`
}

func NewDeceptionGraph() *DeceptionGraph {
	return &DeceptionGraph{
		Nodes: make(map[string]*DeceptionNode),
		Bandit: &MultiArmedBandit{
			Arms:      make(map[string]*BanditArm),
			Algorithm: "ucb1",
			Epsilon:   0.1,
		},
	}
}

func (dg *DeceptionGraph) AddNode(node *DeceptionNode) {
	dg.mu.Lock()
	defer dg.mu.Unlock()
	
	dg.Nodes[node.ID] = node
	dg.Bandit.Arms[node.ID] = &BanditArm{
		ID:    node.ID,
		Pulls: 0,
		Reward: 0.0,
		Value:  0.0,
	}
}

func (dg *DeceptionGraph) SelectOptimalDecoy(ctx context.Context) (*DeceptionNode, error) {
	dg.mu.Lock()
	defer dg.mu.Unlock()
	
	switch dg.Bandit.Algorithm {
	case "ucb1":
		return dg.selectUCB1()
	case "epsilon_greedy":
		return dg.selectEpsilonGreedy()
	default:
		return dg.selectRandom()
	}
}

func (dg *DeceptionGraph) selectUCB1() (*DeceptionNode, error) {
	if len(dg.Nodes) == 0 {
		return nil, fmt.Errorf("no nodes available")
	}
	
	totalPulls := 0
	for _, arm := range dg.Bandit.Arms {
		totalPulls += arm.Pulls
	}
	
	if totalPulls == 0 {
		return dg.selectRandom()
	}
	
	bestScore := -math.Inf(1)
	var bestNode *DeceptionNode
	
	for nodeID, node := range dg.Nodes {
		arm := dg.Bandit.Arms[nodeID]
		if arm.Pulls == 0 {
			return node, nil
		}
		
		confidence := math.Sqrt(2 * math.Log(float64(totalPulls)) / float64(arm.Pulls))
		score := arm.Value + confidence
		
		if score > bestScore {
			bestScore = score
			bestNode = node
		}
	}
	
	return bestNode, nil
}

func (dg *DeceptionGraph) selectEpsilonGreedy() (*DeceptionNode, error) {
	if rand.Float64() < dg.Bandit.Epsilon {
		return dg.selectRandom()
	}
	
	bestValue := -math.Inf(1)
	var bestNode *DeceptionNode
	
	for nodeID, node := range dg.Nodes {
		arm := dg.Bandit.Arms[nodeID]
		if arm.Value > bestValue {
			bestValue = arm.Value
			bestNode = node
		}
	}
	
	if bestNode == nil {
		return dg.selectRandom()
	}
	
	return bestNode, nil
}

func (dg *DeceptionGraph) selectRandom() (*DeceptionNode, error) {
	if len(dg.Nodes) == 0 {
		return nil, fmt.Errorf("no nodes available")
	}
	
	nodes := make([]*DeceptionNode, 0, len(dg.Nodes))
	for _, node := range dg.Nodes {
		nodes = append(nodes, node)
	}
	
	return nodes[rand.Intn(len(nodes))], nil
}

func (dg *DeceptionGraph) UpdateReward(nodeID string, reward float64) {
	dg.mu.Lock()
	defer dg.mu.Unlock()
	
	arm, exists := dg.Bandit.Arms[nodeID]
	if !exists {
		return
	}
	
	arm.Pulls++
	arm.Reward += reward
	arm.Value = arm.Reward / float64(arm.Pulls)
	
	if node, exists := dg.Nodes[nodeID]; exists {
		node.Visits++
		node.LastUsed = time.Now()
		node.Effectiveness = arm.Value
	}
}

func (dg *DeceptionGraph) GetMetrics() map[string]interface{} {
	dg.mu.RLock()
	defer dg.mu.RUnlock()
	
	metrics := map[string]interface{}{
		"total_nodes": len(dg.Nodes),
		"algorithm":   dg.Bandit.Algorithm,
		"epsilon":     dg.Bandit.Epsilon,
	}
	
	bestNodes := make([]map[string]interface{}, 0)
	for nodeID, node := range dg.Nodes {
		arm := dg.Bandit.Arms[nodeID]
		bestNodes = append(bestNodes, map[string]interface{}{
			"id":           nodeID,
			"type":         node.Type,
			"effectiveness": node.Effectiveness,
			"visits":       node.Visits,
			"pulls":        arm.Pulls,
			"value":        arm.Value,
		})
	}
	
	metrics["nodes"] = bestNodes
	return metrics
}

func CreateWebServerDecoy() *DeceptionNode {
	return &DeceptionNode{
		ID:   fmt.Sprintf("web_decoy_%d", rand.Uint64()),
		Type: "web_server",
		Config: map[string]interface{}{
			"port":        80,
			"server_name": "Apache/2.4.41",
			"pages":       []string{"/", "/admin", "/login"},
			"jitter_ms":   rand.Intn(200) + 50,
		},
		Effectiveness: 0.0,
	}
}

func CreateSSHHoneypot() *DeceptionNode {
	return &DeceptionNode{
		ID:   fmt.Sprintf("ssh_honeypot_%d", rand.Uint64()),
		Type: "ssh_server",
		Config: map[string]interface{}{
			"port":        22,
			"banner":      "OpenSSH_8.2p1 Ubuntu-4ubuntu0.5",
			"fake_auth":   true,
			"delay_ms":    rand.Intn(1000) + 500,
		},
		Effectiveness: 0.0,
	}
}

func CreateDatabaseDecoy() *DeceptionNode {
	return &DeceptionNode{
		ID:   fmt.Sprintf("db_decoy_%d", rand.Uint64()),
		Type: "database",
		Config: map[string]interface{}{
			"port":     3306,
			"type":     "mysql",
			"version":  "8.0.28",
			"fake_db":  "production",
			"tables":   []string{"users", "orders", "payments"},
		},
		Effectiveness: 0.0,
	}
}