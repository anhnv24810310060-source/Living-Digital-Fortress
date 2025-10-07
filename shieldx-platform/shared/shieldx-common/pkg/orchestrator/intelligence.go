package orchestrator

import (
	"fmt"
	"sync"
	"time"
)

type ThreatIntelligence struct {
	federation *TIFederation
	mitre      *MITREGraph
	correlator *TTPrealTimeCorrelator
	mu         sync.RWMutex
}

type TIFederation struct {
	nodes      map[string]*FederationNode
	aggregator *SecureAggregator
	mu         sync.RWMutex
}

type FederationNode struct {
	ID        string
	Endpoint  string
	PublicKey []byte
	LastSeen  time.Time
	Trust     float64
}

type SecureAggregator struct {
	contributions map[string]*Contribution
	threshold     int
	privacyBudget float64
}

type Contribution struct {
	NodeID    string
	Data      []byte
	Signature []byte
	Timestamp time.Time
}

type MITREGraph struct {
	techniques map[string]*Technique
	tactics    map[string]*Tactic
	edges      []TechniqueEdge
	mu         sync.RWMutex
}

type Technique struct {
	ID          string
	Name        string
	Tactic      string
	Description string
	Indicators  []string
	Confidence  float64
}

type Tactic struct {
	ID         string
	Name       string
	Techniques []string
	Phase      string
}

type TechniqueEdge struct {
	From   string
	To     string
	Weight float64
}

type TTPrealTimeCorrelator struct {
	campaigns map[string]*Campaign
	patterns  map[string]*AttackPattern
	mu        sync.RWMutex
}

type Campaign struct {
	ID         string
	Name       string
	Techniques []string
	Actors     []string
	StartTime  time.Time
	LastUpdate time.Time
	Confidence float64
}

type AttackPattern struct {
	ID         string
	Sequence   []string
	Timing     []time.Duration
	Confidence float64
	Frequency  int
}

func NewThreatIntelligence() *ThreatIntelligence {
	return &ThreatIntelligence{
		federation: NewTIFederation(),
		mitre:      NewMITREGraph(),
		correlator: NewTTPrealTimeCorrelator(),
	}
}

func NewTIFederation() *TIFederation {
	return &TIFederation{
		nodes: make(map[string]*FederationNode),
		aggregator: &SecureAggregator{
			contributions: make(map[string]*Contribution),
			threshold:     3,
			privacyBudget: 1.0,
		},
	}
}

func (tif *TIFederation) AddNode(id, endpoint string, pubKey []byte) {
	tif.mu.Lock()
	defer tif.mu.Unlock()

	tif.nodes[id] = &FederationNode{
		ID:        id,
		Endpoint:  endpoint,
		PublicKey: pubKey,
		LastSeen:  time.Now(),
		Trust:     0.5,
	}
}

func (tif *TIFederation) ContributeThreatData(nodeID string, data []byte, signature []byte) error {
	tif.mu.Lock()
	defer tif.mu.Unlock()

	node := tif.nodes[nodeID]
	if node == nil {
		return fmt.Errorf("unknown node: %s", nodeID)
	}

	contribution := &Contribution{
		NodeID:    nodeID,
		Data:      data,
		Signature: signature,
		Timestamp: time.Now(),
	}

	tif.aggregator.contributions[nodeID] = contribution
	node.LastSeen = time.Now()

	if len(tif.aggregator.contributions) >= tif.aggregator.threshold {
		go tif.performSecureAggregation()
	}

	return nil
}

func (tif *TIFederation) performSecureAggregation() {
	// Secure multi-party computation for threat intelligence
	// Add differential privacy noise
	// Aggregate without revealing individual contributions
}

func NewMITREGraph() *MITREGraph {
	mg := &MITREGraph{
		techniques: make(map[string]*Technique),
		tactics:    make(map[string]*Tactic),
		edges:      make([]TechniqueEdge, 0),
	}

	mg.loadMITREData()
	return mg
}

func (mg *MITREGraph) loadMITREData() {
	// Load MITRE ATT&CK framework data
	mg.techniques["T1190"] = &Technique{
		ID:          "T1190",
		Name:        "Exploit Public-Facing Application",
		Tactic:      "Initial Access",
		Description: "Adversaries may attempt to take advantage of a weakness in an Internet-facing computer or program",
		Indicators:  []string{"web_exploit", "sql_injection", "xss"},
		Confidence:  0.9,
	}

	mg.techniques["T1059"] = &Technique{
		ID:          "T1059",
		Name:        "Command and Scripting Interpreter",
		Tactic:      "Execution",
		Description: "Adversaries may abuse command and script interpreters to execute commands",
		Indicators:  []string{"cmd_injection", "powershell", "bash"},
		Confidence:  0.8,
	}

	mg.tactics["TA0001"] = &Tactic{
		ID:         "TA0001",
		Name:       "Initial Access",
		Techniques: []string{"T1190", "T1566", "T1078"},
		Phase:      "Entry",
	}

	mg.edges = append(mg.edges, TechniqueEdge{
		From:   "T1190",
		To:     "T1059",
		Weight: 0.7,
	})
}

func (mg *MITREGraph) AnalyzeTechniques(indicators []string) []*Technique {
	mg.mu.RLock()
	defer mg.mu.RUnlock()

	matches := make([]*Technique, 0)

	for _, technique := range mg.techniques {
		for _, indicator := range indicators {
			for _, techIndicator := range technique.Indicators {
				if indicator == techIndicator {
					matches = append(matches, technique)
					break
				}
			}
		}
	}

	return matches
}

func (mg *MITREGraph) PredictNextTechniques(currentTechnique string) []*Technique {
	mg.mu.RLock()
	defer mg.mu.RUnlock()

	predictions := make([]*Technique, 0)

	for _, edge := range mg.edges {
		if edge.From == currentTechnique {
			if technique := mg.techniques[edge.To]; technique != nil {
				predictions = append(predictions, technique)
			}
		}
	}

	return predictions
}

func NewTTPrealTimeCorrelator() *TTPrealTimeCorrelator {
	return &TTPrealTimeCorrelator{
		campaigns: make(map[string]*Campaign),
		patterns:  make(map[string]*AttackPattern),
	}
}

func (ttc *TTPrealTimeCorrelator) CorrelateTechniques(tenantID string, techniques []string, timestamps []time.Time) *Campaign {
	ttc.mu.Lock()
	defer ttc.mu.Unlock()

	// Find existing campaign or create new one
	for _, campaign := range ttc.campaigns {
		if ttc.matchesCampaign(techniques, campaign.Techniques) {
			campaign.LastUpdate = time.Now()
			campaign.Confidence = ttc.calculateConfidence(campaign, techniques)
			return campaign
		}
	}

	// Create new campaign
	campaignID := fmt.Sprintf("campaign_%d", time.Now().UnixNano())
	campaign := &Campaign{
		ID:         campaignID,
		Name:       fmt.Sprintf("Campaign_%s", tenantID),
		Techniques: techniques,
		Actors:     []string{"unknown"},
		StartTime:  time.Now(),
		LastUpdate: time.Now(),
		Confidence: 0.6,
	}

	ttc.campaigns[campaignID] = campaign

	// Update attack patterns
	ttc.updateAttackPatterns(techniques, timestamps)

	return campaign
}

func (ttc *TTPrealTimeCorrelator) matchesCampaign(techniques1, techniques2 []string) bool {
	if len(techniques1) == 0 || len(techniques2) == 0 {
		return false
	}

	matches := 0
	for _, t1 := range techniques1 {
		for _, t2 := range techniques2 {
			if t1 == t2 {
				matches++
				break
			}
		}
	}

	return float64(matches)/float64(len(techniques1)) >= 0.5
}

func (ttc *TTPrealTimeCorrelator) calculateConfidence(campaign *Campaign, newTechniques []string) float64 {
	baseConfidence := campaign.Confidence

	// Increase confidence with more matching techniques
	matches := 0
	for _, nt := range newTechniques {
		for _, ct := range campaign.Techniques {
			if nt == ct {
				matches++
				break
			}
		}
	}

	matchRatio := float64(matches) / float64(len(newTechniques))
	v := baseConfidence + matchRatio*0.2
	if v > 1.0 {
		v = 1.0
	}
	return v
}

func (ttc *TTPrealTimeCorrelator) updateAttackPatterns(techniques []string, timestamps []time.Time) {
	if len(techniques) < 2 {
		return
	}

	// Calculate timing between techniques
	timings := make([]time.Duration, len(timestamps)-1)
	for i := 1; i < len(timestamps); i++ {
		timings[i-1] = timestamps[i].Sub(timestamps[i-1])
	}

	patternID := fmt.Sprintf("pattern_%s", techniques[0])
	pattern := ttc.patterns[patternID]

	if pattern == nil {
		pattern = &AttackPattern{
			ID:         patternID,
			Sequence:   techniques,
			Timing:     timings,
			Confidence: 0.5,
			Frequency:  1,
		}
		ttc.patterns[patternID] = pattern
	} else {
		pattern.Frequency++
		if pattern.Confidence+0.1 > 1.0 {
			pattern.Confidence = 1.0
		} else {
			pattern.Confidence = pattern.Confidence + 0.1
		}
	}
}

func (ti *ThreatIntelligence) GetGlobalThreatLevel() float64 {
	ti.mu.RLock()
	defer ti.mu.RUnlock()

	totalCampaigns := len(ti.correlator.campaigns)
	if totalCampaigns == 0 {
		return 0.0
	}

	totalConfidence := 0.0
	for _, campaign := range ti.correlator.campaigns {
		totalConfidence += campaign.Confidence
	}

	return totalConfidence / float64(totalCampaigns)
}

// min removed to avoid conflicts with other package-level min definitions
