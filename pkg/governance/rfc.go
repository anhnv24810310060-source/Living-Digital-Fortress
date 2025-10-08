package governance

import (
	"fmt"
	"sync"
	"time"
)

type RFCStatus string

const (
	RFCDraft      RFCStatus = "draft"
	RFCProposed   RFCStatus = "proposed"
	RFCDiscussion RFCStatus = "discussion"
	RFCVoting     RFCStatus = "voting"
	RFCAccepted   RFCStatus = "accepted"
	RFCRejected   RFCStatus = "rejected"
	RFCWithdrawn  RFCStatus = "withdrawn"
)

type RFC struct {
	ID            string    `json:"id"`
	Title         string    `json:"title"`
	Author        string    `json:"author"`
	Status        RFCStatus `json:"status"`
	Summary       string    `json:"summary"`
	Motivation    string    `json:"motivation"`
	Specification string    `json:"specification"`
	Rationale     string    `json:"rationale"`
	CreatedAt     time.Time `json:"created_at"`
	UpdatedAt     time.Time `json:"updated_at"`
	VotingEnds    time.Time `json:"voting_ends,omitempty"`
}

type Vote struct {
	ID        string    `json:"id"`
	RFCID     string    `json:"rfc_id"`
	Voter     string    `json:"voter"`
	Decision  bool      `json:"decision"` // true = approve, false = reject
	Reason    string    `json:"reason"`
	Timestamp time.Time `json:"timestamp"`
	Weight    float64   `json:"weight"` // voting weight based on reputation
}

type Comment struct {
	ID        string    `json:"id"`
	RFCID     string    `json:"rfc_id"`
	Author    string    `json:"author"`
	Content   string    `json:"content"`
	Timestamp time.Time `json:"timestamp"`
	ParentID  string    `json:"parent_id,omitempty"` // for threaded comments
}

type Governance struct {
	rfcs     map[string]*RFC
	votes    map[string]*Vote
	comments map[string]*Comment
	mu       sync.RWMutex

	// Governance parameters
	votingPeriod    time.Duration
	quorumThreshold float64 // minimum participation rate
	passThreshold   float64 // minimum approval rate
}

func NewGovernance(votingPeriod time.Duration, quorum, passThreshold float64) *Governance {
	return &Governance{
		rfcs:            make(map[string]*RFC),
		votes:           make(map[string]*Vote),
		comments:        make(map[string]*Comment),
		votingPeriod:    votingPeriod,
		quorumThreshold: quorum,
		passThreshold:   passThreshold,
	}
}

func (g *Governance) SubmitRFC(rfc *RFC) error {
	g.mu.Lock()
	defer g.mu.Unlock()

	if rfc.ID == "" {
		rfc.ID = fmt.Sprintf("rfc_%d", time.Now().UnixNano())
	}

	rfc.Status = RFCDraft
	rfc.CreatedAt = time.Now()
	rfc.UpdatedAt = time.Now()

	g.rfcs[rfc.ID] = rfc
	return nil
}

func (g *Governance) ProposeRFC(rfcID string) error {
	g.mu.Lock()
	defer g.mu.Unlock()

	rfc, exists := g.rfcs[rfcID]
	if !exists {
		return fmt.Errorf("RFC not found: %s", rfcID)
	}

	if rfc.Status != RFCDraft {
		return fmt.Errorf("RFC must be in draft status to propose")
	}

	rfc.Status = RFCProposed
	rfc.UpdatedAt = time.Now()

	return nil
}

func (g *Governance) StartVoting(rfcID string) error {
	g.mu.Lock()
	defer g.mu.Unlock()

	rfc, exists := g.rfcs[rfcID]
	if !exists {
		return fmt.Errorf("RFC not found: %s", rfcID)
	}

	if rfc.Status != RFCDiscussion {
		return fmt.Errorf("RFC must be in discussion status to start voting")
	}

	rfc.Status = RFCVoting
	rfc.VotingEnds = time.Now().Add(g.votingPeriod)
	rfc.UpdatedAt = time.Now()

	return nil
}

func (g *Governance) CastVote(vote *Vote) error {
	g.mu.Lock()
	defer g.mu.Unlock()

	rfc, exists := g.rfcs[vote.RFCID]
	if !exists {
		return fmt.Errorf("RFC not found: %s", vote.RFCID)
	}

	if rfc.Status != RFCVoting {
		return fmt.Errorf("RFC is not in voting status")
	}

	if time.Now().After(rfc.VotingEnds) {
		return fmt.Errorf("voting period has ended")
	}

	// Check if voter already voted
	for _, existingVote := range g.votes {
		if existingVote.RFCID == vote.RFCID && existingVote.Voter == vote.Voter {
			return fmt.Errorf("voter has already cast a vote")
		}
	}

	if vote.ID == "" {
		vote.ID = fmt.Sprintf("vote_%d", time.Now().UnixNano())
	}

	vote.Timestamp = time.Now()
	if vote.Weight == 0 {
		vote.Weight = 1.0 // Default weight
	}

	g.votes[vote.ID] = vote
	return nil
}

func (g *Governance) AddComment(comment *Comment) error {
	g.mu.Lock()
	defer g.mu.Unlock()

	if _, exists := g.rfcs[comment.RFCID]; !exists {
		return fmt.Errorf("RFC not found: %s", comment.RFCID)
	}

	if comment.ID == "" {
		comment.ID = fmt.Sprintf("comment_%d", time.Now().UnixNano())
	}

	comment.Timestamp = time.Now()
	g.comments[comment.ID] = comment

	return nil
}

func (g *Governance) FinalizeVoting(rfcID string) error {
	g.mu.Lock()
	defer g.mu.Unlock()

	rfc, exists := g.rfcs[rfcID]
	if !exists {
		return fmt.Errorf("RFC not found: %s", rfcID)
	}

	if rfc.Status != RFCVoting {
		return fmt.Errorf("RFC is not in voting status")
	}

	if time.Now().Before(rfc.VotingEnds) {
		return fmt.Errorf("voting period has not ended")
	}

	// Count votes
	var totalWeight, approveWeight float64
	voteCount := 0

	for _, vote := range g.votes {
		if vote.RFCID == rfcID {
			totalWeight += vote.Weight
			if vote.Decision {
				approveWeight += vote.Weight
			}
			voteCount++
		}
	}

	// Check quorum
	// For simplicity, assume total eligible voters = 100 (in real system, this would be dynamic)
	participationRate := float64(voteCount) / 100.0
	if participationRate < g.quorumThreshold {
		rfc.Status = RFCRejected
		rfc.UpdatedAt = time.Now()
		return fmt.Errorf("quorum not met: %.2f%% < %.2f%%", participationRate*100, g.quorumThreshold*100)
	}

	// Check approval threshold
	approvalRate := approveWeight / totalWeight
	if approvalRate >= g.passThreshold {
		rfc.Status = RFCAccepted
	} else {
		rfc.Status = RFCRejected
	}

	rfc.UpdatedAt = time.Now()
	return nil
}

func (g *Governance) GetRFC(id string) (*RFC, error) {
	g.mu.RLock()
	defer g.mu.RUnlock()

	rfc, exists := g.rfcs[id]
	if !exists {
		return nil, fmt.Errorf("RFC not found: %s", id)
	}

	return rfc, nil
}

func (g *Governance) ListRFCs(status RFCStatus) []*RFC {
	g.mu.RLock()
	defer g.mu.RUnlock()

	var results []*RFC

	for _, rfc := range g.rfcs {
		if status == "" || rfc.Status == status {
			results = append(results, rfc)
		}
	}

	return results
}

func (g *Governance) GetVotingResults(rfcID string) (map[string]interface{}, error) {
	g.mu.RLock()
	defer g.mu.RUnlock()

	var totalWeight, approveWeight float64
	voteCount := 0

	for _, vote := range g.votes {
		if vote.RFCID == rfcID {
			totalWeight += vote.Weight
			if vote.Decision {
				approveWeight += vote.Weight
			}
			voteCount++
		}
	}

	results := map[string]interface{}{
		"total_votes":    voteCount,
		"total_weight":   totalWeight,
		"approve_weight": approveWeight,
		"approval_rate":  0.0,
		"participation":  float64(voteCount) / 100.0, // Assuming 100 eligible voters
	}

	if totalWeight > 0 {
		results["approval_rate"] = approveWeight / totalWeight
	}

	return results, nil
}
