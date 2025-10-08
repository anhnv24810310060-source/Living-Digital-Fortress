package marketplace

import (
	"fmt"
	"sync"
	"time"
)

type BountyType string

const (
	BugBounty BountyType = "bug"
	DecoyJam  BountyType = "decoy_jam"
	Feature   BountyType = "feature"
	Security  BountyType = "security"
)

type BountyStatus string

const (
	BountyOpen     BountyStatus = "open"
	BountyAssigned BountyStatus = "assigned"
	BountyClosed   BountyStatus = "closed"
	BountyPaid     BountyStatus = "paid"
)

type Bounty struct {
	ID          string       `json:"id"`
	Type        BountyType   `json:"type"`
	Title       string       `json:"title"`
	Description string       `json:"description"`
	Reward      uint64       `json:"reward"`
	Status      BountyStatus `json:"status"`
	Creator     string       `json:"creator"`
	Assignee    string       `json:"assignee,omitempty"`
	Deadline    time.Time    `json:"deadline"`
	CreatedAt   time.Time    `json:"created_at"`
	UpdatedAt   time.Time    `json:"updated_at"`
	Tags        []string     `json:"tags"`
	Difficulty  int          `json:"difficulty"` // 1-5 scale
}

type Submission struct {
	ID        string    `json:"id"`
	BountyID  string    `json:"bounty_id"`
	Hunter    string    `json:"hunter"`
	Content   string    `json:"content"`
	Files     []string  `json:"files"`
	Timestamp time.Time `json:"timestamp"`
	Approved  bool      `json:"approved"`
}

type DecoyJamEntry struct {
	ID            string    `json:"id"`
	Hunter        string    `json:"hunter"`
	DecoyName     string    `json:"decoy_name"`
	Description   string    `json:"description"`
	Complexity    int       `json:"complexity"`    // 1-10 scale
	Creativity    int       `json:"creativity"`    // 1-10 scale
	Effectiveness int       `json:"effectiveness"` // 1-10 scale
	SubmittedAt   time.Time `json:"submitted_at"`
	Score         float64   `json:"score"`
}

type BountyManager struct {
	bounties    map[string]*Bounty
	submissions map[string]*Submission
	decoyJams   map[string]*DecoyJamEntry
	mu          sync.RWMutex
}

func NewBountyManager() *BountyManager {
	return &BountyManager{
		bounties:    make(map[string]*Bounty),
		submissions: make(map[string]*Submission),
		decoyJams:   make(map[string]*DecoyJamEntry),
	}
}

func (bm *BountyManager) CreateBounty(bounty *Bounty) error {
	bm.mu.Lock()
	defer bm.mu.Unlock()

	if bounty.ID == "" {
		bounty.ID = fmt.Sprintf("bounty_%d", time.Now().UnixNano())
	}

	bounty.Status = BountyOpen
	bounty.CreatedAt = time.Now()
	bounty.UpdatedAt = time.Now()

	bm.bounties[bounty.ID] = bounty
	return nil
}

func (bm *BountyManager) GetBounty(id string) (*Bounty, error) {
	bm.mu.RLock()
	defer bm.mu.RUnlock()

	bounty, exists := bm.bounties[id]
	if !exists {
		return nil, fmt.Errorf("bounty not found: %s", id)
	}

	return bounty, nil
}

func (bm *BountyManager) ListBounties(bountyType BountyType, status BountyStatus) []*Bounty {
	bm.mu.RLock()
	defer bm.mu.RUnlock()

	var results []*Bounty

	for _, bounty := range bm.bounties {
		if (bountyType == "" || bounty.Type == bountyType) &&
			(status == "" || bounty.Status == status) {
			results = append(results, bounty)
		}
	}

	return results
}

func (bm *BountyManager) SubmitSolution(submission *Submission) error {
	bm.mu.Lock()
	defer bm.mu.Unlock()

	bounty, exists := bm.bounties[submission.BountyID]
	if !exists {
		return fmt.Errorf("bounty not found: %s", submission.BountyID)
	}

	if bounty.Status != BountyOpen {
		return fmt.Errorf("bounty is not open for submissions")
	}

	if submission.ID == "" {
		submission.ID = fmt.Sprintf("sub_%d", time.Now().UnixNano())
	}

	submission.Timestamp = time.Now()
	bm.submissions[submission.ID] = submission

	return nil
}

func (bm *BountyManager) ApproveSolution(submissionID string) error {
	bm.mu.Lock()
	defer bm.mu.Unlock()

	submission, exists := bm.submissions[submissionID]
	if !exists {
		return fmt.Errorf("submission not found: %s", submissionID)
	}

	bounty, exists := bm.bounties[submission.BountyID]
	if !exists {
		return fmt.Errorf("bounty not found: %s", submission.BountyID)
	}

	submission.Approved = true
	bounty.Status = BountyPaid
	bounty.Assignee = submission.Hunter
	bounty.UpdatedAt = time.Now()

	return nil
}

func (bm *BountyManager) SubmitDecoyJam(entry *DecoyJamEntry) error {
	bm.mu.Lock()
	defer bm.mu.Unlock()

	if entry.ID == "" {
		entry.ID = fmt.Sprintf("jam_%d", time.Now().UnixNano())
	}

	entry.SubmittedAt = time.Now()
	entry.Score = bm.calculateDecoyScore(entry)

	bm.decoyJams[entry.ID] = entry
	return nil
}

func (bm *BountyManager) calculateDecoyScore(entry *DecoyJamEntry) float64 {
	// Weighted scoring: 40% complexity, 30% creativity, 30% effectiveness
	return float64(entry.Complexity)*0.4 + float64(entry.Creativity)*0.3 + float64(entry.Effectiveness)*0.3
}

func (bm *BountyManager) GetDecoyJamLeaderboard(limit int) []*DecoyJamEntry {
	bm.mu.RLock()
	defer bm.mu.RUnlock()

	var entries []*DecoyJamEntry
	for _, entry := range bm.decoyJams {
		entries = append(entries, entry)
	}

	// Sort by score (bubble sort for simplicity)
	for i := 0; i < len(entries)-1; i++ {
		for j := 0; j < len(entries)-i-1; j++ {
			if entries[j].Score < entries[j+1].Score {
				entries[j], entries[j+1] = entries[j+1], entries[j]
			}
		}
	}

	if len(entries) > limit {
		entries = entries[:limit]
	}

	return entries
}

func (bm *BountyManager) GetHunterStats(hunter string) map[string]interface{} {
	bm.mu.RLock()
	defer bm.mu.RUnlock()

	stats := map[string]interface{}{
		"total_submissions":    0,
		"approved_submissions": 0,
		"total_rewards":        uint64(0),
		"decoy_jam_entries":    0,
		"avg_decoy_score":      0.0,
	}

	submissionCount := 0
	approvedCount := 0
	totalRewards := uint64(0)
	jamEntries := 0
	totalJamScore := 0.0

	for _, submission := range bm.submissions {
		if submission.Hunter == hunter {
			submissionCount++
			if submission.Approved {
				approvedCount++
				if bounty, exists := bm.bounties[submission.BountyID]; exists {
					totalRewards += bounty.Reward
				}
			}
		}
	}

	for _, entry := range bm.decoyJams {
		if entry.Hunter == hunter {
			jamEntries++
			totalJamScore += entry.Score
		}
	}

	stats["total_submissions"] = submissionCount
	stats["approved_submissions"] = approvedCount
	stats["total_rewards"] = totalRewards
	stats["decoy_jam_entries"] = jamEntries

	if jamEntries > 0 {
		stats["avg_decoy_score"] = totalJamScore / float64(jamEntries)
	}

	return stats
}
