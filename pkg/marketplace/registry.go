package marketplace

import (
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

type Package struct {
	ID          string            `json:"id"`
	Name        string            `json:"name"`
	Version     string            `json:"version"`
	Author      string            `json:"author"`
	Description string            `json:"description"`
	Tags        []string          `json:"tags"`
	Price       uint64            `json:"price"` // in wei or smallest unit
	Downloads   uint64            `json:"downloads"`
	Rating      float64           `json:"rating"`
	Hash        string            `json:"hash"`
	Metadata    map[string]string `json:"metadata"`
	CreatedAt   time.Time         `json:"created_at"`
	UpdatedAt   time.Time         `json:"updated_at"`
}

type Purchase struct {
	ID        string    `json:"id"`
	PackageID string    `json:"package_id"`
	Buyer     string    `json:"buyer"`
	Price     uint64    `json:"price"`
	Timestamp time.Time `json:"timestamp"`
	TxHash    string    `json:"tx_hash"`
}

type RevenueShare struct {
	Author      string  `json:"author"`
	Platform    string  `json:"platform"`
	AuthorPct   float64 `json:"author_pct"`   // 0.7 = 70%
	PlatformPct float64 `json:"platform_pct"` // 0.3 = 30%
}

type Registry struct {
	packages     map[string]*Package
	purchases    map[string]*Purchase
	revenueShare *RevenueShare
	mu           sync.RWMutex
}

func NewRegistry(authorPct, platformPct float64) *Registry {
	return &Registry{
		packages:  make(map[string]*Package),
		purchases: make(map[string]*Purchase),
		revenueShare: &RevenueShare{
			AuthorPct:   authorPct,
			PlatformPct: platformPct,
		},
	}
}

func (r *Registry) PublishPackage(pkg *Package) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if pkg.ID == "" {
		return fmt.Errorf("package ID required")
	}

	if _, exists := r.packages[pkg.ID]; exists {
		return fmt.Errorf("package already exists: %s", pkg.ID)
	}

	pkg.CreatedAt = time.Now()
	pkg.UpdatedAt = time.Now()
	r.packages[pkg.ID] = pkg

	return nil
}

func (r *Registry) GetPackage(id string) (*Package, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	pkg, exists := r.packages[id]
	if !exists {
		return nil, fmt.Errorf("package not found: %s", id)
	}

	return pkg, nil
}

func (r *Registry) SearchPackages(query string, tags []string) []*Package {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var results []*Package

	for _, pkg := range r.packages {
		if r.matchesSearch(pkg, query, tags) {
			results = append(results, pkg)
		}
	}

	return results
}

func (r *Registry) matchesSearch(pkg *Package, query string, tags []string) bool {
	// Simple text matching
	if query != "" {
		if !contains(pkg.Name, query) && !contains(pkg.Description, query) {
			return false
		}
	}

	// Tag matching
	if len(tags) > 0 {
		for _, requiredTag := range tags {
			found := false
			for _, pkgTag := range pkg.Tags {
				if pkgTag == requiredTag {
					found = true
					break
				}
			}
			if !found {
				return false
			}
		}
	}

	return true
}

func (r *Registry) PurchasePackage(packageID, buyer string, txHash string) (*Purchase, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	pkg, exists := r.packages[packageID]
	if !exists {
		return nil, fmt.Errorf("package not found: %s", packageID)
	}

	purchase := &Purchase{
		ID:        fmt.Sprintf("purchase_%d", time.Now().UnixNano()),
		PackageID: packageID,
		Buyer:     buyer,
		Price:     pkg.Price,
		Timestamp: time.Now(),
		TxHash:    txHash,
	}

	r.purchases[purchase.ID] = purchase
	pkg.Downloads++
	pkg.UpdatedAt = time.Now()

	return purchase, nil
}

func (r *Registry) CalculateRevenue(packageID string) (authorRevenue, platformRevenue uint64, err error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var totalRevenue uint64

	for _, purchase := range r.purchases {
		if purchase.PackageID == packageID {
			totalRevenue += purchase.Price
		}
	}

	authorRevenue = uint64(float64(totalRevenue) * r.revenueShare.AuthorPct)
	platformRevenue = uint64(float64(totalRevenue) * r.revenueShare.PlatformPct)

	return authorRevenue, platformRevenue, nil
}

func (r *Registry) GetTopPackages(limit int) []*Package {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var packages []*Package
	for _, pkg := range r.packages {
		packages = append(packages, pkg)
	}

	// Sort by downloads (simple bubble sort for small datasets)
	for i := 0; i < len(packages)-1; i++ {
		for j := 0; j < len(packages)-i-1; j++ {
			if packages[j].Downloads < packages[j+1].Downloads {
				packages[j], packages[j+1] = packages[j+1], packages[j]
			}
		}
	}

	if len(packages) > limit {
		packages = packages[:limit]
	}

	return packages
}

func (r *Registry) ExportMetrics() ([]byte, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	metrics := map[string]interface{}{
		"total_packages":  len(r.packages),
		"total_purchases": len(r.purchases),
		"revenue_share":   r.revenueShare,
		"timestamp":       time.Now(),
	}

	return json.Marshal(metrics)
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || (len(s) > len(substr) &&
		(s[:len(substr)] == substr || s[len(s)-len(substr):] == substr ||
			containsMiddle(s, substr))))
}

func containsMiddle(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
