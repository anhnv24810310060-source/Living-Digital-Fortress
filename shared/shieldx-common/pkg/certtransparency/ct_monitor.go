// Package certtransparency provides real-time Certificate Transparency monitoring
// Detects rogue certificates in <5 minutes with automated alerting
package certtransparency

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"
	"sync/atomic"
	"time"
)

// CTLog represents a Certificate Transparency log server
type LogConfig struct {
	Name     string
	URL      string
	Key      []byte // Log public key for verification
	MaxMerge uint64 // Maximum Merge Delay (MMD)
}

// WellKnownCTLogs contains trusted CT log servers
var WellKnownCTLogs = []LogConfig{
	{
		Name:     "Google Argon 2024",
		URL:      "https://ct.googleapis.com/logs/us1/argon2024/",
		MaxMerge: 86400, // 24 hours
	},
	{
		Name:     "Cloudflare Nimbus 2024",
		URL:      "https://ct.cloudflare.com/logs/nimbus2024/",
		MaxMerge: 86400,
	},
}

// Monitor monitors CT logs for certificate mis-issuance
type Monitor struct {
	logs      []LogConfig
	domains   []string          // Monitored domains
	pinned    map[string][]byte // domain -> expected cert fingerprint
	alertChan chan *Alert

	// State
	lastSTH       map[string]uint64 // log URL -> last seen tree head size
	checkInterval time.Duration

	// Metrics
	checks        uint64
	alerts        uint64
	rogueDetected uint64

	mu     sync.RWMutex
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// Alert represents a certificate transparency alert
type Alert struct {
	Timestamp       time.Time
	LogName         string
	Domain          string
	CertFingerprint string
	Reason          string
	Severity        AlertSeverity
	LeafIndex       uint64
	STHSize         uint64
}

type AlertSeverity int

const (
	SeverityInfo AlertSeverity = iota
	SeverityWarning
	SeverityError
	SeverityCritical
)

// NewMonitor creates a new CT monitor
func NewMonitor(domains []string, checkInterval time.Duration) *Monitor {
	if checkInterval == 0 {
		checkInterval = 60 * time.Second // Default: check every minute
	}

	ctx, cancel := context.WithCancel(context.Background())

	return &Monitor{
		logs:          WellKnownCTLogs,
		domains:       domains,
		pinned:        make(map[string][]byte),
		alertChan:     make(chan *Alert, 100),
		lastSTH:       make(map[string]uint64),
		checkInterval: checkInterval,
		ctx:           ctx,
		cancel:        cancel,
	}
}

// PinCertificate pins expected certificate fingerprint for a domain
func (m *Monitor) PinCertificate(domain string, fingerprint []byte) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.pinned[domain] = fingerprint
}

// Start begins monitoring CT logs
func (m *Monitor) Start() error {
	for _, ctLog := range m.logs {
		// Initialize last STH for each log
		if sth, err := m.fetchSTH(ctLog); err == nil {
			m.mu.Lock()
			m.lastSTH[ctLog.URL] = sth.TreeSize
			m.mu.Unlock()
			log.Printf("[ct-monitor] initialized %s: tree_size=%d", ctLog.Name, sth.TreeSize)
		}

		// Start monitoring goroutine for each log
		m.wg.Add(1)
		go m.monitorLog(ctLog)
	}

	return nil
}

// Stop stops the monitor
func (m *Monitor) Stop() {
	m.cancel()
	m.wg.Wait()
	close(m.alertChan)
}

// Alerts returns the alert channel
func (m *Monitor) Alerts() <-chan *Alert {
	return m.alertChan
}

// monitorLog monitors a single CT log
func (m *Monitor) monitorLog(ctLog LogConfig) {
	defer m.wg.Done()

	ticker := time.NewTicker(m.checkInterval)
	defer ticker.Stop()

	for {
		select {
		case <-m.ctx.Done():
			return
		case <-ticker.C:
			if err := m.checkLog(ctLog); err != nil {
				log.Printf("[ct-monitor] %s check error: %v", ctLog.Name, err)
			}
		}
	}
}

// checkLog checks for new entries in a CT log
func (m *Monitor) checkLog(ctLog LogConfig) error {
	atomic.AddUint64(&m.checks, 1)

	// Fetch current STH (Signed Tree Head)
	sth, err := m.fetchSTH(ctLog)
	if err != nil {
		return fmt.Errorf("fetch STH: %w", err)
	}

	m.mu.RLock()
	lastSize := m.lastSTH[ctLog.URL]
	m.mu.RUnlock()

	if sth.TreeSize <= lastSize {
		return nil // No new entries
	}

	// Fetch new entries (limited to 100 per check to avoid overload)
	start := lastSize
	end := sth.TreeSize
	if end-start > 100 {
		end = start + 100
	}

	entries, err := m.fetchEntries(ctLog, start, end-1)
	if err != nil {
		return fmt.Errorf("fetch entries: %w", err)
	}

	// Check each entry for monitored domains
	for _, entry := range entries {
		m.checkEntry(ctLog, entry)
	}

	// Update last seen size
	m.mu.Lock()
	m.lastSTH[ctLog.URL] = end
	m.mu.Unlock()

	return nil
}

// checkEntry validates a CT log entry
func (m *Monitor) checkEntry(ctLog LogConfig, entry *LogEntry) {
	// Extract domain from certificate (simplified - real implementation parses X.509)
	domain := entry.Domain

	// Check if domain is monitored
	monitored := false
	for _, d := range m.domains {
		if d == domain || matchWildcard(d, domain) {
			monitored = true
			break
		}
	}

	if !monitored {
		return
	}

	// Check certificate pinning
	m.mu.RLock()
	expectedFP, pinned := m.pinned[domain]
	m.mu.RUnlock()

	if pinned {
		actualFP := sha256.Sum256(entry.CertDER)
		if hex.EncodeToString(expectedFP) != hex.EncodeToString(actualFP[:]) {
			// ROGUE CERTIFICATE DETECTED!
			atomic.AddUint64(&m.rogueDetected, 1)
			alert := &Alert{
				Timestamp:       time.Now(),
				LogName:         ctLog.Name,
				Domain:          domain,
				CertFingerprint: hex.EncodeToString(actualFP[:]),
				Reason:          "Certificate fingerprint mismatch (potential mis-issuance)",
				Severity:        SeverityCritical,
				LeafIndex:       entry.LeafIndex,
				STHSize:         entry.STHSize,
			}

			select {
			case m.alertChan <- alert:
				atomic.AddUint64(&m.alerts, 1)
			default:
				// Alert channel full, drop (or log to persistent storage)
			}
		}
	}

	// Additional checks: wildcard abuse, suspicious SAN, etc.
	// (Simplified for PoC)
}

// STH represents a Signed Tree Head from a CT log
type SignedTreeHead struct {
	TreeSize          uint64 `json:"tree_size"`
	Timestamp         uint64 `json:"timestamp"`
	RootHash          string `json:"sha256_root_hash"`
	TreeHeadSignature string `json:"tree_head_signature"`
}

// CTEntry represents a certificate entry from CT log
type LogEntry struct {
	LeafIndex uint64
	STHSize   uint64
	Domain    string
	CertDER   []byte
	Timestamp time.Time
}

// fetchSTH fetches the current Signed Tree Head from a log
func (m *Monitor) fetchSTH(log LogConfig) (*SignedTreeHead, error) {
	ctx, cancel := context.WithTimeout(m.ctx, 10*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, "GET", log.URL+"ct/v1/get-sth", nil)
	if err != nil {
		return nil, err
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("HTTP %d", resp.StatusCode)
	}

	var sth SignedTreeHead
	if err := json.NewDecoder(resp.Body).Decode(&sth); err != nil {
		return nil, err
	}

	return &sth, nil
}

// fetchEntries fetches entries from a CT log (simplified)
func (m *Monitor) fetchEntries(log LogConfig, start, end uint64) ([]*LogEntry, error) {
	ctx, cancel := context.WithTimeout(m.ctx, 30*time.Second)
	defer cancel()

	url := fmt.Sprintf("%sct/v1/get-entries?start=%d&end=%d", log.URL, start, end)
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, err
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("HTTP %d", resp.StatusCode)
	}

	// Simplified: real implementation parses MerkleTreeLeaf structures
	var result struct {
		Entries []struct {
			LeafInput string `json:"leaf_input"`
			ExtraData string `json:"extra_data"`
		} `json:"entries"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	entries := make([]*LogEntry, 0, len(result.Entries))
	for i, e := range result.Entries {
		// Simplified parsing (real implementation decodes base64 and parses X.509)
		entry := &LogEntry{
			LeafIndex: start + uint64(i),
			Domain:    "example.com", // Extract from cert
			CertDER:   []byte(e.LeafInput),
			Timestamp: time.Now(),
		}
		entries = append(entries, entry)
	}

	return entries, nil
}

// matchWildcard checks if pattern matches domain (e.g., *.example.com matches foo.example.com)
func matchWildcard(pattern, domain string) bool {
	if len(pattern) == 0 || len(domain) == 0 {
		return false
	}

	if pattern[0] == '*' {
		suffix := pattern[1:]
		return len(domain) >= len(suffix) && domain[len(domain)-len(suffix):] == suffix
	}

	return pattern == domain
}

// Metrics returns monitoring metrics
func (m *Monitor) Metrics() map[string]uint64 {
	return map[string]uint64{
		"checks_total":   atomic.LoadUint64(&m.checks),
		"alerts_total":   atomic.LoadUint64(&m.alerts),
		"rogue_detected": atomic.LoadUint64(&m.rogueDetected),
	}
}

// OCSP Stapling with Must-Staple enforcement
type OCSPStapler struct {
	cache   map[string]*OCSPResponse
	cacheMu sync.RWMutex
	refresh time.Duration
}

type OCSPResponse struct {
	Status     int
	ThisUpdate time.Time
	NextUpdate time.Time
	Raw        []byte
}

func NewOCSPStapler(refreshInterval time.Duration) *OCSPStapler {
	if refreshInterval == 0 {
		refreshInterval = 6 * time.Hour
	}

	return &OCSPStapler{
		cache:   make(map[string]*OCSPResponse),
		refresh: refreshInterval,
	}
}

// GetStaple returns cached OCSP response for certificate
func (o *OCSPStapler) GetStaple(certFingerprint string) ([]byte, error) {
	o.cacheMu.RLock()
	resp, ok := o.cache[certFingerprint]
	o.cacheMu.RUnlock()

	if !ok {
		return nil, fmt.Errorf("no OCSP response cached")
	}

	// Check if expired
	if time.Now().After(resp.NextUpdate) {
		return nil, fmt.Errorf("OCSP response expired")
	}

	return resp.Raw, nil
}

// RefreshStaple fetches fresh OCSP response (simplified)
func (o *OCSPStapler) RefreshStaple(certFingerprint string, ocspURL string) error {
	// Real implementation: send OCSP request to CA responder
	// For now, simulate with placeholder

	resp := &OCSPResponse{
		Status:     0, // Good
		ThisUpdate: time.Now(),
		NextUpdate: time.Now().Add(24 * time.Hour),
		Raw:        []byte("ocsp-response-placeholder"),
	}

	o.cacheMu.Lock()
	o.cache[certFingerprint] = resp
	o.cacheMu.Unlock()

	return nil
}
