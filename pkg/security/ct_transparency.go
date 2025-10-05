package security

// Package cttransparency implements Certificate Transparency monitoring
// Real-time monitoring of CT logs for certificate mis-issuance detection

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// CTMonitor monitors Certificate Transparency logs for domain certificates
type CTMonitor struct {
	domains       []string // domains to monitor
	logs          []string // CT log URLs
	client        *http.Client
	alertChan     chan *CertificateAlert
	mu            sync.RWMutex
	running       atomic.Bool
	checkInterval time.Duration

	// Metrics
	certsChecked  atomic.Uint64
	alertsRaised  atomic.Uint64
	lastCheckTime atomic.Int64
}

// CertificateAlert represents a suspicious certificate detected
type CertificateAlert struct {
	Timestamp       time.Time
	Domain          string
	Issuer          string
	SerialNumber    string
	NotBefore       time.Time
	NotAfter        time.Time
	SubjectAltNames []string
	Fingerprint     string
	CTLogURL        string
	Reason          string // "unexpected-issuer", "suspicious-san", "short-validity", etc.
}

// CTLogEntry represents a CT log entry (simplified)
type CTLogEntry struct {
	LeafInput string `json:"leaf_input"`
	ExtraData string `json:"extra_data"`
	Index     uint64 `json:"index"`
}

// CTLogResponse represents CT log API response
type CTLogResponse struct {
	Entries []CTLogEntry `json:"entries"`
}

// NewCTMonitor creates a new CT monitor
func NewCTMonitor(domains []string, checkInterval time.Duration) *CTMonitor {
	// Production CT logs (Google, Cloudflare, DigiCert)
	defaultLogs := []string{
		"https://ct.googleapis.com/logs/argon2024",
		"https://ct.cloudflare.com/logs/nimbus2024",
		"https://ct.digicert.com/log",
	}

	return &CTMonitor{
		domains:       domains,
		logs:          defaultLogs,
		client:        &http.Client{Timeout: 30 * time.Second},
		alertChan:     make(chan *CertificateAlert, 100),
		checkInterval: checkInterval,
	}
}

// Start begins monitoring CT logs
func (m *CTMonitor) Start(ctx context.Context) error {
	if !m.running.CompareAndSwap(false, true) {
		return fmt.Errorf("already running")
	}

	log.Printf("[ct] started monitoring %d domains across %d logs", len(m.domains), len(m.logs))

	ticker := time.NewTicker(m.checkInterval)
	defer ticker.Stop()

	// Initial check
	m.checkAllLogs(ctx)

	for {
		select {
		case <-ctx.Done():
			m.running.Store(false)
			return ctx.Err()
		case <-ticker.C:
			m.checkAllLogs(ctx)
		}
	}
}

// checkAllLogs queries all CT logs for new certificates
func (m *CTMonitor) checkAllLogs(ctx context.Context) {
	for _, logURL := range m.logs {
		go m.checkLog(ctx, logURL)
	}
	m.lastCheckTime.Store(time.Now().Unix())
}

// checkLog queries a single CT log
func (m *CTMonitor) checkLog(ctx context.Context, logURL string) {
	// Get recent entries (last 100)
	// In production, maintain cursor position per log
	endpoint := fmt.Sprintf("%s/ct/v1/get-entries?start=0&end=99", logURL)

	req, err := http.NewRequestWithContext(ctx, "GET", endpoint, nil)
	if err != nil {
		log.Printf("[ct] request error %s: %v", logURL, err)
		return
	}

	resp, err := m.client.Do(req)
	if err != nil {
		log.Printf("[ct] fetch error %s: %v", logURL, err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		log.Printf("[ct] non-200 status %s: %d", logURL, resp.StatusCode)
		return
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Printf("[ct] read error %s: %v", logURL, err)
		return
	}

	var logResp CTLogResponse
	if err := json.Unmarshal(body, &logResp); err != nil {
		log.Printf("[ct] parse error %s: %v", logURL, err)
		return
	}

	// Process entries
	for _, entry := range logResp.Entries {
		m.certsChecked.Add(1)
		m.processEntry(logURL, &entry)
	}
}

// processEntry analyzes a CT log entry for suspicious certificates
func (m *CTMonitor) processEntry(logURL string, entry *CTLogEntry) {
	// Simplified: In production, parse X.509 from leaf_input/extra_data
	// Here we demonstrate the alert logic with mock data

	// Check if certificate matches our monitored domains
	for _, domain := range m.domains {
		// Simplified match (production: parse actual cert Subject/SANs)
		if strings.Contains(entry.LeafInput, domain) {
			// Validate certificate against expected criteria
			alert := m.validateCertificate(logURL, domain, entry)
			if alert != nil {
				m.alertsRaised.Add(1)
				select {
				case m.alertChan <- alert:
				default:
					log.Printf("[ct] alert channel full, dropping alert for %s", domain)
				}
			}
		}
	}
}

// validateCertificate checks if a certificate is legitimate
func (m *CTMonitor) validateCertificate(logURL, domain string, entry *CTLogEntry) *CertificateAlert {
	// Simplified validation logic
	// In production: parse X.509, check issuer, validity period, SANs, etc.

	// Example: Detect unexpected issuer
	expectedIssuers := []string{"Let's Encrypt", "DigiCert", "Cloudflare"}
	issuer := "Unknown" // Parse from cert in production

	isExpected := false
	for _, exp := range expectedIssuers {
		if strings.Contains(issuer, exp) {
			isExpected = true
			break
		}
	}

	if !isExpected {
		return &CertificateAlert{
			Timestamp:    time.Now(),
			Domain:       domain,
			Issuer:       issuer,
			SerialNumber: fmt.Sprintf("%d", entry.Index),
			Fingerprint:  m.computeFingerprint(entry),
			CTLogURL:     logURL,
			Reason:       "unexpected-issuer",
		}
	}

	// Additional checks:
	// - Short validity period (< 90 days suspicious)
	// - Suspicious SANs (e.g., too many domains)
	// - Revocation status
	// - Known malicious patterns

	return nil
}

// computeFingerprint calculates SHA256 fingerprint of cert
func (m *CTMonitor) computeFingerprint(entry *CTLogEntry) string {
	h := sha256.Sum256([]byte(entry.LeafInput))
	return fmt.Sprintf("%x", h)
}

// Alerts returns the alert channel for consuming alerts
func (m *CTMonitor) Alerts() <-chan *CertificateAlert {
	return m.alertChan
}

// Stats returns monitoring statistics
func (m *CTMonitor) Stats() map[string]interface{} {
	return map[string]interface{}{
		"certsChecked": m.certsChecked.Load(),
		"alertsRaised": m.alertsRaised.Load(),
		"lastCheck":    time.Unix(m.lastCheckTime.Load(), 0),
		"domains":      len(m.domains),
		"logs":         len(m.logs),
		"running":      m.running.Load(),
	}
}

// CertificatePinning implements HPKP-style certificate pinning with backup pins
type CertificatePinning struct {
	mu         sync.RWMutex
	pins       map[string][]string // domain -> []pin (base64-encoded SHA256 of SPKI)
	backupPins map[string][]string // backup pins for rotation
	maxAge     time.Duration
}

// NewCertificatePinning creates a new certificate pinning manager
func NewCertificatePinning(maxAge time.Duration) *CertificatePinning {
	return &CertificatePinning{
		pins:       make(map[string][]string),
		backupPins: make(map[string][]string),
		maxAge:     maxAge,
	}
}

// AddPin adds a primary pin for a domain
func (cp *CertificatePinning) AddPin(domain, pin string) {
	cp.mu.Lock()
	cp.pins[domain] = append(cp.pins[domain], pin)
	cp.mu.Unlock()
}

// AddBackupPin adds a backup pin for rotation
func (cp *CertificatePinning) AddBackupPin(domain, pin string) {
	cp.mu.Lock()
	cp.backupPins[domain] = append(cp.backupPins[domain], pin)
	cp.mu.Unlock()
}

// Verify checks if the certificate matches pinned values
func (cp *CertificatePinning) Verify(domain, certFingerprint string) bool {
	cp.mu.RLock()
	defer cp.mu.RUnlock()

	// Check primary pins
	if pins, ok := cp.pins[domain]; ok {
		for _, pin := range pins {
			if pin == certFingerprint {
				return true
			}
		}
	}

	// Check backup pins
	if pins, ok := cp.backupPins[domain]; ok {
		for _, pin := range pins {
			if pin == certFingerprint {
				return true
			}
		}
	}

	return false
}

// RotatePins promotes backup pins to primary
func (cp *CertificatePinning) RotatePins(domain string) {
	cp.mu.Lock()
	defer cp.mu.Unlock()

	if backup, ok := cp.backupPins[domain]; ok {
		cp.pins[domain] = backup
		delete(cp.backupPins, domain)
	}
}

// OCSPStapling manages OCSP stapling with must-staple enforcement
type OCSPStapling struct {
	mu              sync.RWMutex
	responses       map[string]*OCSPResponse // cert serial -> OCSP response
	refreshInterval time.Duration
	mustStaple      bool
}

// OCSPResponse represents an OCSP response
type OCSPResponse struct {
	Status      string // "good", "revoked", "unknown"
	ProducedAt  time.Time
	NextUpdate  time.Time
	RawResponse []byte
}

// NewOCSPStapling creates a new OCSP stapling manager
func NewOCSPStapling(refreshInterval time.Duration, mustStaple bool) *OCSPStapling {
	return &OCSPStapling{
		responses:       make(map[string]*OCSPResponse),
		refreshInterval: refreshInterval,
		mustStaple:      mustStaple,
	}
}

// GetResponse returns the OCSP response for a certificate
func (os *OCSPStapling) GetResponse(serialNumber string) (*OCSPResponse, bool) {
	os.mu.RLock()
	defer os.mu.RUnlock()
	resp, ok := os.responses[serialNumber]
	return resp, ok
}

// UpdateResponse stores a fresh OCSP response
func (os *OCSPStapling) UpdateResponse(serialNumber string, resp *OCSPResponse) {
	os.mu.Lock()
	os.responses[serialNumber] = resp
	os.mu.Unlock()
}

// MustStaple returns whether OCSP stapling is required
func (os *OCSPStapling) MustStaple() bool {
	return os.mustStaple
}

// AutomatedRotation handles automatic certificate rotation
type AutomatedRotation struct {
	renewBefore   time.Duration // renew certificates X days before expiry
	notifyChannel chan string   // channel for rotation notifications
}

// NewAutomatedRotation creates a new automated rotation manager
func NewAutomatedRotation(renewBefore time.Duration) *AutomatedRotation {
	return &AutomatedRotation{
		renewBefore:   renewBefore,
		notifyChannel: make(chan string, 10),
	}
}

// CheckExpiry checks if a certificate needs rotation
func (ar *AutomatedRotation) CheckExpiry(domain string, notAfter time.Time) bool {
	renewAt := notAfter.Add(-ar.renewBefore)
	if time.Now().After(renewAt) {
		select {
		case ar.notifyChannel <- domain:
		default:
		}
		return true
	}
	return false
}

// Notifications returns the notification channel
func (ar *AutomatedRotation) Notifications() <-chan string {
	return ar.notifyChannel
}
