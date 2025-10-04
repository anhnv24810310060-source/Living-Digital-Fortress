// Package certtransparency provides real-time monitoring of Certificate Transparency logs
// to detect certificate mis-issuance and attacks (RFC 6962, RFC 9162).
package certtransparency

import (
	"context"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sync"
	"sync/atomic"
	"time"
)

// CT Log operators
const (
	GooglePilotLog    = "https://ct.googleapis.com/pilot"
	GoogleAviatorLog  = "https://ct.googleapis.com/aviator"
	CloudflareCTLog   = "https://ct.cloudflare.com/logs/nimbus2023"
	DigiCertYetiLog   = "https://yeti2023.ct.digicert.com/log"
)

// Monitor configuration
type MonitorConfig struct {
	Domains         []string      // Domains to monitor
	CTLogs          []string      // CT log URLs to query
	CheckInterval   time.Duration // How often to check logs
	AlertThreshold  time.Duration // Alert if cert issued within this window
	EnableOCSP      bool          // Enable OCSP stapling checks
	EnablePinning   bool          // Enable certificate pinning
}

// Monitor watches CT logs for certificate issuance
type Monitor struct {
	cfg             MonitorConfig
	client          *http.Client
	
	// Certificate pinning
	pins            map[string][][]byte // domain -> list of pinned public key hashes
	backupPins      map[string][][]byte // backup pins
	
	// Monitoring state
	lastCheck       map[string]time.Time // log URL -> last check time
	seenCerts       map[string]bool      // cert fingerprint -> seen
	
	// Alerts
	alerts          chan Alert
	
	// Metrics
	certsFound      uint64
	alertsSent      uint64
	missIssuances   uint64
	pinViolations   uint64
	
	mu              sync.RWMutex
	ctx             context.Context
	cancel          context.CancelFunc
}

// Alert represents a certificate transparency alert
type Alert struct {
	Type        AlertType
	Domain      string
	Issuer      string
	NotBefore   time.Time
	NotAfter    time.Time
	Fingerprint string
	LogURL      string
	Message     string
	Timestamp   time.Time
}

type AlertType string

const (
	AlertTypeNewCert        AlertType = "new_certificate"
	AlertTypeMissIssuance   AlertType = "mis_issuance"
	AlertTypePinViolation   AlertType = "pin_violation"
	AlertTypeOCSPFailure    AlertType = "ocsp_failure"
)

// CTLogEntry represents an entry from a CT log
type CTLogEntry struct {
	LeafInput string `json:"leaf_input"`
	ExtraData string `json:"extra_data"`
}

// CTLogResponse is the response from a CT log query
type CTLogResponse struct {
	Entries []CTLogEntry `json:"entries"`
}

// NewMonitor creates a new CT monitor
func NewMonitor(cfg MonitorConfig) (*Monitor, error) {
	if len(cfg.Domains) == 0 {
		return nil, errors.New("at least one domain required")
	}
	if len(cfg.CTLogs) == 0 {
		// Default to Google Pilot
		cfg.CTLogs = []string{GooglePilotLog, CloudflareCTLog}
	}
	if cfg.CheckInterval == 0 {
		cfg.CheckInterval = 5 * time.Minute
	}
	if cfg.AlertThreshold == 0 {
		cfg.AlertThreshold = 1 * time.Hour
	}
	
	ctx, cancel := context.WithCancel(context.Background())
	
	m := &Monitor{
		cfg:        cfg,
		client:     &http.Client{Timeout: 30 * time.Second},
		lastCheck:  make(map[string]time.Time),
		seenCerts:  make(map[string]bool),
		pins:       make(map[string][][]byte),
		backupPins: make(map[string][][]byte),
		alerts:     make(chan Alert, 100),
		ctx:        ctx,
		cancel:     cancel,
	}
	
	return m, nil
}

// Start begins monitoring CT logs
func (m *Monitor) Start() error {
	// Start monitoring goroutines for each log
	for _, logURL := range m.cfg.CTLogs {
		go m.monitorLog(logURL)
	}
	
	// Start OCSP checker if enabled
	if m.cfg.EnableOCSP {
		go m.checkOCSP()
	}
	
	return nil
}

// Stop stops the monitor
func (m *Monitor) Stop() {
	m.cancel()
	close(m.alerts)
}

// Alerts returns the alert channel
func (m *Monitor) Alerts() <-chan Alert {
	return m.alerts
}

// monitorLog continuously monitors a single CT log
func (m *Monitor) monitorLog(logURL string) {
	ticker := time.NewTicker(m.cfg.CheckInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-m.ctx.Done():
			return
		case <-ticker.C:
			if err := m.queryLog(logURL); err != nil {
				// Log error in production
				continue
			}
		}
	}
}

// queryLog queries a CT log for new certificates
func (m *Monitor) queryLog(logURL string) error {
	m.mu.RLock()
	lastCheck := m.lastCheck[logURL]
	m.mu.RUnlock()
	
	// Query log (simplified - real implementation uses proper CT API)
	url := fmt.Sprintf("%s/ct/v1/get-entries?start=0&end=100", logURL)
	
	req, err := http.NewRequestWithContext(m.ctx, "GET", url, nil)
	if err != nil {
		return err
	}
	
	resp, err := m.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != 200 {
		return fmt.Errorf("CT log returned %d", resp.StatusCode)
	}
	
	var logResp CTLogResponse
	if err := json.NewDecoder(resp.Body).Decode(&logResp); err != nil {
		return err
	}
	
	// Process entries
	for _, entry := range logResp.Entries {
		if err := m.processEntry(entry, logURL); err != nil {
			// Log error but continue
			continue
		}
	}
	
	// Update last check time
	m.mu.Lock()
	m.lastCheck[logURL] = time.Now()
	m.mu.Unlock()
	
	return nil
}

// processEntry processes a single CT log entry
func (m *Monitor) processEntry(entry CTLogEntry, logURL string) error {
	// Decode leaf input (base64 encoded certificate)
	certData, err := base64.StdEncoding.DecodeString(entry.LeafInput)
	if err != nil {
		return err
	}
	
	// Parse certificate
	cert, err := x509.ParseCertificate(certData)
	if err != nil {
		// Try parsing as DER
		return err
	}
	
	// Calculate fingerprint
	fingerprint := sha256.Sum256(cert.Raw)
	fpStr := base64.StdEncoding.EncodeToString(fingerprint[:])
	
	// Check if already seen
	m.mu.RLock()
	seen := m.seenCerts[fpStr]
	m.mu.RUnlock()
	
	if seen {
		return nil
	}
	
	// Mark as seen
	m.mu.Lock()
	m.seenCerts[fpStr] = true
	m.mu.Unlock()
	
	atomic.AddUint64(&m.certsFound, 1)
	
	// Check if cert is for monitored domain
	for _, domain := range m.cfg.Domains {
		if matchesDomain(cert, domain) {
			// Check for suspicious issuance
			if time.Since(cert.NotBefore) < m.cfg.AlertThreshold {
				alert := Alert{
					Type:        AlertTypeNewCert,
					Domain:      domain,
					Issuer:      cert.Issuer.CommonName,
					NotBefore:   cert.NotBefore,
					NotAfter:    cert.NotAfter,
					Fingerprint: fpStr,
					LogURL:      logURL,
					Message:     "New certificate detected",
					Timestamp:   time.Now(),
				}
				
				// Check for mis-issuance indicators
				if isMisIssuance(cert, domain) {
					alert.Type = AlertTypeMissIssuance
					alert.Message = "Potential certificate mis-issuance detected"
					atomic.AddUint64(&m.missIssuances, 1)
				}
				
				// Check certificate pinning if enabled
				if m.cfg.EnablePinning {
					if !m.checkPin(cert, domain) {
						alert.Type = AlertTypePinViolation
						alert.Message = "Certificate pinning violation"
						atomic.AddUint64(&m.pinViolations, 1)
					}
				}
				
				select {
				case m.alerts <- alert:
					atomic.AddUint64(&m.alertsSent, 1)
				default:
					// Alert queue full
				}
			}
		}
	}
	
	return nil
}

// AddPin adds a certificate pin for a domain (HPKP style)
func (m *Monitor) AddPin(domain string, publicKeyHash []byte) {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	if m.pins[domain] == nil {
		m.pins[domain] = make([][]byte, 0)
	}
	m.pins[domain] = append(m.pins[domain], publicKeyHash)
}

// AddBackupPin adds a backup pin (HPKP requires 2+ pins)
func (m *Monitor) AddBackupPin(domain string, publicKeyHash []byte) {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	if m.backupPins[domain] == nil {
		m.backupPins[domain] = make([][]byte, 0)
	}
	m.backupPins[domain] = append(m.backupPins[domain], publicKeyHash)
}

// checkPin verifies a certificate against pinned public keys
func (m *Monitor) checkPin(cert *x509.Certificate, domain string) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	pins, ok := m.pins[domain]
	if !ok || len(pins) == 0 {
		return true // No pins configured
	}
	
	// Hash the certificate's public key
	pubKeyDER, err := x509.MarshalPKIXPublicKey(cert.PublicKey)
	if err != nil {
		return false
	}
	
	pubKeyHash := sha256.Sum256(pubKeyDER)
	
	// Check against primary pins
	for _, pin := range pins {
		if bytesEqual(pin, pubKeyHash[:]) {
			return true
		}
	}
	
	// Check against backup pins
	if backups, ok := m.backupPins[domain]; ok {
		for _, pin := range backups {
			if bytesEqual(pin, pubKeyHash[:]) {
				return true
			}
		}
	}
	
	return false
}

// checkOCSP periodically checks OCSP status of pinned certificates
func (m *Monitor) checkOCSP() {
	ticker := time.NewTicker(1 * time.Hour)
	defer ticker.Stop()
	
	for {
		select {
		case <-m.ctx.Done():
			return
		case <-ticker.C:
			// OCSP checking logic (not fully implemented in this version)
			// Would query OCSP responders for certificate revocation status
		}
	}
}

// Metrics returns current monitoring metrics
func (m *Monitor) Metrics() map[string]uint64 {
	return map[string]uint64{
		"certs_found":     atomic.LoadUint64(&m.certsFound),
		"alerts_sent":     atomic.LoadUint64(&m.alertsSent),
		"miss_issuances":  atomic.LoadUint64(&m.missIssuances),
		"pin_violations":  atomic.LoadUint64(&m.pinViolations),
	}
}

// RotatePin removes old pins and adds new ones (for key rotation)
func (m *Monitor) RotatePin(domain string, oldPin, newPin []byte) {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	// Remove old pin
	if pins, ok := m.pins[domain]; ok {
		filtered := make([][]byte, 0, len(pins))
		for _, p := range pins {
			if !bytesEqual(p, oldPin) {
				filtered = append(filtered, p)
			}
		}
		m.pins[domain] = filtered
	}
	
	// Add new pin
	if m.pins[domain] == nil {
		m.pins[domain] = make([][]byte, 0)
	}
	m.pins[domain] = append(m.pins[domain], newPin)
}

// ---------- Helper Functions ----------

func matchesDomain(cert *x509.Certificate, domain string) bool {
	// Check CN
	if cert.Subject.CommonName == domain {
		return true
	}
	
	// Check SANs
	for _, san := range cert.DNSNames {
		if san == domain || san == "*."+domain {
			return true
		}
	}
	
	return false
}

func isMisIssuance(cert *x509.Certificate, expectedDomain string) bool {
	// Check for suspicious indicators
	
	// 1. Unexpected issuer (not in trusted list)
	trustedIssuers := []string{"Let's Encrypt", "DigiCert", "GlobalSign", "Sectigo"}
	issuerTrusted := false
	for _, trusted := range trustedIssuers {
		if contains(cert.Issuer.CommonName, trusted) {
			issuerTrusted = true
			break
		}
	}
	if !issuerTrusted {
		return true
	}
	
	// 2. Very short validity period (< 1 day)
	if cert.NotAfter.Sub(cert.NotBefore) < 24*time.Hour {
		return true
	}
	
	// 3. Suspicious CN or SANs
	// (In production, check against expected patterns)
	
	return false
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && s[:len(substr)] == substr
}

func bytesEqual(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
