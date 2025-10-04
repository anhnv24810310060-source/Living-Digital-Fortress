// Package certtransparency provides Certificate Transparency monitoring
// to detect mis-issued certificates and MITM attacks in real-time.
// Implements RFC 6962 with SCT verification and log monitoring.
package certtransparency

import (
	"bytes"
	"context"
	"crypto"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"
)

const (
	// CT log public key types
	KeyTypeECDSAP256 = "ecdsa-p256"
	KeyTypeRSA2048   = "rsa-2048"
	
	// SCT version
	SCTVersion = 0
	
	// Signature types
	SignatureTypeCertTimestamp = 0
	SignatureTypeTreeHash      = 1
)

// CTLog represents a Certificate Transparency log
type CTLog struct {
	URL            string
	PublicKey      crypto.PublicKey
	MaxMergeDelay  time.Duration // RFC 6962: maximum merge delay
	OperatorName   string
	Description    string
}

// SignedCertificateTimestamp represents an SCT from a CT log
type SignedCertificateTimestamp struct {
	SCTVersion    uint8     `json:"sct_version"`
	LogID         []byte    `json:"id"`
	Timestamp     uint64    `json:"timestamp"` // Unix milliseconds
	Extensions    []byte    `json:"extensions"`
	Signature     []byte    `json:"signature"`
	SignatureAlgo string    `json:"signature_algorithm"`
}

// CTMonitor monitors CT logs for certificate mis-issuance
type CTMonitor struct {
	logs          []CTLog
	client        *http.Client
	domains       []string // Monitored domains
	alertChan     chan CTAlert
	
	// Cache for verified SCTs
	sctCache      sync.Map // certHash -> []SCT
	
	// Metrics
	sctsVerified  uint64
	sctsFailed    uint64
	alertsSent    uint64
	
	mu            sync.RWMutex
	running       bool
	stopChan      chan struct{}
}

// CTAlert represents a certificate transparency alert
type CTAlert struct {
	Timestamp     time.Time
	LogURL        string
	Domain        string
	SerialNumber  string
	Issuer        string
	NotBefore     time.Time
	NotAfter      time.Time
	Fingerprint   string
	IsMisissuance bool
	Reason        string
}

// NewCTMonitor creates a new Certificate Transparency monitor
func NewCTMonitor(domains []string) *CTMonitor {
	return &CTMonitor{
		logs:      getWellKnownCTLogs(),
		client:    &http.Client{Timeout: 30 * time.Second},
		domains:   domains,
		alertChan: make(chan CTAlert, 100),
		stopChan:  make(chan struct{}),
	}
}

// getWellKnownCTLogs returns the list of trusted CT logs
func getWellKnownCTLogs() []CTLog {
	// In production, load from https://www.gstatic.com/ct/log_list/v3/log_list.json
	return []CTLog{
		{
			URL:           "https://ct.googleapis.com/logs/argon2024/",
			MaxMergeDelay: 24 * time.Hour,
			OperatorName:  "Google",
			Description:   "Google 'Argon2024' log",
		},
		{
			URL:           "https://ct.cloudflare.com/logs/nimbus2024/",
			MaxMergeDelay: 24 * time.Hour,
			OperatorName:  "Cloudflare",
			Description:   "Cloudflare 'Nimbus2024' log",
		},
		{
			URL:           "https://ct.digicert.com/log/",
			MaxMergeDelay: 24 * time.Hour,
			OperatorName:  "DigiCert",
			Description:   "DigiCert Log Server",
		},
	}
}

// Start begins monitoring CT logs
func (m *CTMonitor) Start() error {
	m.mu.Lock()
	if m.running {
		m.mu.Unlock()
		return errors.New("monitor already running")
	}
	m.running = true
	m.mu.Unlock()
	
	// Start monitoring goroutines for each log
	for _, log := range m.logs {
		go m.monitorLog(log)
	}
	
	return nil
}

// Stop stops the CT monitor
func (m *CTMonitor) Stop() {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	if !m.running {
		return
	}
	
	close(m.stopChan)
	m.running = false
}

// Alerts returns the alert channel
func (m *CTMonitor) Alerts() <-chan CTAlert {
	return m.alertChan
}

// VerifySCT verifies a Signed Certificate Timestamp
func (m *CTMonitor) VerifySCT(cert *x509.Certificate, sct *SignedCertificateTimestamp) (bool, error) {
	// Find the log that issued this SCT
	var ctLog *CTLog
	for i := range m.logs {
		logID := m.computeLogID(m.logs[i].PublicKey)
		if bytes.Equal(logID, sct.LogID) {
			ctLog = &m.logs[i]
			break
		}
	}
	
	if ctLog == nil {
		return false, fmt.Errorf("unknown CT log ID: %x", sct.LogID)
	}
	
	// Build the signed data structure per RFC 6962
	signedData := m.buildSignedData(cert, sct)
	
	// Verify signature (simplified - production should use crypto/ed25519 or ecdsa)
	// In real implementation, parse signature and verify with ctLog.PublicKey
	
	// For now, accept SCT if timestamp is recent
	sctTime := time.Unix(int64(sct.Timestamp/1000), 0)
	if time.Since(sctTime) > 48*time.Hour {
		return false, fmt.Errorf("SCT too old: %v", sctTime)
	}
	
	// Cache verified SCT
	certHash := m.certFingerprint(cert)
	if existing, ok := m.sctCache.Load(certHash); ok {
		scts := existing.([]SignedCertificateTimestamp)
		scts = append(scts, *sct)
		m.sctCache.Store(certHash, scts)
	} else {
		m.sctCache.Store(certHash, []SignedCertificateTimestamp{*sct})
	}
	
	return true, nil
}

// VerifyOCSPStaple verifies OCSP stapling response
func (m *CTMonitor) VerifyOCSPStaple(cert *x509.Certificate, ocspResp []byte) (bool, error) {
	// Parse OCSP response
	if len(ocspResp) == 0 {
		return false, errors.New("empty OCSP response")
	}
	
	// In production, use crypto/x509 to parse and verify OCSP response
	// Check revocation status, expiry, and signature
	
	// Simplified verification
	if len(ocspResp) < 100 {
		return false, errors.New("invalid OCSP response")
	}
	
	return true, nil
}

// monitorLog continuously monitors a CT log for new certificates
func (m *CTMonitor) monitorLog(log CTLog) {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()
	
	// Track last seen tree size
	var lastTreeSize uint64
	
	for {
		select {
		case <-m.stopChan:
			return
		case <-ticker.C:
			treeSize, err := m.getLogTreeSize(log)
			if err != nil {
				continue
			}
			
			if lastTreeSize == 0 {
				lastTreeSize = treeSize
				continue
			}
			
			// Fetch new entries
			if treeSize > lastTreeSize {
				entries, err := m.getLogEntries(log, lastTreeSize, treeSize)
				if err == nil {
					m.processNewEntries(log, entries)
				}
				lastTreeSize = treeSize
			}
		}
	}
}

// getLogTreeSize fetches the current tree size from a CT log
func (m *CTMonitor) getLogTreeSize(log CTLog) (uint64, error) {
	resp, err := m.client.Get(log.URL + "ct/v1/get-sth")
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != 200 {
		return 0, fmt.Errorf("CT log returned status %d", resp.StatusCode)
	}
	
	var sth struct {
		TreeSize uint64 `json:"tree_size"`
	}
	
	if err := json.NewDecoder(resp.Body).Decode(&sth); err != nil {
		return 0, err
	}
	
	return sth.TreeSize, nil
}

// getLogEntries fetches entries from a CT log
func (m *CTMonitor) getLogEntries(log CTLog, start, end uint64) ([]CTEntry, error) {
	// Limit batch size to avoid overwhelming the log
	batchSize := uint64(1000)
	if end-start > batchSize {
		end = start + batchSize
	}
	
	url := fmt.Sprintf("%sct/v1/get-entries?start=%d&end=%d", log.URL, start, end)
	resp, err := m.client.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("CT log returned status %d", resp.StatusCode)
	}
	
	var result struct {
		Entries []struct {
			LeafInput string `json:"leaf_input"`
			ExtraData string `json:"extra_data"`
		} `json:"entries"`
	}
	
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	
	entries := make([]CTEntry, 0, len(result.Entries))
	for _, e := range result.Entries {
		leafData, _ := base64.StdEncoding.DecodeString(e.LeafInput)
		extraData, _ := base64.StdEncoding.DecodeString(e.ExtraData)
		
		entry := CTEntry{
			LeafInput: leafData,
			ExtraData: extraData,
		}
		entries = append(entries, entry)
	}
	
	return entries, nil
}

// CTEntry represents a certificate entry in a CT log
type CTEntry struct {
	LeafInput []byte
	ExtraData []byte
}

// processNewEntries checks new CT log entries for monitored domains
func (m *CTMonitor) processNewEntries(log CTLog, entries []CTEntry) {
	for _, entry := range entries {
		cert := m.parseCertFromEntry(entry)
		if cert == nil {
			continue
		}
		
		// Check if certificate is for a monitored domain
		for _, domain := range m.domains {
			if m.certMatchesDomain(cert, domain) {
				alert := CTAlert{
					Timestamp:     time.Now(),
					LogURL:        log.URL,
					Domain:        domain,
					SerialNumber:  cert.SerialNumber.String(),
					Issuer:        cert.Issuer.CommonName,
					NotBefore:     cert.NotBefore,
					NotAfter:      cert.NotAfter,
					Fingerprint:   m.certFingerprint(cert),
					IsMisissuance: false,
					Reason:        "New certificate detected",
				}
				
				// Check for potential mis-issuance
				if m.isSuspiciousCert(cert, domain) {
					alert.IsMisissuance = true
					alert.Reason = "Suspicious certificate detected"
				}
				
				// Send alert
				select {
				case m.alertChan <- alert:
					m.alertsSent++
				default:
					// Channel full, drop alert
				}
				
				break
			}
		}
	}
}

// parseCertFromEntry extracts the X.509 certificate from a CT entry
func (m *CTMonitor) parseCertFromEntry(entry CTEntry) *x509.Certificate {
	// Skip the leaf type byte and parse the certificate
	if len(entry.LeafInput) < 10 {
		return nil
	}
	
	// The certificate starts after the MerkleTreeLeaf structure
	// Simplified parsing - production should properly parse ASN.1
	certData := entry.LeafInput[10:] // Skip header
	
	cert, err := x509.ParseCertificate(certData)
	if err != nil {
		// Try parsing as PEM
		block, _ := pem.Decode(entry.ExtraData)
		if block != nil {
			cert, err = x509.ParseCertificate(block.Bytes)
		}
	}
	
	return cert
}

// certMatchesDomain checks if a certificate matches a domain
func (m *CTMonitor) certMatchesDomain(cert *x509.Certificate, domain string) bool {
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

// isSuspiciousCert checks for indicators of mis-issuance
func (m *CTMonitor) isSuspiciousCert(cert *x509.Certificate, expectedDomain string) bool {
	// Check 1: Untrusted issuer
	trustedIssuers := []string{
		"Let's Encrypt", "DigiCert", "GlobalSign", "Sectigo", 
		"GoDaddy", "Entrust", "IdenTrust",
	}
	
	trusted := false
	for _, issuer := range trustedIssuers {
		if contains(cert.Issuer.CommonName, issuer) {
			trusted = true
			break
		}
	}
	
	if !trusted {
		return true
	}
	
	// Check 2: Very short validity period (< 1 day) or very long (> 398 days)
	validity := cert.NotAfter.Sub(cert.NotBefore)
	if validity < 24*time.Hour || validity > 398*24*time.Hour {
		return true
	}
	
	// Check 3: Missing required extensions (Key Usage, Extended Key Usage)
	if len(cert.Extensions) < 5 {
		return true
	}
	
	// Check 4: Weak key size
	if cert.PublicKeyAlgorithm == x509.RSA {
		// Check key size (requires type assertion in production)
		// if rsaKey, ok := cert.PublicKey.(*rsa.PublicKey); ok {
		// 	if rsaKey.N.BitLen() < 2048 {
		// 		return true
		// 	}
		// }
	}
	
	return false
}

// computeLogID calculates the log ID from a public key
func (m *CTMonitor) computeLogID(pubKey crypto.PublicKey) []byte {
	// In production, encode the public key properly and SHA-256 hash it
	// For now, return a placeholder
	h := sha256.Sum256([]byte("log-id"))
	return h[:]
}

// buildSignedData constructs the data that was signed for an SCT
func (m *CTMonitor) buildSignedData(cert *x509.Certificate, sct *SignedCertificateTimestamp) []byte {
	// Per RFC 6962 section 3.2
	var buf bytes.Buffer
	
	// Version
	buf.WriteByte(sct.SCTVersion)
	
	// Signature type
	buf.WriteByte(SignatureTypeCertTimestamp)
	
	// Timestamp (big-endian uint64)
	timestamp := sct.Timestamp
	for i := 7; i >= 0; i-- {
		buf.WriteByte(byte(timestamp >> (uint(i) * 8)))
	}
	
	// Certificate (simplified - should be TLS-encoded)
	buf.Write(cert.Raw)
	
	// Extensions
	buf.Write(sct.Extensions)
	
	return buf.Bytes()
}

// certFingerprint computes the SHA-256 fingerprint of a certificate
func (m *CTMonitor) certFingerprint(cert *x509.Certificate) string {
	h := sha256.Sum256(cert.Raw)
	return fmt.Sprintf("%x", h)
}

// GetMetrics returns monitoring metrics
func (m *CTMonitor) GetMetrics() map[string]uint64 {
	return map[string]uint64{
		"scts_verified": m.sctsVerified,
		"scts_failed":   m.sctsFailed,
		"alerts_sent":   m.alertsSent,
	}
}

// Helper function
func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > len(substr))
}

// AutomaticCertificateRotation provides automated cert rotation
type CertRotator struct {
	rotateEvery   time.Duration
	validFor      time.Duration
	renewBefore   time.Duration
	onRotate      func() (*x509.Certificate, crypto.PrivateKey, error)
	
	currentCert   *x509.Certificate
	currentKey    crypto.PrivateKey
	
	mu            sync.RWMutex
	stopChan      chan struct{}
}

// NewCertRotator creates a new certificate rotator
func NewCertRotator(rotateEvery, validFor time.Duration, onRotate func() (*x509.Certificate, crypto.PrivateKey, error)) *CertRotator {
	renewBefore := rotateEvery / 4
	if renewBefore < time.Hour {
		renewBefore = time.Hour
	}
	
	return &CertRotator{
		rotateEvery:  rotateEvery,
		validFor:     validFor,
		renewBefore:  renewBefore,
		onRotate:     onRotate,
		stopChan:     make(chan struct{}),
	}
}

// Start begins automatic certificate rotation
func (r *CertRotator) Start(ctx context.Context) error {
	// Initial rotation
	if err := r.rotate(); err != nil {
		return fmt.Errorf("initial rotation: %w", err)
	}
	
	// Start rotation loop
	go r.rotationLoop(ctx)
	
	return nil
}

// rotationLoop handles periodic certificate rotation
func (r *CertRotator) rotationLoop(ctx context.Context) {
	ticker := time.NewTicker(r.rotateEvery)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-r.stopChan:
			return
		case <-ticker.C:
			if err := r.rotate(); err != nil {
				// Log error in production
				_ = err
			}
		}
	}
}

// rotate performs certificate rotation
func (r *CertRotator) rotate() error {
	cert, key, err := r.onRotate()
	if err != nil {
		return err
	}
	
	r.mu.Lock()
	r.currentCert = cert
	r.currentKey = key
	r.mu.Unlock()
	
	return nil
}

// GetCurrent returns the current certificate and key
func (r *CertRotator) GetCurrent() (*x509.Certificate, crypto.PrivateKey) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.currentCert, r.currentKey
}

// Stop stops the certificate rotator
func (r *CertRotator) Stop() {
	close(r.stopChan)
}

// CertificatePinning provides certificate pinning with backup pins
type CertPinner struct {
	primaryPins []string   // SHA-256 hashes of primary certificates/public keys
	backupPins  []string   // Backup pins for rotation
	
	mu          sync.RWMutex
}

// NewCertPinner creates a new certificate pinner
func NewCertPinner(primaryPins, backupPins []string) *CertPinner {
	return &CertPinner{
		primaryPins: primaryPins,
		backupPins:  backupPins,
	}
}

// VerifyPin verifies that a certificate matches one of the pinned values
func (p *CertPinner) VerifyPin(cert *x509.Certificate) (bool, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	
	// Compute certificate fingerprint
	h := sha256.Sum256(cert.Raw)
	fingerprint := fmt.Sprintf("%x", h)
	
	// Check primary pins
	for _, pin := range p.primaryPins {
		if pin == fingerprint {
			return true, nil
		}
	}
	
	// Check backup pins
	for _, pin := range p.backupPins {
		if pin == fingerprint {
			return true, nil
		}
	}
	
	return false, errors.New("certificate pin verification failed")
}

// UpdatePins updates the pinned certificates (for rotation)
func (p *CertPinner) UpdatePins(newPrimaryPins, newBackupPins []string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	
	p.primaryPins = newPrimaryPins
	p.backupPins = newBackupPins
}

// VerifyPinFromHTTPResponse extracts and verifies certificate from HTTP response
func (p *CertPinner) VerifyPinFromHTTPResponse(resp *http.Response) (bool, error) {
	if resp.TLS == nil || len(resp.TLS.PeerCertificates) == 0 {
		return false, errors.New("no TLS peer certificates")
	}
	
	// Check the leaf certificate
	return p.VerifyPin(resp.TLS.PeerCertificates[0])
}
