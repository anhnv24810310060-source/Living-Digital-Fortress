package tlsutil

import (
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"os"
	"strings"
)

// LoadServerMTLS creates a TLS 1.3+ mTLS server configuration
// P0 Requirement: MinVersion MUST be TLS 1.3, client cert REQUIRED
func LoadServerMTLS(certFile, keyFile, caFile string) (*tls.Config, error) {
	cert, err := tls.LoadX509KeyPair(certFile, keyFile)
	if err != nil {
		return nil, fmt.Errorf("failed to load server cert/key: %w", err)
	}
	caPEM, err := os.ReadFile(caFile)
	if err != nil {
		return nil, fmt.Errorf("failed to read CA file: %w", err)
	}
	pool := x509.NewCertPool()
	if !pool.AppendCertsFromPEM(caPEM) {
		return nil, errors.New("failed to parse CA cert")
	}
	return &tls.Config{
		MinVersion:   tls.VersionTLS13, // P0: MUST be TLS 1.3
		Certificates: []tls.Certificate{cert},
		ClientAuth:   tls.RequireAndVerifyClientCert, // P0: mTLS required
		ClientCAs:    pool,
		CipherSuites: []uint16{
			// TLS 1.3 cipher suites (order matters for performance)
			tls.TLS_AES_128_GCM_SHA256,
			tls.TLS_AES_256_GCM_SHA384,
			tls.TLS_CHACHA20_POLY1305_SHA256,
		},
	}, nil
}

// LoadServerMTLSWithSANAllow loads a TLS1.3 mTLS server config and enforces client SAN allowlist
// P0 Requirement: SAN verification for service-to-service authentication
// allowedSANPrefixes is a comma-separated list of prefixes, e.g.:
// "spiffe://shieldx.local/ns/default/sa/orchestrator,spiffe://shieldx.local/ns/default/sa/guardian"
func LoadServerMTLSWithSANAllow(certFile, keyFile, caFile, allowedSANPrefixes string) (*tls.Config, error) {
	cfg, err := LoadServerMTLS(certFile, keyFile, caFile)
	if err != nil {
		return nil, err
	}

	// Parse allowlist
	prefixes := []string{}
	for _, p := range strings.Split(allowedSANPrefixes, ",") {
		t := strings.TrimSpace(p)
		if t != "" {
			prefixes = append(prefixes, t)
		}
	}
	if len(prefixes) == 0 {
		return cfg, nil // No allowlist = allow all valid certs
	}

	// Add SAN verification to certificate verification chain
	base := cfg.VerifyPeerCertificate
	cfg.VerifyPeerCertificate = func(rawCerts [][]byte, chains [][]*x509.Certificate) error {
		// First, run base verification (signature, expiry, etc.)
		if base != nil {
			if err := base(rawCerts, chains); err != nil {
				return fmt.Errorf("base cert verification failed: %w", err)
			}
		}

		// Verify we have at least one certificate
		if len(rawCerts) == 0 {
			return fmt.Errorf("no peer certificate presented")
		}

		// Parse the peer certificate
		cert, err := x509.ParseCertificate(rawCerts[0])
		if err != nil {
			return fmt.Errorf("failed to parse peer cert: %w", err)
		}

		// P0: Check URI SANs (preferred for SPIFFE IDs)
		for _, uri := range cert.URIs {
			s := uri.String()
			for _, pref := range prefixes {
				if strings.HasPrefix(s, pref) {
					return nil // Allowed!
				}
			}
		}

		// Also check DNS SANs
		for _, dns := range cert.DNSNames {
			for _, pref := range prefixes {
				if strings.HasPrefix(dns, pref) {
					return nil // Allowed!
				}
			}
		}

		// Check IP SANs
		for _, ip := range cert.IPAddresses {
			s := ip.String()
			for _, pref := range prefixes {
				if strings.HasPrefix(s, pref) {
					return nil // Allowed!
				}
			}
		}

		// Fallback to CN (deprecated but sometimes used)
		if cert.Subject.CommonName != "" {
			for _, pref := range prefixes {
				if strings.HasPrefix(cert.Subject.CommonName, pref) {
					return nil // Allowed!
				}
			}
		}

		// P0: DENY if no SAN matches allowlist
		return fmt.Errorf("client SAN not in allowlist (checked %d prefixes)", len(prefixes))
	}

	return cfg, nil
}

// LoadClientMTLS creates a TLS 1.3+ mTLS client configuration for service-to-service calls
func LoadClientMTLS(certFile, keyFile, caFile string) (*tls.Config, error) {
	cert, err := tls.LoadX509KeyPair(certFile, keyFile)
	if err != nil {
		return nil, fmt.Errorf("failed to load client cert/key: %w", err)
	}
	caPEM, err := os.ReadFile(caFile)
	if err != nil {
		return nil, fmt.Errorf("failed to read CA file: %w", err)
	}
	pool := x509.NewCertPool()
	if !pool.AppendCertsFromPEM(caPEM) {
		return nil, errors.New("failed to parse CA cert")
	}
	return &tls.Config{
		MinVersion:   tls.VersionTLS13, // P0: MUST be TLS 1.3
		Certificates: []tls.Certificate{cert},
		RootCAs:      pool,
		CipherSuites: []uint16{
			tls.TLS_AES_128_GCM_SHA256,
			tls.TLS_AES_256_GCM_SHA384,
			tls.TLS_CHACHA20_POLY1305_SHA256,
		},
	}, nil
}
