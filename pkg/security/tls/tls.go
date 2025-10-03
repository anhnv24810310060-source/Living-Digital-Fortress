package tlsutil

import (
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"os"
	"strings"
)

func LoadServerMTLS(certFile, keyFile, caFile string) (*tls.Config, error) {
	cert, err := tls.LoadX509KeyPair(certFile, keyFile)
	if err != nil {
		return nil, err
	}
	caPEM, err := os.ReadFile(caFile)
	if err != nil {
		return nil, err
	}
	pool := x509.NewCertPool()
	if !pool.AppendCertsFromPEM(caPEM) {
		return nil, errors.New("failed to parse CA cert")
	}
	return &tls.Config{
		MinVersion:   tls.VersionTLS13,
		Certificates: []tls.Certificate{cert},
		ClientAuth:   tls.RequireAndVerifyClientCert,
		ClientCAs:    pool,
	}, nil
}

// LoadServerMTLSWithSANAllow loads a TLS1.3 mTLS server config and enforces client SAN allowlist via prefix match on URI SANs and/or DNS/IP SANs.
// allowedSANPrefixes is a comma-separated list of prefixes, e.g., "spiffe://shieldx.local/ns/default/sa/,svc-".
func LoadServerMTLSWithSANAllow(certFile, keyFile, caFile, allowedSANPrefixes string) (*tls.Config, error) {
	cfg, err := LoadServerMTLS(certFile, keyFile, caFile)
	if err != nil { return nil, err }
	prefixes := []string{}
	for _, p := range strings.Split(allowedSANPrefixes, ",") {
		t := strings.TrimSpace(p)
		if t != "" { prefixes = append(prefixes, t) }
	}
	if len(prefixes) == 0 { return cfg, nil }
	base := cfg.VerifyPeerCertificate
	cfg.VerifyPeerCertificate = func(rawCerts [][]byte, chains [][]*x509.Certificate) error {
		if base != nil {
			if err := base(rawCerts, chains); err != nil { return err }
		}
		if len(rawCerts) == 0 { return fmt.Errorf("no peer cert") }
		cert, err := x509.ParseCertificate(rawCerts[0])
		if err != nil { return err }
		// check URIs
		for _, uri := range cert.URIs {
			s := uri.String()
			for _, pref := range prefixes {
				if strings.HasPrefix(s, pref) { return nil }
			}
		}
		// check DNS/IP SANs and Subject CN (deprecated)
		for _, dns := range cert.DNSNames {
			for _, pref := range prefixes { if strings.HasPrefix(dns, pref) { return nil } }
		}
		for _, ip := range cert.IPAddresses {
			s := ip.String()
			for _, pref := range prefixes { if strings.HasPrefix(s, pref) { return nil } }
		}
		if cert.Subject.CommonName != "" {
			for _, pref := range prefixes { if strings.HasPrefix(cert.Subject.CommonName, pref) { return nil } }
		}
		return fmt.Errorf("client SAN not allowed")
	}
	return cfg, nil
}
