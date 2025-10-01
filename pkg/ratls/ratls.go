package ratls

import (
    "crypto/rand"
    "crypto/rsa"
    "crypto/tls"
    "crypto/x509"
    "crypto/x509/pkix"
    "encoding/pem"
    "fmt"
    "math/big"
    "net/url"
    "sync"
    "time"
)

// Identity represents a SPIFFE-based workload identity.
type Identity struct {
    TrustDomain string // e.g., "shieldx.local"
    Namespace   string // e.g., "default"
    Service     string // e.g., "ingress"
}

func (id Identity) SPIFFE() string {
    if id.Namespace != "" {
        return fmt.Sprintf("spiffe://%s/ns/%s/sa/%s", id.TrustDomain, id.Namespace, id.Service)
    }
    return fmt.Sprintf("spiffe://%s/%s", id.TrustDomain, id.Service)
}

// AutoIssuer issues short-lived leaf certificates signed by an in-memory CA and rotates them periodically.
type AutoIssuer struct {
    id           Identity
    caCert       *x509.Certificate
    caKey        *rsa.PrivateKey
    rotateEvery  time.Duration
    validity     time.Duration

    mu           sync.RWMutex
    currentLeaf  tls.Certificate
    roots        *x509.CertPool
    stopped      chan struct{}
}

// NewDevIssuer creates an in-memory CA and an auto-rotating issuer for the given identity.
// rotateEvery should be less than validity (e.g., 45m rotate for 60m validity).
func NewDevIssuer(id Identity, rotateEvery, validity time.Duration) (*AutoIssuer, error) {
    if rotateEvery <= 0 || validity <= 0 || rotateEvery >= validity {
        return nil, fmt.Errorf("invalid rotation/validity (rotateEvery=%s, validity=%s)", rotateEvery, validity)
    }
    // Generate CA
    caKey, err := rsa.GenerateKey(rand.Reader, 2048)
    if err != nil { return nil, fmt.Errorf("gen CA key: %w", err) }
    serial, _ := rand.Int(rand.Reader, big.NewInt(1<<62))
    caTpl := &x509.Certificate{
        SerialNumber: serial,
        Subject: pkix.Name{CommonName: "shieldx-dev-ca"},
        NotBefore: time.Now().Add(-1 * time.Minute),
        NotAfter:  time.Now().Add(365 * 24 * time.Hour),
        KeyUsage:  x509.KeyUsageCertSign | x509.KeyUsageCRLSign,
        BasicConstraintsValid: true,
        IsCA: true,
    }
    caDER, err := x509.CreateCertificate(rand.Reader, caTpl, caTpl, &caKey.PublicKey, caKey)
    if err != nil { return nil, fmt.Errorf("create CA cert: %w", err) }
    caCert, err := x509.ParseCertificate(caDER)
    if err != nil { return nil, fmt.Errorf("parse CA cert: %w", err) }

    roots := x509.NewCertPool()
    roots.AddCert(caCert)

    ai := &AutoIssuer{
        id: id,
        caCert: caCert,
        caKey: caKey,
        rotateEvery: rotateEvery,
        validity: validity,
        roots: roots,
        stopped: make(chan struct{}),
    }
    if err := ai.rotateLeaf(); err != nil {
        return nil, err
    }
    go ai.rotateLoop()
    return ai, nil
}

func (ai *AutoIssuer) rotateLoop() {
    t := time.NewTicker(ai.rotateEvery)
    defer t.Stop()
    for {
        select {
        case <-t.C:
            _ = ai.rotateLeaf()
        case <-ai.stopped:
            return
        }
    }
}

func (ai *AutoIssuer) rotateLeaf() error {
    leaf, err := ai.newLeaf()
    if err != nil { return err }
    ai.mu.Lock()
    ai.currentLeaf = leaf
    ai.mu.Unlock()
    return nil
}

func (ai *AutoIssuer) newLeaf() (tls.Certificate, error) {
    key, err := rsa.GenerateKey(rand.Reader, 2048)
    if err != nil { return tls.Certificate{}, fmt.Errorf("gen leaf key: %w", err) }
    serial, _ := rand.Int(rand.Reader, big.NewInt(1<<62))
    spiffeURI, _ := url.Parse(ai.id.SPIFFE())
    now := time.Now()
    tpl := &x509.Certificate{
        SerialNumber: serial,
        Subject: pkix.Name{CommonName: ai.id.Service},
        NotBefore: now.Add(-1 * time.Minute),
        NotAfter:  now.Add(ai.validity),
        KeyUsage:  x509.KeyUsageDigitalSignature | x509.KeyUsageKeyEncipherment,
        ExtKeyUsage: []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth, x509.ExtKeyUsageClientAuth},
        URIs:       []*url.URL{spiffeURI},
        BasicConstraintsValid: true,
    }
    der, err := x509.CreateCertificate(rand.Reader, tpl, ai.caCert, &key.PublicKey, ai.caKey)
    if err != nil { return tls.Certificate{}, fmt.Errorf("create leaf: %w", err) }
    leafPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der})
    keyPEM := pem.EncodeToMemory(&pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(key)})
    cert, err := tls.X509KeyPair(leafPEM, keyPEM)
    if err != nil { return tls.Certificate{}, fmt.Errorf("x509 keypair: %w", err) }
    return cert, nil
}

// Roots returns a cert pool with the CA cert.
func (ai *AutoIssuer) Roots() *x509.CertPool { return ai.roots }

// Stop halts rotation.
func (ai *AutoIssuer) Stop() { close(ai.stopped) }

// ServerTLSConfig returns a tls.Config that presents a short-lived cert and optionally enforces client certificates with SPIFFE trust domain check.
func (ai *AutoIssuer) ServerTLSConfig(requireClientCert bool, expectedTrustDomain string) *tls.Config {
    return &tls.Config{
        MinVersion: tls.VersionTLS13,
        ClientAuth: func() tls.ClientAuthType {
            if requireClientCert { return tls.RequireAndVerifyClientCert }
            return tls.NoClientCert
        }(),
        ClientCAs: ai.roots,
        GetCertificate: func(chi *tls.ClientHelloInfo) (*tls.Certificate, error) {
            ai.mu.RLock(); defer ai.mu.RUnlock()
            return &ai.currentLeaf, nil
        },
        VerifyPeerCertificate: func(rawCerts [][]byte, verifiedChains [][]*x509.Certificate) error {
            if !requireClientCert { return nil }
            if len(rawCerts) == 0 { return fmt.Errorf("no peer cert") }
            cert, err := x509.ParseCertificate(rawCerts[0])
            if err != nil { return fmt.Errorf("parse peer cert: %w", err) }
            // Verify chain to our CA
            if _, err := cert.Verify(x509.VerifyOptions{Roots: ai.roots}); err != nil {
                return fmt.Errorf("verify chain: %w", err)
            }
            // Enforce SPIFFE trust domain
            if expectedTrustDomain != "" {
                ok := false
                for _, u := range cert.URIs {
                    if u.Scheme == "spiffe" && u.Host == expectedTrustDomain {
                        ok = true; break
                    }
                }
                if !ok { return fmt.Errorf("peer spiffe trust domain mismatch") }
            }
            return nil
        },
    }
}

// ClientTLSConfig returns a tls.Config with client cert and the CA roots for mTLS to peers.
func (ai *AutoIssuer) ClientTLSConfig() *tls.Config {
    return &tls.Config{
        MinVersion: tls.VersionTLS13,
        RootCAs:    ai.roots,
        GetClientCertificate: func(cri *tls.CertificateRequestInfo) (*tls.Certificate, error) {
            ai.mu.RLock(); defer ai.mu.RUnlock()
            return &ai.currentLeaf, nil
        },
    }
}

// LeafNotAfter returns the NotAfter time for the current leaf certificate.
// If no current leaf exists, returns zero time and an error.
func (ai *AutoIssuer) LeafNotAfter() (time.Time, error) {
    ai.mu.RLock(); defer ai.mu.RUnlock()
    if len(ai.currentLeaf.Certificate) == 0 {
        return time.Time{}, fmt.Errorf("no current leaf")
    }
    cert, err := x509.ParseCertificate(ai.currentLeaf.Certificate[0])
    if err != nil { return time.Time{}, fmt.Errorf("parse leaf: %w", err) }
    return cert.NotAfter, nil
}

// Deprecated: LoadServerTLS loads a TLS config from cert/key PEM paths. If both empty, returns nil.
func LoadServerTLS(certPath, keyPath string) (*tls.Config, error) {
    if certPath == "" || keyPath == "" { return nil, nil }
    crt, err := tls.LoadX509KeyPair(certPath, keyPath)
    if err != nil { return nil, fmt.Errorf("load keypair: %w", err) }
    return &tls.Config{Certificates: []tls.Certificate{crt}}, nil
}



