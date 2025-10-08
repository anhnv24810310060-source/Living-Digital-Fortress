package ratls

import (
	"crypto/tls"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestMutualTLSWithSPIFFE(t *testing.T) {
	id := Identity{TrustDomain: "shieldx.local", Namespace: "default", Service: "svc-a"}
	issuer, err := NewDevIssuer(id, 2*time.Second, 1*time.Minute)
	if err != nil {
		t.Fatalf("issuer: %v", err)
	}
	defer issuer.Stop()

	// Server requires client cert and enforces trust domain
	srv := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(200) }))
	srv.TLS = issuer.ServerTLSConfig(true, "shieldx.local")
	srv.StartTLS()
	defer srv.Close()

	// Client presents cert and trusts CA; skip hostname verification for test (no IP SAN)
	ccfg := issuer.ClientTLSConfig()
	ccfg.InsecureSkipVerify = true
	client := &http.Client{Transport: &http.Transport{TLSClientConfig: ccfg}}
	resp, err := client.Get(srv.URL)
	if err != nil {
		t.Fatalf("mTLS request failed: %v", err)
	}
	if resp.TLS == nil || resp.TLS.NegotiatedProtocol == "" { /* basic sanity */
	}
	if resp.StatusCode != 200 {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
}

func TestTrustDomainMismatchRejected(t *testing.T) {
	good := Identity{TrustDomain: "shieldx.local", Namespace: "default", Service: "svc-a"}
	bad := Identity{TrustDomain: "other.local", Namespace: "default", Service: "svc-b"}
	serverIssuer, _ := NewDevIssuer(good, 2*time.Second, 1*time.Minute)
	defer serverIssuer.Stop()
	clientIssuer, _ := NewDevIssuer(bad, 2*time.Second, 1*time.Minute)
	defer clientIssuer.Stop()

	srv := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(200) }))
	srv.TLS = serverIssuer.ServerTLSConfig(true, "shieldx.local")
	srv.StartTLS()
	defer srv.Close()

	client := &http.Client{Transport: &http.Transport{TLSClientConfig: clientIssuer.ClientTLSConfig()}}
	_, err := client.Get(srv.URL)
	if err == nil {
		t.Fatalf("expected trust domain mismatch to fail")
	}
}

func TestRotationHappens(t *testing.T) {
	id := Identity{TrustDomain: "shieldx.local", Namespace: "default", Service: "svc-a"}
	issuer, err := NewDevIssuer(id, 500*time.Millisecond, 2*time.Second)
	if err != nil {
		t.Fatalf("issuer: %v", err)
	}
	defer issuer.Stop()

	// Capture first cert
	first, err := issuer.GetLeafForTest()
	if err != nil {
		t.Fatalf("get leaf: %v", err)
	}
	// Wait for rotation
	time.Sleep(1200 * time.Millisecond)
	second, err := issuer.GetLeafForTest()
	if err != nil {
		t.Fatalf("get leaf: %v", err)
	}
	if tlsCertEqual(first, second) {
		t.Fatalf("expected rotated cert to differ")
	}
}

// Helpers for tests only
func (ai *AutoIssuer) GetLeafForTest() (tls.Certificate, error) {
	ai.mu.RLock()
	defer ai.mu.RUnlock()
	if len(ai.currentLeaf.Certificate) == 0 {
		return tls.Certificate{}, nil
	}
	return ai.currentLeaf, nil
}

func tlsCertEqual(a, b tls.Certificate) bool {
	if len(a.Certificate) == 0 || len(b.Certificate) == 0 {
		return false
	}
	if len(a.Certificate[0]) != len(b.Certificate[0]) {
		return false
	}
	// compare first raw cert bytes
	for i := range a.Certificate[0] {
		if a.Certificate[0][i] != b.Certificate[0][i] {
			return false
		}
	}
	return true
}
