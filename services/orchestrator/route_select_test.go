package main

import (
	"net/http/httptest"
	"testing"
	"time"
)

// helper to make backend with health/EWMA/conns
func mkBackend(url string, healthy bool, ewma float64, conns int64, weight float64) *Backend {
	b := &Backend{URL: url, Weight: weight}
	b.Healthy.Store(healthy)
	b.setEWMA(ewma)
	b.Conns = conns
	return b
}

func TestPickBackendLeastConn(t *testing.T) {
	p := &Pool{name: "t", backends: []*Backend{
		mkBackend("http://a", true, 100, 5, 1.0),
		mkBackend("http://b", true, 100, 1, 1.0),
	}}
	b, err := pickBackend(p, LBLeastConnections, "")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if b.URL != "http://b" {
		t.Fatalf("expected b, got %s", b.URL)
	}
}

func TestPickBackendEWMA(t *testing.T) {
	p := &Pool{name: "t", backends: []*Backend{
		mkBackend("http://a", true, 200, 0, 1.0),
		mkBackend("http://b", true, 50, 0, 1.0),
	}}
	b, err := pickBackend(p, LBEWMA, "")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if b.URL != "http://b" {
		t.Fatalf("expected b (lower ewma), got %s", b.URL)
	}
}

func TestPickBackendP2CConsidersPenalty(t *testing.T) {
	// a has lower ewma but very high in-flight; b should be chosen
	p2cPenalty = 10.0
	p := &Pool{name: "t", backends: []*Backend{
		mkBackend("http://a", true, 20, 10, 1.0),
		mkBackend("http://b", true, 30, 0, 1.0),
	}}
	// try multiple times due to randomness in choice of candidates
	good := false
	for i := 0; i < 20; i++ {
		b, err := pickBackend(p, LBP2CEWMA, "")
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		if b.URL == "http://b" {
			good = true
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if !good {
		t.Fatalf("expected to choose b at least once due to cost penalty")
	}
}

func TestRendezvousDeterministic(t *testing.T) {
	p := &Pool{name: "t", backends: []*Backend{
		mkBackend("http://a", true, 0, 0, 1.0),
		mkBackend("http://b", true, 0, 0, 1.0),
		mkBackend("http://c", true, 0, 0, 1.0),
	}}
	k := "user:42"
	b1, _ := pickBackend(p, LBConsistentHash, k)
	b2, _ := pickBackend(p, LBConsistentHash, k)
	if b1.URL != b2.URL {
		t.Fatalf("rendezvous not stable: %s vs %s", b1.URL, b2.URL)
	}
}

func TestHandlePolicyEndpoint(t *testing.T) {
	// set globals minimally
	basePolicy = basePolicy // no-op; ensure compiled
	rr := httptest.NewRecorder()
	req := httptest.NewRequest("GET", "/policy", nil)
	handlePolicy(rr, req)
	if rr.Code != 200 {
		t.Fatalf("status: %d", rr.Code)
	}
	if ct := rr.Header().Get("Content-Type"); ct == "" {
		t.Fatalf("missing content-type")
	}
}
