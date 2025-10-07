package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestMakeRLLimiter(t *testing.T) {
	rl := makeRLLimiter(2)
	hits := 0
	h := rl(func(w http.ResponseWriter, r *http.Request) { hits++; w.WriteHeader(200) })
	rr := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/", nil)
	h(rr, r)
	if rr.Code != 200 {
		t.Fatalf("want 200 got %d", rr.Code)
	}
	rr = httptest.NewRecorder()
	h(rr, r)
	if rr.Code != 200 {
		t.Fatalf("want 200 got %d", rr.Code)
	}
	rr = httptest.NewRecorder()
	h(rr, r)
	if rr.Code != http.StatusTooManyRequests {
		t.Fatalf("want 429 got %d", rr.Code)
	}
	if hits != 2 {
		t.Fatalf("hits=%d", hits)
	}
}

func TestConsumeCreditsClient(t *testing.T) {
	// Fake server: first insufficient, then success
	calls := 0
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if calls == 1 {
			w.WriteHeader(200)
			_ = json.NewEncoder(w).Encode(map[string]any{"success": false, "error": "insufficient"})
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"success": true})
	}))
	defer ts.Close()
	ok, code := consumeCredits(ts.URL, "tenant", 1)
	if ok || code != http.StatusPaymentRequired {
		t.Fatalf("want 402; ok=%v code=%d", ok, code)
	}
	ok, code = consumeCredits(ts.URL, "tenant", 1)
	if !ok || code != http.StatusOK {
		t.Fatalf("want ok; code=%d", code)
	}
}
