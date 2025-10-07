package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

// TestConsumeCreditsSuccess verifies a normal successful debit returns (true,200).
func TestConsumeCreditsSuccess(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("expected POST got %s", r.Method)
		}
		if r.URL.Path != "/credits/consume" {
			t.Fatalf("unexpected path %s", r.URL.Path)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"success": true})
	}))
	defer srv.Close()
	ok, code := consumeCredits(srv.URL, "tenantA", 5)
	if !ok || code != http.StatusOK {
		t.Fatalf("expected success 200 got ok=%v code=%d", ok, code)
	}
}

// TestConsumeCreditsInsufficient verifies insufficient credits path mapping to 402.
func TestConsumeCreditsInsufficient(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{"success": false, "error": "Insufficient balance"})
	}))
	defer srv.Close()
	ok, code := consumeCredits(srv.URL, "tenantA", 10)
	if ok || code != http.StatusPaymentRequired {
		t.Fatalf("expected insufficient (false,402) got ok=%v code=%d", ok, code)
	}
}

// TestConsumeCreditsServiceError verifies non-2xx status propagates service code.
func TestConsumeCreditsServiceError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		_ = json.NewEncoder(w).Encode(map[string]any{"success": false})
	}))
	defer srv.Close()
	ok, code := consumeCredits(srv.URL, "tenantB", 1)
	if ok || code != http.StatusServiceUnavailable {
		t.Fatalf("expected (false,503) got ok=%v code=%d", ok, code)
	}
}

// TestConsumeCreditsBadJSON verifies decode error returns BadGateway.
func TestConsumeCreditsBadJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("not-json"))
	}))
	defer srv.Close()
	ok, code := consumeCredits(srv.URL, "tenantC", 2)
	if ok || code != http.StatusBadGateway {
		t.Fatalf("expected (false,502) got ok=%v code=%d", ok, code)
	}
}
