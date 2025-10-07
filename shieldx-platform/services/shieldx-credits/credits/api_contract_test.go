package main

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestAuthMiddleware_AllowsHealthAndMetrics(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(200) })
	mux.HandleFunc("/metrics", func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(200) })
	mux.HandleFunc("/credits/balance/x", func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(200) })
	h := withAuth("secret", mux)

	// health no auth
	r := httptest.NewRequest("GET", "/health", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if w.Result().StatusCode != 200 {
		t.Fatalf("health should be public, got %d", w.Result().StatusCode)
	}

	// metrics no auth
	r = httptest.NewRequest("GET", "/metrics", nil)
	w = httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if w.Result().StatusCode != 200 {
		t.Fatalf("metrics should be public, got %d", w.Result().StatusCode)
	}

	// protected without token
	r = httptest.NewRequest("GET", "/credits/balance/x", nil)
	w = httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if w.Result().StatusCode != 401 {
		t.Fatalf("expected 401 for protected endpoint, got %d", w.Result().StatusCode)
	}

	// protected with token
	r = httptest.NewRequest("GET", "/credits/balance/x", nil)
	r.Header.Set("Authorization", "Bearer secret")
	w = httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if w.Result().StatusCode == 401 {
		t.Fatalf("unexpected 401 with valid token")
	}
}
