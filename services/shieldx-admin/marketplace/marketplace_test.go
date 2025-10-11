package main

import (
    "encoding/json"
    "net/http"
    "net/http/httptest"
    "strings"
    "testing"

    "shieldx/shared/shieldx-common/pkg/marketplace"
)

/**
 * TestHandleHealth verifies that the /health endpoint returns status healthy.
 */
func TestHandleHealth(t *testing.T) {
    server := &Server{}
    req := httptest.NewRequest("GET", "/health", nil)
    w := httptest.NewRecorder()

    server.handleHealth(w, req)

    res := w.Result()
    defer res.Body.Close()

    if res.StatusCode != http.StatusOK {
        t.Fatalf("expected 200, got %d", res.StatusCode)
    }

    var resp map[string]interface{}
    if err := json.NewDecoder(res.Body).Decode(&resp); err != nil {
        t.Fatalf("invalid JSON response: %v", err)
    }
    if resp["status"] != "healthy" {
        t.Errorf("expected status 'healthy', got %v", resp["status"])
    }
    if resp["service"] != "marketplace" {
        t.Errorf("expected service 'marketplace', got %v", resp["service"])
    }
}

/**
 * TestHandlePackages_Get checks GET /packages returns a JSON array (can be empty).
 */
func TestHandlePackages_Get(t *testing.T) {
    authorPct := 0.7
    platformPct := 0.3
    realRegistry := marketplace.NewRegistry(authorPct, platformPct)
    server := &Server{registry: realRegistry}

    req := httptest.NewRequest("GET", "/packages", nil)
    w := httptest.NewRecorder()
    server.handlePackages(w, req)
    res := w.Result()
    defer res.Body.Close()

    if res.StatusCode != http.StatusOK {
        t.Fatalf("expected 200, got %d", res.StatusCode)
    }
    if ct := res.Header.Get("Content-Type"); ct != "application/json" {
        t.Errorf("expected Content-Type json, got %v", ct)
    }

    // Adjust for slice of pointers, as in your real codebase
    var pkgs []*marketplace.Package
    err := json.NewDecoder(res.Body).Decode(&pkgs)
    if err != nil {
        t.Fatalf("json decode failed: %v", err)
    }
    t.Logf("Got %d packages", len(pkgs)) // 0 is fine initially
}

/**
 * TestHandlePackages_WrongMethod checks that POST returns 405 for /packages.
 */
func TestHandlePackages_WrongMethod(t *testing.T) {
    authorPct := 0.7
    platformPct := 0.3
    realRegistry := marketplace.NewRegistry(authorPct, platformPct)
    server := &Server{registry: realRegistry}

    req := httptest.NewRequest("POST", "/packages", nil)
    w := httptest.NewRecorder()
    server.handlePackages(w, req)
    res := w.Result()
    defer res.Body.Close()

    if res.StatusCode != http.StatusMethodNotAllowed {
        t.Fatalf("expected 405, got %d", res.StatusCode)
    }
}

/**
 * TestHandlePublish_Valid checks POST /packages/publish with valid package.
 */
func TestHandlePublish_Valid(t *testing.T) {
    authorPct := 0.7
    platformPct := 0.3
    realRegistry := marketplace.NewRegistry(authorPct, platformPct)
    server := &Server{registry: realRegistry}

    pkg := marketplace.Package{ID: "pkg-test-1", Name: "Sample"}
    js, _ := json.Marshal(pkg)
    req := httptest.NewRequest("POST", "/packages/publish", strings.NewReader(string(js)))
    req.Header.Set("Content-Type", "application/json")
    w := httptest.NewRecorder()

    server.handlePublish(w, req)
    res := w.Result()
    defer res.Body.Close()

    if res.StatusCode != http.StatusOK {
        t.Fatalf("expected 200, got %d", res.StatusCode)
    }
    var resp map[string]string
    err := json.NewDecoder(res.Body).Decode(&resp)
    if err != nil {
        t.Fatalf("json decode failed: %v", err)
    }
    if resp["status"] != "published" {
        t.Errorf("expected 'published', got %v", resp["status"])
    }
    if resp["package_id"] != "pkg-test-1" {
        t.Errorf("expected package_id 'pkg-test-1', got %v", resp["package_id"])
    }
}

/**
 * TestHandlePublish_InvalidJSON checks POST /packages/publish with bad JSON fails.
 */
func TestHandlePublish_InvalidJSON(t *testing.T) {
    authorPct := 0.7
    platformPct := 0.3
    realRegistry := marketplace.NewRegistry(authorPct, platformPct)
    server := &Server{registry: realRegistry}

    badJSON := "{invalid_json"
    req := httptest.NewRequest("POST", "/packages/publish", strings.NewReader(badJSON))
    req.Header.Set("Content-Type", "application/json")
    w := httptest.NewRecorder()

    server.handlePublish(w, req)
    res := w.Result()
    defer res.Body.Close()

    if res.StatusCode != http.StatusBadRequest {
        t.Fatalf("expected 400, got %d", res.StatusCode)
    }
}

/**
 * TestHandlePublish_WrongMethod checks that GET is not allowed on /packages/publish.
 */
func TestHandlePublish_WrongMethod(t *testing.T) {
    authorPct := 0.7
    platformPct := 0.3
    realRegistry := marketplace.NewRegistry(authorPct, platformPct)
    server := &Server{registry: realRegistry}

    req := httptest.NewRequest("GET", "/packages/publish", nil)
    w := httptest.NewRecorder()
    server.handlePublish(w, req)
    res := w.Result()
    defer res.Body.Close()

    if res.StatusCode != http.StatusMethodNotAllowed {
        t.Fatalf("expected 405, got %d", res.StatusCode)
    }
}
