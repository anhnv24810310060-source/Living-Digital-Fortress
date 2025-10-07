package main

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"shieldx/shared/policy"
	"testing"
)

// Basic golden tests for policy.Evaluate integration semantics used by orchestrator.
// These focus on precedence: Advanced rules (first match) override allowlist / allowAll.
func TestPolicyEvaluatePrecedence(t *testing.T) {
	cfg := policy.Config{
		AllowAll: false,
		Allowed:  []policy.ScopeRule{{Tenant: "t1", Scopes: []string{"s1", "s2"}}},
		Advanced: []policy.AdvancedRule{
			{Tenant: "t1", Scopes: []string{"s1"}, PathPrefix: "/admin", Action: policy.ActionDeny},
			{Tenant: "t1", Scopes: []string{"s2"}, PathPrefix: "/slow", Action: policy.ActionTarpit},
		},
		TarpitMs: 500,
	}

	cases := []struct {
		name   string
		tenant string
		scope  string
		path   string
		want   policy.Action
	}{
		{"deny admin override", "t1", "s1", "/admin/panel", policy.ActionDeny},
		{"tarpit slow path", "t1", "s2", "/slow/op", policy.ActionTarpit},
		{"allow by allowlist", "t1", "s1", "/public", policy.ActionAllow},
		{"deny unknown scope", "t1", "s9", "/anything", policy.ActionDeny},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := policy.Evaluate(cfg, tc.tenant, tc.scope, tc.path)
			if got != tc.want {
				t.Fatalf("want %s got %s", tc.want, got)
			}
		})
	}
}

// TestCircuitBreakerOpen ensures backend transitions to OPEN after threshold failures.
func TestCircuitBreakerOpen(t *testing.T) {
	b := &Backend{URL: "http://b1", Weight: 1.0}
	b.Healthy.Store(true)
	os.Setenv("ORCH_CB_THRESHOLD", "3")
	defer os.Unsetenv("ORCH_CB_THRESHOLD")
	for i := 0; i < 3; i++ {
		recordBackendFailure(b, fmt.Errorf("err"))
	}
	if st := b.cbState.Load(); st != 1 {
		t.Fatalf("expected OPEN(1) got %d", st)
	}
	if b.cbNextProbe.Load() == 0 {
		t.Fatal("expected nextProbe set")
	}
}

// TestHealthHandlerDegraded triggers health handler with many unhealthy backends to get 503.
func TestHealthHandlerDegraded(t *testing.T) {
	// Prepare global pools map (modify then restore)
	poolsMu.Lock()
	old := pools
	pools = map[string]*Pool{}
	p := &Pool{name: "svc"}
	// 4 backends: only 1 healthy (<50%) -> degraded
	for i := 0; i < 4; i++ {
		b := &Backend{URL: fmt.Sprintf("http://b%d", i), Weight: 1.0}
		if i == 0 {
			b.Healthy.Store(true)
		} else {
			b.Healthy.Store(false)
		}
		p.backends = append(p.backends, b)
	}
	pools["svc"] = p
	poolsMu.Unlock()
	// Call handler
	rr := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	handleHealthEnhanced(rr, req)
	if rr.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected 503 degraded got %d", rr.Code)
	}
	// cleanup
	poolsMu.Lock()
	pools = old
	poolsMu.Unlock()
}

// Simple benchmark for policy evaluation under mixed rules.
func BenchmarkPolicyEvaluate(b *testing.B) {
	cfg := policy.Config{
		AllowAll: false,
		Allowed:  []policy.ScopeRule{{Tenant: "t1", Scopes: []string{"s1", "s2", "s3"}}},
	}
	for i := 0; i < 100; i++ {
		cfg.Advanced = append(cfg.Advanced, policy.AdvancedRule{Tenant: "t1", PathPrefix: fmt.Sprintf("/p/%d", i), Action: policy.ActionDeny})
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = policy.Evaluate(cfg, "t1", "s2", fmt.Sprintf("/p/%d/x", i%100))
	}
}
