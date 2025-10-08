package policy

import (
	"os"
	"path/filepath"
	"testing"
)

const sampleRego = `package shieldx

default decision = "deny"

authz := {"decision": decision}

decision = "allow" {
  input.tenant == "t1"
}

route := {"service": "svc-a", "algo": "ewma", "candidates": ["http://a", "http://b"]} {
  input.scope == "prod"
}`

func TestEvaluateRoute(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "pol.rego")
	if err := os.WriteFile(p, []byte(sampleRego), 0o644); err != nil {
		t.Fatalf("write rego: %v", err)
	}
	eng, err := LoadOPA(p)
	if err != nil {
		t.Fatalf("load opa: %v", err)
	}
	out, ok, err := eng.EvaluateRoute(map[string]any{"tenant": "t1", "scope": "prod"})
	if err != nil || !ok {
		t.Fatalf("expected route: %v %v", ok, err)
	}
	if out["service"].(string) != "svc-a" {
		t.Fatalf("service mismatch: %v", out)
	}
}
