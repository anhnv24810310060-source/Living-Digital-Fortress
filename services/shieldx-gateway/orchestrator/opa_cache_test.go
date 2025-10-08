package main

import (
	"os"
	"path/filepath"
	"shieldx/shared/shieldx-common/pkg/policy"
	"testing"
)

const regoAllowAll = `package shieldx

default decision = "allow"

authz := {"decision": decision}
`

func TestEvaluateOPAWithCache(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "pol.rego")
	if err := os.WriteFile(p, []byte(regoAllowAll), 0o644); err != nil {
		t.Fatalf("write rego: %v", err)
	}
	var err error
	opaEng, err = policy.LoadOPA(p)
	if err != nil || opaEng == nil {
		t.Fatalf("load opa: %v", err)
	}
	// First call should compute and populate cache
	if act, ok := evaluateOPAWithCache("t1", "s1", "/", "1.1.1.1"); !ok || string(act) != "allow" {
		t.Fatalf("unexpected decision: %v ok=%v", act, ok)
	}
	// Second call should hit cache (behavioral check: still returns same decision)
	if act, ok := evaluateOPAWithCache("t1", "s1", "/", "1.1.1.1"); !ok || string(act) != "allow" {
		t.Fatalf("unexpected decision (2): %v ok=%v", act, ok)
	}
}
