package main

import "testing"

func TestParseWeightsSpecJSON(t *testing.T) {
	m := parseWeightsSpec(`{"http://a":2.5, "http://b": 0.5}`)
	if len(m) != 2 {
		t.Fatalf("expected 2 entries, got %d", len(m))
	}
	if m["http://a"] != 2.5 {
		t.Fatalf("wrong weight for a: %v", m["http://a"])
	}
	if m["http://b"] != 0.5 {
		t.Fatalf("wrong weight for b: %v", m["http://b"])
	}
}

func TestParseWeightsSpecCSV(t *testing.T) {
	m := parseWeightsSpec("http://a=1.0, http://b=3.0, http://c=0")
	if len(m) != 2 {
		t.Fatalf("expected 2 valid entries (c ignored), got %d", len(m))
	}
	if m["http://a"] != 1.0 {
		t.Fatalf("wrong weight for a: %v", m["http://a"])
	}
	if m["http://b"] != 3.0 {
		t.Fatalf("wrong weight for b: %v", m["http://b"])
	}
}
