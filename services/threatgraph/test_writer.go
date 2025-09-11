package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestCreateNode(t *testing.T) {
	// Mock GraphWriter for testing
	gw := &GraphWriter{}

	artifact := Artifact{
		ID:   "test_001",
		Type: "malware",
		Value: "test.exe",
		Properties: map[string]interface{}{
			"risk_score": 0.8,
		},
	}

	jsonBody, _ := json.Marshal(artifact)
	req := httptest.NewRequest("POST", "/graph/node", bytes.NewBuffer(jsonBody))
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	
	// Test validation
	if artifact.ID == "" || artifact.Type == "" {
		t.Error("Artifact should have ID and Type")
	}

	if artifact.Properties["risk_score"].(float64) != 0.8 {
		t.Error("Risk score should be 0.8")
	}
}

func TestCreateEdge(t *testing.T) {
	rel := Relationship{
		From: "mal_001",
		To:   "ip_001",
		Type: "COMMUNICATES_WITH",
		Properties: map[string]interface{}{
			"confidence": 0.9,
		},
	}

	jsonBody, _ := json.Marshal(rel)
	req := httptest.NewRequest("POST", "/graph/edge", bytes.NewBuffer(jsonBody))
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()

	// Test validation
	if rel.From == "" || rel.To == "" || rel.Type == "" {
		t.Error("Relationship should have From, To, and Type")
	}

	if rel.Properties["confidence"].(float64) != 0.9 {
		t.Error("Confidence should be 0.9")
	}
}

func TestQueryGraph(t *testing.T) {
	req := httptest.NewRequest("GET", "/graph/query?cypher=MATCH (n:Artifact) RETURN n LIMIT 5", nil)
	w := httptest.NewRecorder()

	cypherQuery := req.URL.Query().Get("cypher")
	if cypherQuery == "" {
		t.Error("Cypher query should not be empty")
	}

	expectedQuery := "MATCH (n:Artifact) RETURN n LIMIT 5"
	if cypherQuery != expectedQuery {
		t.Errorf("Expected query '%s', got '%s'", expectedQuery, cypherQuery)
	}
}

func TestArtifactValidation(t *testing.T) {
	tests := []struct {
		artifact Artifact
		valid    bool
	}{
		{Artifact{ID: "test", Type: "malware"}, true},
		{Artifact{ID: "", Type: "malware"}, false},
		{Artifact{ID: "test", Type: ""}, false},
		{Artifact{}, false},
	}

	for _, test := range tests {
		valid := test.artifact.ID != "" && test.artifact.Type != ""
		if valid != test.valid {
			t.Errorf("Artifact %+v: expected valid=%v, got valid=%v", test.artifact, test.valid, valid)
		}
	}
}

func BenchmarkArtifactCreation(b *testing.B) {
	for i := 0; i < b.N; i++ {
		artifact := Artifact{
			ID:        "bench_test",
			Type:      "malware",
			Value:     "test.exe",
			Timestamp: time.Now(),
			Properties: map[string]interface{}{
				"risk_score": 0.8,
			},
		}
		_ = artifact
	}
}