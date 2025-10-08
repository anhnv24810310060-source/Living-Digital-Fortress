package autoheal

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestMeshController_HandleIncident(t *testing.T) {
	controller := NewMeshController()

	// Test incident request
	reqBody := IncidentRequest{
		NodeID:   "node-001",
		Type:     "node_down",
		Severity: "critical",
	}

	jsonBody, _ := json.Marshal(reqBody)
	req := httptest.NewRequest("POST", "/autoheal/incident", bytes.NewBuffer(jsonBody))
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	controller.HandleIncident(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	var response map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &response)

	if !response["success"].(bool) {
		t.Error("Expected success to be true")
	}

	if response["incident_id"].(string) == "" {
		t.Error("Expected incident_id to be set")
	}

	// Wait for recovery to complete
	time.Sleep(3 * time.Second)

	// Check if incident was processed
	if len(controller.incidents) != 1 {
		t.Errorf("Expected 1 incident, got %d", len(controller.incidents))
	}

	// Check if node was created/updated
	if len(controller.nodes) != 1 {
		t.Errorf("Expected 1 node, got %d", len(controller.nodes))
	}
}

func TestMeshController_InvalidMethod(t *testing.T) {
	controller := NewMeshController()

	req := httptest.NewRequest("GET", "/autoheal/incident", nil)
	w := httptest.NewRecorder()

	controller.HandleIncident(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("Expected status 405, got %d", w.Code)
	}
}

func TestMeshController_InvalidJSON(t *testing.T) {
	controller := NewMeshController()

	req := httptest.NewRequest("POST", "/autoheal/incident", bytes.NewBufferString("invalid json"))
	w := httptest.NewRecorder()

	controller.HandleIncident(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected status 400, got %d", w.Code)
	}
}
