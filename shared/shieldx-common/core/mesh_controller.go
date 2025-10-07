package core

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"
)

type MeshController struct {
	incidents map[string]*Incident
	nodes     map[string]*Node
}

type Incident struct {
	ID       string    `json:"id"`
	NodeID   string    `json:"node_id"`
	Type     string    `json:"type"`
	Severity string    `json:"severity"`
	Status   string    `json:"status"`
	Created  time.Time `json:"created"`
}

type Node struct {
	ID       string `json:"id"`
	Name     string `json:"name"`
	Status   string `json:"status"`
	Instance string `json:"instance"`
}

type IncidentRequest struct {
	NodeID   string `json:"node_id"`
	Type     string `json:"type"`
	Severity string `json:"severity"`
}

func NewMeshController() *MeshController {
	return &MeshController{
		incidents: make(map[string]*Incident),
		nodes:     make(map[string]*Node),
	}
}

func (mc *MeshController) HandleIncident(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req IncidentRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	incident := &Incident{
		ID:       fmt.Sprintf("inc_%d", time.Now().UnixNano()),
		NodeID:   req.NodeID,
		Type:     req.Type,
		Severity: req.Severity,
		Status:   "open",
		Created:  time.Now(),
	}

	mc.incidents[incident.ID] = incident

	// Trigger recovery
	go mc.triggerRecovery(incident)

	response := map[string]interface{}{
		"success":     true,
		"incident_id": incident.ID,
		"message":     "Incident reported, recovery initiated",
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (mc *MeshController) triggerRecovery(incident *Incident) {
	log.Printf("Triggering recovery for incident %s", incident.ID)

	// Mock VM spawn
	newInstanceID := fmt.Sprintf("i-%d", time.Now().UnixNano())
	
	log.Printf("Spawning replacement VM: %s for node %s", newInstanceID, incident.NodeID)
	
	// Simulate VM creation delay
	time.Sleep(2 * time.Second)
	
	// Update node with new instance
	if node, exists := mc.nodes[incident.NodeID]; exists {
		node.Instance = newInstanceID
		node.Status = "healthy"
	} else {
		mc.nodes[incident.NodeID] = &Node{
			ID:       incident.NodeID,
			Name:     fmt.Sprintf("node-%s", incident.NodeID),
			Status:   "healthy",
			Instance: newInstanceID,
		}
	}

	// Mark incident as resolved
	incident.Status = "resolved"
	
	log.Printf("Recovery completed for incident %s, new instance: %s", incident.ID, newInstanceID)
}