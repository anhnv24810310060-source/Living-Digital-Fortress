package threatgraph
package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"sync"
	"time"

	"shieldx/shared/metrics"
	otelobs "shieldx/shared/observability/otel"
)

// ThreatGraphService manages threat intelligence graph and scoring
type ThreatGraphService struct {
	mu       sync.RWMutex
	nodes    map[string]*ThreatNode
	edges    map[string][]*ThreatEdge
	registry *metrics.Registry
}

type ThreatNode struct {
	ID         string                 `json:"id"`
	Type       string                 `json:"type"` // ip, domain, hash, behavior
	Value      string                 `json:"value"`
	Score      float64                `json:"score"`
	Confidence float64                `json:"confidence"`
	FirstSeen  time.Time              `json:"first_seen"`
	LastSeen   time.Time              `json:"last_seen"`
	Count      int                    `json:"count"`
	Tags       []string               `json:"tags,omitempty"`
	Metadata   map[string]interface{} `json:"metadata,omitempty"`
}

type ThreatEdge struct {
	From       string                 `json:"from"`
	To         string                 `json:"to"`
	Relation   string                 `json:"relation"` // connects_to, part_of, similar_to
	Weight     float64                `json:"weight"`
	Timestamp  time.Time              `json:"timestamp"`
	Metadata   map[string]interface{} `json:"metadata,omitempty"`
}

type ThreatEvent struct {
	Source     string                 `json:"source"`
	Type       string                 `json:"type"`
	Value      string                 `json:"value"`
	Severity   string                 `json:"severity"`
	Indicators []string               `json:"indicators,omitempty"`
	Metadata   map[string]interface{} `json:"metadata,omitempty"`
}

type ThreatQueryRequest struct {
	NodeID     string   `json:"node_id,omitempty"`
	NodeType   string   `json:"node_type,omitempty"`
	Tags       []string `json:"tags,omitempty"`
	MinScore   float64  `json:"min_score,omitempty"`
	MaxResults int      `json:"max_results,omitempty"`
}

type ThreatQueryResponse struct {
	Nodes   []*ThreatNode  `json:"nodes"`
	Edges   []*ThreatEdge  `json:"edges"`
	Summary map[string]int `json:"summary"`
}

func NewThreatGraphService() *ThreatGraphService {
	reg := metrics.NewRegistry("threatgraph")
	reg.RegisterCounter("threatgraph_events_total", "Total threat events processed")
	reg.RegisterCounter("threatgraph_nodes_total", "Total threat nodes in graph")
	reg.RegisterCounter("threatgraph_queries_total", "Total threat graph queries")
	reg.RegisterGauge("threatgraph_high_risk_nodes", "Number of high-risk nodes")

	return &ThreatGraphService{
		nodes:    make(map[string]*ThreatNode),
		edges:    make(map[string][]*ThreatEdge),
		registry: reg,
	}
}

func (tg *ThreatGraphService) IngestEvent(event ThreatEvent) error {
	tg.mu.Lock()
	defer tg.mu.Unlock()

	nodeID := event.Type + ":" + event.Value
	
	// Update or create node
	node, exists := tg.nodes[nodeID]
	if !exists {
		node = &ThreatNode{
			ID:        nodeID,
			Type:      event.Type,
			Value:     event.Value,
			FirstSeen: time.Now(),
			Tags:      []string{},
			Metadata:  make(map[string]interface{}),
		}
		tg.nodes[nodeID] = node
		tg.registry.IncrementCounter("threatgraph_nodes_total")
	}

	node.LastSeen = time.Now()
	node.Count++

	// Calculate score based on severity
	severityScore := map[string]float64{
		"critical": 1.0,
		"high":     0.8,
		"medium":   0.5,
		"low":      0.2,
	}
	if score, ok := severityScore[event.Severity]; ok {
		node.Score = (node.Score + score) / 2 // Moving average
	}

	// Add source metadata
	if node.Metadata == nil {
		node.Metadata = make(map[string]interface{})
	}
	node.Metadata["last_source"] = event.Source
	node.Metadata["last_severity"] = event.Severity

	// Create edges for indicators
	for _, indicator := range event.Indicators {
		indicatorID := "indicator:" + indicator
		edge := &ThreatEdge{
			From:      nodeID,
			To:        indicatorID,
			Relation:  "has_indicator",
			Weight:    node.Score,
			Timestamp: time.Now(),
		}
		tg.edges[nodeID] = append(tg.edges[nodeID], edge)
	}

	tg.registry.IncrementCounter("threatgraph_events_total")
	
	// Update high-risk gauge
	if node.Score >= 0.7 {
		tg.registry.SetGauge("threatgraph_high_risk_nodes", float64(tg.countHighRiskNodes()))
	}

	return nil
}

func (tg *ThreatGraphService) countHighRiskNodes() int {
	count := 0
	for _, node := range tg.nodes {
		if node.Score >= 0.7 {
			count++
		}
	}
	return count
}

func (tg *ThreatGraphService) Query(req ThreatQueryRequest) *ThreatQueryResponse {
	tg.mu.RLock()
	defer tg.mu.RUnlock()

	tg.registry.IncrementCounter("threatgraph_queries_total")

	nodes := []*ThreatNode{}
	edges := []*ThreatEdge{}
	summary := make(map[string]int)

	maxResults := req.MaxResults
	if maxResults <= 0 {
		maxResults = 100
	}

	// Filter nodes
	for _, node := range tg.nodes {
		if len(nodes) >= maxResults {
			break
		}

		// Filter by node ID
		if req.NodeID != "" && node.ID != req.NodeID {
			continue
		}

		// Filter by type
		if req.NodeType != "" && node.Type != req.NodeType {
			continue
		}

		// Filter by score
		if node.Score < req.MinScore {
			continue
		}

		// Filter by tags
		if len(req.Tags) > 0 {
			hasTag := false
			for _, tag := range req.Tags {
				for _, nodeTag := range node.Tags {
					if tag == nodeTag {
						hasTag = true
						break
					}
				}
			}
			if !hasTag {
				continue
			}
		}

		nodes = append(nodes, node)
		summary[node.Type]++

		// Include related edges
		if relatedEdges, ok := tg.edges[node.ID]; ok {
			edges = append(edges, relatedEdges...)
		}
	}

	return &ThreatQueryResponse{
		Nodes:   nodes,
		Edges:   edges,
		Summary: summary,
	}
}

func (tg *ThreatGraphService) handleIngest(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var event ThreatEvent
	if err := json.NewDecoder(r.Body).Decode(&event); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	if err := tg.IngestEvent(event); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusAccepted)
	json.NewEncoder(w).Encode(map[string]string{"status": "ingested"})
}

func (tg *ThreatGraphService) handleQuery(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req ThreatQueryRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	result := tg.Query(req)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

func (tg *ThreatGraphService) handleStats(w http.ResponseWriter, r *http.Request) {
	tg.mu.RLock()
	defer tg.mu.RUnlock()

	stats := map[string]interface{}{
		"total_nodes":      len(tg.nodes),
		"total_edges":      len(tg.edges),
		"high_risk_nodes":  tg.countHighRiskNodes(),
		"node_types":       make(map[string]int),
	}

	// Count by type
	nodeTypes := stats["node_types"].(map[string]int)
	for _, node := range tg.nodes {
		nodeTypes[node.Type]++
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(stats)
}

func main() {
	port := os.Getenv("THREATGRAPH_PORT")
	if port == "" {
		port = "5011"
	}

	ctx := context.Background()
	shutdown, err := otelobs.InitTracer("threatgraph")
	if err != nil {
		log.Printf("Failed to initialize tracer: %v", err)
	} else {
		defer shutdown(ctx)
	}

	service := NewThreatGraphService()

	mux := http.NewServeMux()
	mux.HandleFunc("/ingest", service.handleIngest)
	mux.HandleFunc("/query", service.handleQuery)
	mux.HandleFunc("/stats", service.handleStats)
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]string{"status": "healthy"})
	})
	mux.HandleFunc("/metrics", func(w http.ResponseWriter, r *http.Request) {
		service.registry.WriteMetrics(w)
	})

	handler := metrics.HTTPMetrics(mux, service.registry, "threatgraph")
	handler = otelobs.WrapHTTPHandler(handler, "threatgraph")

	addr := ":" + port
	log.Printf("ThreatGraph service starting on %s", addr)
	if err := http.ListenAndServe(addr, handler); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
