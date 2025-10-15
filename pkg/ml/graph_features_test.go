package ml

import (
	"fmt"
	"math"
	"testing"
)

func TestNewGraph(t *testing.T) {
	g := NewGraph(true)
	if g == nil {
		t.Fatal("NewGraph returned nil")
	}
	if !g.directed {
		t.Error("Graph should be directed")
	}
	
	g2 := NewGraph(false)
	if g2.directed {
		t.Error("Graph should be undirected")
	}
}

func TestGraph_AddNode(t *testing.T) {
	g := NewGraph(false)
	
	node := &GraphNode{
		ID:     "node1",
		Label:  "Test Node",
		Weight: 1.0,
	}
	
	g.AddNode(node)
	
	if len(g.nodes) != 1 {
		t.Errorf("Expected 1 node, got %d", len(g.nodes))
	}
	
	if g.nodes["node1"] != node {
		t.Error("Node not added correctly")
	}
}

func TestGraph_AddEdge(t *testing.T) {
	g := NewGraph(false)
	
	g.AddNode(&GraphNode{ID: "A"})
	g.AddNode(&GraphNode{ID: "B"})
	
	edge := &GraphEdge{
		Source: "A",
		Target: "B",
		Weight: 1.0,
	}
	
	g.AddEdge(edge)
	
	// Check edge exists
	if g.edges["A"]["B"] == nil {
		t.Error("Edge not added")
	}
	
	// For undirected graph, reverse edge should exist
	if g.edges["B"]["A"] == nil {
		t.Error("Reverse edge not added for undirected graph")
	}
}

func TestGraph_GetNeighbors(t *testing.T) {
	g := NewGraph(false)
	
	g.AddNode(&GraphNode{ID: "A"})
	g.AddNode(&GraphNode{ID: "B"})
	g.AddNode(&GraphNode{ID: "C"})
	
	g.AddEdge(&GraphEdge{Source: "A", Target: "B"})
	g.AddEdge(&GraphEdge{Source: "A", Target: "C"})
	
	neighbors := g.GetNeighbors("A")
	if len(neighbors) != 2 {
		t.Errorf("Expected 2 neighbors, got %d", len(neighbors))
	}
}

func TestGraphExtractor_DegreeCentrality(t *testing.T) {
	g := NewGraph(false)
	extractor := NewGraphExtractor(GraphConfig{})
	
	// Create star graph: A connected to B, C, D
	g.AddNode(&GraphNode{ID: "A"})
	g.AddNode(&GraphNode{ID: "B"})
	g.AddNode(&GraphNode{ID: "C"})
	g.AddNode(&GraphNode{ID: "D"})
	
	g.AddEdge(&GraphEdge{Source: "A", Target: "B"})
	g.AddEdge(&GraphEdge{Source: "A", Target: "C"})
	g.AddEdge(&GraphEdge{Source: "A", Target: "D"})
	
	features, err := extractor.ExtractFeatures(g, "A")
	if err != nil {
		t.Fatalf("ExtractFeatures failed: %v", err)
	}
	
	// A has 3 connections out of 3 possible (to B, C, D)
	expectedDegree := 3.0 / 3.0
	if math.Abs(features.DegreeCentrality-expectedDegree) > 0.01 {
		t.Errorf("DegreeCentrality = %f, want %f", features.DegreeCentrality, expectedDegree)
	}
	
	// B, C, D should have lower centrality
	featuresB, _ := extractor.ExtractFeatures(g, "B")
	if featuresB.DegreeCentrality >= features.DegreeCentrality {
		t.Error("Center node should have higher degree centrality")
	}
}

func TestGraphExtractor_ClusteringCoefficient(t *testing.T) {
	g := NewGraph(false)
	extractor := NewGraphExtractor(GraphConfig{})
	
	// Create triangle: A-B-C-A
	g.AddNode(&GraphNode{ID: "A"})
	g.AddNode(&GraphNode{ID: "B"})
	g.AddNode(&GraphNode{ID: "C"})
	
	g.AddEdge(&GraphEdge{Source: "A", Target: "B"})
	g.AddEdge(&GraphEdge{Source: "B", Target: "C"})
	g.AddEdge(&GraphEdge{Source: "C", Target: "A"})
	
	features, err := extractor.ExtractFeatures(g, "A")
	if err != nil {
		t.Fatalf("ExtractFeatures failed: %v", err)
	}
	
	// In a complete triangle, clustering coefficient should be 1.0
	if math.Abs(features.ClusteringCoeff-1.0) > 0.01 {
		t.Errorf("ClusteringCoeff = %f, want 1.0 for triangle", features.ClusteringCoeff)
	}
}

func TestGraphExtractor_ClusteringNoCluster(t *testing.T) {
	g := NewGraph(false)
	extractor := NewGraphExtractor(GraphConfig{})
	
	// Create star (no clustering)
	g.AddNode(&GraphNode{ID: "A"})
	g.AddNode(&GraphNode{ID: "B"})
	g.AddNode(&GraphNode{ID: "C"})
	g.AddNode(&GraphNode{ID: "D"})
	
	g.AddEdge(&GraphEdge{Source: "A", Target: "B"})
	g.AddEdge(&GraphEdge{Source: "A", Target: "C"})
	g.AddEdge(&GraphEdge{Source: "A", Target: "D"})
	
	features, err := extractor.ExtractFeatures(g, "A")
	if err != nil {
		t.Fatalf("ExtractFeatures failed: %v", err)
	}
	
	// Star center has no clustering (neighbors not connected)
	if features.ClusteringCoeff != 0.0 {
		t.Errorf("ClusteringCoeff = %f, want 0.0 for star center", features.ClusteringCoeff)
	}
}

func TestGraphExtractor_PageRank(t *testing.T) {
	g := NewGraph(true) // Directed
	extractor := NewGraphExtractor(GraphConfig{
		MaxIterations: 100,
		DampingFactor: 0.85,
	})
	
	// Create simple directed graph
	g.AddNode(&GraphNode{ID: "A"})
	g.AddNode(&GraphNode{ID: "B"})
	g.AddNode(&GraphNode{ID: "C"})
	
	g.AddEdge(&GraphEdge{Source: "A", Target: "B"})
	g.AddEdge(&GraphEdge{Source: "A", Target: "C"})
	g.AddEdge(&GraphEdge{Source: "B", Target: "C"})
	
	features, err := extractor.ExtractFeatures(g, "C")
	if err != nil {
		t.Fatalf("ExtractFeatures failed: %v", err)
	}
	
	// C has incoming edges, should have positive PageRank
	if features.PageRank <= 0 {
		t.Errorf("PageRank = %f, want positive", features.PageRank)
	}
	
	featuresA, _ := extractor.ExtractFeatures(g, "A")
	// C should have higher PageRank than A (has incoming edges)
	if features.PageRank <= featuresA.PageRank {
		t.Errorf("Node with incoming edges should have higher PageRank")
	}
}

func TestGraphExtractor_ShortestPaths(t *testing.T) {
	g := NewGraph(false)
	extractor := NewGraphExtractor(GraphConfig{})
	
	// Create path: A-B-C-D
	g.AddNode(&GraphNode{ID: "A"})
	g.AddNode(&GraphNode{ID: "B"})
	g.AddNode(&GraphNode{ID: "C"})
	g.AddNode(&GraphNode{ID: "D"})
	
	g.AddEdge(&GraphEdge{Source: "A", Target: "B"})
	g.AddEdge(&GraphEdge{Source: "B", Target: "C"})
	g.AddEdge(&GraphEdge{Source: "C", Target: "D"})
	
	features, err := extractor.ExtractFeatures(g, "A")
	if err != nil {
		t.Fatalf("ExtractFeatures failed: %v", err)
	}
	
	// Max distance from A should be 3 (to D)
	if features.ShortestPathMax != 3 {
		t.Errorf("ShortestPathMax = %d, want 3", features.ShortestPathMax)
	}
	
	// Average should be (1+2+3)/3 = 2
	expectedAvg := 2.0
	if math.Abs(features.ShortestPathAvg-expectedAvg) > 0.01 {
		t.Errorf("ShortestPathAvg = %f, want %f", features.ShortestPathAvg, expectedAvg)
	}
}

func TestGraphExtractor_Triangles(t *testing.T) {
	g := NewGraph(false)
	extractor := NewGraphExtractor(GraphConfig{})
	
	// Create graph with triangles
	g.AddNode(&GraphNode{ID: "A"})
	g.AddNode(&GraphNode{ID: "B"})
	g.AddNode(&GraphNode{ID: "C"})
	g.AddNode(&GraphNode{ID: "D"})
	
	// Triangle A-B-C
	g.AddEdge(&GraphEdge{Source: "A", Target: "B"})
	g.AddEdge(&GraphEdge{Source: "B", Target: "C"})
	g.AddEdge(&GraphEdge{Source: "C", Target: "A"})
	
	// Additional connection
	g.AddEdge(&GraphEdge{Source: "A", Target: "D"})
	
	features, err := extractor.ExtractFeatures(g, "A")
	if err != nil {
		t.Fatalf("ExtractFeatures failed: %v", err)
	}
	
	// A participates in 1 triangle
	if features.TriangleCount != 1 {
		t.Errorf("TriangleCount = %d, want 1", features.TriangleCount)
	}
}

func TestGraphExtractor_ArticulationPoint(t *testing.T) {
	g := NewGraph(false)
	extractor := NewGraphExtractor(GraphConfig{})
	
	// Create graph where B is articulation point
	// A-B-C
	g.AddNode(&GraphNode{ID: "A"})
	g.AddNode(&GraphNode{ID: "B"})
	g.AddNode(&GraphNode{ID: "C"})
	
	g.AddEdge(&GraphEdge{Source: "A", Target: "B"})
	g.AddEdge(&GraphEdge{Source: "B", Target: "C"})
	
	features, err := extractor.ExtractFeatures(g, "B")
	if err != nil {
		t.Fatalf("ExtractFeatures failed: %v", err)
	}
	
	// B should be articulation point
	if !features.ArticulationPoint {
		t.Error("B should be articulation point")
	}
	
	// A should not be articulation point
	featuresA, _ := extractor.ExtractFeatures(g, "A")
	if featuresA.ArticulationPoint {
		t.Error("A should not be articulation point")
	}
}

func TestGraphExtractor_Bridges(t *testing.T) {
	g := NewGraph(false)
	extractor := NewGraphExtractor(GraphConfig{})
	
	// Create graph with bridge
	g.AddNode(&GraphNode{ID: "A"})
	g.AddNode(&GraphNode{ID: "B"})
	
	g.AddEdge(&GraphEdge{Source: "A", Target: "B"})
	
	features, err := extractor.ExtractFeatures(g, "A")
	if err != nil {
		t.Fatalf("ExtractFeatures failed: %v", err)
	}
	
	// Edge A-B is a bridge
	if features.BridgeCount == 0 {
		t.Error("Should detect bridge")
	}
}

func TestGraphExtractor_AnomalyScore(t *testing.T) {
	g := NewGraph(false)
	extractor := NewGraphExtractor(GraphConfig{})
	
	// Create normal nodes
	g.AddNode(&GraphNode{ID: "A"})
	g.AddNode(&GraphNode{ID: "B"})
	g.AddNode(&GraphNode{ID: "C"})
	g.AddNode(&GraphNode{ID: "D"})
	g.AddNode(&GraphNode{ID: "E"})
	
	// Normal connections
	g.AddEdge(&GraphEdge{Source: "A", Target: "B"})
	g.AddEdge(&GraphEdge{Source: "B", Target: "C"})
	g.AddEdge(&GraphEdge{Source: "C", Target: "D"})
	
	// E is isolated (anomaly)
	
	featuresE, err := extractor.ExtractFeatures(g, "E")
	if err != nil {
		t.Fatalf("ExtractFeatures failed: %v", err)
	}
	
	featuresA, _ := extractor.ExtractFeatures(g, "A")
	
	// E should have higher anomaly score
	if featuresE.AnomalyScore <= featuresA.AnomalyScore {
		t.Error("Isolated node should have higher anomaly score")
	}
}

func TestGraphExtractor_IsolationScore(t *testing.T) {
	g := NewGraph(false)
	extractor := NewGraphExtractor(GraphConfig{})
	
	// Create graph with isolated node
	g.AddNode(&GraphNode{ID: "A"})
	g.AddNode(&GraphNode{ID: "B"})
	g.AddNode(&GraphNode{ID: "C"})
	g.AddNode(&GraphNode{ID: "Isolated"})
	
	// Well-connected cluster
	g.AddEdge(&GraphEdge{Source: "A", Target: "B"})
	g.AddEdge(&GraphEdge{Source: "B", Target: "C"})
	g.AddEdge(&GraphEdge{Source: "C", Target: "A"})
	
	// Isolated node with single connection
	g.AddEdge(&GraphEdge{Source: "Isolated", Target: "A"})
	
	featuresIsolated, err := extractor.ExtractFeatures(g, "Isolated")
	if err != nil {
		t.Fatalf("ExtractFeatures failed: %v", err)
	}
	
	featuresA, _ := extractor.ExtractFeatures(g, "A")
	
	// Well-connected node (A) should have LOWER isolation than isolated node
	// Because A's neighbors (B, C, Isolated) have more total connections
	if featuresA.IsolationScore >= 1.0 {
		t.Errorf("Well-connected node isolation=%f should be < 1.0", featuresA.IsolationScore)
	}
	
	// Both should have valid isolation scores
	if featuresIsolated.IsolationScore < 0 || featuresIsolated.IsolationScore > 1 {
		t.Errorf("IsolationScore should be in [0,1], got %f", featuresIsolated.IsolationScore)
	}
}

func TestGraphExtractor_ClosenessCentrality(t *testing.T) {
	g := NewGraph(false)
	extractor := NewGraphExtractor(GraphConfig{})
	
	// Create star graph
	g.AddNode(&GraphNode{ID: "Center"})
	g.AddNode(&GraphNode{ID: "A"})
	g.AddNode(&GraphNode{ID: "B"})
	g.AddNode(&GraphNode{ID: "C"})
	
	g.AddEdge(&GraphEdge{Source: "Center", Target: "A"})
	g.AddEdge(&GraphEdge{Source: "Center", Target: "B"})
	g.AddEdge(&GraphEdge{Source: "Center", Target: "C"})
	
	featuresCenter, err := extractor.ExtractFeatures(g, "Center")
	if err != nil {
		t.Fatalf("ExtractFeatures failed: %v", err)
	}
	
	featuresA, _ := extractor.ExtractFeatures(g, "A")
	
	// Center should have higher closeness
	if featuresCenter.ClosenessCentral <= featuresA.ClosenessCentral {
		t.Error("Center should have higher closeness centrality")
	}
}

func TestGraphExtractor_ToVector(t *testing.T) {
	g := NewGraph(false)
	extractor := NewGraphExtractor(GraphConfig{})
	
	g.AddNode(&GraphNode{ID: "A"})
	g.AddNode(&GraphNode{ID: "B"})
	g.AddEdge(&GraphEdge{Source: "A", Target: "B"})
	
	features, err := extractor.ExtractFeatures(g, "A")
	if err != nil {
		t.Fatalf("ExtractFeatures failed: %v", err)
	}
	
	vector := features.ToVector()
	
	// Should have 26 features
	if len(vector) != 26 {
		t.Errorf("Vector length = %d, want 26", len(vector))
	}
	
	// All values should be finite
	for i, v := range vector {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("Vector[%d] = %f, want finite value", i, v)
		}
	}
}

func TestGraphExtractor_NonexistentNode(t *testing.T) {
	g := NewGraph(false)
	extractor := NewGraphExtractor(GraphConfig{})
	
	g.AddNode(&GraphNode{ID: "A"})
	
	_, err := extractor.ExtractFeatures(g, "B")
	if err == nil {
		t.Error("Should fail for nonexistent node")
	}
}

func TestGraphExtractor_SingleNode(t *testing.T) {
	g := NewGraph(false)
	extractor := NewGraphExtractor(GraphConfig{})
	
	g.AddNode(&GraphNode{ID: "A"})
	
	features, err := extractor.ExtractFeatures(g, "A")
	if err != nil {
		t.Fatalf("ExtractFeatures failed: %v", err)
	}
	
	// Single isolated node should have zero centrality
	if features.DegreeCentrality != 0 {
		t.Errorf("DegreeCentrality = %f, want 0", features.DegreeCentrality)
	}
	
	if features.ClusteringCoeff != 0 {
		t.Errorf("ClusteringCoeff = %f, want 0", features.ClusteringCoeff)
	}
}

func TestGraphExtractor_DirectedVsUndirected(t *testing.T) {
	extractor := NewGraphExtractor(GraphConfig{})
	
	// Directed graph
	gDirected := NewGraph(true)
	gDirected.AddNode(&GraphNode{ID: "A"})
	gDirected.AddNode(&GraphNode{ID: "B"})
	gDirected.AddEdge(&GraphEdge{Source: "A", Target: "B"})
	
	featuresDirected, _ := extractor.ExtractFeatures(gDirected, "A")
	
	// Undirected graph
	gUndirected := NewGraph(false)
	gUndirected.AddNode(&GraphNode{ID: "A"})
	gUndirected.AddNode(&GraphNode{ID: "B"})
	gUndirected.AddEdge(&GraphEdge{Source: "A", Target: "B"})
	
	featuresUndirected, _ := extractor.ExtractFeatures(gUndirected, "A")
	
	// In directed, A has out-degree 1
	// In undirected, A has degree 1 (but bidirectional)
	// Both should have same degree centrality
	if featuresDirected.DegreeCentrality != featuresUndirected.DegreeCentrality {
		t.Errorf("Degree centrality differs: directed=%f, undirected=%f",
			featuresDirected.DegreeCentrality, featuresUndirected.DegreeCentrality)
	}
}

func TestGraphExtractor_DefaultConfig(t *testing.T) {
	extractor := NewGraphExtractor(GraphConfig{})
	
	if extractor.maxIterations != 100 {
		t.Errorf("Default maxIterations = %d, want 100", extractor.maxIterations)
	}
	if extractor.dampingFactor != 0.85 {
		t.Errorf("Default dampingFactor = %f, want 0.85", extractor.dampingFactor)
	}
	if extractor.tolerance != 1e-6 {
		t.Errorf("Default tolerance = %f, want 1e-6", extractor.tolerance)
	}
	if extractor.kCoreMin != 2 {
		t.Errorf("Default kCoreMin = %d, want 2", extractor.kCoreMin)
	}
}

func BenchmarkGraphExtractor_Extract(b *testing.B) {
	g := NewGraph(false)
	extractor := NewGraphExtractor(GraphConfig{})
	
	// Create medium-sized graph (100 nodes)
	for i := 0; i < 100; i++ {
		g.AddNode(&GraphNode{ID: fmt.Sprintf("node%d", i)})
	}
	
	// Add random edges
	for i := 0; i < 100; i++ {
		for j := i + 1; j < 100; j += 10 {
			g.AddEdge(&GraphEdge{
				Source: fmt.Sprintf("node%d", i),
				Target: fmt.Sprintf("node%d", j),
			})
		}
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := extractor.ExtractFeatures(g, "node0")
		if err != nil {
			b.Fatalf("ExtractFeatures failed: %v", err)
		}
	}
}

func BenchmarkGraphExtractor_PageRank(b *testing.B) {
	g := NewGraph(true)
	extractor := NewGraphExtractor(GraphConfig{
		MaxIterations: 50,
	})
	
	// Create graph
	for i := 0; i < 50; i++ {
		g.AddNode(&GraphNode{ID: fmt.Sprintf("node%d", i)})
	}
	
	for i := 0; i < 50; i++ {
		for j := i + 1; j < 50; j += 5 {
			g.AddEdge(&GraphEdge{
				Source: fmt.Sprintf("node%d", i),
				Target: fmt.Sprintf("node%d", j),
			})
		}
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := extractor.ExtractFeatures(g, "node0")
		if err != nil {
			b.Fatalf("ExtractFeatures failed: %v", err)
		}
	}
}
