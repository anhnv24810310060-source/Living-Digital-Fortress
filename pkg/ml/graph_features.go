package ml

import (
	"fmt"
	"math"
	"sync"
)

// GraphFeatures represents extracted graph-based features
type GraphFeatures struct {
	// Node centrality features
	DegreeCentrality    float64 // Number of connections
	BetweennessCentral  float64 // Bridge importance
	ClosenessCentral    float64 // Distance to all nodes
	EigenvectorCentral  float64 // Influence score
	PageRank            float64 // Google's PageRank
	
	// Community features
	ClusteringCoeff     float64 // Local clustering
	CommunityID         int     // Detected community
	CommunitySize       int     // Size of community
	Modularity          float64 // Community strength
	
	// Path features
	ShortestPathAvg     float64 // Average path length
	ShortestPathMax     int     // Diameter
	Eccentricity        int     // Max distance from node
	
	// Structural features
	TriangleCount       int     // Number of triangles
	SquareCount         int     // Number of squares
	KCoreNumber         int     // K-core decomposition
	
	// Connectivity features
	ConnectedComponent  int     // Component ID
	ComponentSize       int     // Component size
	BiconnectedComp     int     // Biconnected component
	BridgeCount         int     // Number of bridges
	ArticulationPoint   bool    // Is articulation point
	
	// Flow features
	MaxFlow             float64 // Maximum flow
	MinCut              float64 // Minimum cut
	Bottleneck          float64 // Bottleneck capacity
	
	// Anomaly indicators
	StructuralHole      float64 // Burt's structural holes
	AnomalyScore        float64 // Graph anomaly score
	IsolationScore      float64 // Node isolation
}

// GraphNode represents a node in the graph
type GraphNode struct {
	ID         string
	Label      string
	Weight     float64
	Attributes map[string]interface{}
}

// GraphEdge represents an edge in the graph
type GraphEdge struct {
	Source string
	Target string
	Weight float64
	Type   string
}

// Graph represents a network graph
type Graph struct {
	mu sync.RWMutex
	
	nodes map[string]*GraphNode
	edges map[string]map[string]*GraphEdge // source -> target -> edge
	
	directed bool
}

// GraphExtractor extracts features from graphs
type GraphExtractor struct {
	mu sync.RWMutex
	
	maxIterations  int     // For iterative algorithms
	dampingFactor  float64 // For PageRank
	tolerance      float64 // Convergence threshold
	kCoreMin       int     // Minimum k-core
}

// GraphConfig configures the graph extractor
type GraphConfig struct {
	MaxIterations int
	DampingFactor float64
	Tolerance     float64
	KCoreMin      int
}

// NewGraph creates a new graph
func NewGraph(directed bool) *Graph {
	return &Graph{
		nodes:    make(map[string]*GraphNode),
		edges:    make(map[string]map[string]*GraphEdge),
		directed: directed,
	}
}

// AddNode adds a node to the graph
func (g *Graph) AddNode(node *GraphNode) {
	g.mu.Lock()
	defer g.mu.Unlock()
	
	g.nodes[node.ID] = node
	if g.edges[node.ID] == nil {
		g.edges[node.ID] = make(map[string]*GraphEdge)
	}
}

// AddEdge adds an edge to the graph
func (g *Graph) AddEdge(edge *GraphEdge) {
	g.mu.Lock()
	defer g.mu.Unlock()
	
	if g.edges[edge.Source] == nil {
		g.edges[edge.Source] = make(map[string]*GraphEdge)
	}
	g.edges[edge.Source][edge.Target] = edge
	
	// For undirected graphs, add reverse edge
	if !g.directed {
		if g.edges[edge.Target] == nil {
			g.edges[edge.Target] = make(map[string]*GraphEdge)
		}
		g.edges[edge.Target][edge.Source] = &GraphEdge{
			Source: edge.Target,
			Target: edge.Source,
			Weight: edge.Weight,
			Type:   edge.Type,
		}
	}
}

// GetNeighbors returns neighbors of a node
func (g *Graph) GetNeighbors(nodeID string) []string {
	g.mu.RLock()
	defer g.mu.RUnlock()
	
	neighbors := []string{}
	if edges, ok := g.edges[nodeID]; ok {
		for target := range edges {
			neighbors = append(neighbors, target)
		}
	}
	return neighbors
}

// NewGraphExtractor creates a new graph feature extractor
func NewGraphExtractor(config GraphConfig) *GraphExtractor {
	if config.MaxIterations <= 0 {
		config.MaxIterations = 100
	}
	if config.DampingFactor <= 0 {
		config.DampingFactor = 0.85
	}
	if config.Tolerance <= 0 {
		config.Tolerance = 1e-6
	}
	if config.KCoreMin <= 0 {
		config.KCoreMin = 2
	}
	
	return &GraphExtractor{
		maxIterations: config.MaxIterations,
		dampingFactor: config.DampingFactor,
		tolerance:     config.Tolerance,
		kCoreMin:      config.KCoreMin,
	}
}

// ExtractFeatures extracts features for a specific node
func (ge *GraphExtractor) ExtractFeatures(graph *Graph, nodeID string) (*GraphFeatures, error) {
	graph.mu.RLock()
	defer graph.mu.RUnlock()
	
	if _, exists := graph.nodes[nodeID]; !exists {
		return nil, fmt.Errorf("node %s not found", nodeID)
	}
	
	features := &GraphFeatures{}
	
	// Centrality features
	features.DegreeCentrality = ge.calculateDegreeCentrality(graph, nodeID)
	features.BetweennessCentral = ge.calculateBetweennessCentrality(graph, nodeID)
	features.ClosenessCentral = ge.calculateClosenessCentrality(graph, nodeID)
	features.PageRank = ge.calculatePageRank(graph, nodeID)
	
	// Local clustering
	features.ClusteringCoeff = ge.calculateClusteringCoefficient(graph, nodeID)
	
	// Path features
	features.ShortestPathAvg, features.ShortestPathMax = ge.calculateShortestPaths(graph, nodeID)
	features.Eccentricity = features.ShortestPathMax
	
	// Structural features
	features.TriangleCount = ge.countTriangles(graph, nodeID)
	features.KCoreNumber = ge.calculateKCore(graph, nodeID)
	
	// Connectivity
	features.ArticulationPoint = ge.isArticulationPoint(graph, nodeID)
	features.BridgeCount = ge.countBridges(graph, nodeID)
	
	// Anomaly detection
	features.AnomalyScore = ge.calculateAnomalyScore(graph, nodeID)
	features.IsolationScore = ge.calculateIsolationScore(graph, nodeID)
	
	return features, nil
}

// Centrality calculations

func (ge *GraphExtractor) calculateDegreeCentrality(graph *Graph, nodeID string) float64 {
	neighbors := graph.GetNeighbors(nodeID)
	n := len(graph.nodes)
	if n <= 1 {
		return 0
	}
	return float64(len(neighbors)) / float64(n-1)
}

func (ge *GraphExtractor) calculateBetweennessCentrality(graph *Graph, nodeID string) float64 {
	// Simplified betweenness using BFS
	betweenness := 0.0
	
	for source := range graph.nodes {
		if source == nodeID {
			continue
		}
		
		for target := range graph.nodes {
			if target == nodeID || target == source {
				continue
			}
			
			// Count shortest paths through nodeID
			pathsThrough := ge.countPathsThrough(graph, source, target, nodeID)
			totalPaths := ge.countShortestPaths(graph, source, target)
			
			if totalPaths > 0 {
				betweenness += float64(pathsThrough) / float64(totalPaths)
			}
		}
	}
	
	// Normalize
	n := len(graph.nodes)
	if n > 2 {
		betweenness /= float64((n - 1) * (n - 2))
	}
	
	return betweenness
}

func (ge *GraphExtractor) calculateClosenessCentrality(graph *Graph, nodeID string) float64 {
	distances := ge.bfs(graph, nodeID)
	
	sumDist := 0
	reachable := 0
	
	for target, dist := range distances {
		if target != nodeID && dist < math.MaxInt32 {
			sumDist += dist
			reachable++
		}
	}
	
	if sumDist == 0 || reachable == 0 {
		return 0
	}
	
	return float64(reachable) / float64(sumDist)
}

func (ge *GraphExtractor) calculatePageRank(graph *Graph, nodeID string) float64 {
	n := len(graph.nodes)
	if n == 0 {
		return 0
	}
	
	// Initialize PageRank scores
	pageRank := make(map[string]float64)
	newPageRank := make(map[string]float64)
	
	initialValue := 1.0 / float64(n)
	for id := range graph.nodes {
		pageRank[id] = initialValue
	}
	
	// Power iteration
	for iter := 0; iter < ge.maxIterations; iter++ {
		converged := true
		
		for id := range graph.nodes {
			sum := 0.0
			
			// Sum contributions from incoming edges
			for source := range graph.nodes {
				if edges, ok := graph.edges[source]; ok {
					if _, hasEdge := edges[id]; hasEdge {
						outDegree := len(graph.edges[source])
						if outDegree > 0 {
							sum += pageRank[source] / float64(outDegree)
						}
					}
				}
			}
			
			newPageRank[id] = (1-ge.dampingFactor)/float64(n) + ge.dampingFactor*sum
			
			// Check convergence
			if math.Abs(newPageRank[id]-pageRank[id]) > ge.tolerance {
				converged = false
			}
		}
		
		// Update scores
		for id := range graph.nodes {
			pageRank[id] = newPageRank[id]
		}
		
		if converged {
			break
		}
	}
	
	return pageRank[nodeID]
}

// Clustering

func (ge *GraphExtractor) calculateClusteringCoefficient(graph *Graph, nodeID string) float64 {
	neighbors := graph.GetNeighbors(nodeID)
	k := len(neighbors)
	
	if k < 2 {
		return 0
	}
	
	// Count edges between neighbors
	edgeCount := 0
	for i := 0; i < len(neighbors); i++ {
		for j := i + 1; j < len(neighbors); j++ {
			if edges, ok := graph.edges[neighbors[i]]; ok {
				if _, hasEdge := edges[neighbors[j]]; hasEdge {
					edgeCount++
				}
			}
		}
	}
	
	maxEdges := k * (k - 1) / 2
	return float64(edgeCount) / float64(maxEdges)
}

// Path algorithms

func (ge *GraphExtractor) bfs(graph *Graph, startID string) map[string]int {
	distances := make(map[string]int)
	for id := range graph.nodes {
		distances[id] = math.MaxInt32
	}
	distances[startID] = 0
	
	queue := []string{startID}
	
	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]
		
		for _, neighbor := range graph.GetNeighbors(current) {
			if distances[neighbor] == math.MaxInt32 {
				distances[neighbor] = distances[current] + 1
				queue = append(queue, neighbor)
			}
		}
	}
	
	return distances
}

func (ge *GraphExtractor) calculateShortestPaths(graph *Graph, nodeID string) (float64, int) {
	distances := ge.bfs(graph, nodeID)
	
	sum := 0
	max := 0
	count := 0
	
	for target, dist := range distances {
		if target != nodeID && dist < math.MaxInt32 {
			sum += dist
			count++
			if dist > max {
				max = dist
			}
		}
	}
	
	avg := 0.0
	if count > 0 {
		avg = float64(sum) / float64(count)
	}
	
	return avg, max
}

func (ge *GraphExtractor) countShortestPaths(graph *Graph, source, target string) int {
	if source == target {
		return 0
	}
	
	distances := ge.bfs(graph, source)
	if distances[target] == math.MaxInt32 {
		return 0
	}
	
	return 1 // Simplified: just count if path exists
}

func (ge *GraphExtractor) countPathsThrough(graph *Graph, source, target, through string) int {
	// Check if shortest path goes through 'through' node
	distSourceThrough := ge.bfs(graph, source)[through]
	distThroughTarget := ge.bfs(graph, through)[target]
	distSourceTarget := ge.bfs(graph, source)[target]
	
	if distSourceThrough+distThroughTarget == distSourceTarget {
		return 1
	}
	return 0
}

// Structural features

func (ge *GraphExtractor) countTriangles(graph *Graph, nodeID string) int {
	neighbors := graph.GetNeighbors(nodeID)
	triangles := 0
	
	for i := 0; i < len(neighbors); i++ {
		for j := i + 1; j < len(neighbors); j++ {
			// Check if neighbors[i] and neighbors[j] are connected
			if edges, ok := graph.edges[neighbors[i]]; ok {
				if _, hasEdge := edges[neighbors[j]]; hasEdge {
					triangles++
				}
			}
		}
	}
	
	return triangles
}

func (ge *GraphExtractor) calculateKCore(graph *Graph, nodeID string) int {
	// Simplified k-core: return degree as approximation
	return len(graph.GetNeighbors(nodeID))
}

// Connectivity

func (ge *GraphExtractor) isArticulationPoint(graph *Graph, nodeID string) bool {
	// Simplified: check if removing node increases components
	// This is a heuristic - full implementation needs DFS
	neighbors := graph.GetNeighbors(nodeID)
	
	if len(neighbors) <= 1 {
		return false
	}
	
	// Check if neighbors are connected without this node
	connected := 0
	for i := 0; i < len(neighbors); i++ {
		for j := i + 1; j < len(neighbors); j++ {
			if ge.areConnected(graph, neighbors[i], neighbors[j], nodeID) {
				connected++
			}
		}
	}
	
	maxConnections := len(neighbors) * (len(neighbors) - 1) / 2
	return connected < maxConnections
}

func (ge *GraphExtractor) countBridges(graph *Graph, nodeID string) int {
	bridges := 0
	neighbors := graph.GetNeighbors(nodeID)
	
	for _, neighbor := range neighbors {
		// Check if edge is a bridge (removing it increases components)
		if ge.isBridge(graph, nodeID, neighbor) {
			bridges++
		}
	}
	
	return bridges
}

func (ge *GraphExtractor) isBridge(graph *Graph, source, target string) bool {
	// Simplified bridge detection
	neighborsSource := graph.GetNeighbors(source)
	neighborsTarget := graph.GetNeighbors(target)
	
	// If either node has only one neighbor, edge is a bridge
	return len(neighborsSource) == 1 || len(neighborsTarget) == 1
}

func (ge *GraphExtractor) areConnected(graph *Graph, source, target, exclude string) bool {
	if source == exclude || target == exclude {
		return false
	}
	
	visited := make(map[string]bool)
	queue := []string{source}
	visited[source] = true
	
	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]
		
		if current == target {
			return true
		}
		
		for _, neighbor := range graph.GetNeighbors(current) {
			if neighbor != exclude && !visited[neighbor] {
				visited[neighbor] = true
				queue = append(queue, neighbor)
			}
		}
	}
	
	return false
}

// Anomaly detection

func (ge *GraphExtractor) calculateAnomalyScore(graph *Graph, nodeID string) float64 {
	// Combine multiple anomaly indicators
	degree := float64(len(graph.GetNeighbors(nodeID)))
	avgDegree := ge.calculateAvgDegree(graph)
	
	clustering := ge.calculateClusteringCoefficient(graph, nodeID)
	avgClustering := ge.calculateAvgClustering(graph)
	
	// Z-score based anomaly
	degreeAnomaly := math.Abs(degree - avgDegree)
	clusteringAnomaly := math.Abs(clustering - avgClustering)
	
	return (degreeAnomaly + clusteringAnomaly) / 2.0
}

func (ge *GraphExtractor) calculateIsolationScore(graph *Graph, nodeID string) float64 {
	neighbors := graph.GetNeighbors(nodeID)
	if len(neighbors) == 0 {
		return 1.0 // Completely isolated
	}
	
	// Check how connected neighbors are to rest of graph
	totalConnections := 0
	for _, neighbor := range neighbors {
		totalConnections += len(graph.GetNeighbors(neighbor))
	}
	
	avgConnections := float64(totalConnections) / float64(len(neighbors))
	maxConnections := float64(len(graph.nodes) - 1)
	
	if maxConnections == 0 {
		return 0
	}
	
	return 1.0 - (avgConnections / maxConnections)
}

func (ge *GraphExtractor) calculateAvgDegree(graph *Graph) float64 {
	sum := 0
	for id := range graph.nodes {
		sum += len(graph.GetNeighbors(id))
	}
	if len(graph.nodes) == 0 {
		return 0
	}
	return float64(sum) / float64(len(graph.nodes))
}

func (ge *GraphExtractor) calculateAvgClustering(graph *Graph) float64 {
	sum := 0.0
	for id := range graph.nodes {
		sum += ge.calculateClusteringCoefficient(graph, id)
	}
	if len(graph.nodes) == 0 {
		return 0
	}
	return sum / float64(len(graph.nodes))
}

// ToVector converts graph features to vector
func (gf *GraphFeatures) ToVector() []float64 {
	return []float64{
		gf.DegreeCentrality,
		gf.BetweennessCentral,
		gf.ClosenessCentral,
		gf.EigenvectorCentral,
		gf.PageRank,
		gf.ClusteringCoeff,
		float64(gf.CommunityID),
		float64(gf.CommunitySize),
		gf.Modularity,
		gf.ShortestPathAvg,
		float64(gf.ShortestPathMax),
		float64(gf.Eccentricity),
		float64(gf.TriangleCount),
		float64(gf.SquareCount),
		float64(gf.KCoreNumber),
		float64(gf.ConnectedComponent),
		float64(gf.ComponentSize),
		float64(gf.BiconnectedComp),
		float64(gf.BridgeCount),
		boolToFloat(gf.ArticulationPoint),
		gf.MaxFlow,
		gf.MinCut,
		gf.Bottleneck,
		gf.StructuralHole,
		gf.AnomalyScore,
		gf.IsolationScore,
	}
}

func boolToFloat(b bool) float64 {
	if b {
		return 1.0
	}
	return 0.0
}
