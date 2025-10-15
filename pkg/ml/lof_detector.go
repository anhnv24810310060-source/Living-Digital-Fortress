package ml

import (
	"fmt"
	"math"
	"sort"
	"sync"
)

// LOFDetector implements Local Outlier Factor anomaly detection
// More robust to local density variations than global methods
type LOFDetector struct {
	mu         sync.RWMutex
	k          int           // Number of neighbors
	trained    bool
	dataPoints [][]float64   // Training data
	threshold  float64       // LOF threshold for anomaly (typically 1.5-2.0)
}

// NewLOFDetector creates a new LOF detector
// k: number of neighbors (typically 5-20)
// threshold: LOF score threshold for anomaly detection (typically 1.5)
func NewLOFDetector(k int, threshold float64) *LOFDetector {
	return &LOFDetector{
		k:         k,
		threshold: threshold,
	}
}

// Train stores the training data for LOF calculation
func (lof *LOFDetector) Train(data [][]float64) error {
	if len(data) == 0 {
		return fmt.Errorf("no training data provided")
	}

	if len(data) < lof.k {
		return fmt.Errorf("training data size (%d) must be >= k (%d)", len(data), lof.k)
	}

	lof.mu.Lock()
	defer lof.mu.Unlock()

	// Deep copy training data
	lof.dataPoints = make([][]float64, len(data))
	for i, point := range data {
		lof.dataPoints[i] = make([]float64, len(point))
		copy(lof.dataPoints[i], point)
	}

	lof.trained = true
	return nil
}

// Detect checks if a point is an anomaly using LOF algorithm
// Returns (isAnomaly, lofScore)
func (lof *LOFDetector) Detect(point []float64) (bool, float64) {
	lof.mu.RLock()
	defer lof.mu.RUnlock()

	if !lof.trained {
		return false, 0.0
	}

	// Calculate LOF score
	lofScore := lof.calculateLOF(point)

	// Normalize score to 0-1 range for consistency with other detectors
	normalizedScore := math.Min(lofScore/10.0, 1.0)

	// Point is anomaly if LOF > threshold
	isAnomaly := lofScore > lof.threshold

	return isAnomaly, normalizedScore
}

// Algorithm returns the algorithm name
func (lof *LOFDetector) Algorithm() string {
	return "local-outlier-factor"
}

// calculateLOF computes the Local Outlier Factor for a point
func (lof *LOFDetector) calculateLOF(point []float64) float64 {
	// 1. Find k-nearest neighbors
	neighbors := lof.findKNearestNeighbors(point, lof.k)

	// 2. Calculate Local Reachability Density (LRD) for the point
	lrd := lof.calculateLRD(point, neighbors)

	if lrd == 0 || math.IsInf(lrd, 0) {
		return 0.0
	}

	// 3. Calculate average LRD of neighbors
	avgNeighborLRD := 0.0
	for _, neighbor := range neighbors {
		neighborNeighbors := lof.findKNearestNeighbors(neighbor, lof.k)
		neighborLRD := lof.calculateLRD(neighbor, neighborNeighbors)
		avgNeighborLRD += neighborLRD
	}
	avgNeighborLRD /= float64(len(neighbors))

	// 4. LOF = ratio of average neighbor LRD to point's LRD
	// High LOF means point is in less dense area (potential outlier)
	lofScore := avgNeighborLRD / lrd

	return lofScore
}

// findKNearestNeighbors finds k nearest neighbors of a point
func (lof *LOFDetector) findKNearestNeighbors(point []float64, k int) [][]float64 {
	type distancePoint struct {
		distance float64
		point    []float64
	}

	// Calculate distances to all training points
	distances := make([]distancePoint, 0, len(lof.dataPoints))
	for _, dataPoint := range lof.dataPoints {
		// Skip if it's the same point
		if equalPoints(point, dataPoint) {
			continue
		}
		dist := euclideanDistance(point, dataPoint)
		distances = append(distances, distancePoint{dist, dataPoint})
	}

	// Sort by distance
	sort.Slice(distances, func(i, j int) bool {
		return distances[i].distance < distances[j].distance
	})

	// Return k nearest
	neighbors := make([][]float64, 0, k)
	for i := 0; i < k && i < len(distances); i++ {
		neighbors = append(neighbors, distances[i].point)
	}

	return neighbors
}

// calculateKDistance returns the distance to the k-th nearest neighbor
func (lof *LOFDetector) calculateKDistance(point []float64, neighbors [][]float64) float64 {
	if len(neighbors) == 0 {
		return 0
	}
	// k-distance is the distance to the farthest neighbor
	return euclideanDistance(point, neighbors[len(neighbors)-1])
}

// calculateReachabilityDistance calculates reachability distance between two points
// reach-dist(p, o) = max(k-distance(o), dist(p, o))
func (lof *LOFDetector) calculateReachabilityDistance(point, neighbor []float64, neighborNeighbors [][]float64) float64 {
	dist := euclideanDistance(point, neighbor)
	kDist := lof.calculateKDistance(neighbor, neighborNeighbors)
	return math.Max(dist, kDist)
}

// calculateLRD calculates Local Reachability Density
// LRD = 1 / (average reachability distance to neighbors)
func (lof *LOFDetector) calculateLRD(point []float64, neighbors [][]float64) float64 {
	if len(neighbors) == 0 {
		return 0
	}

	sumReachDist := 0.0
	for _, neighbor := range neighbors {
		neighborNeighbors := lof.findKNearestNeighbors(neighbor, lof.k)
		reachDist := lof.calculateReachabilityDistance(point, neighbor, neighborNeighbors)
		sumReachDist += reachDist
	}

	avgReachDist := sumReachDist / float64(len(neighbors))

	if avgReachDist == 0 {
		return math.Inf(1) // Infinite density (very close to neighbors)
	}

	return 1.0 / avgReachDist
}

// euclideanDistance calculates Euclidean distance between two points
func euclideanDistance(a, b []float64) float64 {
	if len(a) != len(b) {
		return math.Inf(1)
	}

	sum := 0.0
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

// equalPoints checks if two points are equal
func equalPoints(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(a[i]-b[i]) > 1e-10 {
			return false
		}
	}
	return true
}
