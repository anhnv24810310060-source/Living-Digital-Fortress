package main

import (
	"encoding/json"
	"math"
	"math/rand"
	"time"
)

// IsolationForest is a lightweight implementation suitable for small to medium datasets.
// It builds random trees up to a height limit and scores points by average path length.
type IsolationForest struct {
	Trees      []*iTree `json:"trees"`
	NumTrees   int      `json:"num_trees"`
	SampleSize int      `json:"sample_size"`
	HeightLim  int      `json:"height_limit"`
}

type iTree struct {
	Root *iNode `json:"root"`
}

type iNode struct {
	Leaf     bool    `json:"leaf"`
	Size     int     `json:"size"`
	Dim      int     `json:"dim"`
	SplitVal float64 `json:"split_val"`
	Left     *iNode  `json:"left"`
	Right    *iNode  `json:"right"`
}

func NewIsolationForest(numTrees, sampleSize int) *IsolationForest {
	if numTrees <= 0 {
		numTrees = 100
	}
	if sampleSize <= 0 {
		sampleSize = 256
	}
	return &IsolationForest{NumTrees: numTrees, SampleSize: sampleSize, HeightLim: int(math.Ceil(math.Log2(float64(sampleSize))))}
}

func (f *IsolationForest) Train(X [][]float64) {
	rand.Seed(time.Now().UnixNano())
	f.Trees = make([]*iTree, f.NumTrees)
	n := len(X)
	for i := 0; i < f.NumTrees; i++ {
		// sample without replacement up to SampleSize
		idxs := rand.Perm(n)
		m := f.SampleSize
		if m > n {
			m = n
		}
		sample := make([][]float64, m)
		for j := 0; j < m; j++ {
			sample[j] = X[idxs[j]]
		}
		t := &iTree{Root: buildTree(sample, 0, f.HeightLim)}
		f.Trees[i] = t
	}
}

func buildTree(X [][]float64, h, hlim int) *iNode {
	if len(X) <= 1 || h >= hlim {
		return &iNode{Leaf: true, Size: len(X)}
	}
	d := len(X[0])
	// choose random dim
	dim := rand.Intn(d)
	// find min/max on this dim
	minv, maxv := X[0][dim], X[0][dim]
	for i := 1; i < len(X); i++ {
		v := X[i][dim]
		if v < minv {
			minv = v
		}
		if v > maxv {
			maxv = v
		}
	}
	if minv == maxv { // cannot split further
		return &iNode{Leaf: true, Size: len(X)}
	}
	split := minv + rand.Float64()*(maxv-minv)
	left := make([][]float64, 0, len(X))
	right := make([][]float64, 0, len(X))
	for _, row := range X {
		if row[dim] < split {
			left = append(left, row)
		} else {
			right = append(right, row)
		}
	}
	if len(left) == 0 || len(right) == 0 {
		return &iNode{Leaf: true, Size: len(X)}
	}
	return &iNode{Leaf: false, Dim: dim, SplitVal: split, Left: buildTree(left, h+1, hlim), Right: buildTree(right, h+1, hlim)}
}

// c(n): average path length of unsuccessful search in Binary Search Tree, used for normalization
func cFactor(n int) float64 {
	if n <= 1 {
		return 0
	}
	return 2.0*(math.Log(float64(n-1))+0.5772156649) - 2.0*float64(n-1)/float64(n)
}

func pathLength(node *iNode, x []float64, h int) float64 {
	if node.Leaf {
		if node.Size <= 1 {
			return float64(h)
		}
		return float64(h) + cFactor(node.Size)
	}
	if x[node.Dim] < node.SplitVal {
		return pathLength(node.Left, x, h+1)
	}
	return pathLength(node.Right, x, h+1)
}

// Score returns anomaly score in [0,1], higher means more anomalous.
func (f *IsolationForest) Score(x []float64) float64 {
	if len(f.Trees) == 0 {
		return 0.0
	}
	sum := 0.0
	for _, t := range f.Trees {
		sum += pathLength(t.Root, x, 0)
	}
	Eh := sum / float64(len(f.Trees))
	c := cFactor(f.SampleSize)
	if c <= 0 {
		c = 1
	}
	return math.Pow(2, -Eh/c)
}

func (f *IsolationForest) SaveJSON() ([]byte, error) { return json.Marshal(f) }
func (f *IsolationForest) LoadJSON(b []byte) error   { return json.Unmarshal(b, f) }
