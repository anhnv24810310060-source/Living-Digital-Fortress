package main

import (
	"fmt"
	"math"
	"sync/atomic"
	"testing"
	"time"
)

// Unit tests for advanced load balancing algorithms
// Target: >= 85% code coverage

func TestSelectBackendRoundRobin(t *testing.T) {
	pool := &Pool{
		name: "test-pool",
		backends: []*Backend{
			{URL: "http://backend-1", Weight: 1.0},
			{URL: "http://backend-2", Weight: 1.0},
			{URL: "http://backend-3", Weight: 1.0},
		},
	}

	// Mark all as healthy
	for _, b := range pool.backends {
		b.Healthy.Store(true)
		b.cbState.Store(0) // CLOSED
	}

	// Test round-robin distribution
	seen := make(map[string]int)
	iterations := 300

	for i := 0; i < iterations; i++ {
		selected := pool.selectBackendRoundRobin()
		if selected == nil {
			t.Fatal("expected backend, got nil")
		}
		seen[selected.URL]++
	}

	// Each backend should be selected ~100 times (300/3)
	for _, count := range seen {
		if count < 90 || count > 110 {
			t.Errorf("uneven distribution: %v", seen)
		}
	}
}

func TestSelectBackendRoundRobin_UnhealthyBackends(t *testing.T) {
	pool := &Pool{
		name: "test-pool",
		backends: []*Backend{
			{URL: "http://backend-1", Weight: 1.0},
			{URL: "http://backend-2", Weight: 1.0},
			{URL: "http://backend-3", Weight: 1.0},
		},
	}

	// Only backend-2 is healthy
	pool.backends[0].Healthy.Store(false)
	pool.backends[1].Healthy.Store(true)
	pool.backends[2].Healthy.Store(false)

	for _, b := range pool.backends {
		b.cbState.Store(0)
	}

	// Should always return backend-2
	for i := 0; i < 10; i++ {
		selected := pool.selectBackendRoundRobin()
		if selected == nil {
			t.Fatal("expected backend-2, got nil")
		}
		if selected.URL != "http://backend-2" {
			t.Errorf("expected backend-2, got %s", selected.URL)
		}
	}
}

func TestSelectBackendLeastConnections(t *testing.T) {
	pool := &Pool{
		name: "test-pool",
		backends: []*Backend{
			{URL: "http://backend-1", Weight: 1.0, Conns: 10},
			{URL: "http://backend-2", Weight: 1.0, Conns: 5},
			{URL: "http://backend-3", Weight: 1.0, Conns: 20},
		},
	}

	for _, b := range pool.backends {
		b.Healthy.Store(true)
		b.cbState.Store(0)
	}

	// Should select backend-2 (least connections = 5)
	selected := pool.selectBackendLeastConnections()
	if selected == nil {
		t.Fatal("expected backend, got nil")
	}
	if selected.URL != "http://backend-2" {
		t.Errorf("expected backend-2, got %s", selected.URL)
	}
}

func TestSelectBackendEWMA(t *testing.T) {
	pool := &Pool{
		name: "test-pool",
		backends: []*Backend{
			{URL: "http://backend-1", Weight: 1.0},
			{URL: "http://backend-2", Weight: 1.0},
			{URL: "http://backend-3", Weight: 1.0},
		},
	}

	for _, b := range pool.backends {
		b.Healthy.Store(true)
		b.cbState.Store(0)
	}

	// Set different EWMA values (simulating latency)
	pool.backends[0].setEWMA(50.0)  // 50ms
	pool.backends[1].setEWMA(10.0)  // 10ms - best
	pool.backends[2].setEWMA(100.0) // 100ms

	penalty := 5.0

	// Should select backend-2 (lowest EWMA)
	selected := pool.selectBackendEWMA(penalty)
	if selected == nil {
		t.Fatal("expected backend, got nil")
	}
	if selected.URL != "http://backend-2" {
		t.Errorf("expected backend-2, got %s", selected.URL)
	}
}

func TestSelectBackendEWMA_WithLoad(t *testing.T) {
	pool := &Pool{
		name: "test-pool",
		backends: []*Backend{
			{URL: "http://backend-1", Weight: 1.0},
			{URL: "http://backend-2", Weight: 1.0},
		},
	}

	for _, b := range pool.backends {
		b.Healthy.Store(true)
		b.cbState.Store(0)
	}

	// backend-1: lower latency but high load
	// backend-2: higher latency but no load
	pool.backends[0].setEWMA(10.0)
	atomic.StoreInt64(&pool.backends[0].Conns, 20)

	pool.backends[1].setEWMA(15.0)
	atomic.StoreInt64(&pool.backends[1].Conns, 0)

	penalty := 5.0
	// backend-1 score: 10 + (20 * 5 / 1) = 110
	// backend-2 score: 15 + (0 * 5 / 1) = 15
	// Should select backend-2

	selected := pool.selectBackendEWMA(penalty)
	if selected == nil {
		t.Fatal("expected backend, got nil")
	}
	if selected.URL != "http://backend-2" {
		t.Errorf("expected backend-2 due to lower score, got %s", selected.URL)
	}
}

func TestSelectBackendP2C(t *testing.T) {
	pool := &Pool{
		name: "test-pool",
		backends: []*Backend{
			{URL: "http://backend-1", Weight: 1.0},
			{URL: "http://backend-2", Weight: 1.0},
			{URL: "http://backend-3", Weight: 1.0},
			{URL: "http://backend-4", Weight: 1.0},
			{URL: "http://backend-5", Weight: 1.0},
		},
	}

	for _, b := range pool.backends {
		b.Healthy.Store(true)
		b.cbState.Store(0)
		b.setEWMA(10.0)
	}

	penalty := 5.0

	// Run P2C multiple times, should select backends (not deterministic)
	seen := make(map[string]bool)
	for i := 0; i < 50; i++ {
		selected := pool.selectBackendP2C(penalty)
		if selected == nil {
			t.Fatal("expected backend, got nil")
		}
		seen[selected.URL] = true
	}

	// Should have sampled at least 2 different backends
	if len(seen) < 2 {
		t.Errorf("P2C should sample multiple backends, got only %d", len(seen))
	}
}

func TestSelectBackendRendezvous(t *testing.T) {
	pool := &Pool{
		name: "test-pool",
		backends: []*Backend{
			{URL: "http://backend-1", Weight: 1.0},
			{URL: "http://backend-2", Weight: 1.0},
			{URL: "http://backend-3", Weight: 1.0},
		},
	}

	for _, b := range pool.backends {
		b.Healthy.Store(true)
		b.cbState.Store(0)
	}

	hashKey := "user-12345"

	// Same hash key should always select same backend (consistency)
	selected1 := pool.selectBackendRendezvous(hashKey)
	selected2 := pool.selectBackendRendezvous(hashKey)
	selected3 := pool.selectBackendRendezvous(hashKey)

	if selected1 == nil || selected2 == nil || selected3 == nil {
		t.Fatal("expected backend, got nil")
	}

	if selected1.URL != selected2.URL || selected2.URL != selected3.URL {
		t.Errorf("rendezvous hash not consistent: %s, %s, %s",
			selected1.URL, selected2.URL, selected3.URL)
	}
}

func TestSelectBackendRendezvous_Distribution(t *testing.T) {
	pool := &Pool{
		name: "test-pool",
		backends: []*Backend{
			{URL: "http://backend-1", Weight: 1.0},
			{URL: "http://backend-2", Weight: 1.0},
			{URL: "http://backend-3", Weight: 1.0},
		},
	}

	for _, b := range pool.backends {
		b.Healthy.Store(true)
		b.cbState.Store(0)
	}

	// Test distribution with many different keys
	seen := make(map[string]int)
	for i := 0; i < 300; i++ {
		hashKey := fmt.Sprintf("user-%d", i)
		selected := pool.selectBackendRendezvous(hashKey)
		if selected != nil {
			seen[selected.URL]++
		}
	}

	// Should be reasonably distributed (each ~100 times out of 300)
	for url, count := range seen {
		if count < 70 || count > 130 {
			t.Errorf("uneven distribution for %s: %d (expected ~100)", url, count)
		}
	}
}

func TestUpdateEWMA(t *testing.T) {
	backend := &Backend{URL: "http://test", Weight: 1.0}
	backend.setEWMA(0)

	// First update (initialization)
	updateEWMA(backend, 100.0)
	if backend.getEWMA() != 100.0 {
		t.Errorf("expected EWMA=100, got %f", backend.getEWMA())
	}

	// Second update (should blend)
	updateEWMA(backend, 50.0)
	ewma := backend.getEWMA()
	// EWMA = 0.3*50 + 0.7*100 = 15 + 70 = 85
	expected := 85.0
	if math.Abs(ewma-expected) > 0.1 {
		t.Errorf("expected EWMA=%.1f, got %.1f", expected, ewma)
	}
}

func TestCircuitBreaker_OpenClose(t *testing.T) {
	backend := &Backend{URL: "http://test", Weight: 1.0}
	backend.Healthy.Store(true)
	backend.cbState.Store(0) // CLOSED
	backend.cbFails.Store(0)

	// Initially should allow traffic
	if !isCircuitClosed(backend) {
		t.Error("circuit should be closed initially")
	}

	// Record 5 failures (threshold)
	for i := 0; i < 5; i++ {
		recordBackendFailure(backend, fmt.Errorf("test error"))
	}

	// Circuit should be OPEN
	state := backend.cbState.Load()
	if state != 1 {
		t.Errorf("expected circuit OPEN (1), got %d", state)
	}

	// Should not allow traffic
	if isCircuitClosed(backend) {
		t.Error("circuit should be open after failures")
	}
}

func TestCircuitBreaker_HalfOpen(t *testing.T) {
	backend := &Backend{URL: "http://test", Weight: 1.0}
	backend.Healthy.Store(true)
	backend.cbState.Store(1) // OPEN

	// Set next probe time in the past (allow probe)
	backend.cbNextProbe.Store(time.Now().Add(-1 * time.Second).UnixNano())

	// Should transition to HALF_OPEN and allow probe
	if !isCircuitClosed(backend) {
		t.Error("circuit should allow probe in half-open state")
	}

	// State should now be HALF_OPEN
	state := backend.cbState.Load()
	if state != 2 {
		t.Errorf("expected circuit HALF_OPEN (2), got %d", state)
	}
}

func TestCircuitBreaker_Recovery(t *testing.T) {
	backend := &Backend{URL: "http://test", Weight: 1.0}
	backend.Healthy.Store(true)
	backend.cbState.Store(2) // HALF_OPEN
	backend.cbFails.Store(3)

	// Record success - should close circuit
	recordBackendSuccess(backend, 50.0)

	state := backend.cbState.Load()
	if state != 0 {
		t.Errorf("expected circuit CLOSED (0) after success, got %d", state)
	}

	fails := backend.cbFails.Load()
	if fails != 0 {
		t.Errorf("expected fail count=0 after recovery, got %d", fails)
	}
}

// TestCircuitBreaker_HalfOpenFailureBackoff verifies that a failure during HALF_OPEN
// transitions the circuit back to OPEN and doubles the probe backoff interval.
func TestCircuitBreaker_HalfOpenFailureBackoff(t *testing.T) {
	backend := &Backend{URL: "http://test", Weight: 1.0}
	backend.Healthy.Store(true)
	// Enter HALF_OPEN state ready for a probe
	backend.cbState.Store(2) // HALF_OPEN
	// Configure small base backoff to make test fast
	t.Setenv("ORCH_CB_BACKOFF_MS", "100")

	before := time.Now()
	recordBackendFailure(backend, fmt.Errorf("probe failure"))

	// Should transition to OPEN
	if st := backend.cbState.Load(); st != 1 {
		t.Fatalf("expected circuit OPEN (1) after half-open failure, got %d", st)
	}

	// Backoff should be roughly base*2 = 200ms in the future
	np := backend.cbNextProbe.Load()
	if np == 0 {
		t.Fatalf("expected next probe timestamp to be set")
	}
	dur := time.Until(time.Unix(0, np))
	if dur < 150*time.Millisecond || dur > 400*time.Millisecond {
		t.Fatalf("expected next probe ~200ms ahead, got %v (now=%v target=%v)", dur, before, time.Unix(0, np))
	}
}

func TestSelectBackend_AlgorithmSelection(t *testing.T) {
	pool := &Pool{
		name: "test-pool",
		backends: []*Backend{
			{URL: "http://backend-1", Weight: 1.0},
			{URL: "http://backend-2", Weight: 1.0},
		},
	}

	for _, b := range pool.backends {
		b.Healthy.Store(true)
		b.cbState.Store(0)
		b.setEWMA(10.0)
	}

	// Test all algorithms
	algorithms := []LBAlgo{
		LBRoundRobin,
		LBLeastConnections,
		LBEWMA,
		LBP2CEWMA,
		LBConsistentHash,
	}

	for _, algo := range algorithms {
		selected := pool.selectBackend(algo, "test-key", 5.0)
		if selected == nil {
			t.Errorf("algorithm %s returned nil", algo)
		}
	}
}

func TestWeightedBackends(t *testing.T) {
	pool := &Pool{
		name: "test-pool",
		backends: []*Backend{
			{URL: "http://backend-1", Weight: 1.0}, // Normal capacity
			{URL: "http://backend-2", Weight: 2.0}, // Double capacity
		},
	}

	for _, b := range pool.backends {
		b.Healthy.Store(true)
		b.cbState.Store(0)
		b.setEWMA(10.0)
	}

	// With EWMA, backend-2 should handle more load due to higher weight
	// Set same connection count
	atomic.StoreInt64(&pool.backends[0].Conns, 10)
	atomic.StoreInt64(&pool.backends[1].Conns, 10)

	penalty := 5.0
	// backend-1 score: 10 + (10 * 5 / 1.0) = 60
	// backend-2 score: 10 + (10 * 5 / 2.0) = 35
	// Should prefer backend-2

	selected := pool.selectBackendEWMA(penalty)
	if selected.URL != "http://backend-2" {
		t.Errorf("expected backend-2 (higher weight), got %s", selected.URL)
	}
}

func BenchmarkSelectBackendRoundRobin(b *testing.B) {
	pool := &Pool{
		name:     "bench-pool",
		backends: make([]*Backend, 10),
	}

	for i := 0; i < 10; i++ {
		pool.backends[i] = &Backend{
			URL:    fmt.Sprintf("http://backend-%d", i),
			Weight: 1.0,
		}
		pool.backends[i].Healthy.Store(true)
		pool.backends[i].cbState.Store(0)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pool.selectBackendRoundRobin()
	}
}

func BenchmarkSelectBackendEWMA(b *testing.B) {
	pool := &Pool{
		name:     "bench-pool",
		backends: make([]*Backend, 10),
	}

	for i := 0; i < 10; i++ {
		pool.backends[i] = &Backend{
			URL:    fmt.Sprintf("http://backend-%d", i),
			Weight: 1.0,
		}
		pool.backends[i].Healthy.Store(true)
		pool.backends[i].cbState.Store(0)
		pool.backends[i].setEWMA(float64(10 + i))
	}

	penalty := 5.0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pool.selectBackendEWMA(penalty)
	}
}

func BenchmarkSelectBackendP2C(b *testing.B) {
	pool := &Pool{
		name:     "bench-pool",
		backends: make([]*Backend, 100),
	}

	for i := 0; i < 100; i++ {
		pool.backends[i] = &Backend{
			URL:    fmt.Sprintf("http://backend-%d", i),
			Weight: 1.0,
		}
		pool.backends[i].Healthy.Store(true)
		pool.backends[i].cbState.Store(0)
		pool.backends[i].setEWMA(10.0)
	}

	penalty := 5.0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pool.selectBackendP2C(penalty)
	}
}
