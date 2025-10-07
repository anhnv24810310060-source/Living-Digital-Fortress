package main

import (
	"fmt"
	"strings"
	"sync/atomic"
	"testing"
)

func TestChooseRendezvousStable(t *testing.T) {
	p := newPool("t", []string{"a", "b", "c"})
	b1, _ := pickBackend(p, LBConsistentHash, "user:123")
	b2, _ := pickBackend(p, LBConsistentHash, "user:123")
	if b1.URL != b2.URL {
		t.Fatalf("expected stable selection, got %s vs %s", b1.URL, b2.URL)
	}
}

func TestValidServiceName(t *testing.T) {
	ok := []string{"ingress", "guardian_1", "svc-01"}
	bad := []string{"", "THIS", strings.Repeat("a", 65), "bad!"}
	for _, s := range ok {
		if !validServiceName(s) {
			t.Fatalf("expected valid: %s", s)
		}
	}
	for _, s := range bad {
		if validServiceName(s) {
			t.Fatalf("expected invalid: %s", s)
		}
	}
}

func TestP2CConsidersConnections(t *testing.T) {
	p := newPool("t", []string{"http://a", "http://b"})
	// set EWMA equal but different connections
	p.backends[0].setEWMA(100)
	p.backends[1].setEWMA(100)
	// backend 0 has more in-flight
	atomic.StoreInt64(&p.backends[0].Conns, 10)
	atomic.StoreInt64(&p.backends[1].Conns, 1)
	// run many picks to see if lower cost is preferred
	better := 0
	for i := 0; i < 200; i++ {
		b, _ := pickBackend(p, LBP2CEWMA, "")
		if b == p.backends[1] {
			better++
		}
	}
	if better < 120 { // >60% picks should prefer backend1
		t.Fatalf("p2c should prefer lower cost backend, picked=%d", better)
	}
}

func TestWeightedRendezvousPrefersHeavier(t *testing.T) {
	p := newPool("t", []string{"http://a", "http://b"})
	p.backends[0].Weight = 1.0 // a
	p.backends[1].Weight = 4.0 // b heavier
	// same healthy status
	p.backends[0].Healthy.Store(true)
	p.backends[1].Healthy.Store(true)
	// measure selection over many keys
	countA, countB := 0, 0
	for i := 0; i < 200; i++ {
		key := fmt.Sprintf("user:%d", i)
		b, _ := pickBackend(p, LBConsistentHash, key)
		if b == p.backends[0] {
			countA++
		} else if b == p.backends[1] {
			countB++
		}
	}
	if countB <= countA {
		t.Fatalf("expected heavier backend to be chosen more often: a=%d b=%d", countA, countB)
	}
}

func TestLbCostWeightImpact(t *testing.T) {
	b1 := &Backend{Weight: 1.0}
	b1.setEWMA(100)
	b2 := &Backend{Weight: 2.0}
	b2.setEWMA(100)
	c1 := lbCost(b1)
	c2 := lbCost(b2)
	if !(c2 < c1) {
		t.Fatalf("expected higher weight to reduce cost: c1=%v c2=%v", c1, c2)
	}
}
