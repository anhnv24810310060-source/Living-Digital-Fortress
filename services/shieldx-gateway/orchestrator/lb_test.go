package main

import "testing"

func TestParseLBAlgoIncludesP2C(t *testing.T) {
	if got := parseLBAlgo("p2c", LBRoundRobin); got != LBP2CEWMA {
		t.Fatalf("expected p2c -> LBP2CEWMA, got %v", got)
	}
}
