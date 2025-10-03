package main

import (
	"testing"
	"time"
)

func TestRateLimiterBasic(t *testing.T) {
    rl := newRateLimiter(3, 200*time.Millisecond)
    key := "ip1"
    if !rl.Allow(key) { t.Fatalf("first should allow") }
    if !rl.Allow(key) { t.Fatalf("second should allow") }
    if !rl.Allow(key) { t.Fatalf("third should allow") }
    if rl.Allow(key) { t.Fatalf("fourth should be blocked") }
    // after window resets, should allow again
    time.Sleep(210 * time.Millisecond)
    if !rl.Allow(key) { t.Fatalf("after reset should allow") }
}
