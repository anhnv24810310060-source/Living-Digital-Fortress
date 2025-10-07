package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// NOTE: These tests exercise a minimal slice of guardian execute flow by instantiating
// a subset of handlers from main() via a helper. To avoid refactoring main heavily right now,
// we duplicate small bits (acceptable for initial scaffolding) and can be consolidated later.

// buildGuardianMux constructs an HTTP mux with only the /guardian/execute and /guardian/status endpoints.
type testJob struct{ ID, Status string }

func buildGuardianTestMux() http.Handler {
	mux := http.NewServeMux()
	jobs := map[string]*testJob{}
	var idCtr uint64
	nextID := func() string {
		return strings.ReplaceAll(time.Now().Format("150405.000000"), ".", "") + fmt.Sprintf("-%d", atomic.AddUint64(&idCtr, 1))
	}

	mux.HandleFunc("/guardian/execute", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", 405)
			return
		}
		var body struct {
			Payload string `json:"payload"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, "bad json", 400)
			return
		}
		if len(body.Payload) == 0 {
			http.Error(w, "payload required", 400)
			return
		}
		id := nextID()
		jobs[id] = &testJob{ID: id, Status: "queued"}
		// Simulate async completion
		go func(j *testJob) { time.Sleep(20 * time.Millisecond); j.Status = "done" }(jobs[id])
		_ = json.NewEncoder(w).Encode(map[string]string{"id": id, "status": "queued"})
	})
	mux.HandleFunc("/guardian/status/", func(w http.ResponseWriter, r *http.Request) {
		id := strings.TrimPrefix(r.URL.Path, "/guardian/status/")
		if j, ok := jobs[id]; ok {
			_ = json.NewEncoder(w).Encode(j)
			return
		}
		http.Error(w, "not found", 404)
	})
	return mux
}

func TestGuardianExecuteValidation(t *testing.T) {
	srv := httptest.NewServer(buildGuardianTestMux())
	defer srv.Close()

	// Missing payload
	resp, err := http.Post(srv.URL+"/guardian/execute", "application/json", strings.NewReader(`{"payload":""}`))
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != 400 {
		t.Fatalf("expected 400 got %d", resp.StatusCode)
	}

	// Valid payload
	resp2, err2 := http.Post(srv.URL+"/guardian/execute", "application/json", strings.NewReader(`{"payload":"echo hi"}`))
	if err2 != nil {
		t.Fatal(err2)
	}
	if resp2.StatusCode != 200 {
		t.Fatalf("expected 200 got %d", resp2.StatusCode)
	}
	var out map[string]string
	if err := json.NewDecoder(resp2.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	if out["status"] != "queued" {
		t.Fatalf("expected status queued got %s", out["status"])
	}
	if out["id"] == "" {
		t.Fatal("expected id in response")
	}

	// Unknown job ID
	resp3, _ := http.Get(srv.URL + "/guardian/status/does-not-exist")
	if resp3.StatusCode != 404 {
		t.Fatalf("expected 404 got %d", resp3.StatusCode)
	}
}

func TestGuardianJobLifecycle(t *testing.T) {
	srv := httptest.NewServer(buildGuardianTestMux())
	defer srv.Close()
	resp, err := http.Post(srv.URL+"/guardian/execute", "application/json", strings.NewReader(`{"payload":"echo hi"}`))
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != 200 {
		t.Fatalf("unexpected code %d", resp.StatusCode)
	}
	var out map[string]string
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	id := out["id"]
	if id == "" {
		t.Fatal("missing id")
	}
	// Poll until status becomes done
	deadline := time.Now().Add(500 * time.Millisecond)
	for {
		if time.Now().After(deadline) {
			t.Fatal("timeout waiting for job done")
		}
		time.Sleep(25 * time.Millisecond)
		r2, _ := http.Get(srv.URL + "/guardian/status/" + id)
		if r2.StatusCode != 200 {
			continue
		}
		var st testJob
		if err := json.NewDecoder(r2.Body).Decode(&st); err == nil {
			if st.Status == "done" {
				return
			}
		}
	}
}

// TestGuardianRateLimiter ensures that exceeding the per-minute limit results in 429 responses.
func TestGuardianRateLimiter(t *testing.T) {
	// Build a mux with a configured low rate limit (5/min) to trigger quickly
	// We'll replicate minimal execution handler with limiter wrapper.
	rlMux := http.NewServeMux()
	// simple limiter reused from production logic (inline minimal variant)
	makeLimiter := func(max int) func(http.HandlerFunc) http.HandlerFunc {
		type bucket struct {
			c int
			w int64
		}
		var mu sync.Mutex
		m := map[string]*bucket{}
		return func(next http.HandlerFunc) http.HandlerFunc {
			return func(w http.ResponseWriter, r *http.Request) {
				ip := "test"
				now := time.Now().Unix() / 60
				mu.Lock()
				b := m[ip]
				if b == nil || b.w != now {
					b = &bucket{w: now}
					m[ip] = b
				}
				b.c++
				c := b.c
				mu.Unlock()
				if c > max {
					http.Error(w, "rate limit exceeded", http.StatusTooManyRequests)
					return
				}
				next(w, r)
			}
		}
	}
	limiter := makeLimiter(5)
	rlMux.HandleFunc("/guardian/execute", limiter(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", 405)
			return
		}
		var body struct {
			Payload string `json:"payload"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, "bad json", 400)
			return
		}
		if body.Payload == "" {
			http.Error(w, "payload required", 400)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]string{"id": "id-1", "status": "queued"})
	}))
	srv := httptest.NewServer(rlMux)
	defer srv.Close()

	// Perform 6 rapid requests (limit is 5)
	for i := 0; i < 6; i++ {
		resp, err := http.Post(srv.URL+"/guardian/execute", "application/json", strings.NewReader(`{"payload":"echo hi"}`))
		if err != nil {
			t.Fatalf("request %d error: %v", i, err)
		}
		if i < 5 && resp.StatusCode != 200 {
			t.Fatalf("expected success for request %d got %d", i, resp.StatusCode)
		}
		if i == 5 && resp.StatusCode != http.StatusTooManyRequests {
			t.Fatalf("expected 429 for 6th request got %d", resp.StatusCode)
		}
	}
}

// TestGuardianConcurrencyLimiter verifies that when the semaphore is full additional
// requests receive 429 Too Many Requests.
func TestGuardianConcurrencyLimiter(t *testing.T) {
	mux := http.NewServeMux()
	// concurrency limit = 2
	sem := make(chan struct{}, 2)
	releaseCh := make(chan struct{})
	mux.HandleFunc("/guardian/execute", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", 405)
			return
		}
		select {
		case sem <- struct{}{}:
			// Hold until released so we can saturate
			go func() {
				<-releaseCh
				<-sem
			}()
			_ = json.NewEncoder(w).Encode(map[string]string{"id": "x", "status": "queued"})
		default:
			http.Error(w, "too many concurrent executions", http.StatusTooManyRequests)
		}
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	// Fire 3 requests rapidly
	var codes []int
	for i := 0; i < 3; i++ {
		resp, err := http.Post(srv.URL+"/guardian/execute", "application/json", strings.NewReader(`{"payload":"echo hi"}`))
		if err != nil {
			t.Fatalf("post %d: %v", i, err)
		}
		codes = append(codes, resp.StatusCode)
	}
	// Expect first two 200, third 429
	if codes[0] != 200 || codes[1] != 200 || codes[2] != http.StatusTooManyRequests {
		t.Fatalf("unexpected codes: %v", codes)
	}
	// Release running requests
	close(releaseCh)
}

// TestGuardianBreakerOpen simulates repeated failures triggering open state resulting in 503.
func TestGuardianBreakerOpen(t *testing.T) {
	mux := http.NewServeMux()
	var breakerOpen bool
	failThreshold := 3
	var fails int
	mux.HandleFunc("/guardian/execute", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", 405)
			return
		}
		if breakerOpen {
			http.Error(w, "service temporarily unavailable", http.StatusServiceUnavailable)
			return
		}
		// Simulate failure path increments
		fails++
		if fails >= failThreshold {
			breakerOpen = true
		}
		_ = json.NewEncoder(w).Encode(map[string]string{"id": fmt.Sprintf("j-%d", fails), "status": "queued"})
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	// First 3 requests succeed (200), 4th should be 503
	for i := 0; i < 4; i++ {
		resp, err := http.Post(srv.URL+"/guardian/execute", "application/json", strings.NewReader(`{"payload":"echo hi"}`))
		if err != nil {
			t.Fatalf("post %d: %v", i, err)
		}
		if i < 3 && resp.StatusCode != 200 {
			t.Fatalf("expected 200 before breaker open got %d", resp.StatusCode)
		}
		if i == 3 && resp.StatusCode != http.StatusServiceUnavailable {
			t.Fatalf("expected 503 after breaker open got %d", resp.StatusCode)
		}
	}
}

// TestGuardianBreakerRecovery simulates breaker half-open recovery after cooldown.
func TestGuardianBreakerRecovery(t *testing.T) {
	mux := http.NewServeMux()
	var state string = "open" // open -> half -> closed
	cooldown := 50 * time.Millisecond
	openedAt := time.Now()
	mux.HandleFunc("/guardian/execute", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", 405)
			return
		}
		switch state {
		case "open":
			if time.Since(openedAt) < cooldown {
				http.Error(w, "service temporarily unavailable", http.StatusServiceUnavailable)
				return
			}
			state = "half" // allow a probe
			_ = json.NewEncoder(w).Encode(map[string]string{"id": "probe", "status": "queued"})
		case "half":
			// success closes breaker
			state = "closed"
			_ = json.NewEncoder(w).Encode(map[string]string{"id": "recover", "status": "queued"})
		default: // closed
			_ = json.NewEncoder(w).Encode(map[string]string{"id": "normal", "status": "queued"})
		}
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()
	// First call during open window => 503
	resp1, _ := http.Post(srv.URL+"/guardian/execute", "application/json", strings.NewReader(`{"payload":"x"}`))
	if resp1.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("expected 503 got %d", resp1.StatusCode)
	}
	// Wait for cooldown then probe (half-open)
	time.Sleep(cooldown + 10*time.Millisecond)
	resp2, _ := http.Post(srv.URL+"/guardian/execute", "application/json", strings.NewReader(`{"payload":"x"}`))
	if resp2.StatusCode != 200 {
		t.Fatalf("expected 200 probe got %d", resp2.StatusCode)
	}
	// Next success should fully close
	resp3, _ := http.Post(srv.URL+"/guardian/execute", "application/json", strings.NewReader(`{"payload":"x"}`))
	if resp3.StatusCode != 200 {
		t.Fatalf("expected 200 after close got %d", resp3.StatusCode)
	}
}

// TestGuardianJobTimeout simulates a job exceeding its runtime and marking timeout.
func TestGuardianJobTimeout(t *testing.T) {
	mux := http.NewServeMux()
	type job struct{ ID, Status string }
	jobs := map[string]*job{}
	var mu sync.Mutex
	mux.HandleFunc("/guardian/execute", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", 405)
			return
		}
		id := "jt1"
		j := &job{ID: id, Status: "queued"}
		mu.Lock()
		jobs[id] = j
		mu.Unlock()
		go func() {
			time.Sleep(30 * time.Millisecond)
			mu.Lock()
			if j.Status == "queued" {
				j.Status = "timeout"
			}
			mu.Unlock()
		}()
		_ = json.NewEncoder(w).Encode(map[string]string{"id": id, "status": "queued"})
	})
	mux.HandleFunc("/guardian/status/", func(w http.ResponseWriter, r *http.Request) {
		id := strings.TrimPrefix(r.URL.Path, "/guardian/status/")
		mu.Lock()
		j := jobs[id]
		mu.Unlock()
		if j == nil {
			http.Error(w, "not found", 404)
			return
		}
		_ = json.NewEncoder(w).Encode(j)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()
	resp, _ := http.Post(srv.URL+"/guardian/execute", "application/json", strings.NewReader(`{"payload":"x"}`))
	if resp.StatusCode != 200 {
		t.Fatalf("expected 200 got %d", resp.StatusCode)
	}
	deadline := time.Now().Add(300 * time.Millisecond)
	for {
		if time.Now().After(deadline) {
			t.Fatal("timeout waiting for job to timeout")
		}
		time.Sleep(20 * time.Millisecond)
		r2, _ := http.Get(srv.URL + "/guardian/status/jt1")
		if r2.StatusCode != 200 {
			continue
		}
		var st job
		_ = json.NewDecoder(r2.Body).Decode(&st)
		if st.Status == "timeout" {
			break
		}
	}
}

// TestGuardianReportPreview ensures long output is truncated in preview.
func TestGuardianReportPreview(t *testing.T) {
	mux := http.NewServeMux()
	type job struct {
		ID, Status, Output, Hash string
		CompletedAt              time.Time
	}
	jobs := map[string]*job{}
	longOut := strings.Repeat("A", 400)
	jobs["j1"] = &job{ID: "j1", Status: "done", Output: longOut, CompletedAt: time.Now()}
	mux.HandleFunc("/guardian/report/", func(w http.ResponseWriter, r *http.Request) {
		id := strings.TrimPrefix(r.URL.Path, "/guardian/report/")
		j := jobs[id]
		if j == nil {
			http.Error(w, "not found", 404)
			return
		}
		preview := j.Output
		if len(preview) > 256 {
			preview = preview[:256] + "..."
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"id": j.ID, "output_preview": preview})
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()
	resp, _ := http.Get(srv.URL + "/guardian/report/j1")
	if resp.StatusCode != 200 {
		t.Fatalf("expected 200 got %d", resp.StatusCode)
	}
	var out map[string]string
	_ = json.NewDecoder(resp.Body).Decode(&out)
	prev := out["output_preview"]
	if len(prev) <= 256 || !strings.HasSuffix(prev, "...") {
		t.Fatalf("expected truncated preview len>256 with ellipsis, got len=%d", len(prev))
	}
}
