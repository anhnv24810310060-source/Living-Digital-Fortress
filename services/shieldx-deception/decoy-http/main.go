package main

import (
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"os"
	"strconv"
	"time"

	"encoding/json"
	"io"
	"shieldx/shared/shieldx-common/pkg/metrics"
)

func rootHandler(w http.ResponseWriter, r *http.Request) {
	// Realism: rotate a set of plausible banners and apply jitter to response time
	banners := currentBanners()
	if len(banners) == 0 {
		banners = []string{"nginx/1.23.4"}
	}
	w.Header().Set("Server", banners[rand.Intn(len(banners))])
	w.Header().Set("Date", time.Now().UTC().Format(http.TimeFormat))
	// Jitter: controlled via env DECOY_JITTER_MS (0..1000)
	if j := getenvInt("DECOY_JITTER_MS", 120); j > 0 {
		time.Sleep(time.Duration(rand.Intn(j)) * time.Millisecond)
	}
	fmt.Fprintf(w, "Welcome to ShieldX decoy. Path: %s\n", r.URL.Path)
}

func healthz(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(200)
	_, _ = w.Write([]byte("ok"))
}

func main() {
	rand.Seed(time.Now().UnixNano())
	port := getenvInt("DECOY_PORT", 8082)
	mux := http.NewServeMux()
	reg := metrics.NewRegistry()
	mHits := metrics.NewCounter("decoy_http_hits_total", "Total decoy http hits")
	mux.HandleFunc("/healthz", healthz)
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) { mHits.Inc(); rootHandler(w, r) })
	reg.Register(mHits)
	mux.Handle("/metrics", reg)

	addr := fmt.Sprintf(":%d", port)
	log.Printf("[decoy-http] listening on %s", addr)
	log.Fatal(http.ListenAndServe(addr, mux))
}

func getenvInt(key string, def int) int {
	v := os.Getenv(key)
	if v == "" {
		return def
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return def
	}
	return n
}

// currentBanners fetches banners from Shapeshifter if configured, else returns defaults.
func currentBanners() []string {
	url := os.Getenv("SHAPESHIFTER_URL")
	if url == "" {
		return []string{"nginx/1.23.4", "Apache/2.4.57 (Unix)", "Caddy", "envoy"}
	}
	resp, err := http.Get(url + "/config/banners")
	if err != nil {
		return []string{"nginx/1.23.4"}
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	var m struct {
		ServerBanners []string `json:"server_banners"`
	}
	if err := json.Unmarshal(b, &m); err != nil {
		return []string{"nginx/1.23.4"}
	}
	return m.ServerBanners
}
