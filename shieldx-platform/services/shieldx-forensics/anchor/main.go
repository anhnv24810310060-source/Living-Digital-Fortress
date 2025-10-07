package main

import (
	"bytes"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"shieldx/pkg/audit"
)

type anchorReport struct {
	Service   string `json:"service"`
	File      string `json:"file"`
	Merkle    string `json:"merkle"`
	Timestamp string `json:"timestamp"`
}

func getenv(key, def string) string {
	v := os.Getenv(key)
	if v == "" {
		return def
	}
	return v
}

func main() {
	// Watches all data/ledger-*.log by default
	dir := getenv("ANCHOR_DIR", "data")
	pattern := getenv("ANCHOR_PATTERN", "ledger-*.log")
	interval := time.Duration(getenvInt("ANCHOR_INTERVAL_SEC", 300)) * time.Second
	webhook := os.Getenv("ANCHOR_WEBHOOK")
	otsWebhook := os.Getenv("ANCHOR_OTS_WEBHOOK") // OpenTimestamps dispatcher or notary bridge

	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(200); _, _ = w.Write([]byte("ok")) })

	go func() {
		t := time.NewTicker(interval)
		defer t.Stop()
		for range t.C {
			files, _ := filepath.Glob(filepath.Join(dir, pattern))
			for _, f := range files {
				mr, err := audit.HashChain(f)
				if err != nil {
					continue
				}
				rep := anchorReport{
					Service:   serviceFromName(f),
					File:      f,
					Merkle:    mr,
					Timestamp: time.Now().UTC().Format(time.RFC3339),
				}
				log.Printf("[anchor] %s %s %s", rep.Service, rep.File, rep.Merkle)
				if webhook != "" {
					postWebhook(webhook, rep)
				}
				if otsWebhook != "" {
					postWebhook(otsWebhook, rep)
				}
			}
		}
	}()

	addr := ":8085"
	if v := os.Getenv("ANCHOR_PORT"); v != "" {
		addr = ":" + v
	}
	log.Printf("[anchor] listening on %s", addr)
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

func serviceFromName(path string) string {
	base := filepath.Base(path)
	// ledger-<service>.log
	if len(base) > len("ledger-.log") && base[:7] == "ledger-" {
		return base[7 : len(base)-4]
	}
	return "unknown"
}

func postWebhook(url string, rep anchorReport) {
	b, _ := json.Marshal(rep)
	req, _ := http.NewRequest(http.MethodPost, url, bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err == nil && resp != nil {
		resp.Body.Close()
	}
}
