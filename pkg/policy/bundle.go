package policy

import (
    "bytes"
    "io"
    "net/http"
    "sync/atomic"
    "time"
)

// EngineHolder allows hot-swapping an OPAEngine safely for readers.
type EngineHolder struct { ptr atomic.Value }

func (h *EngineHolder) Get() *OPAEngine {
    v := h.ptr.Load()
    if v == nil { return nil }
    return v.(*OPAEngine)
}
func (h *EngineHolder) Set(e *OPAEngine) { h.ptr.Store(e) }

// StartBundlePoller fetches a Rego policy from url periodically and updates the engine via holder.Set.
// It uses If-None-Match with ETag to avoid unnecessary reloads.
func StartBundlePoller(url string, interval time.Duration, holder *EngineHolder, client *http.Client, onLoad func(version string, err error)) func() {
    if url == "" { return func(){} }
    if client == nil { client = &http.Client{ Timeout: 10 * time.Second } }
    stop := make(chan struct{})
    go func() {
        var etag string
        ticker := time.NewTicker(interval)
        defer ticker.Stop()
        for {
            req, _ := http.NewRequest(http.MethodGet, url, nil)
            if etag != "" { req.Header.Set("If-None-Match", etag) }
            resp, err := client.Do(req)
            if err != nil { if onLoad!=nil { onLoad("", err) }; goto wait }
            if resp.StatusCode == http.StatusNotModified { if onLoad!=nil { onLoad(etag, nil) }; resp.Body.Close(); goto wait }
            b, err := io.ReadAll(resp.Body)
            resp.Body.Close()
            if err != nil { if onLoad!=nil { onLoad("", err) }; goto wait }
            // Compile new engine
            regoSrc := string(b)
            // Accept either full module or partial query; we assume module text.
            e, err := compileRego(regoSrc)
            if err != nil { if onLoad!=nil { onLoad("", err) }; goto wait }
            holder.Set(e)
            etag = resp.Header.Get("ETag")
            if onLoad != nil { onLoad(etag, nil) }
        wait:
            select {
            case <-stop:
                return
            case <-ticker.C:
            }
        }
    }()
    return func(){ close(stop) }
}

func compileRego(src string) (*OPAEngine, error) {
    r := regoFromSource(src)
    pq, err := r.PrepareForEval(context.Background())
    if err != nil { return nil, err }
    return &OPAEngine{prepared: pq}, nil
}

// regoFromSource creates a rego.Rego from source text using the standard query used elsewhere.
func regoFromSource(src string) rego.Rego {
    return *rego.New(
        rego.Query("data.shieldx.authz.decision"),
        rego.Module("bundle.rego", src),
    )
}



