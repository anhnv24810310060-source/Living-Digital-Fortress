package main

import (
    "crypto/ed25519"
    "encoding/base64"
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "os"
    "path/filepath"
    "strconv"
    "sync"
    "time"

    "shieldx/pkg/ledger"
    "shieldx/pkg/tokens"
    "shieldx/pkg/metrics"
    "github.com/golang-jwt/jwt/v5"
)

type IssueRequest struct {
    Tenant    string `json:"tenant"`
    Scope     string `json:"scope"`
    Audience  string `json:"audience,omitempty"`
    TTLSeconds int64 `json:"ttl_seconds"`
    DPoPJKT   string `json:"dpop_jkt,omitempty"`
}

type IssueResponse struct {
    Token     string `json:"token"`
    Exp       int64  `json:"exp"`
    PubKeyB64 string `json:"pubkey_b64"`
}

type IntrospectRequest struct {
    Token string `json:"token"`
}

type IntrospectResponse struct {
    Active   bool                `json:"active"`
    Claims   tokens.LocatorClaims `json:"claims"`
    Error    string              `json:"error,omitempty"`
}

type RevokeRequest struct {
    Token string `json:"token"`
}

type RevokeResponse struct {
    Revoked bool   `json:"revoked"`
    Error   string `json:"error,omitempty"`
}

var (
    serviceName = "locator"
    ledgerPath  = "data/ledger-locator.log"
    pubKey      ed25519.PublicKey
    privKey     ed25519.PrivateKey
    // revokedNonces maps nonce -> expiry unix seconds
    revokedNonces = map[string]int64{}
    revokedMu     sync.RWMutex
    reg           = metrics.NewRegistry()
    mIssue        = metrics.NewCounter("locator_issue_total", "Total issued tokens")
    mIntrospect   = metrics.NewCounter("locator_introspect_total", "Total introspections")
    mRevoke       = metrics.NewCounter("locator_revoke_total", "Total revokes")
    oidcIssuer    = os.Getenv("OIDC_ISSUER")
    oidcJWKSKID   = os.Getenv("OIDC_JWKS_KID")
    oidcJWKSBase64= os.Getenv("OIDC_JWKS_B64")
)

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(status)
    _ = json.NewEncoder(w).Encode(v)
}

func handleIssue(w http.ResponseWriter, r *http.Request) {
    mIssue.Inc()
    if r.Method != http.MethodPost {
        http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
        return
    }
    var req IssueRequest
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, "bad request", http.StatusBadRequest)
        return
    }
    if req.Tenant == "" || req.Scope == "" || req.TTLSeconds <= 0 {
        http.Error(w, "missing tenant/scope/ttl_seconds", http.StatusBadRequest)
        return
    }
    now := time.Now().Unix()
    nonce, _ := tokens.RandomNonce(16)
    claims := tokens.LocatorClaims{
        Tenant:   req.Tenant,
        Scope:    req.Scope,
        Audience: req.Audience,
        IssuedAt: now,
        Expires:  now + req.TTLSeconds,
        Nonce:    nonce,
        DPoPJKT:  req.DPoPJKT,
    }
    tok, err := tokens.EncodeToken(claims, privKey)
    if err != nil {
        http.Error(w, "failed to sign token", http.StatusInternalServerError)
        _ = ledger.AppendJSONLine(ledgerPath, serviceName, "issue.error", map[string]any{"error": err.Error()})
        return
    }
    _ = ledger.AppendJSONLine(ledgerPath, serviceName, "issue", map[string]any{"tenant": req.Tenant, "scope": req.Scope, "exp": claims.Expires})
    writeJSON(w, http.StatusOK, IssueResponse{Token: tok, Exp: claims.Expires, PubKeyB64: tokens.PublicKeyB64(pubKey)})
}

func handleIntrospect(w http.ResponseWriter, r *http.Request) {
    mIntrospect.Inc()
    if r.Method != http.MethodPost {
        http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
        return
    }
    var req IntrospectRequest
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, "bad request", http.StatusBadRequest)
        return
    }
    // If OIDC is configured and token looks like JWT, attempt OIDC verify for active=true shortcut
    if oidcIssuer != "" && looksLikeJWT(req.Token) {
        if ok := verifyOIDC(req.Token); ok {
            // minimal claims projection: tenant/scope from aud/scope claim if present
            claims := tokens.LocatorClaims{Tenant: "oidc", Scope: "default", IssuedAt: time.Now().Unix(), Expires: time.Now().Add(5*time.Minute).Unix(), Nonce: "oidc"}
            writeJSON(w, http.StatusOK, IntrospectResponse{Active: true, Claims: claims})
            return
        }
    }
    claims, err := tokens.DecodeAndVerify(req.Token, pubKey)
    if err != nil {
        _ = ledger.AppendJSONLine(ledgerPath, serviceName, "introspect.fail", map[string]any{"error": err.Error()})
        writeJSON(w, http.StatusOK, IntrospectResponse{Active: false, Error: err.Error()})
        return
    }
    // Check revocation
    if isRevoked(claims.Nonce) {
        _ = ledger.AppendJSONLine(ledgerPath, serviceName, "introspect.revoked", map[string]any{"nonce": claims.Nonce})
        writeJSON(w, http.StatusOK, IntrospectResponse{Active: false, Claims: claims, Error: "revoked"})
        return
    }
    _ = ledger.AppendJSONLine(ledgerPath, serviceName, "introspect.ok", map[string]any{"tenant": claims.Tenant, "scope": claims.Scope})
    writeJSON(w, http.StatusOK, IntrospectResponse{Active: true, Claims: claims})
}

func handleRevoke(w http.ResponseWriter, r *http.Request) {
    mRevoke.Inc()
    if r.Method != http.MethodPost {
        http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
        return
    }
    var req RevokeRequest
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, "bad request", http.StatusBadRequest)
        return
    }
    claims, err := tokens.DecodeAndVerify(req.Token, pubKey)
    if err != nil {
        _ = ledger.AppendJSONLine(ledgerPath, serviceName, "revoke.fail", map[string]any{"error": err.Error()})
        writeJSON(w, http.StatusOK, RevokeResponse{Revoked: false, Error: err.Error()})
        return
    }
    revokeNonce(claims.Nonce, claims.Expires)
    _ = ledger.AppendJSONLine(ledgerPath, serviceName, "revoke.ok", map[string]any{"tenant": claims.Tenant, "scope": claims.Scope, "nonce": claims.Nonce})
    writeJSON(w, http.StatusOK, RevokeResponse{Revoked: true})
}

func main() {
    port := getenvInt("LOCATOR_PORT", 8080)
    if pkb64 := os.Getenv("LOCATOR_PRIVKEY_B64"); pkb64 != "" {
        var err error
        privKey, err = tokens.PrivateKeyFromB64(pkb64)
        if err != nil {
            log.Fatalf("invalid LOCATOR_PRIVKEY_B64: %v", err)
        }
        pubKey = privKey.Public().(ed25519.PublicKey)
    } else {
        var err error
        pubKey, privKey, err = tokens.GenerateEd25519Keypair()
        if err != nil {
            log.Fatalf("generate keypair: %v", err)
        }
        log.Printf("[locator] ephemeral pubkey (b64): %s", base64.StdEncoding.EncodeToString(pubKey))
    }

    // Load revoked db from disk if exists
    loadRevocations()

    mux := http.NewServeMux()
    mux.HandleFunc("/issue", handleIssue)
    mux.HandleFunc("/introspect", handleIntrospect)
    mux.HandleFunc("/revoke", handleRevoke)
    reg.Register(mIssue); reg.Register(mIntrospect); reg.Register(mRevoke)
    mux.Handle("/metrics", reg)
    mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(200); _, _ = w.Write([]byte("ok")) })

    addr := fmt.Sprintf(":%d", port)
    log.Printf("[locator] listening on %s", addr)
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

func looksLikeJWT(s string) bool {
    // rough check: three parts dot-separated
    n := 0
    for i := 0; i < len(s); i++ { if s[i] == '.' { n++ } }
    return n == 2
}

func verifyOIDC(tok string) bool {
    if oidcJWKSBase64 == "" { return false }
    // Demo-only: accept a single RSA public key from base64 PEM in OIDC_JWKS_B64 (simplified)
    // In production, fetch JWKS from issuer and select by kid.
    pemRaw, err := base64.StdEncoding.DecodeString(oidcJWKSBase64)
    if err != nil { return false }
    key, err := jwt.ParseRSAPublicKeyFromPEM(pemRaw)
    if err != nil { return false }
    parser := jwt.NewParser(jwt.WithValidMethods([]string{"RS256"}))
    claims := jwt.MapClaims{}
    t, err := parser.ParseWithClaims(tok, claims, func(token *jwt.Token) (interface{}, error) { return key, nil })
    if err != nil || !t.Valid { return false }
    // Optional issuer check
    if iss, _ := claims["iss"].(string); oidcIssuer != "" && iss != oidcIssuer { return false }
    return true
}

func isRevoked(nonce string) bool {
    revokedMu.RLock()
    exp, ok := revokedNonces[nonce]
    revokedMu.RUnlock()
    if !ok {
        return false
    }
    now := time.Now().Unix()
    return exp >= now
}

func revokeNonce(nonce string, exp int64) {
    revokedMu.Lock()
    revokedNonces[nonce] = exp
    revokedMu.Unlock()
    persistRevocations()
}

func revokedPath() string { return filepath.Join("data", "revoke.json") }

func persistRevocations() {
    revokedMu.RLock()
    snap := make(map[string]int64, len(revokedNonces))
    for k, v := range revokedNonces { snap[k] = v }
    revokedMu.RUnlock()
    _ = os.MkdirAll("data", 0o755)
    f, err := os.CreateTemp("data", "revoke-*.tmp")
    if err != nil { return }
    enc := json.NewEncoder(f)
    _ = enc.Encode(snap)
    _ = f.Close()
    // atomically replace
    _ = os.Rename(f.Name(), revokedPath())
}

func loadRevocations() {
    b, err := os.ReadFile(revokedPath())
    if err != nil { return }
    var m map[string]int64
    if err := json.Unmarshal(b, &m); err != nil { return }
    now := time.Now().Unix()
    revokedMu.Lock()
    for k, v := range m { if v > now { revokedNonces[k] = v } }
    revokedMu.Unlock()
}



