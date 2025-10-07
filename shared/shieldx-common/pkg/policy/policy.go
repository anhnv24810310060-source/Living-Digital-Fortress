package policy

import (
    "encoding/json"
    "errors"
    "os"
    "path/filepath"
    "strings"
)

// Model (basic allowlist)
type ScopeRule struct {
    Tenant string   `json:"tenant"`
    Scopes []string `json:"scopes"`
}

// Advanced actions
type Action string

const (
    ActionAllow  Action = "allow"
    ActionDeny   Action = "deny"
    ActionDivert Action = "divert"
    ActionTarpit Action = "tarpit"
)

type AdvancedRule struct {
    Tenant     string   `json:"tenant"`
    Scopes     []string `json:"scopes,omitempty"`
    PathPrefix string   `json:"path_prefix,omitempty"`
    Action     Action   `json:"action"`
}

type Config struct {
    AllowAll bool          `json:"allow_all"`
    Allowed  []ScopeRule   `json:"allowed"`
    Advanced []AdvancedRule `json:"advanced,omitempty"`
    TarpitMs int           `json:"tarpit_ms,omitempty"`
}

// Load loads a policy config from a JSON file. If file does not exist, returns a permissive default (AllowAll=true).
func Load(path string) (Config, error) {
    if strings.TrimSpace(path) == "" {
        return Config{AllowAll: true}, nil
    }
    if _, err := os.Stat(path); errors.Is(err, os.ErrNotExist) {
        return Config{AllowAll: true}, nil
    }
    data, err := os.ReadFile(filepath.Clean(path))
    if err != nil {
        return Config{}, err
    }
    var cfg Config
    if err := json.Unmarshal(data, &cfg); err != nil {
        return Config{}, err
    }
    return cfg, nil
}

// IsAllowed checks if a given tenant/scope is permitted by the basic allowlist.
func IsAllowed(cfg Config, tenant string, scope string) bool {
    if cfg.AllowAll {
        return true
    }
    if tenant == "" || scope == "" {
        return false
    }
    for _, r := range cfg.Allowed {
        if r.Tenant == tenant {
            if len(r.Scopes) == 0 {
                return true
            }
            for _, s := range r.Scopes {
                if s == scope {
                    return true
                }
            }
        }
    }
    return false
}

// Evaluate applies Advanced rules first (first-match), otherwise falls back to allowlist/allowAll.
func Evaluate(cfg Config, tenant, scope, path string) Action {
    for _, r := range cfg.Advanced {
        if r.Tenant != "" && r.Tenant != tenant {
            continue
        }
        if len(r.Scopes) > 0 {
            ok := false
            for _, s := range r.Scopes { if s == scope { ok = true; break } }
            if !ok { continue }
        }
        if r.PathPrefix != "" && !strings.HasPrefix(path, r.PathPrefix) {
            continue
        }
        if r.Action != "" { return r.Action }
    }
    if IsAllowed(cfg, tenant, scope) {
        return ActionAllow
    }
    if cfg.AllowAll { return ActionAllow }
    return ActionDeny
}


