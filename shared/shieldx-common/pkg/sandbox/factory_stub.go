package sandbox

import (
    "context"
    "errors"
    "time"
)

// Runner is a minimal interface used by decoy-manager in non-docker builds.
// Returns a string result and error.
type Runner interface {
    Run(ctx context.Context, payload string) (string, error)
}

type noopRunner struct{}

func (noopRunner) Run(_ context.Context, payload string) (string, error) {
    // Echo payload as a trivial placeholder
    return payload, nil
}

// NewFromEnv returns a basic runner for environments without sandbox build tags.
func NewFromEnv() Runner { return noopRunner{} }

// WASMRunner is a stub implementation used when sandbox_wasm is not enabled.
type WASMRunner struct {
    timeout time.Duration
}

func NewWASMRunner(_ []byte, _ string, timeout time.Duration) *WASMRunner {
    return &WASMRunner{timeout: timeout}
}

func (w *WASMRunner) Run(_ context.Context, _ string) (string, error) {
    return "", errors.New("WASM analyzer not enabled (build with -tags sandbox_wasm)")
}
