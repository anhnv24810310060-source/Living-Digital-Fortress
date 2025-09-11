package sandbox

import (
    "context"
    "errors"
    "fmt"
    "time"

    "github.com/tetratelabs/wazero"
    "github.com/tetratelabs/wazero/api"
)

// WASMRunner executes a WASI-like analyzer (no network) to process payload and return a string result.
type WASMRunner struct {
    moduleBytes []byte
    funcName    string
    timeout     time.Duration
}

// NewWASMRunner creates a runner with a compiled module (raw .wasm bytes), function name, and timeout.
func NewWASMRunner(module []byte, funcName string, timeout time.Duration) *WASMRunner {
    return &WASMRunner{moduleBytes: module, funcName: funcName, timeout: timeout}
}

func (w *WASMRunner) Run(ctx context.Context, payload string) (string, error) {
    if len(w.moduleBytes) == 0 { return "", errors.New("empty wasm module") }
    rt := wazero.NewRuntime(ctx)
    defer rt.Close(ctx)
    // No WASI/environment imports to avoid I/O; analyzer must be pure
    mod, err := rt.InstantiateModuleFromBinary(ctx, w.moduleBytes)
    if err != nil { return "", err }
    fn := mod.ExportedFunction(w.funcName)
    if fn == nil { return "", fmt.Errorf("exported function not found: %s", w.funcName) }
    // Pass payload via memory: write bytes and call
    mem := mod.Memory()
    if mem == nil { return "", errors.New("no memory in module") }
    // Very simple ABI: write payload at offset 0, call(fn, len)
    if ok := mem.Write(0, []byte(payload)); !ok { return "", errors.New("mem write failed") }
    ctx2, cancel := context.WithTimeout(ctx, w.timeout)
    defer cancel()
    // Call analyzer: returns (ptr,len) packed in u64: high32=ptr, low32=len
    res, err := fn.Call(ctx2, uint64(len(payload)))
    if err != nil { return "", err }
    if len(res) == 0 { return "", errors.New("no result") }
    v := res[0]
    ptr := uint32(v >> 32)
    ln := uint32(v & 0xffffffff)
    out, ok := mem.Read(ptr, ln)
    if !ok { return "", errors.New("mem read failed") }
    return string(out), nil
}



