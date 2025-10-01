package policy

import (
    "os/exec"
    "path/filepath"
    "testing"
)

func TestBuildHashAndZip(t *testing.T) {
    dir := filepath.Join("..", "..", "policies", "demo")
    b, err := LoadFromDir(dir)
    if err != nil { t.Fatalf("load: %v", err) }
    if _, err := b.Hash(); err != nil { t.Fatalf("hash: %v", err) }
    out := filepath.Join("..", "..", "dist", "test-bundle.zip")
    if err := b.WriteZip(out); err != nil { t.Fatalf("zip: %v", err) }
}

func TestCosignCLIAdapter(t *testing.T) {
    if _, err := exec.LookPath("cosign"); err != nil {
        t.Skip("cosign not installed, skipping")
    }
    dir := filepath.Join("..", "..", "policies", "demo")
    b, err := LoadFromDir(dir)
    if err != nil { t.Fatalf("load: %v", err) }
    digest, err := b.Hash()
    if err != nil { t.Fatalf("hash: %v", err) }
    sig, err := SignDigest(CosignCLI{}, digest)
    if err != nil { t.Fatalf("sign: %v", err) }
    if err := VerifyDigest(CosignCLI{}, digest, sig); err != nil {
        t.Fatalf("verify: %v", err)
    }
}
