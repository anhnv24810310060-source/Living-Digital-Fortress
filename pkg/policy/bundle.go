package policy

import (
    "archive/zip"
    "bytes"
    "crypto/sha256"
    "encoding/hex"
    "encoding/json"
    "fmt"
    "io"
    "os"
    "path/filepath"
    "sort"
    "strings"
    "time"
)

// Manifest describes a policy bundle manifest.
type Manifest struct {
    Name       string            `json:"name"`
    Version    string            `json:"version"`
    CreatedAt  time.Time         `json:"created_at"`
    OPAVersion string            `json:"opa_version,omitempty"`
    Policies   []string          `json:"policies"` // relative paths inside bundle
    Annotations map[string]string `json:"annotations,omitempty"`
}

// Bundle represents a policy bundle in-memory.
type Bundle struct {
    Manifest Manifest
    Files    map[string][]byte // relative path -> content
}

// LoadFromDir builds a Bundle from a directory, reading manifest.json and policy files (*.rego).
func LoadFromDir(dir string) (*Bundle, error) {
    mfPath := filepath.Join(dir, "manifest.json")
    b, err := os.ReadFile(mfPath)
    if err != nil { return nil, fmt.Errorf("read manifest: %w", err) }
    var mf Manifest
    if err := json.Unmarshal(b, &mf); err != nil { return nil, fmt.Errorf("parse manifest: %w", err) }
    files := map[string][]byte{}
    for _, rel := range mf.Policies {
        if !strings.HasSuffix(rel, ".rego") {
            // allow arbitrary files but recommend .rego
        }
        p := filepath.Join(dir, rel)
        data, err := os.ReadFile(p)
        if err != nil { return nil, fmt.Errorf("read policy %s: %w", rel, err) }
        // store using forward slashes
        files[filepath.ToSlash(rel)] = data
    }
    return &Bundle{Manifest: mf, Files: files}, nil
}

// Hash computes a canonical SHA-256 over manifest (without signature fields) and files in sorted order.
func (b *Bundle) Hash() (string, error) {
    // canonicalize manifest by re-marshaling with stable field order via encoding/json
    mf := b.Manifest
    // ensure CreatedAt has RFC3339 format
    if mf.CreatedAt.IsZero() { mf.CreatedAt = time.Now().UTC() }
    mfb, err := json.Marshal(mf)
    if err != nil { return "", fmt.Errorf("marshal manifest: %w", err) }
    h := sha256.New()
    h.Write(mfb)
    // sort file paths
    paths := make([]string, 0, len(b.Files))
    for p := range b.Files { paths = append(paths, p) }
    sort.Strings(paths)
    for _, p := range paths {
        h.Write([]byte("\n--FILE--\n"))
        h.Write([]byte(p))
        h.Write([]byte("\n"))
        h.Write(b.Files[p])
    }
    sum := h.Sum(nil)
    return hex.EncodeToString(sum), nil
}

// WriteZip writes the bundle into a zip file: manifest.json at root and files under their relative paths.
func (b *Bundle) WriteZip(outPath string) error {
    if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil { return err }
    f, err := os.Create(outPath)
    if err != nil { return err }
    defer f.Close()
    zw := zip.NewWriter(f)
    // manifest.json
    mfw, err := zw.Create("manifest.json")
    if err != nil { return err }
    mfb, err := json.MarshalIndent(b.Manifest, "", "  ")
    if err != nil { return err }
    if _, err := mfw.Write(mfb); err != nil { return err }
    // files
    paths := make([]string, 0, len(b.Files))
    for p := range b.Files { paths = append(paths, p) }
    sort.Strings(paths)
    for _, p := range paths {
        w, err := zw.Create(p)
        if err != nil { return err }
        if _, err := w.Write(b.Files[p]); err != nil { return err }
    }
    return zw.Close()
}

// Signer/Verifier interfaces to allow swapping cosign later.
type Signer interface { Sign(digest string, out io.Writer) error }
type Verifier interface { Verify(digest string, sig io.Reader) error }

// NoopSigner writes the digest into the signature as plain text (demo only).
type NoopSigner struct{}
func (NoopSigner) Sign(digest string, out io.Writer) error { _, err := io.WriteString(out, digest); return err }

// NoopVerifier checks equality of provided signature with digest (demo only).
type NoopVerifier struct{}
func (NoopVerifier) Verify(digest string, sig io.Reader) error {
    b, err := io.ReadAll(sig)
    if err != nil { return err }
    if strings.TrimSpace(string(b)) != strings.TrimSpace(digest) {
        return fmt.Errorf("signature mismatch")
    }
    return nil
}

// WriteSignature writes signature bytes to a file path, creating parent directories.
func WriteSignature(path string, sig []byte) error {
    if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil { return err }
    return os.WriteFile(path, sig, 0o644)
}

// ReadSignature reads signature content from file.
func ReadSignature(path string) ([]byte, error) { return os.ReadFile(path) }

// BuildAndWrite builds a bundle from dir and writes a zip to outPath; returns its digest.
func BuildAndWrite(dir, outPath string) (string, error) {
    b, err := LoadFromDir(dir)
    if err != nil { return "", err }
    if err := b.WriteZip(outPath); err != nil { return "", err }
    return b.Hash()
}

// SignDigest uses the provided signer to sign digest and returns signature bytes.
func SignDigest(s Signer, digest string) ([]byte, error) {
    var buf bytes.Buffer
    if err := s.Sign(digest, &buf); err != nil { return nil, err }
    return buf.Bytes(), nil
}

// VerifyDigest uses the provided verifier to verify signature against digest.
func VerifyDigest(v Verifier, digest string, sig []byte) error {
    return v.Verify(digest, bytes.NewReader(sig))
}



