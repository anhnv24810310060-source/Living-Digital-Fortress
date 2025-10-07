package forensics

import (
    "crypto/sha256"
    "encoding/hex"
    "encoding/json"
    "fmt"
    "os"
    "path/filepath"
    "time"
)

// SaveArtifact writes bytes to data/forensics/<service>/<category>/<ts>-<rand>.bin and returns path and sha256 hex
func SaveArtifact(service string, category string, data []byte) (string, string, error) {
    if service == "" { service = "unknown" }
    if category == "" { category = "misc" }
    dir := filepath.Join("data", "forensics", service, category)
    if err := os.MkdirAll(dir, 0o755); err != nil { return "", "", err }
    name := fmt.Sprintf("%d.bin", time.Now().UnixNano())
    path := filepath.Join(dir, name)
    if err := os.WriteFile(path, data, 0o600); err != nil { return "", "", err }
    h := sha256.Sum256(data)
    return path, hex.EncodeToString(h[:]), nil
}

// SaveJSON marshals v to JSON and saves it as an artifact
func SaveJSON(service string, category string, v interface{}) (string, string, error) {
    b, err := json.Marshal(v)
    if err != nil { return "", "", err }
    return SaveArtifact(service, category, b)
}



