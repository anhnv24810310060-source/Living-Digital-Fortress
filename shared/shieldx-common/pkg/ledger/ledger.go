package ledger

import (
    "encoding/json"
    "errors"
    "fmt"
    "os"
    "path/filepath"
    "time"
)

// Record is a simple append-only event used for forensics/audit.
type Record struct {
    Timestamp string      `json:"ts"`
    Service   string      `json:"service"`
    Type      string      `json:"type"`
    Data      interface{} `json:"data,omitempty"`
}

// AppendJSONLine appends a JSON line into the given file path, creating directories if necessary.
func AppendJSONLine(filePath string, service string, eventType string, data interface{}) error {
    if filePath == "" {
        return errors.New("filePath is empty")
    }
    if service == "" {
        service = "unknown"
    }
    rec := Record{
        Timestamp: time.Now().UTC().Format(time.RFC3339Nano),
        Service:   service,
        Type:      eventType,
        Data:      data,
    }
    payload, err := json.Marshal(rec)
    if err != nil {
        return fmt.Errorf("marshal ledger record: %w", err)
    }
    if err := os.MkdirAll(filepath.Dir(filePath), 0o755); err != nil {
        return fmt.Errorf("mkdir: %w", err)
    }
    f, err := os.OpenFile(filePath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
    if err != nil {
        return fmt.Errorf("open ledger: %w", err)
    }
    defer f.Close()
    if _, err := f.Write(append(payload, '\n')); err != nil {
        return fmt.Errorf("write ledger: %w", err)
    }
    return nil
}



