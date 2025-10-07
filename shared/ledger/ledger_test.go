package ledger

import (
    "os"
    "testing"
)

func TestAppendJSONLine(t *testing.T) {
    tmp := t.TempDir() + "/test-ledger.log"
    if err := AppendJSONLine(tmp, "test-service", "unit", map[string]any{"k":"v"}); err != nil {
        t.Fatalf("append failed: %v", err)
    }
    data, err := os.ReadFile(tmp)
    if err != nil || len(data) == 0 { t.Fatalf("expected data written") }
}
