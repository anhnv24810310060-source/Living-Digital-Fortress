package logging

import (
    "log"
    "time"
)

// Simple wrapper; can be replaced with structured logger later.
func Infof(format string, args ...any) {
    log.Printf("INFO  %s "+format, append([]any{time.Now().UTC().Format(time.RFC3339Nano)}, args...)...)
}

func Errorf(format string, args ...any) {
    log.Printf("ERROR %s "+format, append([]any{time.Now().UTC().Format(time.RFC3339Nano)}, args...)...)
}
