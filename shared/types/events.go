package types

import "time"

// TelemetryEvent is a shared minimal event envelope for cross-service communication.
// Full feature vectors & proprietary fields stay internal to producing service until normalized.
type TelemetryEvent struct {
    Timestamp   time.Time
    Source      string
    EventType   string
    TenantID    string
    SessionID   string
    Metadata    map[string]any
    Features    []float64
    ThreatScore float64
}
