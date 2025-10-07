package main

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"
)

// EventType represents different credit operations as immutable events
type EventType string

const (
	EventCreditPurchased  EventType = "CREDIT_PURCHASED"
	EventCreditConsumed   EventType = "CREDIT_CONSUMED"
	EventCreditReserved   EventType = "CREDIT_RESERVED"
	EventCreditCommitted  EventType = "CREDIT_COMMITTED"
	EventCreditCancelled  EventType = "CREDIT_CANCELLED"
	EventCreditRefunded   EventType = "CREDIT_REFUNDED"
	EventThresholdAlerted EventType = "THRESHOLD_ALERTED"
)

// Event is an immutable record of state change - Event Sourcing pattern
type CreditEvent struct {
	EventID     string                 `json:"event_id"`
	EventType   EventType              `json:"event_type"`
	AggregateID string                 `json:"aggregate_id"` // tenant_id
	Timestamp   time.Time              `json:"timestamp"`
	Version     int64                  `json:"version"` // for optimistic locking
	Payload     map[string]interface{} `json:"payload"`
	Metadata    map[string]string      `json:"metadata"` // causation_id, correlation_id, user_id
}

// EventStore handles immutable event persistence with Kafka-like guarantees
type EventStore struct {
	db        *CreditLedger
	mu        sync.RWMutex
	snapshots map[string]*AggregateSnapshot // in-memory cache
	metrics   *EventStoreMetrics
}

// AggregateSnapshot for performance optimization - CQRS read model
type AggregateSnapshot struct {
	AggregateID   string    `json:"aggregate_id"`
	Version       int64     `json:"version"`
	Balance       int64     `json:"balance"`
	ReservedTotal int64     `json:"reserved_total"`
	UpdatedAt     time.Time `json:"updated_at"`
}

type EventStoreMetrics struct {
	EventsAppended         uint64
	SnapshotCreated        uint64
	EventsReplayed         uint64
	OptimisticLockFailures uint64
}

// NewEventStore initializes event-sourced storage with CQRS pattern
func NewEventStore(db *CreditLedger) *EventStore {
	es := &EventStore{
		db:        db,
		snapshots: make(map[string]*AggregateSnapshot),
		metrics:   &EventStoreMetrics{},
	}
	// Load snapshots from DB on startup
	es.loadSnapshots()
	return es
}

// AppendEvent stores immutable event with optimistic locking
// ✅ PHẢI immutable audit logs
// ✅ PHẢI dùng database transactions
func (es *EventStore) AppendEvent(ctx context.Context, event *CreditEvent) error {
	if event.EventID == "" {
		event.EventID = uuid.New().String()
	}
	if event.Timestamp.IsZero() {
		event.Timestamp = time.Now()
	}

	// Optimistic locking check
	es.mu.RLock()
	snapshot := es.snapshots[event.AggregateID]
	es.mu.RUnlock()

	if snapshot != nil && event.Version > 0 && event.Version != snapshot.Version+1 {
		es.metrics.OptimisticLockFailures++
		return fmt.Errorf("optimistic lock failure: expected version %d, got %d",
			snapshot.Version+1, event.Version)
	}

	// Persist to database within transaction
	tx, err := es.db.db.Begin()
	if err != nil {
		return fmt.Errorf("begin transaction: %w", err)
	}
	defer tx.Rollback()

	// Store event in events table
	eventJSON, _ := json.Marshal(event)
	_, err = tx.Exec(`
		INSERT INTO credit_events 
		(event_id, event_type, aggregate_id, timestamp, version, payload, metadata)
		VALUES ($1, $2, $3, $4, $5, $6, $7)
	`, event.EventID, event.EventType, event.AggregateID, event.Timestamp,
		event.Version, eventJSON, mustJSON(event.Metadata))

	if err != nil {
		return fmt.Errorf("insert event: %w", err)
	}

	// Update snapshot (CQRS read model projection)
	if err := es.updateSnapshot(tx, event); err != nil {
		return fmt.Errorf("update snapshot: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit transaction: %w", err)
	}

	es.metrics.EventsAppended++

	// Update in-memory cache
	es.mu.Lock()
	if snapshot == nil {
		snapshot = &AggregateSnapshot{AggregateID: event.AggregateID}
	}
	es.applyEventToSnapshot(snapshot, event)
	es.snapshots[event.AggregateID] = snapshot
	es.mu.Unlock()

	return nil
}

// GetAggregate rebuilds current state from events (time-travel debugging support)
func (es *EventStore) GetAggregate(ctx context.Context, aggregateID string) (*AggregateSnapshot, error) {
	// Try cache first
	es.mu.RLock()
	if snapshot, ok := es.snapshots[aggregateID]; ok {
		es.mu.RUnlock()
		return snapshot, nil
	}
	es.mu.RUnlock()

	// Replay events from database
	rows, err := es.db.db.Query(`
		SELECT event_id, event_type, timestamp, version, payload
		FROM credit_events
		WHERE aggregate_id = $1
		ORDER BY version ASC
	`, aggregateID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	snapshot := &AggregateSnapshot{
		AggregateID: aggregateID,
	}

	for rows.Next() {
		var event CreditEvent
		var payloadJSON []byte
		if err := rows.Scan(&event.EventID, &event.EventType, &event.Timestamp,
			&event.Version, &payloadJSON); err != nil {
			return nil, err
		}
		json.Unmarshal(payloadJSON, &event.Payload)
		es.applyEventToSnapshot(snapshot, &event)
		es.metrics.EventsReplayed++
	}

	// Cache it
	es.mu.Lock()
	es.snapshots[aggregateID] = snapshot
	es.mu.Unlock()

	return snapshot, nil
}

// applyEventToSnapshot projects event to read model
func (es *EventStore) applyEventToSnapshot(snapshot *AggregateSnapshot, event *CreditEvent) {
	snapshot.Version = event.Version
	snapshot.UpdatedAt = event.Timestamp

	switch event.EventType {
	case EventCreditPurchased:
		if amount, ok := event.Payload["amount"].(float64); ok {
			snapshot.Balance += int64(amount)
		}
	case EventCreditConsumed:
		if amount, ok := event.Payload["amount"].(float64); ok {
			snapshot.Balance -= int64(amount)
		}
	case EventCreditReserved:
		if amount, ok := event.Payload["amount"].(float64); ok {
			snapshot.ReservedTotal += int64(amount)
		}
	case EventCreditCommitted:
		if amount, ok := event.Payload["amount"].(float64); ok {
			snapshot.ReservedTotal -= int64(amount)
			snapshot.Balance -= int64(amount)
		}
	case EventCreditCancelled:
		if amount, ok := event.Payload["amount"].(float64); ok {
			snapshot.ReservedTotal -= int64(amount)
		}
	case EventCreditRefunded:
		if amount, ok := event.Payload["amount"].(float64); ok {
			snapshot.Balance += int64(amount)
		}
	}
}

// updateSnapshot updates materialized view in database
func (es *EventStore) updateSnapshot(tx interface {
	Exec(query string, args ...interface{}) (interface{}, error)
}, event *CreditEvent) error {
	// Upsert snapshot
	_, err := tx.Exec(`
		INSERT INTO credit_snapshots (aggregate_id, version, balance, reserved_total, updated_at)
		VALUES ($1, $2, 
			CASE WHEN $3 = 'CREDIT_PURCHASED' THEN COALESCE((SELECT balance FROM credit_snapshots WHERE aggregate_id = $1), 0) + ($4->>'amount')::bigint
			     WHEN $3 = 'CREDIT_CONSUMED' THEN COALESCE((SELECT balance FROM credit_snapshots WHERE aggregate_id = $1), 0) - ($4->>'amount')::bigint
			     ELSE COALESCE((SELECT balance FROM credit_snapshots WHERE aggregate_id = $1), 0)
			END,
			CASE WHEN $3 = 'CREDIT_RESERVED' THEN COALESCE((SELECT reserved_total FROM credit_snapshots WHERE aggregate_id = $1), 0) + ($4->>'amount')::bigint
			     WHEN $3 IN ('CREDIT_COMMITTED', 'CREDIT_CANCELLED') THEN COALESCE((SELECT reserved_total FROM credit_snapshots WHERE aggregate_id = $1), 0) - ($4->>'amount')::bigint
			     ELSE COALESCE((SELECT reserved_total FROM credit_snapshots WHERE aggregate_id = $1), 0)
			END,
			$5)
		ON CONFLICT (aggregate_id) DO UPDATE SET
			version = EXCLUDED.version,
			balance = EXCLUDED.balance,
			reserved_total = EXCLUDED.reserved_total,
			updated_at = EXCLUDED.updated_at
	`, event.AggregateID, event.Version, event.EventType, mustJSON(event.Payload), event.Timestamp)

	es.metrics.SnapshotCreated++
	return err
}

// loadSnapshots pre-loads all snapshots into memory on startup
func (es *EventStore) loadSnapshots() {
	rows, err := es.db.db.Query(`
		SELECT aggregate_id, version, balance, reserved_total, updated_at
		FROM credit_snapshots
		ORDER BY updated_at DESC
		LIMIT 10000
	`)
	if err != nil {
		return
	}
	defer rows.Close()

	es.mu.Lock()
	defer es.mu.Unlock()

	for rows.Next() {
		var snap AggregateSnapshot
		rows.Scan(&snap.AggregateID, &snap.Version, &snap.Balance, &snap.ReservedTotal, &snap.UpdatedAt)
		es.snapshots[snap.AggregateID] = &snap
	}
}

// GetEventHistory returns all events for time-travel debugging
func (es *EventStore) GetEventHistory(ctx context.Context, aggregateID string, fromVersion int64) ([]*CreditEvent, error) {
	rows, err := es.db.db.Query(`
		SELECT event_id, event_type, timestamp, version, payload, metadata
		FROM credit_events
		WHERE aggregate_id = $1 AND version >= $2
		ORDER BY version ASC
	`, aggregateID, fromVersion)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var events []*CreditEvent
	for rows.Next() {
		var event CreditEvent
		var payloadJSON, metadataJSON []byte
		if err := rows.Scan(&event.EventID, &event.EventType, &event.Timestamp,
			&event.Version, &payloadJSON, &metadataJSON); err != nil {
			return nil, err
		}
		event.AggregateID = aggregateID
		json.Unmarshal(payloadJSON, &event.Payload)
		json.Unmarshal(metadataJSON, &event.Metadata)
		events = append(events, &event)
	}

	return events, nil
}

func mustJSON(v interface{}) []byte {
	b, _ := json.Marshal(v)
	return b
}
