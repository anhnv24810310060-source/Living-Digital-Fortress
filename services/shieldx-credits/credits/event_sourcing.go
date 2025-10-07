package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/google/uuid"
)

// EventSourcingEngine implements CQRS pattern with event sourcing
// This provides:
// - Complete audit trail (immutable events)
// - Time-travel debugging capability
// - Horizontal scalability through event replay
// - Eventual consistency guarantees
type EventSourcingEngine struct {
	db              *sql.DB
	eventStore      *EventStore
	commandHandlers map[string]CommandHandler
	eventHandlers   map[string]EventHandler
	projections     map[string]*Projection
	snapshotMgr     *SnapshotManager
	mu              sync.RWMutex
}

// Event types as constants for type safety
const (
	EventTypeCreditsPurchased     = "credits.purchased"
	EventTypeCreditsConsumed      = "credits.consumed"
	EventTypeCreditsReserved      = "credits.reserved"
	EventTypeCreditsCommitted     = "credits.committed"
	EventTypeCreditsCancelled     = "credits.cancelled"
	EventTypeThresholdUpdated     = "threshold.updated"
	EventTypeAccountCreated       = "account.created"
	EventTypeAccountSuspended     = "account.suspended"
	EventTypeAccountReactivated   = "account.reactivated"
)

// Event represents an immutable domain event
type Event struct {
	ID              string                 `json:"id"`
	AggregateID     string                 `json:"aggregate_id"`
	AggregateType   string                 `json:"aggregate_type"`
	EventType       string                 `json:"event_type"`
	EventData       map[string]interface{} `json:"event_data"`
	Metadata        map[string]interface{} `json:"metadata"`
	Version         int64                  `json:"version"`
	Timestamp       time.Time              `json:"timestamp"`
	CausationID     string                 `json:"causation_id,omitempty"`
	CorrelationID   string                 `json:"correlation_id,omitempty"`
}

// Command represents an action to be performed
type Command struct {
	ID            string                 `json:"id"`
	CommandType   string                 `json:"command_type"`
	AggregateID   string                 `json:"aggregate_id"`
	Payload       map[string]interface{} `json:"payload"`
	Metadata      map[string]interface{} `json:"metadata"`
	IdempotencyKey string                `json:"idempotency_key"`
	Timestamp     time.Time              `json:"timestamp"`
}

// CommandHandler processes commands and emits events
type CommandHandler func(ctx context.Context, cmd Command) ([]Event, error)

// EventHandler handles domain events for read model updates
type EventHandler func(ctx context.Context, event Event) error

// EventStore provides event persistence and streaming
type EventStore struct {
	db             *sql.DB
	eventBus       chan Event
	subscribers    []chan Event
	mu             sync.RWMutex
	publishTimeout time.Duration
}

// Projection maintains a read model derived from events
type Projection struct {
	Name          string
	Version       int64
	Handler       EventHandler
	lastProcessed int64
	mu            sync.RWMutex
}

// SnapshotManager optimizes event replay by storing aggregate state snapshots
type SnapshotManager struct {
	db                *sql.DB
	snapshotInterval  int64 // Create snapshot every N events
	retentionDays     int
}

// Snapshot represents aggregate state at a specific version
type Snapshot struct {
	AggregateID   string                 `json:"aggregate_id"`
	AggregateType string                 `json:"aggregate_type"`
	Version       int64                  `json:"version"`
	State         map[string]interface{} `json:"state"`
	Timestamp     time.Time              `json:"timestamp"`
}

// NewEventSourcingEngine creates a new event sourcing engine
func NewEventSourcingEngine(db *sql.DB) (*EventSourcingEngine, error) {
	eventStore := &EventStore{
		db:             db,
		eventBus:       make(chan Event, 1000),
		subscribers:    make([]chan Event, 0),
		publishTimeout: 5 * time.Second,
	}

	snapshotMgr := &SnapshotManager{
		db:               db,
		snapshotInterval: 100, // Snapshot every 100 events
		retentionDays:    90,
	}

	engine := &EventSourcingEngine{
		db:              db,
		eventStore:      eventStore,
		commandHandlers: make(map[string]CommandHandler),
		eventHandlers:   make(map[string]EventHandler),
		projections:     make(map[string]*Projection),
		snapshotMgr:     snapshotMgr,
	}

	// Initialize event store schema
	if err := engine.initializeSchema(); err != nil {
		return nil, fmt.Errorf("failed to initialize schema: %w", err)
	}

	// Start event bus processor
	go engine.processEventBus()

	// Register default command handlers
	engine.registerDefaultHandlers()

	// Start snapshot cleanup job
	go engine.snapshotCleanupJob()

	log.Printf("[event-sourcing] Engine initialized with CQRS pattern")
	return engine, nil
}

// initializeSchema creates necessary tables for event sourcing
func (es *EventSourcingEngine) initializeSchema() error {
	schema := `
	-- Event store table (immutable, append-only)
	CREATE TABLE IF NOT EXISTS event_store (
		id BIGSERIAL PRIMARY KEY,
		event_id UUID UNIQUE NOT NULL,
		aggregate_id VARCHAR(255) NOT NULL,
		aggregate_type VARCHAR(100) NOT NULL,
		event_type VARCHAR(100) NOT NULL,
		event_data JSONB NOT NULL,
		metadata JSONB,
		version BIGINT NOT NULL,
		timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
		causation_id UUID,
		correlation_id UUID,
		CONSTRAINT event_store_unique_version UNIQUE (aggregate_id, version)
	);

	-- Optimized indexes for event querying
	CREATE INDEX IF NOT EXISTS idx_event_store_aggregate 
		ON event_store(aggregate_id, version);
	CREATE INDEX IF NOT EXISTS idx_event_store_type 
		ON event_store(event_type, timestamp DESC);
	CREATE INDEX IF NOT EXISTS idx_event_store_correlation 
		ON event_store(correlation_id) WHERE correlation_id IS NOT NULL;
	CREATE INDEX IF NOT EXISTS idx_event_store_timestamp 
		ON event_store(timestamp DESC);

	-- Snapshot table for performance optimization
	CREATE TABLE IF NOT EXISTS aggregate_snapshots (
		id BIGSERIAL PRIMARY KEY,
		aggregate_id VARCHAR(255) NOT NULL,
		aggregate_type VARCHAR(100) NOT NULL,
		version BIGINT NOT NULL,
		state JSONB NOT NULL,
		timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
		CONSTRAINT snapshot_unique UNIQUE (aggregate_id, version)
	);

	CREATE INDEX IF NOT EXISTS idx_snapshots_aggregate 
		ON aggregate_snapshots(aggregate_id, version DESC);

	-- Projection tracking table
	CREATE TABLE IF NOT EXISTS projection_tracking (
		projection_name VARCHAR(100) PRIMARY KEY,
		last_processed_event_id BIGINT NOT NULL,
		last_processed_version BIGINT NOT NULL,
		last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
		status VARCHAR(50) DEFAULT 'active'
	);

	-- Command deduplication table
	CREATE TABLE IF NOT EXISTS command_log (
		command_id UUID PRIMARY KEY,
		command_type VARCHAR(100) NOT NULL,
		aggregate_id VARCHAR(255) NOT NULL,
		idempotency_key VARCHAR(255) UNIQUE,
		status VARCHAR(50) NOT NULL,
		created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
		completed_at TIMESTAMP WITH TIME ZONE,
		error_message TEXT
	);

	CREATE INDEX IF NOT EXISTS idx_command_log_idempotency 
		ON command_log(idempotency_key) WHERE idempotency_key IS NOT NULL;

	-- Read model: Credit balances (CQRS read side)
	CREATE TABLE IF NOT EXISTS credit_balances_read_model (
		tenant_id VARCHAR(255) PRIMARY KEY,
		balance BIGINT NOT NULL DEFAULT 0,
		reserved BIGINT NOT NULL DEFAULT 0,
		available BIGINT GENERATED ALWAYS AS (balance - reserved) STORED,
		total_consumed BIGINT NOT NULL DEFAULT 0,
		total_purchased BIGINT NOT NULL DEFAULT 0,
		transaction_count INTEGER NOT NULL DEFAULT 0,
		last_transaction_at TIMESTAMP WITH TIME ZONE,
		created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
		updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
		version BIGINT NOT NULL DEFAULT 1
	);

	CREATE INDEX IF NOT EXISTS idx_credit_balances_available 
		ON credit_balances_read_model(available) WHERE available > 0;
	CREATE INDEX IF NOT EXISTS idx_credit_balances_updated 
		ON credit_balances_read_model(updated_at DESC);
	`

	_, err := es.db.Exec(schema)
	if err != nil {
		return fmt.Errorf("schema creation failed: %w", err)
	}

	log.Printf("[event-sourcing] Schema initialized successfully")
	return nil
}

// RegisterCommandHandler registers a command handler
func (es *EventSourcingEngine) RegisterCommandHandler(commandType string, handler CommandHandler) {
	es.mu.Lock()
	defer es.mu.Unlock()
	es.commandHandlers[commandType] = handler
	log.Printf("[event-sourcing] Registered command handler: %s", commandType)
}

// RegisterEventHandler registers an event handler for projections
func (es *EventSourcingEngine) RegisterEventHandler(eventType string, handler EventHandler) {
	es.mu.Lock()
	defer es.mu.Unlock()
	es.eventHandlers[eventType] = handler
	log.Printf("[event-sourcing] Registered event handler: %s", eventType)
}

// HandleCommand processes a command and emits events
func (es *EventSourcingEngine) HandleCommand(ctx context.Context, cmd Command) error {
	// Check for duplicate command (idempotency)
	if cmd.IdempotencyKey != "" {
		exists, err := es.isCommandProcessed(ctx, cmd.IdempotencyKey)
		if err != nil {
			return fmt.Errorf("idempotency check failed: %w", err)
		}
		if exists {
			log.Printf("[event-sourcing] Duplicate command detected: %s", cmd.IdempotencyKey)
			return nil // Already processed
		}
	}

	// Log command attempt
	if err := es.logCommandAttempt(ctx, cmd); err != nil {
		return fmt.Errorf("failed to log command: %w", err)
	}

	// Find and execute handler
	es.mu.RLock()
	handler, ok := es.commandHandlers[cmd.CommandType]
	es.mu.RUnlock()

	if !ok {
		return fmt.Errorf("no handler registered for command type: %s", cmd.CommandType)
	}

	// Execute command handler
	events, err := handler(ctx, cmd)
	if err != nil {
		_ = es.logCommandFailure(ctx, cmd, err)
		return fmt.Errorf("command handler failed: %w", err)
	}

	// Persist events to event store
	for _, event := range events {
		if err := es.eventStore.AppendEvent(ctx, event); err != nil {
			_ = es.logCommandFailure(ctx, cmd, err)
			return fmt.Errorf("failed to append event: %w", err)
		}
	}

	// Mark command as completed
	_ = es.logCommandSuccess(ctx, cmd)

	// Check if snapshot is needed
	if len(events) > 0 {
		lastEvent := events[len(events)-1]
		if lastEvent.Version%es.snapshotMgr.snapshotInterval == 0 {
			go es.createSnapshotAsync(cmd.AggregateID, lastEvent.AggregateType)
		}
	}

	return nil
}

// AppendEvent appends an event to the event store
func (store *EventStore) AppendEvent(ctx context.Context, event Event) error {
	// Ensure event has ID and timestamp
	if event.ID == "" {
		event.ID = uuid.New().String()
	}
	if event.Timestamp.IsZero() {
		event.Timestamp = time.Now()
	}

	eventDataJSON, _ := json.Marshal(event.EventData)
	metadataJSON, _ := json.Marshal(event.Metadata)

	_, err := store.db.ExecContext(ctx, `
		INSERT INTO event_store (
			event_id, aggregate_id, aggregate_type, event_type,
			event_data, metadata, version, timestamp,
			causation_id, correlation_id
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
	`,
		event.ID, event.AggregateID, event.AggregateType, event.EventType,
		eventDataJSON, metadataJSON, event.Version, event.Timestamp,
		nullString(event.CausationID), nullString(event.CorrelationID),
	)

	if err != nil {
		return fmt.Errorf("failed to insert event: %w", err)
	}

	// Publish to event bus (non-blocking)
	select {
	case store.eventBus <- event:
		// Event published successfully
	case <-time.After(store.publishTimeout):
		log.Printf("[event-store] WARNING: Event bus full, event %s not published", event.ID)
	}

	return nil
}

// GetAggregateEvents retrieves all events for an aggregate
func (store *EventStore) GetAggregateEvents(ctx context.Context, aggregateID string, fromVersion int64) ([]Event, error) {
	rows, err := store.db.QueryContext(ctx, `
		SELECT event_id, aggregate_id, aggregate_type, event_type,
		       event_data, metadata, version, timestamp,
		       causation_id, correlation_id
		FROM event_store
		WHERE aggregate_id = $1 AND version >= $2
		ORDER BY version ASC
	`, aggregateID, fromVersion)

	if err != nil {
		return nil, fmt.Errorf("query failed: %w", err)
	}
	defer rows.Close()

	events := make([]Event, 0)
	for rows.Next() {
		var event Event
		var eventDataJSON, metadataJSON []byte
		var causationID, correlationID sql.NullString

		err := rows.Scan(
			&event.ID, &event.AggregateID, &event.AggregateType, &event.EventType,
			&eventDataJSON, &metadataJSON, &event.Version, &event.Timestamp,
			&causationID, &correlationID,
		)
		if err != nil {
			return nil, fmt.Errorf("scan failed: %w", err)
		}

		_ = json.Unmarshal(eventDataJSON, &event.EventData)
		_ = json.Unmarshal(metadataJSON, &event.Metadata)

		if causationID.Valid {
			event.CausationID = causationID.String
		}
		if correlationID.Valid {
			event.CorrelationID = correlationID.String
		}

		events = append(events, event)
	}

	return events, nil
}

// CreateSnapshot stores aggregate state snapshot
func (sm *SnapshotManager) CreateSnapshot(ctx context.Context, snapshot Snapshot) error {
	stateJSON, _ := json.Marshal(snapshot.State)

	_, err := sm.db.ExecContext(ctx, `
		INSERT INTO aggregate_snapshots (
			aggregate_id, aggregate_type, version, state, timestamp
		) VALUES ($1, $2, $3, $4, $5)
		ON CONFLICT (aggregate_id, version) DO NOTHING
	`, snapshot.AggregateID, snapshot.AggregateType, snapshot.Version, stateJSON, snapshot.Timestamp)

	return err
}

// GetLatestSnapshot retrieves the most recent snapshot for an aggregate
func (sm *SnapshotManager) GetLatestSnapshot(ctx context.Context, aggregateID string) (*Snapshot, error) {
	var snapshot Snapshot
	var stateJSON []byte

	err := sm.db.QueryRowContext(ctx, `
		SELECT aggregate_id, aggregate_type, version, state, timestamp
		FROM aggregate_snapshots
		WHERE aggregate_id = $1
		ORDER BY version DESC
		LIMIT 1
	`, aggregateID).Scan(
		&snapshot.AggregateID, &snapshot.AggregateType,
		&snapshot.Version, &stateJSON, &snapshot.Timestamp,
	)

	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}

	_ = json.Unmarshal(stateJSON, &snapshot.State)
	return &snapshot, nil
}

// processEventBus distributes events to subscribers
func (es *EventSourcingEngine) processEventBus() {
	for event := range es.eventStore.eventBus {
		// Find matching event handlers
		es.mu.RLock()
		handler, ok := es.eventHandlers[event.EventType]
		es.mu.RUnlock()

		if ok {
			// Process asynchronously to avoid blocking
			go func(e Event) {
				ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
				defer cancel()

				if err := handler(ctx, e); err != nil {
					log.Printf("[event-sourcing] Event handler failed for %s: %v", e.EventType, err)
				}
			}(event)
		}

		// Notify subscribers
		es.eventStore.mu.RLock()
		for _, subscriber := range es.eventStore.subscribers {
			select {
			case subscriber <- event:
			default:
				// Subscriber buffer full, skip
			}
		}
		es.eventStore.mu.RUnlock()
	}
}

// Subscribe creates a new event subscription channel
func (store *EventStore) Subscribe() <-chan Event {
	store.mu.Lock()
	defer store.mu.Unlock()

	ch := make(chan Event, 100)
	store.subscribers = append(store.subscribers, ch)
	return ch
}

// createSnapshotAsync creates a snapshot in the background
func (es *EventSourcingEngine) createSnapshotAsync(aggregateID, aggregateType string) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Get all events for aggregate
	events, err := es.eventStore.GetAggregateEvents(ctx, aggregateID, 0)
	if err != nil {
		log.Printf("[snapshot] Failed to get events: %v", err)
		return
	}

	if len(events) == 0 {
		return
	}

	// Rebuild aggregate state from events
	state := make(map[string]interface{})
	for _, event := range events {
		es.applyEventToState(state, event)
	}

	// Create snapshot
	snapshot := Snapshot{
		AggregateID:   aggregateID,
		AggregateType: aggregateType,
		Version:       events[len(events)-1].Version,
		State:         state,
		Timestamp:     time.Now(),
	}

	if err := es.snapshotMgr.CreateSnapshot(ctx, snapshot); err != nil {
		log.Printf("[snapshot] Failed to create snapshot: %v", err)
		return
	}

	log.Printf("[snapshot] Created snapshot for %s at version %d", aggregateID, snapshot.Version)
}

// applyEventToState updates aggregate state based on event
func (es *EventSourcingEngine) applyEventToState(state map[string]interface{}, event Event) {
	switch event.EventType {
	case EventTypeCreditsPurchased:
		balance, _ := state["balance"].(float64)
		amount, _ := event.EventData["amount"].(float64)
		state["balance"] = balance + amount

	case EventTypeCreditsConsumed:
		balance, _ := state["balance"].(float64)
		amount, _ := event.EventData["amount"].(float64)
		state["balance"] = balance - amount

	case EventTypeCreditsReserved:
		reserved, _ := state["reserved"].(float64)
		amount, _ := event.EventData["amount"].(float64)
		state["reserved"] = reserved + amount

	case EventTypeCreditsCommitted:
		reserved, _ := state["reserved"].(float64)
		amount, _ := event.EventData["amount"].(float64)
		state["reserved"] = reserved - amount
	}
}

// snapshotCleanupJob periodically removes old snapshots
func (es *EventSourcingEngine) snapshotCleanupJob() {
	ticker := time.NewTicker(24 * time.Hour)
	defer ticker.Stop()

	for range ticker.C {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
		cutoff := time.Now().AddDate(0, 0, -es.snapshotMgr.retentionDays)

		result, err := es.db.ExecContext(ctx, `
			DELETE FROM aggregate_snapshots
			WHERE timestamp < $1
			AND (aggregate_id, version) NOT IN (
				SELECT aggregate_id, MAX(version)
				FROM aggregate_snapshots
				GROUP BY aggregate_id
			)
		`, cutoff)

		cancel()

		if err != nil {
			log.Printf("[snapshot] Cleanup failed: %v", err)
			continue
		}

		rows, _ := result.RowsAffected()
		log.Printf("[snapshot] Cleanup completed: %d snapshots removed", rows)
	}
}

// Helper functions
func (es *EventSourcingEngine) isCommandProcessed(ctx context.Context, idempotencyKey string) (bool, error) {
	var exists bool
	err := es.db.QueryRowContext(ctx, `
		SELECT EXISTS(SELECT 1 FROM command_log WHERE idempotency_key = $1 AND status = 'completed')
	`, idempotencyKey).Scan(&exists)
	return exists, err
}

func (es *EventSourcingEngine) logCommandAttempt(ctx context.Context, cmd Command) error {
	_, err := es.db.ExecContext(ctx, `
		INSERT INTO command_log (command_id, command_type, aggregate_id, idempotency_key, status)
		VALUES ($1, $2, $3, $4, 'processing')
		ON CONFLICT (command_id) DO NOTHING
	`, cmd.ID, cmd.CommandType, cmd.AggregateID, nullString(cmd.IdempotencyKey))
	return err
}

func (es *EventSourcingEngine) logCommandSuccess(ctx context.Context, cmd Command) error {
	_, err := es.db.ExecContext(ctx, `
		UPDATE command_log SET status = 'completed', completed_at = NOW()
		WHERE command_id = $1
	`, cmd.ID)
	return err
}

func (es *EventSourcingEngine) logCommandFailure(ctx context.Context, cmd Command, cmdErr error) error {
	_, err := es.db.ExecContext(ctx, `
		UPDATE command_log SET status = 'failed', completed_at = NOW(), error_message = $2
		WHERE command_id = $1
	`, cmd.ID, cmdErr.Error())
	return err
}

func nullString(s string) sql.NullString {
	if s == "" {
		return sql.NullString{Valid: false}
	}
	return sql.NullString{String: s, Valid: true}
}

// registerDefaultHandlers sets up standard command handlers
func (es *EventSourcingEngine) registerDefaultHandlers() {
	// Purchase credits command
	es.RegisterCommandHandler("purchase_credits", func(ctx context.Context, cmd Command) ([]Event, error) {
		tenantID := cmd.AggregateID
		amount, ok := cmd.Payload["amount"].(float64)
		if !ok || amount <= 0 {
			return nil, fmt.Errorf("invalid amount")
		}

		// Get current version
		version, err := es.getCurrentVersion(ctx, tenantID)
		if err != nil {
			return nil, err
		}

		event := Event{
			ID:            uuid.New().String(),
			AggregateID:   tenantID,
			AggregateType: "credit_account",
			EventType:     EventTypeCreditsPurchased,
			EventData: map[string]interface{}{
				"amount":      amount,
				"description": cmd.Payload["description"],
			},
			Metadata:      cmd.Metadata,
			Version:       version + 1,
			Timestamp:     time.Now(),
			CorrelationID: cmd.ID,
		}

		return []Event{event}, nil
	})

	// Consume credits command with balance check
	es.RegisterCommandHandler("consume_credits", func(ctx context.Context, cmd Command) ([]Event, error) {
		tenantID := cmd.AggregateID
		amount, ok := cmd.Payload["amount"].(float64)
		if !ok || amount <= 0 {
			return nil, fmt.Errorf("invalid amount")
		}

		// Rebuild current state
		currentBalance, err := es.getAggregateBalance(ctx, tenantID)
		if err != nil {
			return nil, err
		}

		if currentBalance < int64(amount) {
			return nil, fmt.Errorf("insufficient credits: have %d, need %.0f", currentBalance, amount)
		}

		version, err := es.getCurrentVersion(ctx, tenantID)
		if err != nil {
			return nil, err
		}

		event := Event{
			ID:            uuid.New().String(),
			AggregateID:   tenantID,
			AggregateType: "credit_account",
			EventType:     EventTypeCreditsConsumed,
			EventData: map[string]interface{}{
				"amount":       amount,
				"description":  cmd.Payload["description"],
				"balance_after": float64(currentBalance) - amount,
			},
			Metadata:      cmd.Metadata,
			Version:       version + 1,
			Timestamp:     time.Now(),
			CorrelationID: cmd.ID,
		}

		return []Event{event}, nil
	})

	// Read model update handler
	es.RegisterEventHandler(EventTypeCreditsPurchased, es.updateReadModelPurchase)
	es.RegisterEventHandler(EventTypeCreditsConsumed, es.updateReadModelConsume)
}

// getCurrentVersion gets the latest version for an aggregate
func (es *EventSourcingEngine) getCurrentVersion(ctx context.Context, aggregateID string) (int64, error) {
	var version sql.NullInt64
	err := es.db.QueryRowContext(ctx, `
		SELECT MAX(version) FROM event_store WHERE aggregate_id = $1
	`, aggregateID).Scan(&version)

	if err != nil {
		return 0, err
	}
	if !version.Valid {
		return 0, nil
	}
	return version.Int64, nil
}

// getAggregateBalance calculates current balance from events (or snapshot)
func (es *EventSourcingEngine) getAggregateBalance(ctx context.Context, aggregateID string) (int64, error) {
	// Try to get latest snapshot first
	snapshot, err := es.snapshotMgr.GetLatestSnapshot(ctx, aggregateID)
	
	var balance int64
	var fromVersion int64 = 0
	
	if snapshot != nil {
		if bal, ok := snapshot.State["balance"].(float64); ok {
			balance = int64(bal)
		}
		fromVersion = snapshot.Version + 1
	}

	// Get events since snapshot
	events, err := es.eventStore.GetAggregateEvents(ctx, aggregateID, fromVersion)
	if err != nil {
		return 0, err
	}

	// Apply events to balance
	for _, event := range events {
		switch event.EventType {
		case EventTypeCreditsPurchased:
			amount, _ := event.EventData["amount"].(float64)
			balance += int64(amount)
		case EventTypeCreditsConsumed:
			amount, _ := event.EventData["amount"].(float64)
			balance -= int64(amount)
		}
	}

	return balance, nil
}

// updateReadModelPurchase updates read model on purchase event
func (es *EventSourcingEngine) updateReadModelPurchase(ctx context.Context, event Event) error {
	amount, _ := event.EventData["amount"].(float64)

	_, err := es.db.ExecContext(ctx, `
		INSERT INTO credit_balances_read_model (
			tenant_id, balance, total_purchased, transaction_count, last_transaction_at
		) VALUES ($1, $2, $2, 1, $3)
		ON CONFLICT (tenant_id) DO UPDATE SET
			balance = credit_balances_read_model.balance + $2,
			total_purchased = credit_balances_read_model.total_purchased + $2,
			transaction_count = credit_balances_read_model.transaction_count + 1,
			last_transaction_at = $3,
			updated_at = NOW(),
			version = credit_balances_read_model.version + 1
	`, event.AggregateID, int64(amount), event.Timestamp)

	return err
}

// updateReadModelConsume updates read model on consume event
func (es *EventSourcingEngine) updateReadModelConsume(ctx context.Context, event Event) error {
	amount, _ := event.EventData["amount"].(float64)

	_, err := es.db.ExecContext(ctx, `
		UPDATE credit_balances_read_model SET
			balance = balance - $2,
			total_consumed = total_consumed + $2,
			transaction_count = transaction_count + 1,
			last_transaction_at = $3,
			updated_at = NOW(),
			version = version + 1
		WHERE tenant_id = $1
	`, event.AggregateID, int64(amount), event.Timestamp)

	return err
}

// Close shuts down the event sourcing engine
func (es *EventSourcingEngine) Close() error {
	close(es.eventStore.eventBus)
	return es.db.Close()
}
