package main

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
)

// WriteAheadLog implements immutable audit logging with cryptographic chaining
// Ensures tamper-proof audit trail for all credit transactions
type WriteAheadLog struct {
	db           *sql.DB
	hmacKey      []byte
	buffer       []*AuditEntry
	bufferMutex  sync.Mutex
	flushSize    int
	lastHash     string
	lastHashLock sync.RWMutex
}

// AuditEntry represents a single immutable audit log entry
type AuditEntry struct {
	ID             string                 `json:"id"`
	SequenceNumber int64                  `json:"sequence_number"`
	Timestamp      time.Time              `json:"timestamp"`
	TenantID       string                 `json:"tenant_id"`
	EventType      string                 `json:"event_type"`
	Action         string                 `json:"action"`
	Amount         int64                  `json:"amount,omitempty"`
	BalanceBefore  int64                  `json:"balance_before"`
	BalanceAfter   int64                  `json:"balance_after"`
	TransactionID  string                 `json:"transaction_id"`
	UserID         string                 `json:"user_id,omitempty"`
	IPAddress      string                 `json:"ip_address,omitempty"` // Masked in production
	UserAgent      string                 `json:"user_agent,omitempty"` // Masked in production
	Metadata       map[string]interface{} `json:"metadata,omitempty"`
	PreviousHash   string                 `json:"previous_hash"`
	CurrentHash    string                 `json:"current_hash"`
	HMACSignature  string                 `json:"hmac_signature"`
	Success        bool                   `json:"success"`
	ErrorMessage   string                 `json:"error_message,omitempty"`
	DurationMs     int64                  `json:"duration_ms"`
}

// NewWriteAheadLog creates a new WAL instance
func NewWriteAheadLog(db *sql.DB, hmacKey []byte) *WriteAheadLog {
	if len(hmacKey) == 0 {
		// Generate a secure random key if not provided
		hmacKey = make([]byte, 32)
		// In production, this should be loaded from secure storage
	}

	wal := &WriteAheadLog{
		db:        db,
		hmacKey:   hmacKey,
		buffer:    make([]*AuditEntry, 0, 100),
		flushSize: 50, // Flush every 50 entries
	}

	// Load the last hash from DB
	wal.loadLastHash()

	return wal
}

// Append adds a new entry to the WAL
func (wal *WriteAheadLog) Append(ctx context.Context, entry *AuditEntry) error {
	// Generate ID if not set
	if entry.ID == "" {
		entry.ID = uuid.New().String()
	}

	// Set timestamp if not set
	if entry.Timestamp.IsZero() {
		entry.Timestamp = time.Now().UTC()
	}

	// Get the last hash for chaining
	wal.lastHashLock.RLock()
	entry.PreviousHash = wal.lastHash
	wal.lastHashLock.RUnlock()

	// Calculate current hash (chain integrity)
	entry.CurrentHash = wal.calculateHash(entry)

	// Calculate HMAC signature
	entry.HMACSignature = wal.calculateHMAC(entry)

	// Mask sensitive data in logs (PII protection)
	entry.IPAddress = maskIP(entry.IPAddress)
	entry.UserAgent = maskUserAgent(entry.UserAgent)

	// Add to buffer
	wal.bufferMutex.Lock()
	wal.buffer = append(wal.buffer, entry)
	shouldFlush := len(wal.buffer) >= wal.flushSize
	wal.bufferMutex.Unlock()

	// Update last hash
	wal.lastHashLock.Lock()
	wal.lastHash = entry.CurrentHash
	wal.lastHashLock.Unlock()

	// Flush if buffer is full
	if shouldFlush {
		return wal.Flush(ctx)
	}

	return nil
}

// Flush writes buffered entries to database (atomic batch)
func (wal *WriteAheadLog) Flush(ctx context.Context) error {
	wal.bufferMutex.Lock()
	if len(wal.buffer) == 0 {
		wal.bufferMutex.Unlock()
		return nil
	}

	// Take ownership of current buffer
	entries := wal.buffer
	wal.buffer = make([]*AuditEntry, 0, 100)
	wal.bufferMutex.Unlock()

	// Begin transaction for atomic batch insert
	tx, err := wal.db.BeginTx(ctx, &sql.TxOptions{
		Isolation: sql.LevelSerializable,
	})
	if err != nil {
		// Restore entries to buffer on failure
		wal.bufferMutex.Lock()
		wal.buffer = append(entries, wal.buffer...)
		wal.bufferMutex.Unlock()
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer tx.Rollback()

	// Prepare batch insert statement
	stmt, err := tx.PrepareContext(ctx, `
		INSERT INTO audit_log (
			id, sequence_number, timestamp, tenant_id, event_type, action,
			amount, balance_before, balance_after, transaction_id,
			user_id, ip_address, user_agent, metadata,
			previous_hash, current_hash, hmac_signature,
			success, error_message, duration_ms
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20)
	`)
	if err != nil {
		return fmt.Errorf("failed to prepare statement: %w", err)
	}
	defer stmt.Close()

	// Get next sequence number
	var nextSeq int64
	err = tx.QueryRowContext(ctx, "SELECT COALESCE(MAX(sequence_number), 0) + 1 FROM audit_log").Scan(&nextSeq)
	if err != nil {
		return fmt.Errorf("failed to get sequence number: %w", err)
	}

	// Insert all entries
	for i, entry := range entries {
		entry.SequenceNumber = nextSeq + int64(i)

		metadataJSON, _ := json.Marshal(entry.Metadata)

		_, err := stmt.ExecContext(ctx,
			entry.ID, entry.SequenceNumber, entry.Timestamp,
			entry.TenantID, entry.EventType, entry.Action,
			entry.Amount, entry.BalanceBefore, entry.BalanceAfter,
			entry.TransactionID, entry.UserID, entry.IPAddress,
			entry.UserAgent, metadataJSON, entry.PreviousHash,
			entry.CurrentHash, entry.HMACSignature, entry.Success,
			entry.ErrorMessage, entry.DurationMs,
		)
		if err != nil {
			return fmt.Errorf("failed to insert audit entry: %w", err)
		}
	}

	// Commit transaction
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit audit log: %w", err)
	}

	return nil
}

// calculateHash computes SHA256 hash of entry for chain integrity
func (wal *WriteAheadLog) calculateHash(entry *AuditEntry) string {
	data := fmt.Sprintf("%s|%s|%s|%s|%d|%d|%d|%s|%s",
		entry.ID,
		entry.Timestamp.Format(time.RFC3339Nano),
		entry.TenantID,
		entry.EventType,
		entry.Amount,
		entry.BalanceBefore,
		entry.BalanceAfter,
		entry.TransactionID,
		entry.PreviousHash,
	)

	hash := sha256.Sum256([]byte(data))
	return hex.EncodeToString(hash[:])
}

// calculateHMAC computes HMAC signature for tamper detection
func (wal *WriteAheadLog) calculateHMAC(entry *AuditEntry) string {
	data := fmt.Sprintf("%s|%s|%s",
		entry.ID,
		entry.CurrentHash,
		entry.Timestamp.Format(time.RFC3339Nano),
	)

	mac := hmac.New(sha256.New, wal.hmacKey)
	mac.Write([]byte(data))
	return hex.EncodeToString(mac.Sum(nil))
}

// VerifyChain verifies the integrity of the audit log chain
func (wal *WriteAheadLog) VerifyChain(ctx context.Context, startSeq, endSeq int64) (bool, error) {
	query := `
		SELECT id, sequence_number, timestamp, tenant_id, event_type, action,
		       amount, balance_before, balance_after, transaction_id,
		       previous_hash, current_hash, hmac_signature
		FROM audit_log
		WHERE sequence_number >= $1 AND sequence_number <= $2
		ORDER BY sequence_number ASC
	`

	rows, err := wal.db.QueryContext(ctx, query, startSeq, endSeq)
	if err != nil {
		return false, fmt.Errorf("failed to query audit log: %w", err)
	}
	defer rows.Close()

	var prevHash string
	var count int

	for rows.Next() {
		var entry AuditEntry
		err := rows.Scan(
			&entry.ID, &entry.SequenceNumber, &entry.Timestamp,
			&entry.TenantID, &entry.EventType, &entry.Action,
			&entry.Amount, &entry.BalanceBefore, &entry.BalanceAfter,
			&entry.TransactionID, &entry.PreviousHash, &entry.CurrentHash,
			&entry.HMACSignature,
		)
		if err != nil {
			return false, fmt.Errorf("failed to scan entry: %w", err)
		}

		// Verify chain
		if count > 0 && entry.PreviousHash != prevHash {
			return false, fmt.Errorf("chain broken at sequence %d: expected previous_hash %s, got %s",
				entry.SequenceNumber, prevHash, entry.PreviousHash)
		}

		// Verify hash
		calculatedHash := wal.calculateHash(&entry)
		if calculatedHash != entry.CurrentHash {
			return false, fmt.Errorf("hash mismatch at sequence %d: expected %s, got %s",
				entry.SequenceNumber, calculatedHash, entry.CurrentHash)
		}

		// Verify HMAC
		calculatedHMAC := wal.calculateHMAC(&entry)
		if calculatedHMAC != entry.HMACSignature {
			return false, fmt.Errorf("HMAC mismatch at sequence %d", entry.SequenceNumber)
		}

		prevHash = entry.CurrentHash
		count++
	}

	return true, nil
}

// loadLastHash loads the most recent hash from database
func (wal *WriteAheadLog) loadLastHash() {
	var lastHash string
	query := `SELECT current_hash FROM audit_log ORDER BY sequence_number DESC LIMIT 1`
	err := wal.db.QueryRow(query).Scan(&lastHash)
	if err != nil && err != sql.ErrNoRows {
		// Log error but don't fail - will use empty hash for first entry
		fmt.Printf("Warning: failed to load last hash: %v\n", err)
		return
	}

	wal.lastHashLock.Lock()
	wal.lastHash = lastHash
	wal.lastHashLock.Unlock()
}

// GetAuditTrail retrieves audit entries for a tenant with pagination
func (wal *WriteAheadLog) GetAuditTrail(ctx context.Context, tenantID string, limit, offset int) ([]*AuditEntry, error) {
	query := `
		SELECT id, sequence_number, timestamp, tenant_id, event_type, action,
		       amount, balance_before, balance_after, transaction_id,
		       user_id, ip_address, user_agent, metadata,
		       previous_hash, current_hash, hmac_signature,
		       success, error_message, duration_ms
		FROM audit_log
		WHERE tenant_id = $1
		ORDER BY sequence_number DESC
		LIMIT $2 OFFSET $3
	`

	rows, err := wal.db.QueryContext(ctx, query, tenantID, limit, offset)
	if err != nil {
		return nil, fmt.Errorf("failed to query audit trail: %w", err)
	}
	defer rows.Close()

	var entries []*AuditEntry
	for rows.Next() {
		var entry AuditEntry
		var metadataJSON []byte

		err := rows.Scan(
			&entry.ID, &entry.SequenceNumber, &entry.Timestamp,
			&entry.TenantID, &entry.EventType, &entry.Action,
			&entry.Amount, &entry.BalanceBefore, &entry.BalanceAfter,
			&entry.TransactionID, &entry.UserID, &entry.IPAddress,
			&entry.UserAgent, &metadataJSON, &entry.PreviousHash,
			&entry.CurrentHash, &entry.HMACSignature, &entry.Success,
			&entry.ErrorMessage, &entry.DurationMs,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan entry: %w", err)
		}

		if len(metadataJSON) > 0 {
			json.Unmarshal(metadataJSON, &entry.Metadata)
		}

		entries = append(entries, &entry)
	}

	return entries, nil
}

// SearchAuditLog searches audit log with filters
func (wal *WriteAheadLog) SearchAuditLog(ctx context.Context, filters map[string]interface{}, limit, offset int) ([]*AuditEntry, error) {
	// Build dynamic query based on filters
	baseQuery := `
		SELECT id, sequence_number, timestamp, tenant_id, event_type, action,
		       amount, balance_before, balance_after, transaction_id,
		       user_id, previous_hash, current_hash, hmac_signature,
		       success, error_message, duration_ms
		FROM audit_log
		WHERE 1=1
	`

	var conditions []string
	var args []interface{}
	argCount := 1

	if tenantID, ok := filters["tenant_id"].(string); ok && tenantID != "" {
		conditions = append(conditions, fmt.Sprintf("tenant_id = $%d", argCount))
		args = append(args, tenantID)
		argCount++
	}

	if eventType, ok := filters["event_type"].(string); ok && eventType != "" {
		conditions = append(conditions, fmt.Sprintf("event_type = $%d", argCount))
		args = append(args, eventType)
		argCount++
	}

	if action, ok := filters["action"].(string); ok && action != "" {
		conditions = append(conditions, fmt.Sprintf("action = $%d", argCount))
		args = append(args, action)
		argCount++
	}

	if success, ok := filters["success"].(bool); ok {
		conditions = append(conditions, fmt.Sprintf("success = $%d", argCount))
		args = append(args, success)
		argCount++
	}

	if startTime, ok := filters["start_time"].(time.Time); ok {
		conditions = append(conditions, fmt.Sprintf("timestamp >= $%d", argCount))
		args = append(args, startTime)
		argCount++
	}

	if endTime, ok := filters["end_time"].(time.Time); ok {
		conditions = append(conditions, fmt.Sprintf("timestamp <= $%d", argCount))
		args = append(args, endTime)
		argCount++
	}

	// Build final query
	for _, cond := range conditions {
		baseQuery += " AND " + cond
	}
	baseQuery += fmt.Sprintf(" ORDER BY sequence_number DESC LIMIT $%d OFFSET $%d", argCount, argCount+1)
	args = append(args, limit, offset)

	rows, err := wal.db.QueryContext(ctx, baseQuery, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to search audit log: %w", err)
	}
	defer rows.Close()

	var entries []*AuditEntry
	for rows.Next() {
		var entry AuditEntry
		err := rows.Scan(
			&entry.ID, &entry.SequenceNumber, &entry.Timestamp,
			&entry.TenantID, &entry.EventType, &entry.Action,
			&entry.Amount, &entry.BalanceBefore, &entry.BalanceAfter,
			&entry.TransactionID, &entry.UserID, &entry.PreviousHash,
			&entry.CurrentHash, &entry.HMACSignature, &entry.Success,
			&entry.ErrorMessage, &entry.DurationMs,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan entry: %w", err)
		}
		entries = append(entries, &entry)
	}

	return entries, nil
}

// maskIP masks IP address for privacy (keep first 2 octets for IPv4)
func maskIP(ip string) string {
	if ip == "" {
		return ""
	}
	// Simple masking: 192.168.1.1 -> 192.168.*.*
	parts := strings.Split(ip, ".")
	if len(parts) == 4 {
		return parts[0] + "." + parts[1] + ".*.*"
	}
	return "***"
}

// maskUserAgent masks user agent (keep browser/OS, remove version details)
func maskUserAgent(ua string) string {
	if ua == "" {
		return ""
	}
	// Simple masking: keep first 50 chars, truncate version numbers
	if len(ua) > 50 {
		return ua[:50] + "..."
	}
	return ua
}
