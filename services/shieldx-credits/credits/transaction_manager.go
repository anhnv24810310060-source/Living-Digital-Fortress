package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"time"

	"github.com/google/uuid"
)

// TransactionManager handles distributed ACID transactions for credits
// Implements two-phase commit protocol for reserve -> commit/cancel flows
type TransactionManager struct {
	db *sql.DB
}

// TransactionState represents the state in two-phase commit
type TransactionState string

const (
	StatePending    TransactionState = "pending"
	StatePrepared   TransactionState = "prepared"
	StateCommitted  TransactionState = "committed"
	StateAborted    TransactionState = "aborted"
	StateRolledBack TransactionState = "rolled_back"
)

// DistributedTransaction represents a two-phase commit transaction
type DistributedTransaction struct {
	ID          string                 `json:"id"`
	TenantID    string                 `json:"tenant_id"`
	Type        string                 `json:"type"` // reserve, consume, topup
	Amount      int64                  `json:"amount"`
	State       TransactionState       `json:"state"`
	PreparedAt  *time.Time             `json:"prepared_at"`
	CommittedAt *time.Time             `json:"committed_at"`
	AbortedAt   *time.Time             `json:"aborted_at"`
	Timeout     time.Duration          `json:"timeout"`
	Metadata    map[string]interface{} `json:"metadata"`
	ParentTxID  string                 `json:"parent_tx_id"` // For nested transactions
	RetryCount  int                    `json:"retry_count"`
	LastError   string                 `json:"last_error,omitempty"`
	CreatedAt   time.Time              `json:"created_at"`
}

// NewTransactionManager creates a new transaction manager
func NewTransactionManager(db *sql.DB) *TransactionManager {
	return &TransactionManager{db: db}
}

// BeginDistributed starts a new distributed transaction with timeout
func (tm *TransactionManager) BeginDistributed(ctx context.Context, tenantID, txType string, amount int64, timeout time.Duration) (*DistributedTransaction, error) {
	if timeout == 0 {
		timeout = 30 * time.Second // Default timeout
	}

	dtx := &DistributedTransaction{
		ID:        uuid.New().String(),
		TenantID:  tenantID,
		Type:      txType,
		Amount:    amount,
		State:     StatePending,
		Timeout:   timeout,
		CreatedAt: time.Now(),
		Metadata:  make(map[string]interface{}),
	}

	// Persist transaction state
	query := `
		INSERT INTO distributed_transactions 
		(tx_id, tenant_id, tx_type, amount, state, timeout_seconds, created_at, metadata)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
	`
	metadataJSON, _ := json.Marshal(dtx.Metadata)
	_, err := tm.db.ExecContext(ctx, query,
		dtx.ID, dtx.TenantID, dtx.Type, dtx.Amount, dtx.State,
		int(dtx.Timeout.Seconds()), dtx.CreatedAt, metadataJSON,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create distributed transaction: %w", err)
	}

	return dtx, nil
}

// Prepare phase 1: Lock resources and validate
func (tm *TransactionManager) Prepare(ctx context.Context, dtx *DistributedTransaction) error {
	// Start DB transaction with serializable isolation
	tx, err := tm.db.BeginTx(ctx, &sql.TxOptions{
		Isolation: sql.LevelSerializable,
	})
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer tx.Rollback()

	// Check for timeout
	if time.Since(dtx.CreatedAt) > dtx.Timeout {
		dtx.State = StateAborted
		dtx.LastError = "transaction timeout in prepare phase"
		now := time.Now()
		dtx.AbortedAt = &now
		tm.updateTransactionState(ctx, dtx)
		return fmt.Errorf("transaction timeout")
	}

	// Lock the account row using SELECT FOR UPDATE
	var currentBalance, currentReserved int64
	query := `
		SELECT balance, reserved 
		FROM credit_accounts 
		WHERE tenant_id = $1 
		FOR UPDATE
	`
	err = tx.QueryRowContext(ctx, query, dtx.TenantID).Scan(&currentBalance, &currentReserved)
	if err != nil {
		if err == sql.ErrNoRows {
			// Create account if not exists
			insertQuery := `
				INSERT INTO credit_accounts (tenant_id, balance, reserved)
				VALUES ($1, 0, 0)
				ON CONFLICT (tenant_id) DO NOTHING
			`
			if _, err := tx.ExecContext(ctx, insertQuery, dtx.TenantID); err != nil {
				return fmt.Errorf("failed to create account: %w", err)
			}
			currentBalance = 0
			currentReserved = 0
		} else {
			return fmt.Errorf("failed to lock account: %w", err)
		}
	}

	// Validate based on transaction type
	switch dtx.Type {
	case "reserve", "consume":
		// Check sufficient balance (available = balance - reserved)
		available := currentBalance - currentReserved
		if available < dtx.Amount {
			dtx.State = StateAborted
			dtx.LastError = fmt.Sprintf("insufficient balance: need %d, have %d available", dtx.Amount, available)
			now := time.Now()
			dtx.AbortedAt = &now
			tm.updateTransactionState(ctx, dtx)
			return fmt.Errorf("insufficient balance")
		}

		// Reserve the amount
		if dtx.Type == "reserve" {
			updateQuery := `
				UPDATE credit_accounts 
				SET reserved = reserved + $1, updated_at = NOW()
				WHERE tenant_id = $2
			`
			if _, err := tx.ExecContext(ctx, updateQuery, dtx.Amount, dtx.TenantID); err != nil {
				return fmt.Errorf("failed to reserve credits: %w", err)
			}
		}

	case "topup":
		// Topup is always allowed (within reasonable limits)
		if dtx.Amount <= 0 || dtx.Amount > 1000000000 { // 1B credits max per topup
			dtx.State = StateAborted
			dtx.LastError = "invalid topup amount"
			now := time.Now()
			dtx.AbortedAt = &now
			tm.updateTransactionState(ctx, dtx)
			return fmt.Errorf("invalid topup amount")
		}
	default:
		return fmt.Errorf("unknown transaction type: %s", dtx.Type)
	}

	// Update transaction state to prepared
	now := time.Now()
	dtx.State = StatePrepared
	dtx.PreparedAt = &now
	updateQuery := `
		UPDATE distributed_transactions
		SET state = $1, prepared_at = $2
		WHERE tx_id = $3
	`
	if _, err := tx.ExecContext(ctx, updateQuery, dtx.State, dtx.PreparedAt, dtx.ID); err != nil {
		return fmt.Errorf("failed to update transaction state: %w", err)
	}

	return tx.Commit()
}

// Commit phase 2: Apply the changes permanently
func (tm *TransactionManager) Commit(ctx context.Context, dtx *DistributedTransaction) error {
	// Verify transaction is in prepared state
	if dtx.State != StatePrepared {
		return fmt.Errorf("transaction not in prepared state: %s", dtx.State)
	}

	tx, err := tm.db.BeginTx(ctx, &sql.TxOptions{
		Isolation: sql.LevelSerializable,
	})
	if err != nil {
		return fmt.Errorf("failed to begin commit transaction: %w", err)
	}
	defer tx.Rollback()

	// Apply the actual balance change
	switch dtx.Type {
	case "reserve":
		// Reserve already applied in Prepare, just mark committed
		// Reserved amount stays until explicit commit of parent transaction

	case "consume":
		// Deduct from balance
		updateQuery := `
			UPDATE credit_accounts 
			SET balance = balance - $1, updated_at = NOW()
			WHERE tenant_id = $2 AND balance >= $1
		`
		result, err := tx.ExecContext(ctx, updateQuery, dtx.Amount, dtx.TenantID)
		if err != nil {
			return fmt.Errorf("failed to consume credits: %w", err)
		}
		rows, _ := result.RowsAffected()
		if rows == 0 {
			return fmt.Errorf("insufficient balance during commit")
		}

	case "topup":
		// Add to balance
		updateQuery := `
			UPDATE credit_accounts 
			SET balance = balance + $1, updated_at = NOW()
			WHERE tenant_id = $2
		`
		if _, err := tx.ExecContext(ctx, updateQuery, dtx.Amount, dtx.TenantID); err != nil {
			return fmt.Errorf("failed to topup credits: %w", err)
		}
	}

	// Update transaction state to committed
	now := time.Now()
	dtx.State = StateCommitted
	dtx.CommittedAt = &now
	updateQuery := `
		UPDATE distributed_transactions
		SET state = $1, committed_at = $2
		WHERE tx_id = $3
	`
	if _, err := tx.ExecContext(ctx, updateQuery, dtx.State, dtx.CommittedAt, dtx.ID); err != nil {
		return fmt.Errorf("failed to update transaction state: %w", err)
	}

	return tx.Commit()
}

// Abort rolls back the transaction
func (tm *TransactionManager) Abort(ctx context.Context, dtx *DistributedTransaction) error {
	tx, err := tm.db.BeginTx(ctx, &sql.TxOptions{
		Isolation: sql.LevelSerializable,
	})
	if err != nil {
		return fmt.Errorf("failed to begin abort transaction: %w", err)
	}
	defer tx.Rollback()

	// If prepared, release reserved credits
	if dtx.State == StatePrepared && dtx.Type == "reserve" {
		updateQuery := `
			UPDATE credit_accounts 
			SET reserved = GREATEST(reserved - $1, 0), updated_at = NOW()
			WHERE tenant_id = $2
		`
		if _, err := tx.ExecContext(ctx, updateQuery, dtx.Amount, dtx.TenantID); err != nil {
			return fmt.Errorf("failed to release reserved credits: %w", err)
		}
	}

	// Mark transaction as aborted
	now := time.Now()
	dtx.State = StateAborted
	dtx.AbortedAt = &now
	updateQuery := `
		UPDATE distributed_transactions
		SET state = $1, aborted_at = $2, last_error = $3
		WHERE tx_id = $4
	`
	if _, err := tx.ExecContext(ctx, updateQuery, dtx.State, dtx.AbortedAt, dtx.LastError, dtx.ID); err != nil {
		return fmt.Errorf("failed to update transaction state: %w", err)
	}

	return tx.Commit()
}

// CommitReservation converts a reserved amount to actual consumption
func (tm *TransactionManager) CommitReservation(ctx context.Context, tenantID, reservationID string) error {
	tx, err := tm.db.BeginTx(ctx, &sql.TxOptions{
		Isolation: sql.LevelSerializable,
	})
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer tx.Rollback()

	// Get the reservation
	var amount int64
	var state string
	query := `
		SELECT amount, state 
		FROM distributed_transactions 
		WHERE tx_id = $1 AND tenant_id = $2 AND tx_type = 'reserve'
		FOR UPDATE
	`
	err = tx.QueryRowContext(ctx, query, reservationID, tenantID).Scan(&amount, &state)
	if err != nil {
		return fmt.Errorf("reservation not found: %w", err)
	}

	if state != string(StatePrepared) {
		return fmt.Errorf("reservation not in prepared state: %s", state)
	}

	// Move from reserved to consumed (deduct from both balance and reserved)
	updateQuery := `
		UPDATE credit_accounts 
		SET balance = balance - $1,
		    reserved = GREATEST(reserved - $1, 0),
		    updated_at = NOW()
		WHERE tenant_id = $2 AND balance >= $1
	`
	result, err := tx.ExecContext(ctx, updateQuery, amount, tenantID)
	if err != nil {
		return fmt.Errorf("failed to commit reservation: %w", err)
	}
	rows, _ := result.RowsAffected()
	if rows == 0 {
		return fmt.Errorf("insufficient balance to commit reservation")
	}

	// Mark reservation as committed
	now := time.Now()
	updateTxQuery := `
		UPDATE distributed_transactions
		SET state = $1, committed_at = $2
		WHERE tx_id = $3
	`
	if _, err := tx.ExecContext(ctx, updateTxQuery, StateCommitted, now, reservationID); err != nil {
		return fmt.Errorf("failed to update reservation state: %w", err)
	}

	return tx.Commit()
}

// CancelReservation releases a reserved amount back to available balance
func (tm *TransactionManager) CancelReservation(ctx context.Context, tenantID, reservationID string) error {
	tx, err := tm.db.BeginTx(ctx, &sql.TxOptions{
		Isolation: sql.LevelSerializable,
	})
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer tx.Rollback()

	// Get the reservation
	var amount int64
	var state string
	query := `
		SELECT amount, state 
		FROM distributed_transactions 
		WHERE tx_id = $1 AND tenant_id = $2 AND tx_type = 'reserve'
		FOR UPDATE
	`
	err = tx.QueryRowContext(ctx, query, reservationID, tenantID).Scan(&amount, &state)
	if err != nil {
		return fmt.Errorf("reservation not found: %w", err)
	}

	if state != string(StatePrepared) {
		return fmt.Errorf("reservation not in prepared state: %s", state)
	}

	// Release reserved amount
	updateQuery := `
		UPDATE credit_accounts 
		SET reserved = GREATEST(reserved - $1, 0),
		    updated_at = NOW()
		WHERE tenant_id = $2
	`
	if _, err := tx.ExecContext(ctx, updateQuery, amount, tenantID); err != nil {
		return fmt.Errorf("failed to cancel reservation: %w", err)
	}

	// Mark reservation as aborted
	now := time.Now()
	updateTxQuery := `
		UPDATE distributed_transactions
		SET state = $1, aborted_at = $2, last_error = 'cancelled by user'
		WHERE tx_id = $3
	`
	if _, err := tx.ExecContext(ctx, updateTxQuery, StateAborted, now, reservationID); err != nil {
		return fmt.Errorf("failed to update reservation state: %w", err)
	}

	return tx.Commit()
}

// CleanupExpiredTransactions removes old completed/aborted transactions
func (tm *TransactionManager) CleanupExpiredTransactions(ctx context.Context, retentionDays int) error {
	query := `
		DELETE FROM distributed_transactions
		WHERE (state = $1 OR state = $2)
		  AND created_at < NOW() - INTERVAL '1 day' * $3
	`
	result, err := tm.db.ExecContext(ctx, query, StateCommitted, StateAborted, retentionDays)
	if err != nil {
		return fmt.Errorf("failed to cleanup transactions: %w", err)
	}
	rows, _ := result.RowsAffected()
	if rows > 0 {
		fmt.Printf("Cleaned up %d expired transactions\n", rows)
	}
	return nil
}

// RecoverStalledTransactions handles transactions that exceeded timeout
func (tm *TransactionManager) RecoverStalledTransactions(ctx context.Context) error {
	// Find prepared transactions that are past timeout
	query := `
		SELECT tx_id, tenant_id, tx_type, amount, timeout_seconds, created_at
		FROM distributed_transactions
		WHERE state = $1
		  AND created_at < NOW() - (timeout_seconds || ' seconds')::INTERVAL
	`
	rows, err := tm.db.QueryContext(ctx, query, StatePrepared)
	if err != nil {
		return fmt.Errorf("failed to query stalled transactions: %w", err)
	}
	defer rows.Close()

	var stalledCount int
	for rows.Next() {
		var txID, tenantID, txType string
		var amount int64
		var timeoutSeconds int
		var createdAt time.Time

		if err := rows.Scan(&txID, &tenantID, &txType, &amount, &timeoutSeconds, &createdAt); err != nil {
			continue
		}

		// Abort the stalled transaction
		dtx := &DistributedTransaction{
			ID:        txID,
			TenantID:  tenantID,
			Type:      txType,
			Amount:    amount,
			State:     StatePrepared,
			Timeout:   time.Duration(timeoutSeconds) * time.Second,
			CreatedAt: createdAt,
			LastError: "transaction timeout - auto-aborted",
		}

		if err := tm.Abort(ctx, dtx); err != nil {
			fmt.Printf("Failed to abort stalled transaction %s: %v\n", txID, err)
		} else {
			stalledCount++
		}
	}

	if stalledCount > 0 {
		fmt.Printf("Recovered %d stalled transactions\n", stalledCount)
	}

	return nil
}

func (tm *TransactionManager) updateTransactionState(ctx context.Context, dtx *DistributedTransaction) error {
	query := `
		UPDATE distributed_transactions
		SET state = $1, last_error = $2, aborted_at = $3
		WHERE tx_id = $4
	`
	_, err := tm.db.ExecContext(ctx, query, dtx.State, dtx.LastError, dtx.AbortedAt, dtx.ID)
	return err
}
