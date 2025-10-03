package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"crypto/aes"
	"crypto/cipher"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"

	"shieldx/pkg/ledger"

	"context"

	"github.com/google/uuid"
	_ "github.com/lib/pq"
	"github.com/redis/go-redis/v9"
)

type CreditLedger struct {
	db           *sql.DB
	auditHMACKey []byte
	rdb          *redis.Client
	metrics      *CreditsMetrics
}

type ConsumeRequest struct {
	TenantID       string `json:"tenant_id"`
	Amount         int64  `json:"amount"`
	Description    string `json:"description"`
	Reference      string `json:"reference"`
	IdempotencyKey string `json:"idempotency_key"`
}

type PurchaseRequest struct {
	TenantID       string `json:"tenant_id"`
	Amount         int64  `json:"amount"`
	PaymentMethod  string `json:"payment_method"`
	PaymentToken   string `json:"payment_token"`
	IdempotencyKey string `json:"idempotency_key"`
}

type ReserveRequest struct {
	TenantID       string `json:"tenant_id"`
	Amount         int64  `json:"amount"`
	TTLSeconds     int64  `json:"ttl_seconds"`
	IdempotencyKey string `json:"idempotency_key"`
}

type CommitRequest struct {
	TenantID       string `json:"tenant_id"`
	ReservationID  string `json:"reservation_id"`
	IdempotencyKey string `json:"idempotency_key"`
}

type CancelRequest struct {
	TenantID      string `json:"tenant_id"`
	ReservationID string `json:"reservation_id"`
}

type CreditResponse struct {
	Success       bool   `json:"success"`
	TransactionID string `json:"transaction_id,omitempty"`
	Balance       int64  `json:"balance,omitempty"`
	Message       string `json:"message,omitempty"`
	Error         string `json:"error,omitempty"`
}

func NewCreditLedger(dbURL string) (*CreditLedger, error) {
	db, err := sql.Open("postgres", dbURL)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}

	if err := db.Ping(); err != nil {
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	db.SetMaxOpenConns(50)
	db.SetMaxIdleConns(10)
	db.SetConnMaxLifetime(5 * time.Minute)

	key := os.Getenv("AUDIT_HMAC_KEY")
	if key == "" {
		log.Printf("WARNING: AUDIT_HMAC_KEY not set; audit log chain will use a weak default. Set AUDIT_HMAC_KEY for production.")
	}
	ledger := &CreditLedger{db: db, auditHMACKey: []byte(key)}
	if os.Getenv("MIGRATE_ON_START") != "true" {
		log.Printf("Skipping DB migrations (set MIGRATE_ON_START=true after taking a backup to run migrations)")
		return ledger, nil
	}
	// Optional safety: backup before migrate
	if os.Getenv("BACKUP_BEFORE_MIGRATE") == "true" {
		if err := backupDatabaseOnce(dbURL); err != nil {
			log.Printf("[credits] pre-migration backup failed: %v", err)
		} else {
			log.Printf("[credits] pre-migration backup completed")
		}
	}
	if err := ledger.migrate(); err != nil {
		return nil, fmt.Errorf("migration failed: %w", err)
	}

	return ledger, nil
}

// initRedis configures optional Redis client for balance cache
func (cl *CreditLedger) initRedis(addr, password string) error {
	if addr == "" {
		return nil
	}
	rdb := redis.NewClient(&redis.Options{Addr: addr, Password: password})
	if err := rdb.Ping(context.Background()).Err(); err != nil {
		return err
	}
	cl.rdb = rdb
	return nil
}

func (cl *CreditLedger) balanceKey(tenant string) string { return "credits:bal:" + tenant }
func (cl *CreditLedger) delBalanceCache(tenant string) {
	if cl.rdb == nil || tenant == "" {
		return
	}
	_ = cl.rdb.Del(context.Background(), cl.balanceKey(tenant)).Err()
}
func (cl *CreditLedger) setBalanceCache(tenant string, bal int64) {
	if cl.rdb == nil || tenant == "" {
		return
	}
	_ = cl.rdb.SetEx(context.Background(), cl.balanceKey(tenant), strconv.FormatInt(bal, 10), 60*time.Second).Err()
}

func (cl *CreditLedger) migrate() error {
	query := `
    -- required for gen_random_uuid()
    CREATE EXTENSION IF NOT EXISTS pgcrypto;
	CREATE TABLE IF NOT EXISTS credit_accounts (
		tenant_id VARCHAR(255) PRIMARY KEY,
		balance BIGINT NOT NULL DEFAULT 0,
		reserved_funds BIGINT NOT NULL DEFAULT 0,
		total_spent BIGINT NOT NULL DEFAULT 0,
		total_purchased BIGINT NOT NULL DEFAULT 0,
		created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
		updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
		CONSTRAINT positive_balance CHECK (balance >= 0),
		CONSTRAINT positive_reserved CHECK (reserved_funds >= 0)
	);

	CREATE TABLE IF NOT EXISTS credit_transactions (
		transaction_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
		tenant_id VARCHAR(255) NOT NULL,
		type VARCHAR(50) NOT NULL,
		amount BIGINT NOT NULL,
		description TEXT,
		reference VARCHAR(255),
		status VARCHAR(50) NOT NULL DEFAULT 'pending',
		created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
		processed_at TIMESTAMP WITH TIME ZONE,
		metadata JSONB
	);

	CREATE TABLE IF NOT EXISTS idempotency_keys (
		key VARCHAR(255) PRIMARY KEY,
		tenant_id VARCHAR(255) NOT NULL,
		transaction_id UUID NOT NULL,
		created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
		expires_at TIMESTAMP WITH TIME ZONE NOT NULL
	);

	-- Reservations for two-phase spending
	CREATE TABLE IF NOT EXISTS credit_reservations (
		reservation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
		tenant_id VARCHAR(255) NOT NULL,
		amount BIGINT NOT NULL,
		status VARCHAR(20) NOT NULL DEFAULT 'active',
		created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
		expires_at TIMESTAMP WITH TIME ZONE NOT NULL
	);
	CREATE INDEX IF NOT EXISTS idx_resv_tenant_status ON credit_reservations(tenant_id, status);
    CREATE INDEX IF NOT EXISTS idx_resv_expires ON credit_reservations(expires_at);

	CREATE INDEX IF NOT EXISTS idx_credit_transactions_tenant ON credit_transactions(tenant_id);
	CREATE INDEX IF NOT EXISTS idx_credit_transactions_type ON credit_transactions(type);
	CREATE INDEX IF NOT EXISTS idx_idempotency_expires ON idempotency_keys(expires_at);

	-- Tamper-evident audit logs (immutable)
	CREATE TABLE IF NOT EXISTS audit_logs (
		id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
		tenant_id VARCHAR(255) NOT NULL,
		transaction_id UUID NOT NULL,
		action VARCHAR(50) NOT NULL,
		amount BIGINT NOT NULL,
		prev_hash VARCHAR(128) NOT NULL,
		hash VARCHAR(128) NOT NULL,
		created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
	);
	CREATE INDEX IF NOT EXISTS idx_audit_tenant ON audit_logs(tenant_id, created_at DESC);

	-- Prevent UPDATE/DELETE on audit_logs (immutability)
	CREATE OR REPLACE FUNCTION audit_logs_no_update_delete() RETURNS trigger AS $$
	BEGIN
		RAISE EXCEPTION 'audit_logs are immutable';
		RETURN NULL;
	END;
	$$ LANGUAGE plpgsql;

	DO $$ BEGIN
		IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'audit_logs_immutable') THEN
			CREATE TRIGGER audit_logs_immutable
			BEFORE UPDATE OR DELETE ON audit_logs
			FOR EACH ROW EXECUTE FUNCTION audit_logs_no_update_delete();
		END IF;
	END $$;`
	_, err := cl.db.Exec(query)
	return err
}

// SetAlertThreshold allows configuring a low-balance alert for a tenant
func (cl *CreditLedger) SetAlertThreshold(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var body struct {
		TenantID  string `json:"tenant_id"`
		Threshold int64  `json:"threshold"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}
	if body.TenantID == "" || body.Threshold < 0 {
		http.Error(w, "invalid tenant_id/threshold", http.StatusBadRequest)
		return
	}
	_, err := cl.db.Exec(`CREATE TABLE IF NOT EXISTS credit_alerts(tenant_id VARCHAR(255) PRIMARY KEY, threshold BIGINT NOT NULL DEFAULT 0, updated_at TIMESTAMPTZ DEFAULT NOW());`)
	if err != nil {
		http.Error(w, "failed", http.StatusInternalServerError)
		return
	}
	_, err = cl.db.Exec(`INSERT INTO credit_alerts(tenant_id, threshold, updated_at) VALUES ($1,$2,NOW()) ON CONFLICT(tenant_id) DO UPDATE SET threshold=EXCLUDED.threshold, updated_at=NOW()`, body.TenantID, body.Threshold)
	if err != nil {
		http.Error(w, "failed", http.StatusInternalServerError)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"success": true})
}

// startAlertWatcher runs in background to log alert events when balance drops below configured threshold
func (cl *CreditLedger) startAlertWatcher() {
	ticker := time.NewTicker(5 * time.Minute)
	go func() {
		for range ticker.C {
			// Expire stale reservations
			if err := cl.expireReservations(); err != nil {
				log.Printf("reservation expiry: %v", err)
			}
			rows, err := cl.db.Query(`SELECT a.tenant_id, a.threshold, COALESCE(c.balance,0) FROM credit_alerts a LEFT JOIN credit_accounts c ON a.tenant_id=c.tenant_id WHERE a.threshold > 0`)
			if err != nil {
				log.Printf("alert watcher query: %v", err)
				continue
			}
			for rows.Next() {
				var tenant string
				var th, bal int64
				if err := rows.Scan(&tenant, &th, &bal); err == nil {
					if bal <= th {
						_ = ledger.AppendJSONLine("data/ledger-credits-alerts.log", "credits", "low_balance", map[string]any{"tenant_id": tenant, "balance": bal, "threshold": th})
					}
				}
			}
			rows.Close()
		}
	}()
}

func (cl *CreditLedger) ConsumeCredits(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req ConsumeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	if req.TenantID == "" || req.Amount <= 0 {
		http.Error(w, "Missing required fields", http.StatusBadRequest)
		return
	}

	if req.IdempotencyKey == "" {
		req.IdempotencyKey = r.Header.Get("Idempotency-Key")
	}
	if req.IdempotencyKey == "" {
		req.IdempotencyKey = uuid.New().String()
	}

	if existingTxn, err := cl.checkIdempotency(req.IdempotencyKey, req.TenantID); err == nil && existingTxn != "" {
		balance, _ := cl.getBalance(req.TenantID)
		response := CreditResponse{
			Success:       true,
			TransactionID: existingTxn,
			Balance:       balance,
			Message:       "Credits already consumed (idempotent)",
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
		return
	}

	var txnID string
	var err error
	for attempt := 0; attempt < 3; attempt++ {
		txnID, err = cl.consumeCreditsAtomic(req.TenantID, req.Amount, req.Description, req.Reference, req.IdempotencyKey)
		if err == nil || !isRetryableTxErr(err) {
			break
		}
		time.Sleep(time.Duration(50*(attempt+1)) * time.Millisecond)
	}
	if err != nil {
		log.Printf("Failed to consume credits: %v", err)

		var status = http.StatusInternalServerError
		var response CreditResponse
		if strings.Contains(err.Error(), "insufficient") {
			status = http.StatusPaymentRequired
			response = CreditResponse{Success: false, Error: "Insufficient credits"}
			if cl.metrics != nil {
				cl.metrics.Ops.Inc(map[string]string{"op": "consume", "result": "insufficient"})
			}
		} else {
			response = CreditResponse{Success: false, Error: "Failed to consume credits"}
			if cl.metrics != nil {
				cl.metrics.Ops.Inc(map[string]string{"op": "consume", "result": "error"})
			}
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_ = json.NewEncoder(w).Encode(response)
		return
	}

	// Append security event log (masking reference) and invalidate cache before reading fresh balance
	maskedRef := req.Reference
	if len(maskedRef) > 6 {
		maskedRef = maskedRef[:2] + "***" + maskedRef[len(maskedRef)-2:]
	}
	_ = ledger.AppendJSONLine("data/ledger-credits.log", "credits", "consume", map[string]any{
		"tenant_id": req.TenantID,
		"amount":    req.Amount,
		"ref":       maskedRef,
		"txn_id":    txnID,
	})
	// Invalidate cache before reading fresh balance
	cl.delBalanceCache(req.TenantID)
	balance, _ := cl.getBalance(req.TenantID)
	response := CreditResponse{
		Success:       true,
		TransactionID: txnID,
		Balance:       balance,
		Message:       fmt.Sprintf("Successfully consumed %d credits", req.Amount),
	}
	if cl.metrics != nil {
		cl.metrics.Ops.Inc(map[string]string{"op": "consume", "result": "ok"})
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (cl *CreditLedger) PurchaseCredits(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req PurchaseRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	if req.TenantID == "" || req.Amount <= 0 {
		http.Error(w, "Missing required fields", http.StatusBadRequest)
		return
	}

	if req.IdempotencyKey == "" {
		req.IdempotencyKey = r.Header.Get("Idempotency-Key")
	}
	if req.IdempotencyKey == "" {
		req.IdempotencyKey = uuid.New().String()
	}

	if existingTxn, err := cl.checkIdempotency(req.IdempotencyKey, req.TenantID); err == nil && existingTxn != "" {
		balance, _ := cl.getBalance(req.TenantID)
		response := CreditResponse{
			Success:       true,
			TransactionID: existingTxn,
			Balance:       balance,
			Message:       "Credits already purchased (idempotent)",
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
		return
	}

	paymentResult, err := cl.processPayment(req)
	if err != nil {
		log.Printf("Payment processing failed: %v", err)
		response := CreditResponse{
			Success: false,
			Error:   "Payment processing failed",
		}
		if cl.metrics != nil {
			cl.metrics.Ops.Inc(map[string]string{"op": "purchase", "result": "pay_fail"})
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
		return
	}

	var txnID string
	txnID, err = cl.addCreditsAtomic(req.TenantID, req.Amount, "purchase",
		fmt.Sprintf("Credit purchase via %s", req.PaymentMethod),
		paymentResult.Reference, req.IdempotencyKey)
	if isRetryableTxErr(err) {
		// simple retry once for purchases
		time.Sleep(75 * time.Millisecond)
		txnID, err = cl.addCreditsAtomic(req.TenantID, req.Amount, "purchase",
			fmt.Sprintf("Credit purchase via %s", req.PaymentMethod),
			paymentResult.Reference, req.IdempotencyKey)
	}

	if err != nil {
		log.Printf("Failed to add credits: %v", err)
		response := CreditResponse{
			Success: false,
			Error:   "Failed to add credits",
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
		return
	}

	// Append audit log for purchase with masked payment reference
	maskedRef := paymentResult.Reference
	if len(maskedRef) > 6 {
		maskedRef = maskedRef[:2] + "***" + maskedRef[len(maskedRef)-2:]
	}
	_ = ledger.AppendJSONLine("data/ledger-credits.log", "credits", "purchase", map[string]any{
		"tenant_id": req.TenantID,
		"amount":    req.Amount,
		"method":    req.PaymentMethod,
		"ref":       maskedRef,
		"txn_id":    txnID,
	})
	cl.delBalanceCache(req.TenantID)
	balance, _ := cl.getBalance(req.TenantID)
	response := CreditResponse{
		Success:       true,
		TransactionID: txnID,
		Balance:       balance,
		Message:       fmt.Sprintf("Successfully purchased %d credits", req.Amount),
	}
	if cl.metrics != nil {
		cl.metrics.Ops.Inc(map[string]string{"op": "purchase", "result": "ok"})
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// ReserveCredits atomically moves amount from balance to reserved_funds and creates a reservation with TTL
func (cl *CreditLedger) ReserveCredits(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req ReserveRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}
	if req.TenantID == "" || req.Amount <= 0 {
		http.Error(w, "Missing required fields", http.StatusBadRequest)
		return
	}
	if req.TTLSeconds <= 0 || req.TTLSeconds > 3600*24 {
		req.TTLSeconds = 900 // default 15m
	}
	if req.IdempotencyKey == "" {
		req.IdempotencyKey = r.Header.Get("Idempotency-Key")
	}
	var id string
	var err error
	for attempt := 0; attempt < 3; attempt++ {
		id, err = cl.reserveAtomic(req.TenantID, req.Amount, time.Duration(req.TTLSeconds)*time.Second, req.IdempotencyKey)
		if err == nil || !isRetryableTxErr(err) {
			break
		}
		time.Sleep(time.Duration(50*(attempt+1)) * time.Millisecond)
	}
	if err != nil {
		log.Printf("reserve failed: %v", err)
		if strings.Contains(err.Error(), "insufficient") {
			if cl.metrics != nil {
				cl.metrics.Ops.Inc(map[string]string{"op": "reserve", "result": "insufficient"})
			}
			writeJSON(w, http.StatusOK, map[string]any{"success": false, "error": "Insufficient credits"})
			return
		}
		if cl.metrics != nil {
			cl.metrics.Ops.Inc(map[string]string{"op": "reserve", "result": "error"})
		}
		http.Error(w, "Failed to reserve", http.StatusInternalServerError)
		return
	}
	if cl.metrics != nil {
		cl.metrics.Ops.Inc(map[string]string{"op": "reserve", "result": "ok"})
	}
	_ = ledger.AppendJSONLine("data/ledger-credits.log", "credits", "reserve", map[string]any{
		"tenant_id":      req.TenantID,
		"amount":         req.Amount,
		"reservation_id": id,
		"ttl_sec":        req.TTLSeconds,
	})
	cl.delBalanceCache(req.TenantID)
	bal, _ := cl.getBalance(req.TenantID)
	writeJSON(w, http.StatusOK, map[string]any{"success": true, "reservation_id": id, "balance": bal})
}

// CommitReservation deducts reserved_funds and marks reservation completed
func (cl *CreditLedger) CommitReservation(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req CommitRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}
	if req.TenantID == "" || req.ReservationID == "" {
		http.Error(w, "Missing required fields", http.StatusBadRequest)
		return
	}
	if req.IdempotencyKey == "" {
		req.IdempotencyKey = r.Header.Get("Idempotency-Key")
	}
	var txnID string
	var err error
	for attempt := 0; attempt < 3; attempt++ {
		txnID, err = cl.commitReservationAtomic(req.TenantID, req.ReservationID, req.IdempotencyKey)
		if err == nil || !isRetryableTxErr(err) {
			break
		}
		time.Sleep(time.Duration(50*(attempt+1)) * time.Millisecond)
	}
	if err != nil {
		log.Printf("commit failed: %v", err)
		if cl.metrics != nil {
			cl.metrics.Ops.Inc(map[string]string{"op": "commit", "result": "error"})
		}
		http.Error(w, "Failed to commit", http.StatusBadRequest)
		return
	}
	if cl.metrics != nil {
		cl.metrics.Ops.Inc(map[string]string{"op": "commit", "result": "ok"})
	}
	_ = ledger.AppendJSONLine("data/ledger-credits.log", "credits", "commit", map[string]any{
		"tenant_id":      req.TenantID,
		"reservation_id": req.ReservationID,
		"txn_id":         txnID,
	})
	cl.delBalanceCache(req.TenantID)
	bal, _ := cl.getBalance(req.TenantID)
	writeJSON(w, http.StatusOK, map[string]any{"success": true, "transaction_id": txnID, "balance": bal})
}

// CancelReservation returns reserved amount to balance if active
func (cl *CreditLedger) CancelReservation(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req CancelRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}
	if req.TenantID == "" || req.ReservationID == "" {
		http.Error(w, "Missing required fields", http.StatusBadRequest)
		return
	}
	var err error
	for attempt := 0; attempt < 3; attempt++ {
		err = cl.cancelReservationAtomic(req.TenantID, req.ReservationID)
		if err == nil || !isRetryableTxErr(err) {
			break
		}
		time.Sleep(time.Duration(50*(attempt+1)) * time.Millisecond)
	}
	if err != nil {
		if cl.metrics != nil {
			cl.metrics.Ops.Inc(map[string]string{"op": "cancel", "result": "error"})
		}
		http.Error(w, "Failed to cancel", http.StatusBadRequest)
		return
	}
	if cl.metrics != nil {
		cl.metrics.Ops.Inc(map[string]string{"op": "cancel", "result": "ok"})
	}
	_ = ledger.AppendJSONLine("data/ledger-credits.log", "credits", "cancel", map[string]any{
		"tenant_id":      req.TenantID,
		"reservation_id": req.ReservationID,
	})
	cl.delBalanceCache(req.TenantID)
	bal, _ := cl.getBalance(req.TenantID)
	writeJSON(w, http.StatusOK, map[string]any{"success": true, "balance": bal})
}

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func (cl *CreditLedger) GetBalance(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	path := strings.TrimPrefix(r.URL.Path, "/credits/balance/")
	tenantID := strings.Split(path, "/")[0]

	if tenantID == "" {
		http.Error(w, "Missing tenant ID", http.StatusBadRequest)
		return
	}

	balance, err := cl.getBalance(tenantID)
	if err != nil {
		log.Printf("Failed to get balance: %v", err)
		http.Error(w, "Failed to get balance", http.StatusInternalServerError)
		return
	}

	response := map[string]interface{}{
		"tenant_id": tenantID,
		"balance":   balance,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (cl *CreditLedger) consumeCreditsAtomic(tenantID string, amount int64, description, reference, idempotencyKey string) (string, error) {
	tx, err := cl.db.Begin()
	if err != nil {
		return "", fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer tx.Rollback()

	// Enforce strongest isolation to prevent write skew and ensure no negative balances under concurrency
	if _, err := tx.Exec(`SET TRANSACTION ISOLATION LEVEL SERIALIZABLE`); err != nil {
		return "", fmt.Errorf("failed to set isolation level: %w", err)
	}

	_, err = tx.Exec(`
		INSERT INTO credit_accounts (tenant_id) 
		VALUES ($1) 
		ON CONFLICT (tenant_id) DO NOTHING`, tenantID)
	if err != nil {
		return "", fmt.Errorf("failed to ensure account exists: %w", err)
	}

	// Atomic deduction with guard prevents negative balances under concurrency
	var txnID string
	res, err := tx.Exec(`
		UPDATE credit_accounts
		SET balance = balance - $1,
			total_spent = total_spent + $1,
			updated_at = NOW()
		WHERE tenant_id = $2 AND balance >= $1`, amount, tenantID)
	if err != nil {
		return "", fmt.Errorf("failed to update balance: %w", err)
	}
	rows, _ := res.RowsAffected()
	if rows == 0 {
		return "", fmt.Errorf("insufficient credits")
	}

	// Record transaction after successful balance update
	err = tx.QueryRow(`
		INSERT INTO credit_transactions (tenant_id, type, amount, description, reference, status, processed_at)
		VALUES ($1, 'consume', $2, $3, $4, 'completed', NOW())
		RETURNING transaction_id`,
		tenantID, amount, description, reference).Scan(&txnID)
	if err != nil {
		return "", fmt.Errorf("failed to create transaction: %w", err)
	}

	_, err = tx.Exec(`
		INSERT INTO idempotency_keys (key, tenant_id, transaction_id, expires_at)
		VALUES ($1, $2, $3, NOW() + INTERVAL '24 hours')
		ON CONFLICT (key) DO NOTHING`,
		idempotencyKey, tenantID, txnID)
	if err != nil {
		return "", fmt.Errorf("failed to store idempotency key: %w", err)
	}

	// Immutable audit log entry
	if err := cl.writeAudit(tx, tenantID, txnID, "consume", amount); err != nil {
		return "", fmt.Errorf("failed to write audit: %v", err)
	}

	if err := tx.Commit(); err != nil {
		return "", fmt.Errorf("failed to commit transaction: %w", err)
	}

	return txnID, nil
}

func (cl *CreditLedger) addCreditsAtomic(tenantID string, amount int64, txnType, description, reference, idempotencyKey string) (string, error) {
	tx, err := cl.db.Begin()
	if err != nil {
		return "", fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer tx.Rollback()

	// Enforce strongest isolation to ensure invariants under concurrency
	if _, err := tx.Exec(`SET TRANSACTION ISOLATION LEVEL SERIALIZABLE`); err != nil {
		return "", fmt.Errorf("failed to set isolation level: %w", err)
	}

	_, err = tx.Exec(`
		INSERT INTO credit_accounts (tenant_id) 
		VALUES ($1) 
		ON CONFLICT (tenant_id) DO NOTHING`, tenantID)
	if err != nil {
		return "", fmt.Errorf("failed to ensure account exists: %w", err)
	}

	// Update balance first
	_, err = tx.Exec(`
		UPDATE credit_accounts 
		SET balance = balance + $1, 
			total_purchased = total_purchased + $1,
			updated_at = NOW()
		WHERE tenant_id = $2`, amount, tenantID)
	if err != nil {
		return "", fmt.Errorf("failed to update balance: %w", err)
	}

	// Optionally encrypt sensitive payment reference at-rest (PCI)
	encRef := reference
	if strings.EqualFold(txnType, "purchase") {
		if r2, err := cl.encryptAtRest(reference); err == nil && r2 != "" {
			encRef = r2
		}
	}

	// Then record transaction
	var txnID string
	err = tx.QueryRow(`
		INSERT INTO credit_transactions (tenant_id, type, amount, description, reference, status, processed_at)
		VALUES ($1, $2, $3, $4, $5, 'completed', NOW())
		RETURNING transaction_id`,
		tenantID, txnType, amount, description, encRef).Scan(&txnID)
	if err != nil {
		return "", fmt.Errorf("failed to create transaction: %w", err)
	}

	_, err = tx.Exec(`
		INSERT INTO idempotency_keys (key, tenant_id, transaction_id, expires_at)
		VALUES ($1, $2, $3, NOW() + INTERVAL '24 hours')
		ON CONFLICT (key) DO NOTHING`,
		idempotencyKey, tenantID, txnID)
	if err != nil {
		return "", fmt.Errorf("failed to store idempotency key: %w", err)
	}

	// Immutable audit log entry
	if err := cl.writeAudit(tx, tenantID, txnID, txnType, amount); err != nil {
		return "", fmt.Errorf("failed to write audit: %v", err)
	}

	if err := tx.Commit(); err != nil {
		return "", fmt.Errorf("failed to commit transaction: %w", err)
	}

	return txnID, nil
}

func (cl *CreditLedger) getBalance(tenantID string) (int64, error) {
	// Try cache first
	if cl.rdb != nil {
		if s, err := cl.rdb.Get(context.Background(), cl.balanceKey(tenantID)).Result(); err == nil && s != "" {
			if v, err2 := strconv.ParseInt(s, 10, 64); err2 == nil {
				return v, nil
			}
		}
	}
	var balance int64
	err := cl.db.QueryRow(`
		SELECT COALESCE(balance, 0) FROM credit_accounts WHERE tenant_id = $1`, tenantID).Scan(&balance)
	if err != nil {
		if err == sql.ErrNoRows {
			return 0, nil
		}
		return 0, err
	}
	// Update cache
	cl.setBalanceCache(tenantID, balance)
	return balance, nil
}

// reserveAtomic moves amount from balance to reserved_funds and creates a reservation row
func (cl *CreditLedger) reserveAtomic(tenantID string, amount int64, ttl time.Duration, idempotencyKey string) (string, error) {
	tx, err := cl.db.Begin()
	if err != nil {
		return "", err
	}
	defer tx.Rollback()

	if _, err := tx.Exec(`SET TRANSACTION ISOLATION LEVEL SERIALIZABLE`); err != nil {
		return "", fmt.Errorf("failed to set isolation level: %w", err)
	}

	_, err = tx.Exec(`INSERT INTO credit_accounts (tenant_id) VALUES ($1) ON CONFLICT (tenant_id) DO NOTHING`, tenantID)
	if err != nil {
		return "", err
	}

	res, err := tx.Exec(`UPDATE credit_accounts SET balance = balance - $1, reserved_funds = reserved_funds + $1, updated_at=NOW() WHERE tenant_id=$2 AND balance >= $1`, amount, tenantID)
	if err != nil {
		return "", err
	}
	if rows, _ := res.RowsAffected(); rows == 0 {
		return "", fmt.Errorf("insufficient credits")
	}

	var resvID string
	sec := int64(ttl.Seconds())
	if err := tx.QueryRow(`INSERT INTO credit_reservations(tenant_id, amount, expires_at) VALUES ($1,$2, NOW() + ($3 * INTERVAL '1 second')) RETURNING reservation_id`, tenantID, amount, sec).Scan(&resvID); err != nil {
		return "", err
	}
	if idempotencyKey != "" {
		_, _ = tx.Exec(`INSERT INTO idempotency_keys(key, tenant_id, transaction_id, expires_at) VALUES ($1,$2,$3,NOW()+INTERVAL '24 hours') ON CONFLICT (key) DO NOTHING`, idempotencyKey, tenantID, uuid.New())
	}
	if err := tx.Commit(); err != nil {
		return "", err
	}
	return resvID, nil
}

func (cl *CreditLedger) commitReservationAtomic(tenantID, reservationID, idempotencyKey string) (string, error) {
	tx, err := cl.db.Begin()
	if err != nil {
		return "", err
	}
	defer tx.Rollback()

	if _, err := tx.Exec(`SET TRANSACTION ISOLATION LEVEL SERIALIZABLE`); err != nil {
		return "", fmt.Errorf("failed to set isolation level: %w", err)
	}

	var amount int64
	var status string
	err = tx.QueryRow(`SELECT amount, status FROM credit_reservations WHERE reservation_id=$1 AND tenant_id=$2 AND expires_at > NOW() FOR UPDATE`, reservationID, tenantID).Scan(&amount, &status)
	if err != nil {
		return "", err
	}
	if status != "active" {
		return "", fmt.Errorf("reservation not active")
	}

	// deduct from reserved_funds, finalize spend
	_, err = tx.Exec(`UPDATE credit_accounts SET reserved_funds = reserved_funds - $1, total_spent = total_spent + $1, updated_at=NOW() WHERE tenant_id=$2 AND reserved_funds >= $1`, amount, tenantID)
	if err != nil {
		return "", err
	}

	var txnID string
	if err := tx.QueryRow(`INSERT INTO credit_transactions(tenant_id, type, amount, description, reference, status, processed_at) VALUES ($1,'consume',$2,'reservation_commit',$3,'completed',NOW()) RETURNING transaction_id`, tenantID, amount, reservationID).Scan(&txnID); err != nil {
		return "", err
	}
	_, err = tx.Exec(`UPDATE credit_reservations SET status='committed' WHERE reservation_id=$1`, reservationID)
	if err != nil {
		return "", err
	}
	if idempotencyKey != "" {
		_, _ = tx.Exec(`INSERT INTO idempotency_keys(key, tenant_id, transaction_id, expires_at) VALUES ($1,$2,$3,NOW()+INTERVAL '24 hours') ON CONFLICT (key) DO NOTHING`, idempotencyKey, tenantID, txnID)
	}
	if err := cl.writeAudit(tx, tenantID, txnID, "consume", amount); err != nil {
		return "", err
	}
	if err := tx.Commit(); err != nil {
		return "", err
	}
	return txnID, nil
}

func (cl *CreditLedger) cancelReservationAtomic(tenantID, reservationID string) error {
	tx, err := cl.db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	if _, err := tx.Exec(`SET TRANSACTION ISOLATION LEVEL SERIALIZABLE`); err != nil {
		return fmt.Errorf("failed to set isolation level: %w", err)
	}
	var amount int64
	var status string
	if err := tx.QueryRow(`SELECT amount, status FROM credit_reservations WHERE reservation_id=$1 AND tenant_id=$2 FOR UPDATE`, reservationID, tenantID).Scan(&amount, &status); err != nil {
		return err
	}
	if status != "active" {
		return fmt.Errorf("reservation not active")
	}
	// return funds
	if _, err := tx.Exec(`UPDATE credit_accounts SET balance = balance + $1, reserved_funds = reserved_funds - $1, updated_at=NOW() WHERE tenant_id=$2 AND reserved_funds >= $1`, amount, tenantID); err != nil {
		return err
	}
	if _, err := tx.Exec(`UPDATE credit_reservations SET status='canceled' WHERE reservation_id=$1`, reservationID); err != nil {
		return err
	}
	return tx.Commit()
}

// GetUsageReport returns aggregates for a tenant
func (cl *CreditLedger) GetUsageReport(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	tenantID := r.URL.Query().Get("tenant_id")
	if tenantID == "" {
		http.Error(w, "tenant_id required", http.StatusBadRequest)
		return
	}
	var spent, purchased, reserved int64
	err := cl.db.QueryRow(`SELECT total_spent, total_purchased, reserved_funds FROM credit_accounts WHERE tenant_id=$1`, tenantID).Scan(&spent, &purchased, &reserved)
	if err == sql.ErrNoRows {
		writeJSON(w, http.StatusOK, map[string]any{"tenant_id": tenantID, "total_spent": 0, "total_purchased": 0, "reserved": 0})
		return
	}
	if err != nil {
		http.Error(w, "failed", http.StatusInternalServerError)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"tenant_id": tenantID, "total_spent": spent, "total_purchased": purchased, "reserved": reserved})
}

func (cl *CreditLedger) checkIdempotency(key, tenantID string) (string, error) {
	var txnID string
	err := cl.db.QueryRow(`
		SELECT transaction_id FROM idempotency_keys 
		WHERE key = $1 AND tenant_id = $2 AND expires_at > NOW()`,
		key, tenantID).Scan(&txnID)

	if err == sql.ErrNoRows {
		return "", fmt.Errorf("key not found")
	}
	return txnID, err
}

// writeAudit appends a tamper-evident log entry chained by HMAC(prev_hash || fields)
func (cl *CreditLedger) writeAudit(tx *sql.Tx, tenantID, transactionID, action string, amount int64) error {
	var prevHash string
	_ = tx.QueryRow(`SELECT hash FROM audit_logs WHERE tenant_id=$1 ORDER BY created_at DESC LIMIT 1`, tenantID).Scan(&prevHash)
	mac := hmac.New(sha256.New, cl.auditHMACKey)
	mac.Write([]byte(prevHash))
	mac.Write([]byte(tenantID))
	mac.Write([]byte(transactionID))
	mac.Write([]byte(action))
	mac.Write([]byte(fmt.Sprintf("%d", amount)))
	sum := mac.Sum(nil)
	newHash := hex.EncodeToString(sum)

	_, err := tx.Exec(`
		INSERT INTO audit_logs (tenant_id, transaction_id, action, amount, prev_hash, hash)
		VALUES ($1,$2,$3,$4,$5,$6)`, tenantID, transactionID, action, amount, prevHash, newHash)
	return err
}

// GetHistory returns recent transactions without exposing sensitive metadata
func (cl *CreditLedger) GetHistory(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	tenantID := r.URL.Query().Get("tenant_id")
	if tenantID == "" {
		http.Error(w, "tenant_id is required", http.StatusBadRequest)
		return
	}
	limit := 50
	if v := r.URL.Query().Get("limit"); v != "" {
		var parsed int
		fmt.Sscanf(v, "%d", &parsed)
		if parsed > 0 && parsed <= 200 {
			limit = parsed
		}
	}
	cursor := r.URL.Query().Get("cursor")

	// If cursor provided, fetch its created_at to build keyset pagination predicate
	var cutoff time.Time
	var haveCutoff bool
	if cursor != "" {
		var err error
		cutoff, err = cl.lookupTxnCreatedAt(tenantID, cursor)
		if err != nil {
			http.Error(w, "invalid cursor", http.StatusBadRequest)
			return
		}
		haveCutoff = true
	}

	// Keyset pagination for stable high-performance paging
	// Order by created_at DESC, transaction_id DESC to disambiguate ties
	var rows *sql.Rows
	var err error
	if haveCutoff {
		rows, err = cl.db.Query(`
			SELECT transaction_id, type, amount, COALESCE(description,''), COALESCE(reference,''), created_at
			FROM credit_transactions
			WHERE tenant_id=$1 AND (created_at < $2 OR (created_at = $2 AND transaction_id < $3))
			ORDER BY created_at DESC, transaction_id DESC
			LIMIT $4`, tenantID, cutoff, cursor, limit)
	} else {
		rows, err = cl.db.Query(`
			SELECT transaction_id, type, amount, COALESCE(description,''), COALESCE(reference,''), created_at
			FROM credit_transactions
			WHERE tenant_id=$1
			ORDER BY created_at DESC, transaction_id DESC
			LIMIT $2`, tenantID, limit)
	}
	if err != nil {
		log.Printf("Failed to fetch history: %v", err)
		http.Error(w, "Failed to fetch history", http.StatusInternalServerError)
		return
	}
	defer rows.Close()

	type item struct {
		TransactionID string    `json:"transaction_id"`
		Type          string    `json:"type"`
		Amount        int64     `json:"amount"`
		Description   string    `json:"description"`
		Reference     string    `json:"reference"`
		CreatedAt     time.Time `json:"created_at"`
	}
	out := make([]item, 0, limit)
	var lastTxn string
	for rows.Next() {
		var it item
		if err := rows.Scan(&it.TransactionID, &it.Type, &it.Amount, &it.Description, &it.Reference, &it.CreatedAt); err == nil {
			// Mask or suppress reference to avoid leaking payment info
			if strings.HasPrefix(it.Reference, "enc:v1:") {
				// Encrypted at rest; do not expose
				it.Reference = "****"
			} else if len(it.Reference) > 6 {
				it.Reference = it.Reference[:2] + "***" + it.Reference[len(it.Reference)-2:]
			}
			out = append(out, it)
			lastTxn = it.TransactionID
		}
	}
	// Compute next cursor if page full
	nextCursor := ""
	if len(out) == limit && lastTxn != "" {
		nextCursor = lastTxn
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"tenant_id":   tenantID,
		"items":       out,
		"count":       len(out),
		"next_cursor": nextCursor,
	})
}

// lookupTxnCreatedAt finds the created_at timestamp for a given transaction id within a tenant.
func (cl *CreditLedger) lookupTxnCreatedAt(tenantID, txnID string) (time.Time, error) {
	var ts time.Time
	err := cl.db.QueryRow(`SELECT created_at FROM credit_transactions WHERE tenant_id=$1 AND transaction_id=$2`, tenantID, txnID).Scan(&ts)
	if err != nil {
		return time.Time{}, err
	}
	return ts, nil
}

type PaymentResult struct {
	Success   bool   `json:"success"`
	Reference string `json:"reference"`
	Message   string `json:"message"`
}

func (cl *CreditLedger) processPayment(req PurchaseRequest) (*PaymentResult, error) {
	if req.PaymentMethod == "" || req.PaymentToken == "" {
		return nil, fmt.Errorf("invalid payment method or token")
	}

	time.Sleep(100 * time.Millisecond)

	reference := fmt.Sprintf("pay_%s_%d", req.PaymentMethod, time.Now().Unix())

	return &PaymentResult{
		Success:   true,
		Reference: reference,
		Message:   "Payment processed successfully",
	}, nil
}

func (cl *CreditLedger) HasSufficientCredits(tenantID string, amount int64) (bool, error) {
	balance, err := cl.getBalance(tenantID)
	if err != nil {
		return false, err
	}
	return balance >= amount, nil
}

func (cl *CreditLedger) cleanupExpiredKeys() {
	ticker := time.NewTicker(1 * time.Hour)
	defer ticker.Stop()

	for range ticker.C {
		_, err := cl.db.Exec(`DELETE FROM idempotency_keys WHERE expires_at < NOW()`)
		if err != nil {
			log.Printf("Failed to cleanup expired keys: %v", err)
		}
	}
}

func (cl *CreditLedger) Close() error {
	return cl.db.Close()
}

// expireReservations moves expired active reservations back to balance and marks them expired
func (cl *CreditLedger) expireReservations() error {
	tx, err := cl.db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	if _, err := tx.Exec(`SET TRANSACTION ISOLATION LEVEL SERIALIZABLE`); err != nil {
		return fmt.Errorf("failed to set isolation level: %w", err)
	}
	rows, err := tx.Query(`SELECT reservation_id, tenant_id, amount FROM credit_reservations WHERE status='active' AND expires_at <= NOW() FOR UPDATE SKIP LOCKED`)
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var id, tenant string
		var amt int64
		if err := rows.Scan(&id, &tenant, &amt); err != nil {
			continue
		}
		if _, err := tx.Exec(`UPDATE credit_accounts SET balance = balance + $1, reserved_funds = reserved_funds - $1, updated_at=NOW() WHERE tenant_id=$2 AND reserved_funds >= $1`, amt, tenant); err != nil {
			return err
		}
		if _, err := tx.Exec(`UPDATE credit_reservations SET status='expired' WHERE reservation_id=$1`, id); err != nil {
			return err
		}
		cl.delBalanceCache(tenant)
	}
	return tx.Commit()
}

// isRetryableTxErr detects serialization/deadlock errors that can be retried safely
func isRetryableTxErr(err error) bool {
	if err == nil {
		return false
	}
	s := strings.ToLower(err.Error())
	return strings.Contains(s, "could not serialize access") || strings.Contains(s, "deadlock detected") || strings.Contains(s, "serialization failure")
}

// backupDatabaseOnce performs a best-effort pg_dump of the provided database URL
func backupDatabaseOnce(dbURL string) error {
	if cmd := os.Getenv("BACKUP_CMD"); cmd != "" {
		c := execCommand("bash", "-lc", cmd)
		return c.Run()
	}
	if _, err := execLookPath("pg_dump"); err != nil {
		return nil
	}
	ts := time.Now().Format("20060102_150405")
	out := "/tmp/credits_" + ts + ".dump"
	c := execCommand("bash", "-lc", "pg_dump --dbname='"+dbURL+"' -Fc -f '"+out+"'")
	return c.Run()
}

// indirections for testability
var execCommand = func(name string, arg ...string) *exec.Cmd { return exec.Command(name, arg...) }
var execLookPath = exec.LookPath

// encryptAtRest encrypts sensitive strings using AES-GCM if PAYMENT_ENC_KEY is configured.
// Output format: enc:v1:<base64(nonce||ciphertext)>. If key missing/invalid, returns plaintext.
func (cl *CreditLedger) encryptAtRest(plaintext string) (string, error) {
	if plaintext == "" {
		return plaintext, nil
	}
	keyStr := os.Getenv("PAYMENT_ENC_KEY")
	if keyStr == "" {
		return plaintext, nil
	}
	// Accept raw base64 or with base64: prefix
	keyStr = strings.TrimPrefix(keyStr, "base64:")
	key, err := base64.StdEncoding.DecodeString(keyStr)
	if err != nil {
		// Fallback: use bytes directly, but require valid AES key size
		key = []byte(keyStr)
	}
	if !(len(key) == 16 || len(key) == 24 || len(key) == 32) {
		// invalid key length, skip encryption quietly
		return plaintext, nil
	}
	block, err := aes.NewCipher(key)
	if err != nil {
		return plaintext, nil
	}
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return plaintext, nil
	}
	nonce := make([]byte, gcm.NonceSize())
	if _, err := rand.Read(nonce); err != nil {
		return plaintext, nil
	}
	ct := gcm.Seal(nil, nonce, []byte(plaintext), nil)
	payload := append(nonce, ct...)
	return "enc:v1:" + base64.StdEncoding.EncodeToString(payload), nil
}
