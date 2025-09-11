package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/google/uuid"
	_ "github.com/lib/pq"
)

type CreditLedger struct {
	db *sql.DB
}

type ConsumeRequest struct {
	TenantID      string `json:"tenant_id"`
	Amount        int64  `json:"amount"`
	Description   string `json:"description"`
	Reference     string `json:"reference"`
	IdempotencyKey string `json:"idempotency_key"`
}

type PurchaseRequest struct {
	TenantID      string `json:"tenant_id"`
	Amount        int64  `json:"amount"`
	PaymentMethod string `json:"payment_method"`
	PaymentToken  string `json:"payment_token"`
	IdempotencyKey string `json:"idempotency_key"`
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

	ledger := &CreditLedger{db: db}
	if err := ledger.migrate(); err != nil {
		return nil, fmt.Errorf("migration failed: %w", err)
	}

	return ledger, nil
}

func (cl *CreditLedger) migrate() error {
	query := `
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

	CREATE INDEX IF NOT EXISTS idx_credit_transactions_tenant ON credit_transactions(tenant_id);
	CREATE INDEX IF NOT EXISTS idx_credit_transactions_type ON credit_transactions(type);
	CREATE INDEX IF NOT EXISTS idx_idempotency_expires ON idempotency_keys(expires_at);`

	_, err := cl.db.Exec(query)
	return err
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

	txnID, err := cl.consumeCreditsAtomic(req.TenantID, req.Amount, req.Description, req.Reference, req.IdempotencyKey)
	if err != nil {
		log.Printf("Failed to consume credits: %v", err)
		
		var response CreditResponse
		if strings.Contains(err.Error(), "insufficient") {
			response = CreditResponse{
				Success: false,
				Error:   "Insufficient credits",
			}
		} else {
			response = CreditResponse{
				Success: false,
				Error:   "Failed to consume credits",
			}
		}
		
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
		return
	}

	balance, _ := cl.getBalance(req.TenantID)
	response := CreditResponse{
		Success:       true,
		TransactionID: txnID,
		Balance:       balance,
		Message:       fmt.Sprintf("Successfully consumed %d credits", req.Amount),
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
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
		return
	}

	txnID, err := cl.addCreditsAtomic(req.TenantID, req.Amount, "purchase", 
		fmt.Sprintf("Credit purchase via %s", req.PaymentMethod), 
		paymentResult.Reference, req.IdempotencyKey)
	
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

	balance, _ := cl.getBalance(req.TenantID)
	response := CreditResponse{
		Success:       true,
		TransactionID: txnID,
		Balance:       balance,
		Message:       fmt.Sprintf("Successfully purchased %d credits", req.Amount),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
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

	_, err = tx.Exec(`
		INSERT INTO credit_accounts (tenant_id) 
		VALUES ($1) 
		ON CONFLICT (tenant_id) DO NOTHING`, tenantID)
	if err != nil {
		return "", fmt.Errorf("failed to ensure account exists: %w", err)
	}

	var currentBalance int64
	err = tx.QueryRow(`
		SELECT balance FROM credit_accounts 
		WHERE tenant_id = $1 
		FOR UPDATE`, tenantID).Scan(&currentBalance)
	if err != nil {
		return "", fmt.Errorf("failed to get current balance: %w", err)
	}

	if currentBalance < amount {
		return "", fmt.Errorf("insufficient credits: have %d, need %d", currentBalance, amount)
	}

	var txnID string
	err = tx.QueryRow(`
		INSERT INTO credit_transactions (tenant_id, type, amount, description, reference, status, processed_at)
		VALUES ($1, 'consume', $2, $3, $4, 'completed', NOW())
		RETURNING transaction_id`, 
		tenantID, amount, description, reference).Scan(&txnID)
	if err != nil {
		return "", fmt.Errorf("failed to create transaction: %w", err)
	}

	_, err = tx.Exec(`
		UPDATE credit_accounts 
		SET balance = balance - $1, 
			total_spent = total_spent + $1,
			updated_at = NOW()
		WHERE tenant_id = $2`, amount, tenantID)
	if err != nil {
		return "", fmt.Errorf("failed to update balance: %w", err)
	}

	_, err = tx.Exec(`
		INSERT INTO idempotency_keys (key, tenant_id, transaction_id, expires_at)
		VALUES ($1, $2, $3, NOW() + INTERVAL '24 hours')
		ON CONFLICT (key) DO NOTHING`, 
		idempotencyKey, tenantID, txnID)
	if err != nil {
		return "", fmt.Errorf("failed to store idempotency key: %w", err)
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

	_, err = tx.Exec(`
		INSERT INTO credit_accounts (tenant_id) 
		VALUES ($1) 
		ON CONFLICT (tenant_id) DO NOTHING`, tenantID)
	if err != nil {
		return "", fmt.Errorf("failed to ensure account exists: %w", err)
	}

	var txnID string
	err = tx.QueryRow(`
		INSERT INTO credit_transactions (tenant_id, type, amount, description, reference, status, processed_at)
		VALUES ($1, $2, $3, $4, $5, 'completed', NOW())
		RETURNING transaction_id`, 
		tenantID, txnType, amount, description, reference).Scan(&txnID)
	if err != nil {
		return "", fmt.Errorf("failed to create transaction: %w", err)
	}

	_, err = tx.Exec(`
		UPDATE credit_accounts 
		SET balance = balance + $1, 
			total_purchased = total_purchased + $1,
			updated_at = NOW()
		WHERE tenant_id = $2`, amount, tenantID)
	if err != nil {
		return "", fmt.Errorf("failed to update balance: %w", err)
	}

	_, err = tx.Exec(`
		INSERT INTO idempotency_keys (key, tenant_id, transaction_id, expires_at)
		VALUES ($1, $2, $3, NOW() + INTERVAL '24 hours')
		ON CONFLICT (key) DO NOTHING`, 
		idempotencyKey, tenantID, txnID)
	if err != nil {
		return "", fmt.Errorf("failed to store idempotency key: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return "", fmt.Errorf("failed to commit transaction: %w", err)
	}

	return txnID, nil
}

func (cl *CreditLedger) getBalance(tenantID string) (int64, error) {
	var balance int64
	err := cl.db.QueryRow(`
		SELECT COALESCE(balance, 0) FROM credit_accounts WHERE tenant_id = $1`, tenantID).Scan(&balance)
	if err != nil {
		if err == sql.ErrNoRows {
			return 0, nil
		}
		return 0, err
	}
	return balance, nil
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