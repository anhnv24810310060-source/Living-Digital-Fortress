-- Migration: Add distributed transactions and Write-Ahead Log for ACID compliance
-- Version: 000004
-- Description: Two-phase commit support + immutable audit trail with cryptographic chaining

-- Distributed transactions table for two-phase commit protocol
CREATE TABLE IF NOT EXISTS distributed_transactions (
    tx_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(255) NOT NULL,
    tx_type VARCHAR(50) NOT NULL CHECK (tx_type IN ('reserve', 'consume', 'topup')),
    amount BIGINT NOT NULL CHECK (amount > 0),
    state VARCHAR(50) NOT NULL DEFAULT 'pending' CHECK (state IN ('pending', 'prepared', 'committed', 'aborted', 'rolled_back')),
    timeout_seconds INTEGER NOT NULL DEFAULT 30,
    parent_tx_id UUID,
    retry_count INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    prepared_at TIMESTAMP WITH TIME ZONE,
    committed_at TIMESTAMP WITH TIME ZONE,
    aborted_at TIMESTAMP WITH TIME ZONE,
    CONSTRAINT distributed_transactions_timeout CHECK (timeout_seconds > 0 AND timeout_seconds <= 300)
);

-- Immutable audit log with cryptographic chain
CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sequence_number BIGINT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(255) NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    action VARCHAR(100) NOT NULL,
    amount BIGINT,
    balance_before BIGINT NOT NULL,
    balance_after BIGINT NOT NULL,
    transaction_id UUID NOT NULL,
    user_id VARCHAR(255),
    ip_address VARCHAR(100), -- Masked for privacy
    user_agent TEXT, -- Masked for privacy  
    metadata JSONB DEFAULT '{}'::jsonb,
    previous_hash VARCHAR(64) NOT NULL,
    current_hash VARCHAR(64) NOT NULL,
    hmac_signature VARCHAR(64) NOT NULL,
    success BOOLEAN NOT NULL DEFAULT true,
    error_message TEXT,
    duration_ms BIGINT NOT NULL DEFAULT 0,
    CONSTRAINT audit_log_sequence_unique UNIQUE (sequence_number)
);

-- Indexes for distributed transactions
CREATE INDEX IF NOT EXISTS idx_distributed_transactions_tenant ON distributed_transactions(tenant_id);
CREATE INDEX IF NOT EXISTS idx_distributed_transactions_state ON distributed_transactions(state);
CREATE INDEX IF NOT EXISTS idx_distributed_transactions_created ON distributed_transactions(created_at);
CREATE INDEX IF NOT EXISTS idx_distributed_transactions_type ON distributed_transactions(tx_type);
CREATE INDEX IF NOT EXISTS idx_distributed_transactions_timeout ON distributed_transactions(created_at, state, timeout_seconds) 
    WHERE state = 'prepared'; -- For stalled transaction recovery

-- Indexes for audit log (optimized for queries)
CREATE INDEX IF NOT EXISTS idx_audit_log_sequence ON audit_log(sequence_number DESC);
CREATE INDEX IF NOT EXISTS idx_audit_log_tenant ON audit_log(tenant_id, sequence_number DESC);
CREATE INDEX IF NOT EXISTS idx_audit_log_transaction ON audit_log(transaction_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_log_event_type ON audit_log(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_log_action ON audit_log(action);
CREATE INDEX IF NOT EXISTS idx_audit_log_success ON audit_log(success) WHERE success = false; -- Failed operations

-- Composite index for audit searches
CREATE INDEX IF NOT EXISTS idx_audit_log_tenant_time ON audit_log(tenant_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_log_event_time ON audit_log(event_type, timestamp DESC);

-- Partial index for active transactions
CREATE INDEX IF NOT EXISTS idx_distributed_tx_active ON distributed_transactions(tenant_id, state, created_at)
    WHERE state IN ('pending', 'prepared');

-- GIN index for metadata searches
CREATE INDEX IF NOT EXISTS idx_distributed_tx_metadata ON distributed_transactions USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_audit_log_metadata ON audit_log USING GIN (metadata);

-- Sequence for audit log ordering
CREATE SEQUENCE IF NOT EXISTS audit_log_sequence START 1;

-- Function to auto-assign sequence numbers
CREATE OR REPLACE FUNCTION assign_audit_sequence()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.sequence_number IS NULL THEN
        NEW.sequence_number := nextval('audit_log_sequence');
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to ensure sequence assignment
CREATE TRIGGER trigger_assign_audit_sequence
    BEFORE INSERT ON audit_log
    FOR EACH ROW
    EXECUTE FUNCTION assign_audit_sequence();

-- View for transaction statistics
CREATE OR REPLACE VIEW transaction_stats AS
SELECT
    tenant_id,
    tx_type,
    state,
    COUNT(*) as count,
    SUM(amount) as total_amount,
    AVG(amount) as avg_amount,
    AVG(EXTRACT(EPOCH FROM (COALESCE(committed_at, aborted_at, NOW()) - created_at))) as avg_duration_seconds,
    COUNT(*) FILTER (WHERE state = 'committed') as success_count,
    COUNT(*) FILTER (WHERE state = 'aborted') as failed_count,
    MAX(created_at) as last_transaction_at
FROM distributed_transactions
GROUP BY tenant_id, tx_type, state;

-- View for audit trail summary
CREATE OR REPLACE VIEW audit_summary AS
SELECT
    tenant_id,
    event_type,
    action,
    DATE_TRUNC('hour', timestamp) as hour,
    COUNT(*) as event_count,
    SUM(amount) FILTER (WHERE amount > 0) as total_amount,
    COUNT(*) FILTER (WHERE success = true) as success_count,
    COUNT(*) FILTER (WHERE success = false) as failure_count,
    AVG(duration_ms) as avg_duration_ms,
    MAX(duration_ms) as max_duration_ms
FROM audit_log
GROUP BY tenant_id, event_type, action, DATE_TRUNC('hour', timestamp);

-- View for suspicious activity detection
CREATE OR REPLACE VIEW suspicious_activity AS
SELECT
    tenant_id,
    COUNT(*) as failed_attempts,
    MAX(timestamp) as last_attempt,
    array_agg(DISTINCT ip_address) as ip_addresses,
    array_agg(DISTINCT action) as attempted_actions
FROM audit_log
WHERE success = false
  AND timestamp > NOW() - INTERVAL '1 hour'
GROUP BY tenant_id
HAVING COUNT(*) > 10; -- More than 10 failures in an hour

-- Function to verify audit log chain integrity
CREATE OR REPLACE FUNCTION verify_audit_chain(start_seq BIGINT, end_seq BIGINT)
RETURNS TABLE (
    is_valid BOOLEAN,
    broken_at BIGINT,
    error_message TEXT
) AS $$
DECLARE
    prev_hash VARCHAR(64);
    curr_record RECORD;
    first_iteration BOOLEAN := true;
BEGIN
    FOR curr_record IN
        SELECT sequence_number, previous_hash, current_hash
        FROM audit_log
        WHERE sequence_number >= start_seq AND sequence_number <= end_seq
        ORDER BY sequence_number ASC
    LOOP
        IF NOT first_iteration THEN
            IF curr_record.previous_hash != prev_hash THEN
                RETURN QUERY SELECT false, curr_record.sequence_number, 
                    'Chain broken: expected ' || prev_hash || ', got ' || curr_record.previous_hash;
                RETURN;
            END IF;
        END IF;
        
        prev_hash := curr_record.current_hash;
        first_iteration := false;
    END LOOP;
    
    RETURN QUERY SELECT true, NULL::BIGINT, NULL::TEXT;
END;
$$ LANGUAGE plpgsql;

-- Function to cleanup old completed transactions (retention policy)
CREATE OR REPLACE FUNCTION cleanup_old_transactions(retention_days INTEGER DEFAULT 90)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM distributed_transactions
    WHERE (state = 'committed' OR state = 'aborted')
      AND created_at < NOW() - (retention_days || ' days')::INTERVAL;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to recover stalled transactions (auto-abort after timeout)
CREATE OR REPLACE FUNCTION recover_stalled_transactions()
RETURNS INTEGER AS $$
DECLARE
    recovered_count INTEGER;
BEGIN
    UPDATE distributed_transactions
    SET state = 'aborted',
        aborted_at = NOW(),
        last_error = 'Transaction timeout - auto-aborted by recovery process'
    WHERE state = 'prepared'
      AND created_at < NOW() - (timeout_seconds || ' seconds')::INTERVAL;
    
    GET DIAGNOSTICS recovered_count = ROW_COUNT;
    RETURN recovered_count;
END;
$$ LANGUAGE plpgsql;

-- Add constraint to prevent negative balance in credit_accounts (if not exists)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'credit_accounts_balance_non_negative'
    ) THEN
        ALTER TABLE credit_accounts ADD CONSTRAINT credit_accounts_balance_non_negative CHECK (balance >= 0);
    END IF;
    
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'credit_accounts_reserved_non_negative'
    ) THEN
        ALTER TABLE credit_accounts ADD CONSTRAINT credit_accounts_reserved_non_negative CHECK (reserved >= 0);
    END IF;
    
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'credit_accounts_reserved_lte_balance'
    ) THEN
        ALTER TABLE credit_accounts ADD CONSTRAINT credit_accounts_reserved_lte_balance CHECK (reserved <= balance);
    END IF;
END $$;

-- Grant permissions (adjust as needed)
-- GRANT SELECT, INSERT ON audit_log TO credits_readonly;
-- GRANT SELECT, INSERT, UPDATE ON distributed_transactions TO credits_service;

-- Create periodic cleanup job (requires pg_cron extension)
-- SELECT cron.schedule('cleanup-old-transactions', '0 2 * * *', 'SELECT cleanup_old_transactions(90)');
-- SELECT cron.schedule('recover-stalled-transactions', '*/5 * * * *', 'SELECT recover_stalled_transactions()');

COMMENT ON TABLE distributed_transactions IS 'Two-phase commit transactions for ACID compliance across distributed operations';
COMMENT ON TABLE audit_log IS 'Immutable audit trail with cryptographic chaining for tamper detection';
COMMENT ON COLUMN audit_log.previous_hash IS 'SHA256 hash of previous entry for chain integrity';
COMMENT ON COLUMN audit_log.current_hash IS 'SHA256 hash of current entry';
COMMENT ON COLUMN audit_log.hmac_signature IS 'HMAC signature for tamper detection';
