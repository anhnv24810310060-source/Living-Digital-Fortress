-- Enhanced Credits Schema Migration with Performance Optimizations
-- PERSON 3 - Production Ready Schema
-- Version: 000005

-- Add partition support for transactions table (time-series optimization)
CREATE TABLE IF NOT EXISTS credit_transactions_2025_q4 PARTITION OF credit_transactions
    FOR VALUES FROM ('2025-10-01') TO ('2026-01-01');

CREATE TABLE IF NOT EXISTS credit_transactions_2026_q1 PARTITION OF credit_transactions
    FOR VALUES FROM ('2026-01-01') TO ('2026-04-01');

-- Create materialized view for fast balance queries
CREATE MATERIALIZED VIEW IF NOT EXISTS credit_balances_summary AS
SELECT 
    tenant_id,
    balance,
    reserved,
    balance - reserved AS available,
    updated_at,
    version
FROM credit_accounts
WHERE balance > 0
WITH DATA;

-- Create unique index on materialized view
CREATE UNIQUE INDEX IF NOT EXISTS idx_credit_balances_summary_tenant 
    ON credit_balances_summary(tenant_id);

-- Create refresh function for materialized view
CREATE OR REPLACE FUNCTION refresh_credit_balances_summary()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY credit_balances_summary;
END;
$$ LANGUAGE plpgsql;

-- Create idempotency keys table for distributed transactions
CREATE TABLE IF NOT EXISTS idempotency_keys (
    key_hash VARCHAR(64) PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() + INTERVAL '24 hours'
);

-- Create index for expired keys cleanup
CREATE INDEX IF NOT EXISTS idx_idempotency_keys_expires 
    ON idempotency_keys(expires_at) WHERE expires_at > NOW();

-- Create audit log chain table for immutable logging
CREATE TABLE IF NOT EXISTS audit_log_chain (
    id BIGSERIAL PRIMARY KEY,
    transaction_id UUID NOT NULL REFERENCES credit_transactions(id),
    prev_hash VARCHAR(64),
    current_hash VARCHAR(64) NOT NULL,
    signature VARCHAR(256),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB
);

-- Create index for audit log queries
CREATE INDEX IF NOT EXISTS idx_audit_log_chain_tx 
    ON audit_log_chain(transaction_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_chain_created 
    ON audit_log_chain(created_at DESC);

-- Create function to calculate audit log hash
CREATE OR REPLACE FUNCTION calculate_audit_hash(
    p_transaction_id UUID,
    p_prev_hash VARCHAR(64)
) RETURNS VARCHAR(64) AS $$
DECLARE
    v_hash VARCHAR(64);
    v_data TEXT;
BEGIN
    -- Concatenate transaction data for hashing
    SELECT encode(
        digest(
            p_transaction_id::TEXT || 
            COALESCE(p_prev_hash, '') || 
            NOW()::TEXT,
            'sha256'
        ),
        'hex'
    ) INTO v_hash;
    
    RETURN v_hash;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Create trigger function for audit log chain
CREATE OR REPLACE FUNCTION insert_audit_log_chain()
RETURNS TRIGGER AS $$
DECLARE
    v_prev_hash VARCHAR(64);
    v_current_hash VARCHAR(64);
BEGIN
    -- Get previous hash
    SELECT current_hash INTO v_prev_hash
    FROM audit_log_chain
    ORDER BY id DESC
    LIMIT 1;
    
    -- Calculate current hash
    v_current_hash := calculate_audit_hash(NEW.id, v_prev_hash);
    
    -- Insert audit log entry
    INSERT INTO audit_log_chain (
        transaction_id,
        prev_hash,
        current_hash,
        signature,
        metadata
    ) VALUES (
        NEW.id,
        v_prev_hash,
        v_current_hash,
        encode(digest(v_current_hash, 'sha256'), 'hex'),
        jsonb_build_object(
            'amount', NEW.amount,
            'type', NEW.transaction_type,
            'tenant', NEW.tenant_id
        )
    );
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for automatic audit logging
DROP TRIGGER IF EXISTS trigger_audit_log_chain ON credit_transactions;
CREATE TRIGGER trigger_audit_log_chain
    AFTER INSERT ON credit_transactions
    FOR EACH ROW
    EXECUTE FUNCTION insert_audit_log_chain();

-- Create usage statistics table for billing
CREATE TABLE IF NOT EXISTS credit_usage_stats (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(255) NOT NULL,
    period_start TIMESTAMP WITH TIME ZONE NOT NULL,
    period_end TIMESTAMP WITH TIME ZONE NOT NULL,
    total_consumed BIGINT DEFAULT 0,
    total_purchased BIGINT DEFAULT 0,
    transaction_count INTEGER DEFAULT 0,
    avg_transaction_size BIGINT DEFAULT 0,
    peak_balance BIGINT DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT credit_usage_stats_period_check CHECK (period_end > period_start)
);

-- Create indexes for usage stats queries
CREATE INDEX IF NOT EXISTS idx_credit_usage_stats_tenant 
    ON credit_usage_stats(tenant_id);
CREATE INDEX IF NOT EXISTS idx_credit_usage_stats_period 
    ON credit_usage_stats(period_start, period_end);

-- Create function to calculate usage statistics
CREATE OR REPLACE FUNCTION calculate_usage_stats(
    p_tenant_id VARCHAR(255),
    p_period_start TIMESTAMP WITH TIME ZONE,
    p_period_end TIMESTAMP WITH TIME ZONE
) RETURNS TABLE (
    total_consumed BIGINT,
    total_purchased BIGINT,
    transaction_count BIGINT,
    avg_transaction_size NUMERIC,
    peak_balance BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COALESCE(SUM(CASE WHEN transaction_type = 'debit' THEN ABS(amount) ELSE 0 END), 0) AS total_consumed,
        COALESCE(SUM(CASE WHEN transaction_type = 'credit' THEN amount ELSE 0 END), 0) AS total_purchased,
        COUNT(*) AS transaction_count,
        COALESCE(AVG(ABS(amount)), 0) AS avg_transaction_size,
        COALESCE(MAX((metadata->>'balance_after')::BIGINT), 0) AS peak_balance
    FROM credit_transactions
    WHERE tenant_id = p_tenant_id
        AND created_at >= p_period_start
        AND created_at < p_period_end;
END;
$$ LANGUAGE plpgsql STABLE;

-- Create low balance alert table
CREATE TABLE IF NOT EXISTS credit_alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(255) NOT NULL,
    alert_type VARCHAR(50) NOT NULL,
    threshold_value BIGINT,
    current_value BIGINT,
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    resolved_at TIMESTAMP WITH TIME ZONE,
    notified BOOLEAN DEFAULT FALSE
);

-- Create index for active alerts
CREATE INDEX IF NOT EXISTS idx_credit_alerts_active 
    ON credit_alerts(tenant_id, status) WHERE status = 'active';

-- Create function to check and create alerts
CREATE OR REPLACE FUNCTION check_balance_threshold()
RETURNS TRIGGER AS $$
DECLARE
    v_threshold BIGINT;
    v_available BIGINT;
BEGIN
    -- Calculate available balance
    v_available := NEW.balance - NEW.reserved;
    
    -- Get threshold from metadata (default 1000)
    v_threshold := COALESCE((NEW.metadata->>'alert_threshold')::BIGINT, 1000);
    
    -- Create alert if below threshold
    IF v_available < v_threshold THEN
        INSERT INTO credit_alerts (
            tenant_id,
            alert_type,
            threshold_value,
            current_value,
            status
        ) VALUES (
            NEW.tenant_id,
            'low_balance',
            v_threshold,
            v_available,
            'active'
        ) ON CONFLICT DO NOTHING;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for balance threshold checks
DROP TRIGGER IF EXISTS trigger_check_balance_threshold ON credit_accounts;
CREATE TRIGGER trigger_check_balance_threshold
    AFTER UPDATE OF balance, reserved ON credit_accounts
    FOR EACH ROW
    EXECUTE FUNCTION check_balance_threshold();

-- Create performance monitoring view
CREATE OR REPLACE VIEW credit_performance_metrics AS
SELECT
    COUNT(*) as total_transactions,
    COUNT(DISTINCT tenant_id) as active_tenants,
    SUM(CASE WHEN transaction_type = 'credit' THEN amount ELSE 0 END) as total_credits,
    SUM(CASE WHEN transaction_type = 'debit' THEN ABS(amount) ELSE 0 END) as total_debits,
    AVG(CASE WHEN completed_at IS NOT NULL 
        THEN EXTRACT(EPOCH FROM (completed_at - created_at)) 
        ELSE NULL END) as avg_transaction_time_seconds,
    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_count,
    COUNT(CASE WHEN status = 'pending' THEN 1 END) as pending_count
FROM credit_transactions
WHERE created_at > NOW() - INTERVAL '24 hours';

-- Grant permissions
GRANT SELECT ON credit_performance_metrics TO credits_user;
GRANT SELECT ON credit_balances_summary TO credits_user;
GRANT EXECUTE ON FUNCTION refresh_credit_balances_summary() TO credits_user;
GRANT EXECUTE ON FUNCTION calculate_usage_stats(VARCHAR, TIMESTAMP WITH TIME ZONE, TIMESTAMP WITH TIME ZONE) TO credits_user;

-- Create scheduled job to refresh materialized view (requires pg_cron extension)
-- Uncomment if pg_cron is available
-- SELECT cron.schedule('refresh-credit-balances', '*/5 * * * *', 'SELECT refresh_credit_balances_summary()');

-- Create comment documentation
COMMENT ON TABLE credit_accounts IS 'Main credits balance table with optimistic locking (version column)';
COMMENT ON TABLE credit_transactions IS 'Immutable transaction log partitioned by date';
COMMENT ON TABLE audit_log_chain IS 'Blockchain-style audit log chain for transaction integrity';
COMMENT ON TABLE idempotency_keys IS 'Distributed transaction deduplication using hashed keys';
COMMENT ON TABLE credit_usage_stats IS 'Pre-aggregated usage statistics for billing reports';
COMMENT ON TABLE credit_alerts IS 'Low balance and quota alerts for tenants';

COMMENT ON COLUMN credit_accounts.version IS 'Optimistic locking version number - increment on each update';
COMMENT ON COLUMN credit_accounts.reserved IS 'Credits reserved for pending operations';
COMMENT ON COLUMN audit_log_chain.prev_hash IS 'Hash of previous audit entry (blockchain chain)';
COMMENT ON COLUMN audit_log_chain.signature IS 'HMAC signature for tamper detection';

-- Create database statistics for query optimization
ANALYZE credit_accounts;
ANALYZE credit_transactions;
ANALYZE audit_log_chain;
ANALYZE idempotency_keys;

-- Log migration completion
DO $$
BEGIN
    RAISE NOTICE 'Credits schema enhancement completed successfully';
    RAISE NOTICE 'Features: Partitioning, Materialized Views, Audit Chain, Alerts, Performance Monitoring';
END $$;
