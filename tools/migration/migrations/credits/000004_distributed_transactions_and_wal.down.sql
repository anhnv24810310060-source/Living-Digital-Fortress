-- Rollback migration: Remove distributed transactions and WAL tables
-- Version: 000004
-- Description: Drops two-phase commit and audit log infrastructure

-- Drop cron jobs if they exist
-- SELECT cron.unschedule('cleanup-old-transactions');
-- SELECT cron.unschedule('recover-stalled-transactions');

-- Drop views
DROP VIEW IF EXISTS suspicious_activity CASCADE;
DROP VIEW IF EXISTS audit_summary CASCADE;
DROP VIEW IF EXISTS transaction_stats CASCADE;

-- Drop functions
DROP FUNCTION IF EXISTS verify_audit_chain(BIGINT, BIGINT) CASCADE;
DROP FUNCTION IF EXISTS cleanup_old_transactions(INTEGER) CASCADE;
DROP FUNCTION IF EXISTS recover_stalled_transactions() CASCADE;
DROP FUNCTION IF EXISTS assign_audit_sequence() CASCADE;

-- Drop triggers
DROP TRIGGER IF EXISTS trigger_assign_audit_sequence ON audit_log;

-- Drop sequences
DROP SEQUENCE IF EXISTS audit_log_sequence CASCADE;

-- Drop indexes
DROP INDEX IF EXISTS idx_audit_log_metadata CASCADE;
DROP INDEX IF EXISTS idx_distributed_tx_metadata CASCADE;
DROP INDEX IF EXISTS idx_distributed_tx_active CASCADE;
DROP INDEX IF EXISTS idx_audit_log_event_time CASCADE;
DROP INDEX IF EXISTS idx_audit_log_tenant_time CASCADE;
DROP INDEX IF EXISTS idx_audit_log_success CASCADE;
DROP INDEX IF EXISTS idx_audit_log_action CASCADE;
DROP INDEX IF EXISTS idx_audit_log_event_type CASCADE;
DROP INDEX IF EXISTS idx_audit_log_timestamp CASCADE;
DROP INDEX IF EXISTS idx_audit_log_transaction CASCADE;
DROP INDEX IF EXISTS idx_audit_log_tenant CASCADE;
DROP INDEX IF EXISTS idx_audit_log_sequence CASCADE;
DROP INDEX IF EXISTS idx_distributed_transactions_timeout CASCADE;
DROP INDEX IF EXISTS idx_distributed_transactions_type CASCADE;
DROP INDEX IF EXISTS idx_distributed_transactions_created CASCADE;
DROP INDEX IF EXISTS idx_distributed_transactions_state CASCADE;
DROP INDEX IF EXISTS idx_distributed_transactions_tenant CASCADE;

-- Drop tables
DROP TABLE IF EXISTS audit_log CASCADE;
DROP TABLE IF EXISTS distributed_transactions CASCADE;

-- Remove constraints from credit_accounts if they were added
ALTER TABLE IF EXISTS credit_accounts DROP CONSTRAINT IF EXISTS credit_accounts_reserved_lte_balance;
