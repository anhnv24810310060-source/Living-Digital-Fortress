-- Rollback: Drop credits schema
-- Version: 000001

-- Drop triggers
DROP TRIGGER IF EXISTS update_credits_updated_at ON credits;

-- Drop functions
DROP FUNCTION IF EXISTS update_updated_at_column();

-- Drop indexes
DROP INDEX IF EXISTS idx_transactions_type;
DROP INDEX IF EXISTS idx_transactions_created_at;
DROP INDEX IF EXISTS idx_transactions_status;
DROP INDEX IF EXISTS idx_transactions_user_id;
DROP INDEX IF EXISTS idx_credits_user_id_unique;
DROP INDEX IF EXISTS idx_credits_user_id;

-- Drop tables
DROP TABLE IF EXISTS transactions;
DROP TABLE IF EXISTS credits;

-- Drop extension (only if not used by other tables)
-- DROP EXTENSION IF EXISTS pgcrypto;
