-- Rollback: Drop cdefnet schema
-- Version: 000001

-- Drop triggers
DROP TRIGGER IF EXISTS update_iocs_timestamp ON iocs;

-- Drop functions
DROP FUNCTION IF EXISTS update_iocs_updated_at();
DROP FUNCTION IF EXISTS cleanup_expired_iocs();

-- Drop indexes
DROP INDEX IF EXISTS idx_audit_details;
DROP INDEX IF EXISTS idx_audit_created_at;
DROP INDEX IF EXISTS idx_audit_event_type;
DROP INDEX IF EXISTS idx_audit_tenant;
DROP INDEX IF EXISTS idx_audit_service;
DROP INDEX IF EXISTS idx_audit_timestamp;
DROP INDEX IF EXISTS idx_iocs_unique;
DROP INDEX IF EXISTS idx_iocs_last_seen;
DROP INDEX IF EXISTS idx_iocs_first_seen;
DROP INDEX IF EXISTS idx_iocs_type;
DROP INDEX IF EXISTS idx_iocs_value_hash;
DROP INDEX IF EXISTS idx_iocs_tenant_hash;

-- Drop tables
DROP TABLE IF EXISTS audit_log;
DROP TABLE IF EXISTS iocs;

-- Drop extension (only if not used by other tables)
-- DROP EXTENSION IF EXISTS "uuid-ossp";
