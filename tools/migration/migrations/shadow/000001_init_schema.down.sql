-- Rollback: Drop shadow schema
-- Version: 000001

-- Drop views
DROP VIEW IF EXISTS shadow_divergence_summary;

-- Drop indexes
DROP INDEX IF EXISTS idx_shadow_metrics_tags;
DROP INDEX IF EXISTS idx_shadow_metrics_created_at;
DROP INDEX IF EXISTS idx_shadow_metrics_name;
DROP INDEX IF EXISTS idx_shadow_evaluations_created_at;
DROP INDEX IF EXISTS idx_shadow_evaluations_divergence;
DROP INDEX IF EXISTS idx_shadow_evaluations_request_id;

-- Drop tables
DROP TABLE IF EXISTS shadow_metrics;
DROP TABLE IF EXISTS shadow_evaluations;

-- Drop extension (only if not used by other tables)
-- DROP EXTENSION IF EXISTS pgcrypto;
