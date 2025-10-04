-- Rollback Production Enhancements
-- Version: 000006_production_enhancements.down.sql

-- Drop helper functions
DROP FUNCTION IF EXISTS get_system_health();
DROP FUNCTION IF EXISTS get_compliance_summary();
DROP FUNCTION IF EXISTS get_event_sourcing_stats();

-- Drop performance monitoring tables
DROP TABLE IF EXISTS service_health_checks CASCADE;
DROP TABLE IF EXISTS performance_metrics CASCADE;

-- Drop security tables
DROP TABLE IF EXISTS security_alerts CASCADE;
DROP TABLE IF EXISTS audit_trail CASCADE;

-- Drop synthetic data tables
DROP TABLE IF EXISTS synthetic_transactions CASCADE;
DROP TABLE IF EXISTS synthetic_user_profiles CASCADE;
DROP TABLE IF EXISTS synthetic_data_batches CASCADE;

-- Drop compliance tables
DROP TABLE IF EXISTS compliance_reports CASCADE;
DROP TABLE IF EXISTS compliance_evidence CASCADE;
DROP TABLE IF EXISTS compliance_findings CASCADE;
DROP TABLE IF EXISTS compliance_check_results CASCADE;
DROP TABLE IF EXISTS compliance_controls CASCADE;
DROP TABLE IF EXISTS compliance_frameworks CASCADE;

-- Drop deployment tables
DROP TABLE IF EXISTS canary_deployments CASCADE;
DROP TABLE IF EXISTS feature_flag_evaluations CASCADE;
DROP TABLE IF EXISTS feature_flags CASCADE;
DROP TABLE IF EXISTS deployment_history CASCADE;

-- Drop DR tables
DROP TABLE IF EXISTS failover_events CASCADE;
DROP TABLE IF EXISTS replication_log CASCADE;
DROP TABLE IF EXISTS cloud_regions CASCADE;

-- Drop sharding tables
DROP TABLE IF EXISTS distributed_tx_log CASCADE;
DROP TABLE IF EXISTS shard_data CASCADE;

-- Drop event sourcing tables
DROP TABLE IF EXISTS credit_balances_read_model CASCADE;
DROP TABLE IF EXISTS command_log CASCADE;
DROP TABLE IF EXISTS projection_tracking CASCADE;
DROP TABLE IF EXISTS aggregate_snapshots CASCADE;
DROP TABLE IF EXISTS event_store_2024_12 CASCADE;
DROP TABLE IF EXISTS event_store_2024_11 CASCADE;
DROP TABLE IF EXISTS event_store_2024_10 CASCADE;
DROP TABLE IF EXISTS event_store CASCADE;
