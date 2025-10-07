-- ShieldX Database Initialization Script
-- Creates databases and base schemas for all services

-- Create separate databases for each service
CREATE DATABASE IF NOT EXISTS credits_db;
CREATE DATABASE IF NOT EXISTS contauth_db;
CREATE DATABASE IF NOT EXISTS shadow_db;
CREATE DATABASE IF NOT EXISTS guardian_db;

-- Connect to credits_db
\c credits_db

-- Credits database schema
CREATE TABLE IF NOT EXISTS credits_transactions (
    id BIGSERIAL PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    amount DECIMAL(10, 2) NOT NULL,
    transaction_type VARCHAR(50) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_tenant_user ON credits_transactions(tenant_id, user_id);
CREATE INDEX IF NOT EXISTS idx_created_at ON credits_transactions(created_at);

CREATE TABLE IF NOT EXISTS tenant_quotas (
    tenant_id VARCHAR(255) PRIMARY KEY,
    total_credits DECIMAL(10, 2) NOT NULL DEFAULT 0,
    used_credits DECIMAL(10, 2) NOT NULL DEFAULT 0,
    quota_limit DECIMAL(10, 2),
    last_reset TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS usage_metrics (
    id BIGSERIAL PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    service_name VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10, 4) NOT NULL,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_tenant_service ON usage_metrics(tenant_id, service_name);
CREATE INDEX IF NOT EXISTS idx_recorded_at ON usage_metrics(recorded_at);

-- Connect to contauth_db
\c contauth_db

-- Continuous Authentication database schema
CREATE TABLE IF NOT EXISTS session_telemetry (
    id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    keystroke_data JSONB,
    mouse_data JSONB,
    device_fingerprint JSONB,
    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_session ON session_telemetry(session_id);
CREATE INDEX IF NOT EXISTS idx_user ON session_telemetry(user_id);
CREATE INDEX IF NOT EXISTS idx_collected_at ON session_telemetry(collected_at);

CREATE TABLE IF NOT EXISTS risk_scores (
    id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    risk_score DECIMAL(5, 4) NOT NULL,
    risk_factors JSONB,
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_risk_session ON risk_scores(session_id);
CREATE INDEX IF NOT EXISTS idx_risk_calculated_at ON risk_scores(calculated_at);

CREATE TABLE IF NOT EXISTS auth_decisions (
    id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    decision VARCHAR(50) NOT NULL,
    confidence DECIMAL(5, 4) NOT NULL,
    reason TEXT,
    decided_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_auth_session ON auth_decisions(session_id);
CREATE INDEX IF NOT EXISTS idx_auth_decided_at ON auth_decisions(decided_at);

CREATE TABLE IF NOT EXISTS user_baselines (
    user_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    keystroke_profile JSONB,
    mouse_profile JSONB,
    device_profile JSONB,
    baseline_samples INT DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_baseline_tenant ON user_baselines(tenant_id);

-- Connect to shadow_db
\c shadow_db

-- Shadow evaluation database schema
CREATE TABLE IF NOT EXISTS evaluation_rules (
    id BIGSERIAL PRIMARY KEY,
    rule_name VARCHAR(255) NOT NULL UNIQUE,
    rule_type VARCHAR(100) NOT NULL,
    rule_definition JSONB NOT NULL,
    enabled BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS test_results (
    id BIGSERIAL PRIMARY KEY,
    rule_id BIGINT REFERENCES evaluation_rules(id),
    test_name VARCHAR(255) NOT NULL,
    test_input JSONB,
    expected_output JSONB,
    actual_output JSONB,
    passed BOOLEAN NOT NULL,
    execution_time_ms INT,
    tested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_test_rule ON test_results(rule_id);
CREATE INDEX IF NOT EXISTS idx_test_tested_at ON test_results(tested_at);

CREATE TABLE IF NOT EXISTS performance_metrics (
    id BIGSERIAL PRIMARY KEY,
    rule_id BIGINT REFERENCES evaluation_rules(id),
    metric_type VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10, 4) NOT NULL,
    sample_size INT,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_perf_rule ON performance_metrics(rule_id);
CREATE INDEX IF NOT EXISTS idx_perf_recorded_at ON performance_metrics(recorded_at);

-- Connect to guardian_db
\c guardian_db

-- Guardian (Sandbox) database schema
CREATE TABLE IF NOT EXISTS sandbox_executions (
    id BIGSERIAL PRIMARY KEY,
    execution_id VARCHAR(255) NOT NULL UNIQUE,
    tenant_id VARCHAR(255) NOT NULL,
    request_data JSONB,
    sandbox_type VARCHAR(50) NOT NULL,
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    status VARCHAR(50) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_exec_execution ON sandbox_executions(execution_id);
CREATE INDEX IF NOT EXISTS idx_exec_tenant ON sandbox_executions(tenant_id);
CREATE INDEX IF NOT EXISTS idx_exec_started_at ON sandbox_executions(started_at);

CREATE TABLE IF NOT EXISTS threat_detections (
    id BIGSERIAL PRIMARY KEY,
    execution_id VARCHAR(255) NOT NULL,
    threat_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    indicators JSONB,
    confidence DECIMAL(5, 4) NOT NULL,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_threat_execution ON threat_detections(execution_id);
CREATE INDEX IF NOT EXISTS idx_threat_detected_at ON threat_detections(detected_at);
CREATE INDEX IF NOT EXISTS idx_threat_severity ON threat_detections(severity);

CREATE TABLE IF NOT EXISTS forensic_data (
    id BIGSERIAL PRIMARY KEY,
    execution_id VARCHAR(255) NOT NULL,
    data_type VARCHAR(100) NOT NULL,
    data_content JSONB,
    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_forensic_execution ON forensic_data(execution_id);
CREATE INDEX IF NOT EXISTS idx_forensic_data_type ON forensic_data(data_type);

-- Create read-only user for replicas
CREATE USER shieldx_reader WITH PASSWORD 'reader_pass_2025';
GRANT CONNECT ON DATABASE credits_db TO shieldx_reader;
GRANT CONNECT ON DATABASE contauth_db TO shieldx_reader;
GRANT CONNECT ON DATABASE shadow_db TO shieldx_reader;
GRANT CONNECT ON DATABASE guardian_db TO shieldx_reader;

\c credits_db
GRANT SELECT ON ALL TABLES IN SCHEMA public TO shieldx_reader;

\c contauth_db
GRANT SELECT ON ALL TABLES IN SCHEMA public TO shieldx_reader;

\c shadow_db
GRANT SELECT ON ALL TABLES IN SCHEMA public TO shieldx_reader;

\c guardian_db
GRANT SELECT ON ALL TABLES IN SCHEMA public TO shieldx_reader;
