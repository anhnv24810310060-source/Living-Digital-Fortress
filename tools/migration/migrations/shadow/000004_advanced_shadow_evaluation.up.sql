-- Advanced shadow evaluation schema with performance optimizations
-- Migration: 000004_advanced_shadow_evaluation.up.sql

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS pg_trgm; -- For text search optimizations

-- Advanced evaluations table with comprehensive metrics
CREATE TABLE IF NOT EXISTS shadow_evaluations_advanced (
    eval_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    rule_id VARCHAR(255) NOT NULL,
    rule_name VARCHAR(255) NOT NULL,
    rule_type VARCHAR(100) NOT NULL,
    rule_config JSONB NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    config JSONB NOT NULL, -- EvaluationConfig JSON
    
    -- Status tracking
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    error_message TEXT,
    
    -- Confusion matrix
    true_positives INTEGER DEFAULT 0,
    false_positives INTEGER DEFAULT 0,
    true_negatives INTEGER DEFAULT 0,
    false_negatives INTEGER DEFAULT 0,
    
    -- Core metrics
    precision DOUBLE PRECISION DEFAULT 0,
    recall_rate DOUBLE PRECISION DEFAULT 0,
    f1_score DOUBLE PRECISION DEFAULT 0,
    accuracy DOUBLE PRECISION DEFAULT 0,
    false_positive_rate DOUBLE PRECISION DEFAULT 0,
    false_negative_rate DOUBLE PRECISION DEFAULT 0,
    
    -- Advanced metrics
    matthews_corr_coef DOUBLE PRECISION DEFAULT 0,
    cohen_kappa DOUBLE PRECISION DEFAULT 0,
    auc_roc DOUBLE PRECISION DEFAULT 0,
    
    -- Statistical significance
    chi_square DOUBLE PRECISION DEFAULT 0,
    p_value DOUBLE PRECISION DEFAULT 1,
    is_significant BOOLEAN DEFAULT FALSE,
    
    -- Performance metrics
    execution_time_ms BIGINT DEFAULT 0,
    throughput_qps DOUBLE PRECISION DEFAULT 0,
    avg_latency_ms DOUBLE PRECISION DEFAULT 0,
    p95_latency_ms DOUBLE PRECISION DEFAULT 0,
    p99_latency_ms DOUBLE PRECISION DEFAULT 0,
    
    -- Deployment recommendation
    deployment_recommendation VARCHAR(50),
    confidence_score DOUBLE PRECISION DEFAULT 0,
    risk_level VARCHAR(20),
    recommendations JSONB,
    
    -- Results storage
    results JSONB,
    performance_data JSONB,
    
    -- Sample metadata
    sample_size INTEGER NOT NULL,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    
    CONSTRAINT valid_status CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    CONSTRAINT valid_risk_level CHECK (risk_level IN ('low', 'medium', 'high') OR risk_level IS NULL),
    CONSTRAINT positive_sample_size CHECK (sample_size > 0)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_shadow_eval_adv_tenant ON shadow_evaluations_advanced(tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_shadow_eval_adv_status ON shadow_evaluations_advanced(status) WHERE status IN ('pending', 'running');
CREATE INDEX IF NOT EXISTS idx_shadow_eval_adv_rule ON shadow_evaluations_advanced(rule_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_shadow_eval_adv_deployment ON shadow_evaluations_advanced(deployment_recommendation) WHERE completed_at IS NOT NULL;

-- GIN index for JSONB queries
CREATE INDEX IF NOT EXISTS idx_shadow_eval_adv_config ON shadow_evaluations_advanced USING GIN(config);
CREATE INDEX IF NOT EXISTS idx_shadow_eval_adv_results ON shadow_evaluations_advanced USING GIN(results);

-- Canary deployment tracking
CREATE TABLE IF NOT EXISTS canary_deployments (
    canary_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    rule_id VARCHAR(255) NOT NULL,
    eval_id UUID NOT NULL,
    
    -- Deployment config
    status VARCHAR(50) NOT NULL DEFAULT 'active',
    percentage INTEGER NOT NULL,
    
    -- Metrics
    requests_total BIGINT DEFAULT 0,
    errors_total BIGINT DEFAULT 0,
    error_rate DOUBLE PRECISION DEFAULT 0,
    avg_latency_ms DOUBLE PRECISION DEFAULT 0,
    p95_latency_ms DOUBLE PRECISION DEFAULT 0,
    throughput_qps DOUBLE PRECISION DEFAULT 0,
    
    -- Rollback information
    rollback_reason TEXT,
    
    -- Timestamps
    start_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    end_time TIMESTAMPTZ,
    
    CONSTRAINT valid_canary_status CHECK (status IN ('active', 'promoted', 'rolled_back')),
    CONSTRAINT valid_percentage CHECK (percentage > 0 AND percentage <= 100),
    CONSTRAINT valid_error_rate CHECK (error_rate >= 0 AND error_rate <= 1),
    
    FOREIGN KEY (eval_id) REFERENCES shadow_evaluations_advanced(eval_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_canary_status ON canary_deployments(status, start_time DESC);
CREATE INDEX IF NOT EXISTS idx_canary_rule ON canary_deployments(rule_id, start_time DESC);
CREATE INDEX IF NOT EXISTS idx_canary_eval ON canary_deployments(eval_id);

-- Enhanced traffic samples with partitioning support (for high volume)
CREATE TABLE IF NOT EXISTS traffic_samples_enhanced (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Network metadata
    timestamp TIMESTAMPTZ NOT NULL,
    source_ip INET NOT NULL,
    dest_ip INET NOT NULL,
    protocol VARCHAR(20) NOT NULL,
    port INTEGER NOT NULL,
    
    -- Request data
    payload TEXT,
    headers JSONB,
    user_agent TEXT,
    method VARCHAR(10),
    uri TEXT,
    query_params JSONB,
    
    -- Classification
    is_attack BOOLEAN NOT NULL DEFAULT FALSE,
    attack_type VARCHAR(100),
    attack_confidence DOUBLE PRECISION,
    threat_score INTEGER,
    
    -- Geographic data
    source_country VARCHAR(2),
    source_asn INTEGER,
    
    -- Response data
    response_status INTEGER,
    response_size INTEGER,
    response_time_ms DOUBLE PRECISION,
    
    -- Metadata
    tenant_id VARCHAR(255),
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    CONSTRAINT valid_threat_score CHECK (threat_score >= 0 AND threat_score <= 100),
    CONSTRAINT valid_response_status CHECK (response_status >= 100 AND response_status < 600 OR response_status IS NULL)
);

-- Partitioning by timestamp for efficient queries and archival
-- Production would use declarative partitioning: PARTITION BY RANGE (timestamp)

-- Indexes for traffic samples
CREATE INDEX IF NOT EXISTS idx_traffic_enhanced_timestamp ON traffic_samples_enhanced(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_traffic_enhanced_attack ON traffic_samples_enhanced(is_attack, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_traffic_enhanced_source_ip ON traffic_samples_enhanced(source_ip, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_traffic_enhanced_tenant ON traffic_samples_enhanced(tenant_id, timestamp DESC) WHERE tenant_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_traffic_enhanced_attack_type ON traffic_samples_enhanced(attack_type) WHERE attack_type IS NOT NULL;

-- GIN indexes for JSONB columns
CREATE INDEX IF NOT EXISTS idx_traffic_enhanced_headers ON traffic_samples_enhanced USING GIN(headers);
CREATE INDEX IF NOT EXISTS idx_traffic_enhanced_query_params ON traffic_samples_enhanced USING GIN(query_params);
CREATE INDEX IF NOT EXISTS idx_traffic_enhanced_metadata ON traffic_samples_enhanced USING GIN(metadata);

-- Text search index for payload
CREATE INDEX IF NOT EXISTS idx_traffic_enhanced_payload_search ON traffic_samples_enhanced USING GIN(to_tsvector('english', COALESCE(payload, '')));

-- Rule deployment history (immutable audit trail)
CREATE TABLE IF NOT EXISTS rule_deployment_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    rule_id VARCHAR(255) NOT NULL,
    eval_id UUID,
    canary_id UUID,
    
    -- Deployment metadata
    deployment_type VARCHAR(50) NOT NULL, -- 'full', 'canary', 'rollback'
    deployment_status VARCHAR(50) NOT NULL,
    deployed_by VARCHAR(255),
    
    -- Version control
    rule_version INTEGER NOT NULL DEFAULT 1,
    previous_version INTEGER,
    
    -- Deployment config
    config JSONB,
    
    -- Results
    success BOOLEAN NOT NULL,
    error_message TEXT,
    
    -- Performance after deployment
    post_deploy_metrics JSONB,
    
    -- Immutable audit trail
    prev_hash VARCHAR(128) NOT NULL DEFAULT 'genesis',
    hash VARCHAR(128) NOT NULL,
    
    -- Timestamps
    deployed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    CONSTRAINT valid_deployment_type CHECK (deployment_type IN ('full', 'canary', 'rollback', 'shadow')),
    CONSTRAINT valid_deployment_status CHECK (deployment_status IN ('pending', 'active', 'completed', 'failed', 'rolled_back'))
);

CREATE INDEX IF NOT EXISTS idx_rule_deploy_history_rule ON rule_deployment_history(rule_id, deployed_at DESC);
CREATE INDEX IF NOT EXISTS idx_rule_deploy_history_status ON rule_deployment_history(deployment_status, deployed_at DESC);
CREATE INDEX IF NOT EXISTS idx_rule_deploy_history_type ON rule_deployment_history(deployment_type, deployed_at DESC);

-- Prevent UPDATE/DELETE on deployment history (immutable)
CREATE OR REPLACE FUNCTION rule_deployment_history_immutable() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'rule_deployment_history is immutable';
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'rule_deployment_history_no_modify') THEN
        CREATE TRIGGER rule_deployment_history_no_modify
        BEFORE UPDATE OR DELETE ON rule_deployment_history
        FOR EACH ROW EXECUTE FUNCTION rule_deployment_history_immutable();
    END IF;
END $$;

-- Performance optimization: Automatic cleanup of old traffic samples
-- Run daily via cron or pg_cron extension
CREATE TABLE IF NOT EXISTS shadow_maintenance_config (
    key VARCHAR(255) PRIMARY KEY,
    value JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

INSERT INTO shadow_maintenance_config (key, value) VALUES 
('traffic_retention_days', '90'::JSONB),
('archive_enabled', 'true'::JSONB),
('cleanup_batch_size', '10000'::JSONB)
ON CONFLICT (key) DO NOTHING;

-- View for deployment recommendations
CREATE OR REPLACE VIEW v_deployment_recommendations AS
SELECT 
    eval_id,
    rule_id,
    rule_name,
    tenant_id,
    deployment_recommendation,
    confidence_score,
    risk_level,
    precision,
    recall_rate,
    f1_score,
    false_positive_rate,
    is_significant,
    p_value,
    created_at,
    completed_at
FROM shadow_evaluations_advanced
WHERE status = 'completed'
  AND completed_at >= NOW() - INTERVAL '30 days'
ORDER BY completed_at DESC;

-- View for active canaries
CREATE OR REPLACE VIEW v_active_canaries AS
SELECT 
    c.canary_id,
    c.rule_id,
    c.eval_id,
    c.percentage,
    c.error_rate,
    c.avg_latency_ms,
    c.throughput_qps,
    c.start_time,
    e.rule_name,
    e.tenant_id,
    EXTRACT(EPOCH FROM (NOW() - c.start_time))/60 AS runtime_minutes
FROM canary_deployments c
JOIN shadow_evaluations_advanced e ON c.eval_id = e.eval_id
WHERE c.status = 'active'
ORDER BY c.start_time DESC;

-- Function to archive old traffic samples
CREATE OR REPLACE FUNCTION archive_old_traffic_samples() RETURNS INTEGER AS $$
DECLARE
    retention_days INTEGER;
    batch_size INTEGER;
    archived_count INTEGER := 0;
BEGIN
    -- Get config
    SELECT (value::TEXT)::INTEGER INTO retention_days 
    FROM shadow_maintenance_config WHERE key = 'traffic_retention_days';
    
    SELECT (value::TEXT)::INTEGER INTO batch_size 
    FROM shadow_maintenance_config WHERE key = 'cleanup_batch_size';
    
    -- Delete old records (or move to archive table in production)
    DELETE FROM traffic_samples_enhanced
    WHERE timestamp < NOW() - (retention_days || ' days')::INTERVAL
      AND id IN (
          SELECT id FROM traffic_samples_enhanced
          WHERE timestamp < NOW() - (retention_days || ' days')::INTERVAL
          ORDER BY timestamp
          LIMIT batch_size
      );
    
    GET DIAGNOSTICS archived_count = ROW_COUNT;
    
    RETURN archived_count;
END;
$$ LANGUAGE plpgsql;

COMMENT ON TABLE shadow_evaluations_advanced IS 'Advanced shadow evaluation with comprehensive metrics and statistical analysis';
COMMENT ON TABLE canary_deployments IS 'Canary deployment tracking with automatic rollback support';
COMMENT ON TABLE traffic_samples_enhanced IS 'Enhanced traffic samples for high-fidelity evaluation';
COMMENT ON TABLE rule_deployment_history IS 'Immutable audit trail of all rule deployments';
