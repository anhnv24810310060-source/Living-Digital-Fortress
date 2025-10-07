-- Migration: Advanced Shadow Evaluation with Bayesian A/B Testing
-- Version: 000002
-- Description: Implements Champion/Challenger pattern with statistical rigor

-- A/B Tests configuration table
CREATE TABLE IF NOT EXISTS ab_tests (
    test_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    champion_id UUID NOT NULL,
    challenger_id UUID NOT NULL,
    
    -- Test parameters
    min_sample_size BIGINT NOT NULL DEFAULT 1000,
    max_duration_seconds INTEGER NOT NULL DEFAULT 86400, -- 24 hours default
    confidence_level FLOAT NOT NULL DEFAULT 0.95,
    minimum_detectable_effect FLOAT NOT NULL DEFAULT 0.01,
    probability_threshold FLOAT NOT NULL DEFAULT 0.95,
    
    -- Safety thresholds
    max_error_rate FLOAT NOT NULL DEFAULT 0.05,
    max_latency_p95 FLOAT NOT NULL DEFAULT 1000.0,
    
    -- Auto-rollback configuration
    auto_rollback BOOLEAN NOT NULL DEFAULT true,
    rollback_on_error_rate FLOAT NOT NULL DEFAULT 0.10,
    rollback_on_latency FLOAT NOT NULL DEFAULT 2000.0,
    
    -- Status tracking
    status VARCHAR(50) NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'completed', 'rolled_back')),
    started_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    winner_variant_id UUID,
    decision_reason TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    CONSTRAINT ab_tests_confidence_check CHECK (confidence_level > 0 AND confidence_level < 1),
    CONSTRAINT ab_tests_probability_check CHECK (probability_threshold > 0.5 AND probability_threshold < 1),
    CONSTRAINT ab_tests_min_samples_check CHECK (min_sample_size >= 100)
);

-- Test variants table (Champion and Challenger)
CREATE TABLE IF NOT EXISTS test_variants (
    variant_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    test_id UUID NOT NULL REFERENCES ab_tests(test_id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    type VARCHAR(50) NOT NULL CHECK (type IN ('champion', 'challenger')),
    rule_id VARCHAR(255) NOT NULL,
    rule_config JSONB NOT NULL DEFAULT '{}'::jsonb,
    traffic_percentage FLOAT NOT NULL DEFAULT 50.0,
    
    -- Bayesian statistics (Beta distribution parameters)
    successes BIGINT NOT NULL DEFAULT 0,
    failures BIGINT NOT NULL DEFAULT 0,
    total_samples BIGINT NOT NULL DEFAULT 0,
    alpha_prior FLOAT NOT NULL DEFAULT 1.0,
    beta_prior FLOAT NOT NULL DEFAULT 1.0,
    posterior_alpha FLOAT,
    posterior_beta FLOAT,
    
    -- Derived metrics
    mean_conversion_rate FLOAT,
    credible_interval_lower FLOAT,
    credible_interval_upper FLOAT,
    probability_beat_champion FLOAT,
    
    -- Performance metrics
    avg_latency_ms FLOAT,
    p95_latency_ms FLOAT,
    p99_latency_ms FLOAT,
    error_rate FLOAT,
    
    -- Status
    status VARCHAR(50) NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'winner', 'loser', 'rolled_back')),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    CONSTRAINT test_variants_traffic_check CHECK (traffic_percentage >= 0 AND traffic_percentage <= 100),
    CONSTRAINT test_variants_stats_check CHECK (successes >= 0 AND failures >= 0 AND total_samples >= 0)
);

-- Individual test samples for detailed analysis
CREATE TABLE IF NOT EXISTS test_samples (
    sample_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    variant_id UUID NOT NULL REFERENCES test_variants(variant_id) ON DELETE CASCADE,
    success BOOLEAN NOT NULL,
    latency_ms FLOAT NOT NULL,
    error_type VARCHAR(100),
    metadata JSONB DEFAULT '{}'::jsonb,
    recorded_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    CONSTRAINT test_samples_latency_check CHECK (latency_ms >= 0)
);

-- Canary deployment tracking
CREATE TABLE IF NOT EXISTS canary_deployments (
    deployment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    rule_id VARCHAR(255) NOT NULL,
    rule_name VARCHAR(255) NOT NULL,
    rule_config JSONB NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    
    -- Canary configuration
    canary_percentage FLOAT NOT NULL DEFAULT 5.0,
    duration_minutes INTEGER NOT NULL DEFAULT 60,
    auto_rollback BOOLEAN NOT NULL DEFAULT true,
    
    -- Health checks
    health_check_interval_seconds INTEGER NOT NULL DEFAULT 60,
    max_error_rate FLOAT NOT NULL DEFAULT 0.05,
    max_latency_p95 FLOAT NOT NULL DEFAULT 1000.0,
    min_sample_size INTEGER NOT NULL DEFAULT 100,
    
    -- Status tracking
    status VARCHAR(50) NOT NULL DEFAULT 'deploying' CHECK (status IN ('deploying', 'monitoring', 'promoting', 'completed', 'rolled_back', 'failed')),
    current_percentage FLOAT NOT NULL DEFAULT 0,
    total_requests BIGINT NOT NULL DEFAULT 0,
    successful_requests BIGINT NOT NULL DEFAULT 0,
    failed_requests BIGINT NOT NULL DEFAULT 0,
    current_error_rate FLOAT,
    current_avg_latency FLOAT,
    current_p95_latency FLOAT,
    
    -- Timeline
    started_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    promoted_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    rollback_reason TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    CONSTRAINT canary_deployments_percentage_check CHECK (canary_percentage > 0 AND canary_percentage <= 100)
);

-- Rule deployment history for audit trail
CREATE TABLE IF NOT EXISTS rule_deployment_history (
    history_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    rule_id VARCHAR(255) NOT NULL,
    rule_name VARCHAR(255) NOT NULL,
    rule_config JSONB NOT NULL,
    deployment_type VARCHAR(50) NOT NULL CHECK (deployment_type IN ('direct', 'canary', 'ab_test', 'shadow')),
    tenant_id VARCHAR(255) NOT NULL,
    
    -- Deployment metadata
    deployed_by VARCHAR(255),
    deployment_method VARCHAR(100),
    test_results JSONB,
    
    -- Status
    status VARCHAR(50) NOT NULL CHECK (status IN ('deployed', 'rolled_back', 'superseded')),
    deployed_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    rolled_back_at TIMESTAMP WITH TIME ZONE,
    rollback_reason TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Indexes for A/B tests
CREATE INDEX IF NOT EXISTS idx_ab_tests_tenant ON ab_tests(tenant_id);
CREATE INDEX IF NOT EXISTS idx_ab_tests_status ON ab_tests(status);
CREATE INDEX IF NOT EXISTS idx_ab_tests_started ON ab_tests(started_at DESC);

-- Indexes for test variants
CREATE INDEX IF NOT EXISTS idx_test_variants_test ON test_variants(test_id);
CREATE INDEX IF NOT EXISTS idx_test_variants_status ON test_variants(status);
CREATE INDEX IF NOT EXISTS idx_test_variants_type ON test_variants(type);

-- Indexes for test samples (partitioned for performance)
CREATE INDEX IF NOT EXISTS idx_test_samples_variant ON test_samples(variant_id, recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_test_samples_recorded ON test_samples(recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_test_samples_success ON test_samples(success);

-- Indexes for canary deployments
CREATE INDEX IF NOT EXISTS idx_canary_deployments_tenant ON canary_deployments(tenant_id);
CREATE INDEX IF NOT EXISTS idx_canary_deployments_status ON canary_deployments(status);
CREATE INDEX IF NOT EXISTS idx_canary_deployments_started ON canary_deployments(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_canary_deployments_rule ON canary_deployments(rule_id);

-- Indexes for deployment history
CREATE INDEX IF NOT EXISTS idx_rule_deployment_history_rule ON rule_deployment_history(rule_id);
CREATE INDEX IF NOT EXISTS idx_rule_deployment_history_tenant ON rule_deployment_history(tenant_id);
CREATE INDEX IF NOT EXISTS idx_rule_deployment_history_deployed ON rule_deployment_history(deployed_at DESC);
CREATE INDEX IF NOT EXISTS idx_rule_deployment_history_status ON rule_deployment_history(status);

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_ab_tests_tenant_status ON ab_tests(tenant_id, status, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_test_variants_test_type ON test_variants(test_id, type, status);

-- GIN indexes for JSON fields
CREATE INDEX IF NOT EXISTS idx_test_variants_config ON test_variants USING GIN (rule_config);
CREATE INDEX IF NOT EXISTS idx_test_samples_metadata ON test_samples USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_canary_deployments_config ON canary_deployments USING GIN (rule_config);

-- Update trigger for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_ab_tests_updated_at
    BEFORE UPDATE ON ab_tests
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_test_variants_updated_at
    BEFORE UPDATE ON test_variants
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_canary_deployments_updated_at
    BEFORE UPDATE ON canary_deployments
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- View: Active A/B tests summary
CREATE OR REPLACE VIEW active_ab_tests_summary AS
SELECT
    t.test_id,
    t.name,
    t.tenant_id,
    t.status,
    t.started_at,
    champion.variant_id as champion_id,
    champion.mean_conversion_rate as champion_rate,
    champion.total_samples as champion_samples,
    challenger.variant_id as challenger_id,
    challenger.mean_conversion_rate as challenger_rate,
    challenger.total_samples as challenger_samples,
    challenger.probability_beat_champion,
    (champion.total_samples + challenger.total_samples) as total_samples,
    t.min_sample_size,
    EXTRACT(EPOCH FROM (NOW() - t.started_at)) as duration_seconds,
    t.max_duration_seconds
FROM ab_tests t
JOIN test_variants champion ON t.champion_id = champion.variant_id
JOIN test_variants challenger ON t.challenger_id = challenger.variant_id
WHERE t.status = 'running';

-- View: Canary deployment status
CREATE OR REPLACE VIEW canary_deployment_status AS
SELECT
    deployment_id,
    rule_id,
    rule_name,
    tenant_id,
    status,
    current_percentage,
    total_requests,
    successful_requests,
    failed_requests,
    CASE 
        WHEN total_requests > 0 THEN (failed_requests::float / total_requests)
        ELSE 0
    END as actual_error_rate,
    max_error_rate,
    current_avg_latency,
    current_p95_latency,
    max_latency_p95,
    EXTRACT(EPOCH FROM (NOW() - started_at)) / 60 as elapsed_minutes,
    duration_minutes,
    CASE
        WHEN status = 'monitoring' AND total_requests >= min_sample_size 
             AND (failed_requests::float / NULLIF(total_requests, 0)) <= max_error_rate
             AND current_p95_latency <= max_latency_p95
        THEN 'healthy'
        WHEN status = 'monitoring' AND total_requests >= min_sample_size
        THEN 'unhealthy'
        ELSE 'insufficient_data'
    END as health_status
FROM canary_deployments
WHERE status IN ('deploying', 'monitoring', 'promoting');

-- Function: Calculate statistical significance
CREATE OR REPLACE FUNCTION calculate_statistical_significance(
    test_id_param UUID
) RETURNS TABLE (
    is_significant BOOLEAN,
    p_value FLOAT,
    effect_size FLOAT,
    recommendation TEXT
) AS $$
DECLARE
    champion_stats RECORD;
    challenger_stats RECORD;
    pooled_rate FLOAT;
    z_score FLOAT;
    std_error FLOAT;
BEGIN
    -- Get champion stats
    SELECT successes, failures, total_samples, mean_conversion_rate
    INTO champion_stats
    FROM test_variants
    WHERE test_id = test_id_param AND type = 'champion';
    
    -- Get challenger stats
    SELECT successes, failures, total_samples, mean_conversion_rate
    INTO challenger_stats
    FROM test_variants
    WHERE test_id = test_id_param AND type = 'challenger';
    
    -- Calculate pooled rate
    pooled_rate := (champion_stats.successes + challenger_stats.successes)::float /
                   (champion_stats.total_samples + challenger_stats.total_samples);
    
    -- Calculate standard error
    std_error := SQRT(pooled_rate * (1 - pooled_rate) * 
                     (1.0/champion_stats.total_samples + 1.0/challenger_stats.total_samples));
    
    -- Calculate z-score
    IF std_error > 0 THEN
        z_score := (challenger_stats.mean_conversion_rate - champion_stats.mean_conversion_rate) / std_error;
    ELSE
        z_score := 0;
    END IF;
    
    -- Calculate effect size (Cohen's d)
    effect_size := challenger_stats.mean_conversion_rate - champion_stats.mean_conversion_rate;
    
    -- Simple p-value approximation (two-tailed)
    p_value := 2 * (1 - 0.5 * (1 + erf(ABS(z_score) / SQRT(2))));
    
    -- Determine significance
    is_significant := p_value < 0.05 AND ABS(effect_size) > 0.01;
    
    -- Generate recommendation
    IF is_significant AND effect_size > 0 THEN
        recommendation := 'Deploy challenger - statistically significant improvement';
    ELSIF is_significant AND effect_size < 0 THEN
        recommendation := 'Keep champion - challenger performs worse';
    ELSIF champion_stats.total_samples + challenger_stats.total_samples < 1000 THEN
        recommendation := 'Continue test - insufficient samples';
    ELSE
        recommendation := 'No significant difference - continue monitoring or conclude';
    END IF;
    
    RETURN QUERY SELECT is_significant, p_value, effect_size, recommendation;
END;
$$ LANGUAGE plpgsql;

-- Function: Auto-promote canary if healthy
CREATE OR REPLACE FUNCTION auto_promote_canary()
RETURNS TABLE (promoted_count INTEGER) AS $$
DECLARE
    deployment RECORD;
    count_promoted INTEGER := 0;
BEGIN
    FOR deployment IN
        SELECT * FROM canary_deployments
        WHERE status = 'monitoring'
          AND auto_rollback = true
          AND total_requests >= min_sample_size
          AND (failed_requests::float / NULLIF(total_requests, 0)) <= max_error_rate
          AND current_p95_latency <= max_latency_p95
          AND EXTRACT(EPOCH FROM (NOW() - started_at)) >= (duration_minutes * 60)
    LOOP
        -- Promote to 100%
        UPDATE canary_deployments
        SET status = 'promoting',
            current_percentage = 100.0,
            promoted_at = NOW()
        WHERE deployment_id = deployment.deployment_id;
        
        count_promoted := count_promoted + 1;
    END LOOP;
    
    RETURN QUERY SELECT count_promoted;
END;
$$ LANGUAGE plpgsql;

-- Function: Auto-rollback unhealthy canaries
CREATE OR REPLACE FUNCTION auto_rollback_unhealthy_canaries()
RETURNS TABLE (rolled_back_count INTEGER) AS $$
DECLARE
    deployment RECORD;
    count_rolled_back INTEGER := 0;
    rollback_msg TEXT;
BEGIN
    FOR deployment IN
        SELECT * FROM canary_deployments
        WHERE status IN ('deploying', 'monitoring')
          AND auto_rollback = true
          AND total_requests >= min_sample_size
          AND ((failed_requests::float / NULLIF(total_requests, 0)) > max_error_rate
               OR current_p95_latency > max_latency_p95)
    LOOP
        -- Determine rollback reason
        IF (deployment.failed_requests::float / NULLIF(deployment.total_requests, 0)) > deployment.max_error_rate THEN
            rollback_msg := 'Error rate exceeded threshold: ' || 
                           (deployment.failed_requests::float / deployment.total_requests)::TEXT || 
                           ' > ' || deployment.max_error_rate::TEXT;
        ELSE
            rollback_msg := 'P95 latency exceeded threshold: ' || 
                           deployment.current_p95_latency::TEXT || 
                           ' > ' || deployment.max_latency_p95::TEXT;
        END IF;
        
        -- Rollback
        UPDATE canary_deployments
        SET status = 'rolled_back',
            current_percentage = 0,
            completed_at = NOW(),
            rollback_reason = rollback_msg
        WHERE deployment_id = deployment.deployment_id;
        
        count_rolled_back := count_rolled_back + 1;
    END LOOP;
    
    RETURN QUERY SELECT count_rolled_back;
END;
$$ LANGUAGE plpgsql;

COMMENT ON TABLE ab_tests IS 'Bayesian A/B testing configuration for Champion/Challenger evaluation';
COMMENT ON TABLE test_variants IS 'Individual variants (Champion/Challenger) with Beta distribution parameters';
COMMENT ON TABLE test_samples IS 'Individual test samples for detailed statistical analysis';
COMMENT ON TABLE canary_deployments IS 'Canary deployment tracking with automated health checks';
COMMENT ON TABLE rule_deployment_history IS 'Audit trail of all rule deployments for compliance';
