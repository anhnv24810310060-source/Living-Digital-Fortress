-- Migration: Create shadow initial schema
-- Version: 000001
-- Description: Setup shadow evaluation and metrics tables

-- Enable required extensions for UUID/random id generation
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Create shadow_evaluations table
CREATE TABLE IF NOT EXISTS shadow_evaluations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id VARCHAR(255) NOT NULL,
    shadow_result JSONB NOT NULL,
    primary_result JSONB NOT NULL,
    divergence BOOLEAN NOT NULL DEFAULT FALSE,
    divergence_details JSONB,
    latency_ms INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT shadow_latency_positive CHECK (latency_ms >= 0)
);

-- Create shadow_metrics table
CREATE TABLE IF NOT EXISTS shadow_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    metric_name VARCHAR(255) NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    tags JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_shadow_evaluations_request_id ON shadow_evaluations(request_id);
CREATE INDEX IF NOT EXISTS idx_shadow_evaluations_divergence ON shadow_evaluations(divergence);
CREATE INDEX IF NOT EXISTS idx_shadow_evaluations_created_at ON shadow_evaluations(created_at);
CREATE INDEX IF NOT EXISTS idx_shadow_metrics_name ON shadow_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_shadow_metrics_created_at ON shadow_metrics(created_at);
CREATE INDEX IF NOT EXISTS idx_shadow_metrics_tags ON shadow_metrics USING GIN (tags);

-- Create view for divergence summary
CREATE OR REPLACE VIEW shadow_divergence_summary AS
SELECT 
    DATE_TRUNC('hour', created_at) as hour,
    COUNT(*) as total_evaluations,
    SUM(CASE WHEN divergence THEN 1 ELSE 0 END) as divergent_count,
    AVG(latency_ms) as avg_latency_ms,
    MAX(latency_ms) as max_latency_ms
FROM shadow_evaluations
GROUP BY DATE_TRUNC('hour', created_at)
ORDER BY hour DESC;
