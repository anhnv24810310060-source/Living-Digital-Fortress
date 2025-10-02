-- Migration: Create cdefnet initial schema
-- Version: 000001
-- Description: Setup IOCs and audit log tables for CDefNet

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create IOCs table
CREATE TABLE IF NOT EXISTS iocs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id_hash VARCHAR(64) NOT NULL,
    ioc_type VARCHAR(50) NOT NULL,
    value_hash VARCHAR(64) NOT NULL,
    confidence DECIMAL(3,2) CHECK (confidence >= 0 AND confidence <= 1),
    ttl INTEGER NOT NULL,
    first_seen TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_seen TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    aggregated_count INTEGER DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT iocs_ttl_positive CHECK (ttl > 0),
    CONSTRAINT iocs_count_positive CHECK (aggregated_count > 0)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_iocs_tenant_hash ON iocs(tenant_id_hash);
CREATE INDEX IF NOT EXISTS idx_iocs_value_hash ON iocs(value_hash);
CREATE INDEX IF NOT EXISTS idx_iocs_type ON iocs(ioc_type);
CREATE INDEX IF NOT EXISTS idx_iocs_first_seen ON iocs(first_seen);
CREATE INDEX IF NOT EXISTS idx_iocs_last_seen ON iocs(last_seen);
CREATE UNIQUE INDEX IF NOT EXISTS idx_iocs_unique ON iocs(tenant_id_hash, value_hash, ioc_type);

-- Create cleanup function
CREATE OR REPLACE FUNCTION cleanup_expired_iocs() RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM iocs WHERE first_seen + INTERVAL '1 second' * ttl < NOW();
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Create audit log table
CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    service VARCHAR(50) NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    tenant_hash VARCHAR(64),
    details JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for audit log
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_service ON audit_log(service);
CREATE INDEX IF NOT EXISTS idx_audit_tenant ON audit_log(tenant_hash);
CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_log(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_created_at ON audit_log(created_at);
CREATE INDEX IF NOT EXISTS idx_audit_details ON audit_log USING GIN (details);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_iocs_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    NEW.last_seen = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for iocs table
CREATE TRIGGER update_iocs_timestamp BEFORE UPDATE ON iocs
    FOR EACH ROW EXECUTE FUNCTION update_iocs_updated_at();
