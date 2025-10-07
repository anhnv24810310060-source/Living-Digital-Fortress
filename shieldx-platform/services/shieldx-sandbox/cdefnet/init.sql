-- CDefNet Database Initialization

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
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_iocs_tenant_hash ON iocs(tenant_id_hash);
CREATE INDEX IF NOT EXISTS idx_iocs_value_hash ON iocs(value_hash);
CREATE INDEX IF NOT EXISTS idx_iocs_type ON iocs(ioc_type);
CREATE INDEX IF NOT EXISTS idx_iocs_first_seen ON iocs(first_seen);
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

CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_service ON audit_log(service);
CREATE INDEX IF NOT EXISTS idx_audit_tenant ON audit_log(tenant_hash);

-- Insert sample data for testing
INSERT INTO iocs (tenant_id_hash, ioc_type, value_hash, confidence, ttl) VALUES
('e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'hash', '5d41402abc4b2a76b9719d911017c592', 0.9, 3600),
('e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'domain', 'c4ca4238a0b923820dcc509a6f75849b', 0.8, 7200),
('e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'ip', 'c81e728d9d4c2f636f067f89cc14862c', 0.7, 1800)
ON CONFLICT (tenant_id_hash, value_hash, ioc_type) DO NOTHING;