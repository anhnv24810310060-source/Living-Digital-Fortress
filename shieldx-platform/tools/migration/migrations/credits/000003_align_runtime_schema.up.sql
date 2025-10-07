-- Align credits DB schema with runtime tables used by services/credits
-- Safe-guarded with IF NOT EXISTS and checks to avoid destructive changes.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- accounts table used at runtime
CREATE TABLE IF NOT EXISTS credit_accounts (
  tenant_id VARCHAR(255) PRIMARY KEY,
  balance BIGINT NOT NULL DEFAULT 0,
  reserved_funds BIGINT NOT NULL DEFAULT 0,
  total_spent BIGINT NOT NULL DEFAULT 0,
  total_purchased BIGINT NOT NULL DEFAULT 0,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CONSTRAINT positive_balance CHECK (balance >= 0),
  CONSTRAINT positive_reserved CHECK (reserved_funds >= 0)
);

-- transactions history
CREATE TABLE IF NOT EXISTS credit_transactions (
  transaction_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id VARCHAR(255) NOT NULL,
  type VARCHAR(50) NOT NULL,
  amount BIGINT NOT NULL,
  description TEXT,
  reference VARCHAR(255),
  status VARCHAR(50) NOT NULL DEFAULT 'pending',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  processed_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_credit_tx_tenant ON credit_transactions(tenant_id);
CREATE INDEX IF NOT EXISTS idx_credit_tx_type ON credit_transactions(type);
CREATE INDEX IF NOT EXISTS idx_credit_tx_created ON credit_transactions(created_at);

-- idempotency store
CREATE TABLE IF NOT EXISTS idempotency_keys (
  key VARCHAR(255) PRIMARY KEY,
  tenant_id VARCHAR(255) NOT NULL,
  transaction_id UUID NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  expires_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_idem_expires ON idempotency_keys(expires_at);

-- reservations
CREATE TABLE IF NOT EXISTS credit_reservations (
  reservation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id VARCHAR(255) NOT NULL,
  amount BIGINT NOT NULL,
  status VARCHAR(20) NOT NULL DEFAULT 'active',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  expires_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_resv_tenant_status ON credit_reservations(tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_resv_expires ON credit_reservations(expires_at);

-- immutable audit logs
CREATE TABLE IF NOT EXISTS audit_logs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id VARCHAR(255) NOT NULL,
  transaction_id UUID NOT NULL,
  action VARCHAR(50) NOT NULL,
  amount BIGINT NOT NULL,
  prev_hash VARCHAR(128) NOT NULL,
  hash VARCHAR(128) NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_audit_tenant ON audit_logs(tenant_id, created_at DESC);

DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'audit_logs_immutable') THEN
    CREATE OR REPLACE FUNCTION audit_logs_no_update_delete() RETURNS trigger AS $$
    BEGIN
      RAISE EXCEPTION 'audit_logs are immutable';
      RETURN NULL;
    END;$$ LANGUAGE plpgsql;

    CREATE TRIGGER audit_logs_immutable
    BEFORE UPDATE OR DELETE ON audit_logs
    FOR EACH ROW EXECUTE FUNCTION audit_logs_no_update_delete();
  END IF;
END $$;
