-- Performance indexes for high-volume credits history and idempotency lookups
CREATE INDEX IF NOT EXISTS idx_credit_txn_tenant_created_id
  ON credit_transactions(tenant_id, created_at DESC, transaction_id DESC);

CREATE INDEX IF NOT EXISTS idx_idem_tenant_expires
  ON idempotency_keys(tenant_id, expires_at DESC);
