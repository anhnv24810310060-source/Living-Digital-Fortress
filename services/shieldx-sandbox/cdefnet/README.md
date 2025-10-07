# CDefNet - Collective Defense Network

## Overview

CDefNet enables secure, privacy-preserving threat intelligence sharing between tenants in the ShieldX ecosystem.

## Features

- **Privacy-First**: PII scrubbing, differential privacy, k-anonymity
- **Secure Storage**: PostgreSQL with hashed IOCs and tenant isolation
- **Rate Limiting**: Per-IP request limiting
- **Authentication**: Bearer token authentication
- **Audit Logging**: Complete audit trail for compliance

## API Endpoints

### Submit IOC
```
POST /v1/submit-ioc
Authorization: Bearer <token>
Content-Type: application/json

{
  "tenant_id": "demo",
  "ioc_type": "hash",
  "value": "5d41402abc4b2a76b9719d911017c592",
  "confidence": 0.8,
  "ttl": 3600
}
```

### Query IOC
```
GET /v1/query-ioc?type=hash&value=5d41402abc4b2a76b9719d911017c592
```

### Health Check
```
GET /health
```

## Environment Variables

- `CDEFNET_PORT`: Server port (default: 8090)
- `DATABASE_URL`: PostgreSQL connection string

## Database Schema

```sql
CREATE TABLE iocs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id_hash VARCHAR(64) NOT NULL,
    ioc_type VARCHAR(50) NOT NULL,
    value_hash VARCHAR(64) NOT NULL,
    confidence DECIMAL(3,2) CHECK (confidence >= 0 AND confidence <= 1),
    ttl INTEGER NOT NULL,
    first_seen TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_seen TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    aggregated_count INTEGER DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

## Privacy Guarantees

1. **PII Scrubbing**: All IOC values are scrubbed of PII before hashing
2. **Tenant Isolation**: Tenant IDs are hashed with salt
3. **Differential Privacy**: Query results include calibrated noise
4. **K-Anonymity**: Aggregation ensures minimum group sizes

## Testing

```bash
cd services/cdefnet/privacy
go test -v
```

## Deployment

```bash
# Set environment
export DATABASE_URL="postgres://user:pass@localhost/cdefnet?sslmode=disable"
export CDEFNET_PORT="8090"

# Run service
cd services/cdefnet
go run .
```

## Week 1 Acceptance Criteria ✅

- [x] REST endpoint POST /v1/submit-ioc stores hashed IOCs
- [x] PII scrubbing before hashing
- [x] PostgreSQL storage with proper indexing
- [x] Rate limiting and authentication
- [x] Comprehensive tests for anonymizer
- [x] Audit logging
- [x] Health checks and metrics
- [x] Graceful shutdown