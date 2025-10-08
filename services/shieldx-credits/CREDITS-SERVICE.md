 
-----

# 🛡️ ShieldX Credits Service

[](https://golang.org)
[](https://opensource.org/licenses/Apache-2.0)
[](https://www.docker.com/)

**ShieldX Credits Service** manages resource consumption and billing for tenants.

## 📋 Table of Contents

  - [🎯 Overview](https://www.google.com/search?q=%23-overview)
      - [Key Features](https://www.google.com/search?q=%23key-features)
      - [Technology Stack](https://www.google.com/search?q=%23technology-stack)
  - [🏗️ Architecture](https://www.google.com/search?q=%23%EF%B8%8F-architecture)
  - [🚀 Quick Start](https://www.google.com/search?q=%23-quick-start)
      - [Prerequisites](https://www.google.com/search?q=%23prerequisites)
      - [Installation](https://www.google.com/search?q=%23installation)
  - [📡 API Reference](https://www.google.com/search?q=%23-api-reference)
      - [Check Balance](https://www.google.com/search?q=%23check-balance)
      - [Consume Credits](https://www.google.com/search?q=%23consume-credits)
      - [Add Credits](https://www.google.com/search?q=%23add-credits)
      - [Transaction History](https://www.google.com/search?q=%23transaction-history)
      - [Usage Statistics](https://www.google.com/search?q=%23usage-statistics)
  - [🗄️ Database Schema](https://www.google.com/search?q=%23%EF%B8%8F-database-schema)
  - [💻 Development](https://www.google.com/search?q=%23-development)
      - [Project Structure](https://www.google.com/search?q=%23project-structure)
      - [Credit Pricing Model](https://www.google.com/search?q=%23credit-pricing-model)
  - [🧪 Testing](https://www.google.com/search?q=%23-testing)
  - [📊 Monitoring](https://www.google.com/search?q=%23-monitoring)
  - [🔧 Troubleshooting](https://www.google.com/search?q=%23-troubleshooting)
  - [🤝 Contributing](https://www.google.com/search?q=%23-contributing)
  - [📄 License](https://www.google.com/search?q=%23-license)

-----

## 🎯 Overview

### Key Features

  - **Credit Balance Management**: Tracks the credit balance of tenants.
  - **Usage Metering**: Measures the consumption of resources.
  - **Transaction History**: Provides a detailed history of all transactions.
  - **Billing Integration**: Integrates with payment systems.
  - **Quota Enforcement**: Controls usage limits.
  - **Real-time Monitoring**: Monitors usage in real-time.

### Technology Stack

| Component | Technology |
| :--- | :--- |
| **Language** | Go 1.25+ |
| **Framework** | Gin Web Framework |
| **Database** | PostgreSQL 15+ |
| **Cache** | Redis 7+ |
| **Metrics** | Prometheus |

-----

## 🏗️ Architecture

```plaintext
┌─────────────────────────────────────┐
│      ShieldX Credits Service        │
│           (Port 5004)             │
├─────────────────────────────────────┤
│          HTTP API Layer             │
│ - Balance queries                   │
│ - Consumption tracking              │
│ - Transaction history               │
├─────────────────────────────────────┤
│       Business Logic Layer          │
│ - Credit Service                    │
│ - Transaction Service               │
│ - Billing Service                   │
├─────────────────────────────────────┤
│        Data Access Layer            │
│ - PostgreSQL Repository             │
│ - Redis Cache                       │
└─────────────────────────────────────┘
       │                      │
┌──────▼──────┐        ┌──────▼──────┐
│  PostgreSQL │        │    Redis    │
└─────────────┘        └─────────────┘
```

-----

## 🚀 Quick Start

### Prerequisites

  - Go 1.25+
  - PostgreSQL 15+
  - Redis 7+
  - Docker & Docker Compose

### Installation

```bash
# Clone the repository
git clone https://github.com/shieldx-bot/shieldx.git
cd shieldx/services/shieldx-credits

# Install dependencies
go mod download

# Setup PostgreSQL
docker run -d \
  --name shieldx-postgres \
  -e POSTGRES_USER=credits_user \
  -e POSTGRES_PASSWORD=credits_pass \
  -e POSTGRES_DB=shieldx_credits \
  -p 5432:5432 \
  postgres:15-alpine

# Setup Redis
docker run -d \
  --name shieldx-redis \
  -p 6379:6379 \
  redis:7-alpine

# Configure environment
export CREDITS_PORT=5004
export CREDITS_DB_HOST=localhost
export CREDITS_DB_USER=credits_user
export CREDITS_DB_PASSWORD=credits_pass
export CREDITS_DB_NAME=shieldx_credits
export CREDITS_REDIS_HOST=localhost

# Run migrations
migrate -path ./migrations \
  -database "postgresql://credits_user:credits_pass@localhost:5432/shieldx_credits?sslmode=disable" \
  up

# Build & Run
go build -o bin/shieldx-credits cmd/server/main.go
./bin/shieldx-credits

# Verify
curl http://localhost:5004/health
```

-----

## 📡 API Reference

**Base URL:** `http://localhost:5004/api/v1`

### Check Balance

`GET /credits/balance/:tenant_id`

\<details\>\<summary\>View Response Example\</summary\>

```json
{
  "tenant_id": "tenant-123",
  "balance": 50000,
  "currency": "credits",
  "last_updated": "2025-10-08T16:30:00Z"
}
```

\</details\>

### Consume Credits

`POST /credits/consume`

\<details\>\<summary\>View Request Example\</summary\>

```json
{
  "tenant_id": "tenant-123",
  "amount": 100,
  "description": "API call - WAF inspection",
  "metadata": {
    "service": "waf",
    "request_id": "req-789"
  }
}
```

\</details\>
\<details\>\<summary\>View Response Example (200 OK)\</summary\>

```json
{
  "transaction_id": "txn-456",
  "tenant_id": "tenant-123",
  "amount": 100,
  "balance_before": 50000,
  "balance_after": 49900,
  "timestamp": "2025-10-08T16:31:00Z"
}
```

\</details\>

### Add Credits

`POST /credits/add`

\<details\>\<summary\>View Request Example\</summary\>

```json
{
  "tenant_id": "tenant-123",
  "amount": 10000,
  "description": "Monthly subscription top-up",
  "payment_id": "pay-xyz-abc"
}
```

\</details\>

### Transaction History

`GET /credits/history/:tenant_id?limit=100`

\<details\>\<summary\>View Response Example\</summary\>

```json
{
  "data": [
    {
      "transaction_id": "txn-456",
      "tenant_id": "tenant-123",
      "type": "debit",
      "amount": 100,
      "balance_after": 49900,
      "description": "API call - WAF inspection",
      "timestamp": "2025-10-08T16:31:00Z"
    }
  ],
  "pagination": {
    "total": 1500,
    "limit": 100,
    "offset": 0
  }
}
```

\</details\>

### Usage Statistics

`GET /credits/stats/:tenant_id?period=daily`

\<details\>\<summary\>View Response Example\</summary\>

```json
{
  "tenant_id": "tenant-123",
  "period": "daily",
  "data": [
    {
      "date": "2025-10-08",
      "consumed": 5000,
      "added": 0,
      "balance_end": 49900
    }
  ],
  "total_consumed": 150000,
  "average_daily": 5000
}
```

\</details\>

-----

## 🗄️ Database Schema

### `credit_balances` Table

```sql
CREATE TABLE credit_balances (
    tenant_id UUID PRIMARY KEY,
    balance BIGINT NOT NULL DEFAULT 0,
    currency VARCHAR(20) DEFAULT 'credits',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT balance_non_negative CHECK (balance >= 0)
);
CREATE INDEX idx_credit_balances_updated ON credit_balances(updated_at DESC);
```

### `credit_transactions` Table

```sql
CREATE TABLE credit_transactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    type VARCHAR(20) NOT NULL, -- 'credit' or 'debit'
    amount BIGINT NOT NULL,
    balance_before BIGINT NOT NULL,
    balance_after BIGINT NOT NULL,
    description TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX idx_transactions_tenant ON credit_transactions(tenant_id);
CREATE INDEX idx_transactions_created ON credit_transactions(created_at DESC);
```

### `usage_quotas` Table

```sql
CREATE TABLE usage_quotas (
    tenant_id UUID PRIMARY KEY,
    daily_limit BIGINT,
    monthly_limit BIGINT,
    current_daily_usage BIGINT DEFAULT 0,
    current_monthly_usage BIGINT DEFAULT 0,
    reset_daily_at TIMESTAMP WITH TIME ZONE,
    reset_monthly_at TIMESTAMP WITH TIME ZONE
);
```

-----

## 💻 Development

### Project Structure

```
shieldx-credits/
├── cmd/server/main.go
├── internal/
│   ├── api/handlers/
│   │   ├── credits.go
│   │   └── stats.go
│   ├── service/
│   │   ├── credit_service.go
│   │   └── transaction_service.go
│   ├── repository/
│   │   └── postgres/
│   └── models/
├── migrations/
└── tests/
```

### Credit Pricing Model

```go
// Example pricing model
const (
    CostPerAPICall       = 1   // 1 credit
    CostPerWAFInspection = 2   // 2 credits
    CostPerSandboxRun    = 100 // 100 credits
    CostPerMLAnalysis    = 50  // 50 credits
)
```

-----

## 🧪 Testing

```bash
# Run all tests
go test ./... -v

# Test credit service logic
go test ./internal/service -v

# Run integration tests
go test ./tests/integration -v

# Run benchmarks
go test -bench=. ./tests/benchmark
```

### Example Test

```go
func TestCreditService_Consume(t *testing.T) {
    // Setup mock repository
    mockRepo := new(mocks.CreditRepository)
    mockRepo.On("GetBalance", mock.Anything, "tenant-123").Return(int64(1000), nil)
    mockRepo.On("DeductCredits", mock.Anything, "tenant-123", int64(100), "Test consumption").Return(nil)

    service := NewCreditService(mockRepo)
    
    req := &ConsumeRequest{
        TenantID:    "tenant-123",
        Amount:      100,
        Description: "Test consumption",
    }
    
    err := service.ConsumeCredits(context.Background(), req)
    assert.NoError(t, err)
    mockRepo.AssertExpectations(t)
}
```

-----

## 📊 Monitoring

### Prometheus Metrics

```
shieldx_credits_balance{tenant_id}          # Current balance per tenant
shieldx_credits_consumed_total{tenant_id}   # Total consumed credits per tenant
shieldx_credits_added_total{tenant_id}      # Total added credits per tenant
shieldx_credits_transactions_total          # Total number of transactions
```

-----

## 🔧 Troubleshooting

### Insufficient Credits Error

```bash
# Check the current balance for a tenant
curl -H "Authorization: Bearer <token>" http://localhost:5004/api/v1/credits/balance/tenant-123

# Add credits if needed
curl -X POST -H "Authorization: Bearer <token>" http://localhost:5004/api/v1/credits/add \
  -d '{"tenant_id":"tenant-123", "amount":10000, "description": "Manual top-up"}'
```

### Transaction Failed

```bash
# Check the service logs for database errors or other issues
journalctl -u shieldx-credits -f

# Check the transaction history for the tenant
curl -H "Authorization: Bearer <token>" http://localhost:5004/api/v1/credits/history/tenant-123

# Directly query the database to verify the last transactions
psql -d shieldx_credits -c "SELECT * FROM credit_transactions WHERE tenant_id='tenant-123' ORDER BY created_at DESC LIMIT 5;"
```

-----

## 🤝 Contributing

See `CONTRIBUTING.md` for details.

-----

## 📄 License

This project is licensed under the [Apache License 2.0](https://github.com/shieldx-bot/shieldx/blob/main/LICENSE).