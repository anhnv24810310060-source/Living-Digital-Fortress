# Database Layer - Production Documentation

## Overview
Production-grade database layer với connection pooling, migrations, automated backups, và read replica routing cho Living Digital Fortress.

## Architecture

### Components
1. **Connection Management** (`pkg/database/connection.go`)
   - Primary/replica connection pooling
   - Round-robin load balancing cho read queries
   - Configurable pool settings
   - Automatic connection lifecycle management

2. **Health Monitoring** (`pkg/database/health.go`)
   - Periodic health checks (primary + replicas)
   - Replication lag detection
   - Auto-reconnection cho unhealthy replicas
   - Connection pool metrics

3. **Migration System** (`pkg/database/migration.go`)
   - golang-migrate integration
   - Versioned migrations (up/down)
   - Advisory locks để prevent concurrent migrations
   - Migration history tracking

4. **Backup/Restore** (`pkg/database/backup.go`)
   - Automated backups with pg_dump
   - Parallel backup/restore jobs
   - Compression support
   - Retention policy management
   - Scheduled backups

## Features

### 1. Connection Pooling
```go
config := database.DBConfig{
    Host:     "localhost",
    Port:     5432,
    User:     "postgres",
    Password: "secret",
    DBName:   "fortress",
    
    // Pool settings
    MaxOpenConns:    25,
    MaxIdleConns:    5,
    ConnMaxLifetime: 5 * time.Minute,
    ConnMaxIdleTime: 1 * time.Minute,
    
    // Read replicas
    ReplicaHosts: []string{
        "replica1.example.com",
        "replica2.example.com",
    },
}

db, err := database.NewDatabase(config)
```

**Metrics tracked:**
- Open connections
- Idle connections
- Wait count/duration
- Query latency
- Slow queries (>1s)
- Error count

### 2. Read/Write Splitting
```go
// Write queries → Primary
result, err := db.Exec(ctx, "INSERT INTO users VALUES ($1, $2)", id, name)

// Read queries → Replicas (round-robin)
rows, err := db.Query(ctx, "SELECT * FROM users WHERE active = true")

// Transactions → Primary only
err := db.WithTransaction(ctx, func(tx *sql.Tx) error {
    // Multiple operations in transaction
    return nil
})
```

### 3. Health Checks
```go
// Create health checker
hc := database.NewHealthChecker(db, 30*time.Second, 5*time.Second)

// Start periodic checks
go hc.Start()

// Check current status
status := hc.Check()
if !status.PrimaryHealthy {
    log.Error("Primary database is down!")
}

// Check replication lag
lags, err := hc.CheckReplicationLag(ctx)
for replicaID, lag := range lags {
    if lag > 30*time.Second {
        log.Warn("Replica %d is lagging by %v", replicaID, lag)
    }
}

// Auto-reconnect unhealthy replicas
err := hc.AutoReconnect(ctx)
```

### 4. Migration System
```go
//go:embed migrations/credits/*.sql
var creditsMigrations embed.FS

// Create migration manager
mm, err := database.NewMigrationManager(db, creditsMigrations, "migrations/credits")

// Run all pending migrations
err = mm.Up()

// Rollback last migration
err = mm.Down()

// Migrate to specific version
err = mm.Migrate(5)

// Get current version
version, dirty, err := mm.Version()

// Auto-migrate with lock protection
err = database.AutoMigrate(ctx, db, creditsMigrations, "migrations/credits")
```

**Migration structure:**
```
migrations/
├── credits/
│   ├── 000001_init_schema.up.sql
│   ├── 000001_init_schema.down.sql
│   ├── 000002_add_indexes.up.sql
│   └── 000002_add_indexes.down.sql
├── shadow/
│   ├── 000001_init_schema.up.sql
│   └── 000001_init_schema.down.sql
└── cdefnet/
    ├── 000001_init_schema.up.sql
    └── 000001_init_schema.down.sql
```

### 5. Automated Backups
```go
// Configure backup manager
config := database.BackupConfig{
    BackupDir:       "/var/backups/postgres",
    RetentionDays:   7,
    Compression:     true,
    IncludeSchema:   true,
    IncludeData:     true,
    ParallelJobs:    4,
    CustomFormat:    true, // pg_dump custom format
}

bm, err := database.NewBackupManager(db, config)

// Create backup
backupPath, err := bm.Backup(ctx)

// List available backups
backups, err := bm.ListBackups()

// Restore from backup
err = bm.Restore(ctx, backupPath)

// Verify backup integrity
err = bm.VerifyBackup(ctx, backupPath)

// Cleanup old backups
deleted, err := bm.CleanupOldBackups()

// Schedule automated backups
schedule := database.NewBackupSchedule(bm, 24*time.Hour)
go schedule.Start(ctx)
```

**CLI backup tool:**
```bash
# Create backup
./scripts/backup-database.sh backup

# List backups
./scripts/backup-database.sh list

# Restore from backup
./scripts/backup-database.sh restore /var/backups/postgres/fortress_20240115_120000.dump

# Verify backup
./scripts/backup-database.sh verify /var/backups/postgres/fortress_20240115_120000.dump

# Cleanup old backups
./scripts/backup-database.sh cleanup
```

## Migration Files

### Credits Service
- **000001_init_schema.up.sql**: Credits and transactions tables
  - UUID primary keys with `gen_random_uuid()`
  - Balance constraints (non-negative)
  - Transaction types: credit, debit, reserve, release
  - Automatic timestamp updates

### Shadow Service
- **000001_init_schema.up.sql**: Shadow evaluation tracking
  - Evaluation results with divergence detection
  - Metrics collection
  - GIN indexes for JSONB queries
  - Divergence summary view

### CDefNet Service
- **000001_init_schema.up.sql**: IOC and audit log tables
  - IOC storage with TTL
  - Automatic cleanup function
  - Audit logging with JSONB details
  - Tenant isolation via hashing

## Configuration

### Environment Variables
```bash
# Database connection
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_NAME=fortress
PGPASSWORD=your_password

# Connection pool
DB_MAX_OPEN_CONNS=25
DB_MAX_IDLE_CONNS=5
DB_CONN_MAX_LIFETIME=5m
DB_CONN_MAX_IDLE_TIME=1m

# Replicas (comma-separated)
DB_REPLICA_HOSTS=replica1.example.com,replica2.example.com

# Backup settings
BACKUP_DIR=/var/backups/postgres
RETENTION_DAYS=7
COMPRESSION=true
PARALLEL_JOBS=4
```

### Production Settings
```go
config := database.DBConfig{
    // Production-grade pool sizing
    MaxOpenConns:    100,  // Max concurrent connections
    MaxIdleConns:    10,   // Keep warm connections
    ConnMaxLifetime: 1 * time.Hour,
    ConnMaxIdleTime: 5 * time.Minute,
    
    // Aggressive timeouts
    ConnectTimeout:   10 * time.Second,
    StatementTimeout: 30 * time.Second,
    
    // TLS required
    SSLMode: "require",
}
```

## Monitoring

### Database Metrics
```go
// Get comprehensive stats
stats := db.GetStats()

// Returns:
{
    "primary": {
        "open_connections": 15,
        "in_use": 5,
        "idle": 10,
        "wait_count": 100,
        "wait_duration_ms": 150,
        "max_idle_closed": 5,
        "max_lifetime_closed": 2
    },
    "replicas": [...],
    "metrics": {
        "primary_queries": 50000,
        "replica_queries": 150000,
        "errors": 10,
        "slow_queries": 5,
        "total_latency_ms": 250000
    }
}
```

### Health Endpoints
```go
// Prometheus metrics endpoint
http.HandleFunc("/metrics/database", func(w http.ResponseWriter, r *http.Request) {
    stats := db.GetStats()
    json.NewEncoder(w).Encode(stats)
})

// Health check endpoint
http.HandleFunc("/health/database", func(w http.ResponseWriter, r *http.Request) {
    if !hc.IsHealthy() {
        w.WriteHeader(http.StatusServiceUnavailable)
        return
    }
    w.WriteHeader(http.StatusOK)
})
```

## Best Practices

### 1. Connection Management
- ✅ Use connection pooling cho tất cả connections
- ✅ Set appropriate pool limits based on load
- ✅ Monitor connection wait times
- ✅ Use read replicas cho read-heavy queries
- ❌ Không open/close connections manually

### 2. Migrations
- ✅ Always test migrations trên staging trước
- ✅ Use transactions trong migrations
- ✅ Write reversible migrations (down files)
- ✅ Use advisory locks để prevent concurrent migrations
- ❌ Không modify deployed migrations

### 3. Backups
- ✅ Schedule daily automated backups
- ✅ Test restore process regularly
- ✅ Store backups off-site
- ✅ Verify backup integrity after creation
- ❌ Không rely solely on replication cho backups

### 4. Queries
- ✅ Use parameterized queries (prevent SQL injection)
- ✅ Set statement timeouts
- ✅ Route reads to replicas
- ✅ Use transactions cho multi-step operations
- ❌ Không execute long-running queries on primary

## Security

### 1. Credentials
- Passwords stored trong environment variables
- Use secrets management (Vault, AWS Secrets Manager)
- Rotate credentials regularly
- Use different credentials cho different services

### 2. SSL/TLS
```go
config.SSLMode = "require"  // Minimum
config.SSLMode = "verify-ca" // Better
config.SSLMode = "verify-full" // Best
```

### 3. Access Control
- Use principle of least privilege
- Separate read-only users cho replicas
- Audit database access logs
- Implement row-level security where needed

## Troubleshooting

### Connection Pool Exhaustion
```bash
# Symptoms: High wait_count, high wait_duration
# Solution: Increase MaxOpenConns or optimize queries
```

### Replication Lag
```bash
# Check lag
lags, _ := hc.CheckReplicationLag(ctx)

# Solutions:
# 1. Reduce write load on primary
# 2. Upgrade replica hardware
# 3. Check network latency
```

### Migration Failures
```bash
# Check dirty state
version, dirty, _ := mm.Version()

# Force to clean state (last resort)
mm.Force(version)

# Or rollback and re-run
mm.Down()
mm.Up()
```

### Backup Issues
```bash
# Verify backup
./scripts/backup-database.sh verify <backup_file>

# Check backup logs
journalctl -u backup-database.service

# Test restore on non-production
DB_NAME=test_restore ./scripts/backup-database.sh restore <backup_file>
```

## Performance Optimization

### 1. Indexes
- Add indexes cho frequently queried columns
- Use partial indexes cho filtered queries
- Monitor index usage with `pg_stat_user_indexes`

### 2. Query Optimization
- Use EXPLAIN ANALYZE để identify slow queries
- Add appropriate indexes
- Consider materialized views cho complex aggregations

### 3. Connection Pooling
- Tune MaxOpenConns based on workload
- Monitor wait times
- Use PgBouncer for connection pooling at scale

## Files Created
```
pkg/database/
├── connection.go      (350 lines) - Connection pooling & replica routing
├── health.go          (200 lines) - Health checks & monitoring
├── migration.go       (250 lines) - Migration management
└── backup.go          (350 lines) - Backup/restore automation

migrations/
├── credits/
│   ├── 000001_init_schema.up.sql    (50 lines)
│   └── 000001_init_schema.down.sql  (20 lines)
├── shadow/
│   ├── 000001_init_schema.up.sql    (40 lines)
│   └── 000001_init_schema.down.sql  (15 lines)
└── cdefnet/
    ├── 000001_init_schema.up.sql    (70 lines)
    └── 000001_init_schema.down.sql  (25 lines)

scripts/
└── backup-database.sh  (250 lines) - CLI backup tool
```

**Total**: ~1,620 lines of production database infrastructure

---
**Phase 3 Complete** ✅ Database Layer modernization với enterprise-grade features
