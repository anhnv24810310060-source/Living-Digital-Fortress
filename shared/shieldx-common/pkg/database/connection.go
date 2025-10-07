package database

import (
	"context"
	"database/sql"
	"fmt"
	"sync"
	"time"

	_ "github.com/lib/pq"
)

// DBConfig configuration for database connection
type DBConfig struct {
	Host     string
	Port     int
	User     string
	Password string
	DBName   string
	SSLMode  string

	// Connection pool settings
	MaxOpenConns    int
	MaxIdleConns    int
	ConnMaxLifetime time.Duration
	ConnMaxIdleTime time.Duration

	// Timeouts
	ConnectTimeout   time.Duration
	StatementTimeout time.Duration

	// Read replica
	ReplicaHosts []string
}

// Database represents a database connection with primary and replicas
type Database struct {
	Primary  *sql.DB
	Replicas []*sql.DB
	config   DBConfig
	mu       sync.RWMutex
	metrics  *DBMetrics
	rrIndex  int // Round-robin index for replicas
}

// DBMetrics tracks database metrics
type DBMetrics struct {
	mu               sync.RWMutex
	PrimaryQueries   int64
	ReplicaQueries   int64
	Errors           int64
	SlowQueries      int64
	ConnectionErrors int64
	TotalLatencyMs   int64
}

// NewDatabase creates a new database connection with replicas
func NewDatabase(config DBConfig) (*Database, error) {
	// Set defaults
	if config.MaxOpenConns == 0 {
		config.MaxOpenConns = 25
	}
	if config.MaxIdleConns == 0 {
		config.MaxIdleConns = 5
	}
	if config.ConnMaxLifetime == 0 {
		config.ConnMaxLifetime = 5 * time.Minute
	}
	if config.ConnMaxIdleTime == 0 {
		config.ConnMaxIdleTime = 1 * time.Minute
	}
	if config.ConnectTimeout == 0 {
		config.ConnectTimeout = 10 * time.Second
	}
	if config.StatementTimeout == 0 {
		config.StatementTimeout = 30 * time.Second
	}
	if config.SSLMode == "" {
		config.SSLMode = "require"
	}

	db := &Database{
		config:  config,
		metrics: &DBMetrics{},
	}

	// Connect to primary
	primary, err := db.connectToDB(config.Host, config.Port, config.DBName, config.User, config.Password)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to primary database: %w", err)
	}
	db.Primary = primary

	// Connect to replicas
	for _, replicaHost := range config.ReplicaHosts {
		replica, err := db.connectToDB(replicaHost, config.Port, config.DBName, config.User, config.Password)
		if err != nil {
			// Log warning but continue
			fmt.Printf("Warning: Failed to connect to replica %s: %v\n", replicaHost, err)
			continue
		}
		db.Replicas = append(db.Replicas, replica)
	}

	return db, nil
}

// connectToDB creates a database connection
func (db *Database) connectToDB(host string, port int, dbName, user, password string) (*sql.DB, error) {
	// Build DSN
	dsn := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=%s connect_timeout=%d statement_timeout=%d",
		host, port, user, password, dbName, db.config.SSLMode,
		int(db.config.ConnectTimeout.Seconds()),
		int(db.config.StatementTimeout.Milliseconds()),
	)

	conn, err := sql.Open("postgres", dsn)
	if err != nil {
		return nil, fmt.Errorf("failed to open database connection: %w", err)
	}

	// Set connection pool parameters
	conn.SetMaxOpenConns(db.config.MaxOpenConns)
	conn.SetMaxIdleConns(db.config.MaxIdleConns)
	conn.SetConnMaxLifetime(db.config.ConnMaxLifetime)
	conn.SetConnMaxIdleTime(db.config.ConnMaxIdleTime)

	// Test connection
	ctx, cancel := context.WithTimeout(context.Background(), db.config.ConnectTimeout)
	defer cancel()

	if err := conn.PingContext(ctx); err != nil {
		conn.Close()
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	return conn, nil
}

// GetPrimary returns the primary database connection
func (db *Database) GetPrimary() *sql.DB {
	return db.Primary
}

// GetReplica returns a replica connection using round-robin
func (db *Database) GetReplica() *sql.DB {
	db.mu.Lock()
	defer db.mu.Unlock()

	if len(db.Replicas) == 0 {
		// Fall back to primary if no replicas
		return db.Primary
	}

	replica := db.Replicas[db.rrIndex]
	db.rrIndex = (db.rrIndex + 1) % len(db.Replicas)

	return replica
}

// Query executes a read query on a replica (or primary if no replicas)
func (db *Database) Query(ctx context.Context, query string, args ...interface{}) (*sql.Rows, error) {
	start := time.Now()
	defer func() {
		latency := time.Since(start).Milliseconds()
		db.metrics.mu.Lock()
		db.metrics.TotalLatencyMs += latency
		if latency > 1000 {
			db.metrics.SlowQueries++
		}
		db.metrics.mu.Unlock()
	}()

	conn := db.GetReplica()

	db.metrics.mu.Lock()
	db.metrics.ReplicaQueries++
	db.metrics.mu.Unlock()

	rows, err := conn.QueryContext(ctx, query, args...)
	if err != nil {
		db.metrics.mu.Lock()
		db.metrics.Errors++
		db.metrics.mu.Unlock()
		return nil, err
	}

	return rows, nil
}

// QueryRow executes a read query for a single row on a replica
func (db *Database) QueryRow(ctx context.Context, query string, args ...interface{}) *sql.Row {
	conn := db.GetReplica()

	db.metrics.mu.Lock()
	db.metrics.ReplicaQueries++
	db.metrics.mu.Unlock()

	return conn.QueryRowContext(ctx, query, args...)
}

// Exec executes a write query on the primary database
func (db *Database) Exec(ctx context.Context, query string, args ...interface{}) (sql.Result, error) {
	start := time.Now()
	defer func() {
		latency := time.Since(start).Milliseconds()
		db.metrics.mu.Lock()
		db.metrics.TotalLatencyMs += latency
		if latency > 1000 {
			db.metrics.SlowQueries++
		}
		db.metrics.mu.Unlock()
	}()

	db.metrics.mu.Lock()
	db.metrics.PrimaryQueries++
	db.metrics.mu.Unlock()

	result, err := db.Primary.ExecContext(ctx, query, args...)
	if err != nil {
		db.metrics.mu.Lock()
		db.metrics.Errors++
		db.metrics.mu.Unlock()
		return nil, err
	}

	return result, nil
}

// Begin starts a transaction on the primary database
func (db *Database) Begin(ctx context.Context) (*sql.Tx, error) {
	return db.Primary.BeginTx(ctx, nil)
}

// BeginTx starts a transaction with options on the primary database
func (db *Database) BeginTx(ctx context.Context, opts *sql.TxOptions) (*sql.Tx, error) {
	return db.Primary.BeginTx(ctx, opts)
}

// GetStats returns database statistics
func (db *Database) GetStats() map[string]interface{} {
	primaryStats := db.Primary.Stats()

	replicaStats := make([]sql.DBStats, len(db.Replicas))
	for i, replica := range db.Replicas {
		replicaStats[i] = replica.Stats()
	}

	db.metrics.mu.RLock()
	defer db.metrics.mu.RUnlock()

	return map[string]interface{}{
		"primary": map[string]interface{}{
			"open_connections":    primaryStats.OpenConnections,
			"in_use":              primaryStats.InUse,
			"idle":                primaryStats.Idle,
			"wait_count":          primaryStats.WaitCount,
			"wait_duration_ms":    primaryStats.WaitDuration.Milliseconds(),
			"max_idle_closed":     primaryStats.MaxIdleClosed,
			"max_lifetime_closed": primaryStats.MaxLifetimeClosed,
		},
		"replicas": replicaStats,
		"metrics": map[string]interface{}{
			"primary_queries":  db.metrics.PrimaryQueries,
			"replica_queries":  db.metrics.ReplicaQueries,
			"errors":           db.metrics.Errors,
			"slow_queries":     db.metrics.SlowQueries,
			"total_latency_ms": db.metrics.TotalLatencyMs,
		},
	}
}

// Ping checks connectivity to all databases
func (db *Database) Ping(ctx context.Context) error {
	if err := db.Primary.PingContext(ctx); err != nil {
		return fmt.Errorf("primary database ping failed: %w", err)
	}

	for i, replica := range db.Replicas {
		if err := replica.PingContext(ctx); err != nil {
			return fmt.Errorf("replica %d ping failed: %w", i, err)
		}
	}

	return nil
}

// Close closes all database connections
func (db *Database) Close() error {
	var errs []error

	if err := db.Primary.Close(); err != nil {
		errs = append(errs, fmt.Errorf("failed to close primary: %w", err))
	}

	for i, replica := range db.Replicas {
		if err := replica.Close(); err != nil {
			errs = append(errs, fmt.Errorf("failed to close replica %d: %w", i, err))
		}
	}

	if len(errs) > 0 {
		return fmt.Errorf("errors closing database connections: %v", errs)
	}

	return nil
}

// WithTransaction executes a function within a transaction
func (db *Database) WithTransaction(ctx context.Context, fn func(*sql.Tx) error) error {
	tx, err := db.Begin(ctx)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}

	defer func() {
		if p := recover(); p != nil {
			tx.Rollback()
			panic(p)
		}
	}()

	if err := fn(tx); err != nil {
		if rbErr := tx.Rollback(); rbErr != nil {
			return fmt.Errorf("transaction error: %v, rollback error: %w", err, rbErr)
		}
		return err
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit transaction: %w", err)
	}

	return nil
}

// ParseDSN parses a PostgreSQL DSN string
func ParseDSN(dsn string) (DBConfig, error) {
	// Simple DSN parser - for production use a proper parser
	// Format: postgres://user:password@host:port/dbname
	config := DBConfig{
		Port:    5432,
		SSLMode: "require",
	}

	// TODO: Implement full DSN parsing
	// For now, return basic config

	return config, nil
}
