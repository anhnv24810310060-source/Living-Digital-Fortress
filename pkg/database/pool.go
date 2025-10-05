package database

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"sync"
	"time"

	_ "github.com/lib/pq"
)

// PoolConfig defines advanced connection pool configuration
type PoolConfig struct {
	DSN                 string
	MaxOpenConns        int
	MaxIdleConns        int
	ConnMaxLifetime     time.Duration
	ConnMaxIdleTime     time.Duration
	HealthCheckInterval time.Duration
	SlowQueryThreshold  time.Duration
	EnableQueryLogging  bool
}

// Pool wraps sql.DB with advanced monitoring and health checks
type Pool struct {
	db              *sql.DB
	config          PoolConfig
	metrics         *PoolMetrics
	slowQueryLogger *SlowQueryLogger
	healthChecker   *PoolHealthChecker
	mu              sync.RWMutex
	lastHealthCheck time.Time
	healthy         bool
}

// PoolMetrics tracks connection pool statistics
type PoolMetrics struct {
	mu                sync.RWMutex
	TotalQueries      int64
	SlowQueries       int64
	FailedQueries     int64
	TotalLatency      time.Duration
	ActiveConnections int
	IdleConnections   int
}

// SlowQueryLogger tracks slow queries for optimization
type SlowQueryLogger struct {
	threshold time.Duration
	queries   []SlowQuery
	mu        sync.Mutex
	maxSize   int
}

type SlowQuery struct {
	Query     string
	Duration  time.Duration
	Timestamp time.Time
}

// PoolHealthChecker monitors database health at the connection pool level.
// Renamed from HealthChecker to avoid collision with cluster-level checker in health.go
type PoolHealthChecker struct {
	pool     *Pool
	interval time.Duration
	stopCh   chan struct{}
}

// NewPool creates optimized database connection pool
func NewPool(config PoolConfig) (*Pool, error) {
	// Set sensible defaults
	if config.MaxOpenConns == 0 {
		config.MaxOpenConns = 100
	}
	if config.MaxIdleConns == 0 {
		config.MaxIdleConns = 25
	}
	if config.ConnMaxLifetime == 0 {
		config.ConnMaxLifetime = 30 * time.Minute
	}
	if config.ConnMaxIdleTime == 0 {
		config.ConnMaxIdleTime = 10 * time.Minute
	}
	if config.HealthCheckInterval == 0 {
		config.HealthCheckInterval = 30 * time.Second
	}
	if config.SlowQueryThreshold == 0 {
		config.SlowQueryThreshold = 100 * time.Millisecond
	}

	// Open database connection
	db, err := sql.Open("postgres", config.DSN)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	// Configure connection pool
	db.SetMaxOpenConns(config.MaxOpenConns)
	db.SetMaxIdleConns(config.MaxIdleConns)
	db.SetConnMaxLifetime(config.ConnMaxLifetime)
	db.SetConnMaxIdleTime(config.ConnMaxIdleTime)

	// Initial health check
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := db.PingContext(ctx); err != nil {
		db.Close()
		return nil, fmt.Errorf("database ping failed: %w", err)
	}

	pool := &Pool{
		db:      db,
		config:  config,
		metrics: &PoolMetrics{},
		slowQueryLogger: &SlowQueryLogger{
			threshold: config.SlowQueryThreshold,
			maxSize:   100,
			queries:   make([]SlowQuery, 0),
		},
		healthy: true,
	}

	// Start health checker
	pool.healthChecker = &PoolHealthChecker{pool: pool, interval: config.HealthCheckInterval, stopCh: make(chan struct{})}
	go pool.healthChecker.start()

	log.Printf("[db] Connection pool initialized: max_open=%d, max_idle=%d, lifetime=%v",
		config.MaxOpenConns, config.MaxIdleConns, config.ConnMaxLifetime)

	return pool, nil
}

// QueryContext executes query with monitoring
func (p *Pool) QueryContext(ctx context.Context, query string, args ...interface{}) (*sql.Rows, error) {
	start := time.Now()

	rows, err := p.db.QueryContext(ctx, query, args...)

	duration := time.Since(start)
	p.recordQuery(query, duration, err)

	return rows, err
}

// QueryRowContext executes single-row query with monitoring
func (p *Pool) QueryRowContext(ctx context.Context, query string, args ...interface{}) *sql.Row {
	start := time.Now()

	row := p.db.QueryRowContext(ctx, query, args...)

	duration := time.Since(start)
	p.recordQuery(query, duration, nil)

	return row
}

// ExecContext executes statement with monitoring
func (p *Pool) ExecContext(ctx context.Context, query string, args ...interface{}) (sql.Result, error) {
	start := time.Now()

	result, err := p.db.ExecContext(ctx, query, args...)

	duration := time.Since(start)
	p.recordQuery(query, duration, err)

	return result, err
}

// BeginTx starts transaction
func (p *Pool) BeginTx(ctx context.Context, opts *sql.TxOptions) (*sql.Tx, error) {
	return p.db.BeginTx(ctx, opts)
}

// recordQuery tracks query metrics
func (p *Pool) recordQuery(query string, duration time.Duration, err error) {
	p.metrics.mu.Lock()
	p.metrics.TotalQueries++
	p.metrics.TotalLatency += duration

	if err != nil {
		p.metrics.FailedQueries++
	}

	if duration > p.config.SlowQueryThreshold {
		p.metrics.SlowQueries++
		p.slowQueryLogger.record(query, duration)
	}
	p.metrics.mu.Unlock()

	if p.config.EnableQueryLogging && duration > p.config.SlowQueryThreshold {
		log.Printf("[db] SLOW QUERY (%v): %s", duration, truncateQuery(query, 100))
	}
}

// GetStats returns current pool statistics
func (p *Pool) GetStats() sql.DBStats {
	stats := p.db.Stats()

	p.metrics.mu.Lock()
	p.metrics.ActiveConnections = stats.InUse
	p.metrics.IdleConnections = stats.Idle
	p.metrics.mu.Unlock()

	return stats
}

// GetMetrics returns custom metrics
func (p *Pool) GetMetrics() map[string]interface{} {
	p.metrics.mu.RLock()
	defer p.metrics.mu.RUnlock()

	stats := p.GetStats()
	avgLatency := time.Duration(0)
	if p.metrics.TotalQueries > 0 {
		avgLatency = p.metrics.TotalLatency / time.Duration(p.metrics.TotalQueries)
	}

	return map[string]interface{}{
		"total_queries":      p.metrics.TotalQueries,
		"slow_queries":       p.metrics.SlowQueries,
		"failed_queries":     p.metrics.FailedQueries,
		"avg_latency_ms":     avgLatency.Milliseconds(),
		"active_connections": stats.InUse,
		"idle_connections":   stats.Idle,
		"max_open_conns":     stats.MaxOpenConnections,
		"wait_count":         stats.WaitCount,
		"wait_duration_ms":   stats.WaitDuration.Milliseconds(),
		"healthy":            p.healthy,
	}
}

// GetSlowQueries returns recent slow queries
func (p *Pool) GetSlowQueries() []SlowQuery {
	return p.slowQueryLogger.getRecent(20)
}

// IsHealthy returns current health status
func (p *Pool) IsHealthy() bool {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.healthy
}

// Close closes database connection pool
func (p *Pool) Close() error {
	if p.healthChecker != nil {
		close(p.healthChecker.stopCh)
	}
	return p.db.Close()
}

// record adds slow query to log
func (sql *SlowQueryLogger) record(query string, duration time.Duration) {
	sql.mu.Lock()
	defer sql.mu.Unlock()

	sq := SlowQuery{
		Query:     truncateQuery(query, 200),
		Duration:  duration,
		Timestamp: time.Now(),
	}

	sql.queries = append(sql.queries, sq)

	// Keep only recent queries
	if len(sql.queries) > sql.maxSize {
		sql.queries = sql.queries[len(sql.queries)-sql.maxSize:]
	}
}

// getRecent returns recent slow queries
func (sql *SlowQueryLogger) getRecent(limit int) []SlowQuery {
	sql.mu.Lock()
	defer sql.mu.Unlock()

	if len(sql.queries) < limit {
		limit = len(sql.queries)
	}

	result := make([]SlowQuery, limit)
	copy(result, sql.queries[len(sql.queries)-limit:])

	return result
}

// start begins health check routine
func (hc *PoolHealthChecker) start() {
	ticker := time.NewTicker(hc.interval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			hc.check()
		case <-hc.stopCh:
			return
		}
	}
}

// check performs health check
func (hc *PoolHealthChecker) check() {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	start := time.Now()
	err := hc.pool.db.PingContext(ctx)
	duration := time.Since(start)

	hc.pool.mu.Lock()
	hc.pool.lastHealthCheck = time.Now()
	hc.pool.healthy = (err == nil)
	hc.pool.mu.Unlock()

	if err != nil {
		log.Printf("[db] HEALTH CHECK FAILED (%v): %v", duration, err)
	} else if duration > 1*time.Second {
		log.Printf("[db] HEALTH CHECK SLOW (%v)", duration)
	}
}

// truncateQuery truncates query string for logging
func truncateQuery(query string, maxLen int) string {
	// Remove extra whitespace
	cleaned := ""
	lastSpace := false
	for _, ch := range query {
		if ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' {
			if !lastSpace {
				cleaned += " "
				lastSpace = true
			}
		} else {
			cleaned += string(ch)
			lastSpace = false
		}
	}

	if len(cleaned) > maxLen {
		return cleaned[:maxLen] + "..."
	}
	return cleaned
}

// OptimizePool tunes connection pool based on workload
func (p *Pool) OptimizePool() {
	stats := p.GetStats()

	// If wait count is high, consider increasing max open connections (cap at 200)
	if stats.WaitCount > 100 && stats.MaxOpenConnections < 200 {
		newMax := stats.MaxOpenConnections + 10
		p.db.SetMaxOpenConns(newMax)
		log.Printf("[db] Increased max open connections to %d (wait_count=%d)", newMax, stats.WaitCount)
	}

	// If idle connections are high relative to in-use, reduce idle pool slightly
	if stats.Idle > stats.InUse*2 && stats.Idle > 20 { // heuristic
		newIdle := stats.Idle - 5
		if newIdle > 10 {
			p.db.SetMaxIdleConns(newIdle)
			log.Printf("[db] Adjusted max idle connections to %d", newIdle)
		}
	}
}
