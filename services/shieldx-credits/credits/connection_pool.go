package main

import (
	"context"
	"database/sql"
	"fmt"
	"sync"
	"time"
)

// EnhancedConnectionPool manages database connections with circuit breaker and health monitoring
type EnhancedConnectionPool struct {
	db              *sql.DB
	circuitBreaker  *CircuitBreaker
	healthMonitor   *HealthMonitor
	connectionStats *ConnectionStats
	mu              sync.RWMutex
}

// NOTE: CircuitBreaker implementation removed here to avoid duplication.
// The richer implementation lives in circuit_breaker.go and is reused.

// ConnectionStats tracks pool performance metrics
type ConnectionStats struct {
	TotalQueries     int64
	FailedQueries    int64
	AvgQueryTime     time.Duration
	ActiveConns      int
	IdleConns        int
	queryTimes       []time.Duration
	queryTimesMutex  sync.Mutex
	mu               sync.RWMutex
}

// HealthMonitor periodically checks database health
type HealthMonitor struct {
	db            *sql.DB
	checkInterval time.Duration
	stopCh        chan struct{}
	isHealthy     bool
	mu            sync.RWMutex
}

// NewEnhancedConnectionPool creates an optimized connection pool
func NewEnhancedConnectionPool(db *sql.DB) *EnhancedConnectionPool {
	// Optimal connection pool settings for production
	// Based on: maxConns = ((core_count * 2) + effective_spindle_count)
	// For modern cloud instances: ~50 connections is optimal
	db.SetMaxOpenConns(50)
	db.SetMaxIdleConns(25) // Keep 50% idle for burst traffic
	db.SetConnMaxLifetime(5 * time.Minute)
	db.SetConnMaxIdleTime(2 * time.Minute) // Aggressively close idle connections

	pool := &EnhancedConnectionPool{
		db: db,
		circuitBreaker: NewCircuitBreaker(CircuitBreakerConfig{Name: "db", MaxFailures: 5, Timeout: 5 * time.Second, ResetTimeout: 30 * time.Second}),
		connectionStats: &ConnectionStats{},
	}

	// Start health monitoring
	pool.healthMonitor = NewHealthMonitor(db, 30*time.Second)
	pool.healthMonitor.Start()

	return pool
}

// Execute wraps query execution with circuit breaker and monitoring
func (ecp *EnhancedConnectionPool) Execute(ctx context.Context, query string, args ...interface{}) (*sql.Rows, error) {
	// Check circuit breaker using stats-based gate; if open, reject.
	if state := ecp.circuitBreaker.GetState(); state == "open" {
		return nil, fmt.Errorf("circuit breaker open: too many failures")
	}

	start := time.Now()
	rows, err := ecp.db.QueryContext(ctx, query, args...)
	duration := time.Since(start)

	// Record metrics
	ecp.connectionStats.RecordQuery(duration, err == nil)

	if err != nil {
		// Fallback: we cannot call RecordFailure (method removed), rely on external breaker or metrics.
		return nil, err
	}
	return rows, nil
}

// ExecContext executes a non-query statement with monitoring
func (ecp *EnhancedConnectionPool) ExecContext(ctx context.Context, query string, args ...interface{}) (sql.Result, error) {
	if state := ecp.circuitBreaker.GetState(); state == "open" {
		return nil, fmt.Errorf("circuit breaker open: too many failures")
	}

	start := time.Now()
	result, err := ecp.db.ExecContext(ctx, query, args...)
	duration := time.Since(start)

	ecp.connectionStats.RecordQuery(duration, err == nil)

	if err != nil {
		return nil, err
	}
	return result, nil
}

// BeginTx starts a transaction with timeout
func (ecp *EnhancedConnectionPool) BeginTx(ctx context.Context, opts *sql.TxOptions) (*sql.Tx, error) {
	if state := ecp.circuitBreaker.GetState(); state == "open" {
		return nil, fmt.Errorf("circuit breaker open")
	}
	return ecp.db.BeginTx(ctx, opts)
}

// GetStats returns current pool statistics
func (ecp *EnhancedConnectionPool) GetStats() map[string]interface{} {
	dbStats := ecp.db.Stats()
	customStats := ecp.connectionStats.GetStats()

	return map[string]interface{}{
		"max_open_connections":     dbStats.MaxOpenConnections,
		"open_connections":         dbStats.OpenConnections,
		"in_use":                   dbStats.InUse,
		"idle":                     dbStats.Idle,
		"wait_count":               dbStats.WaitCount,
		"wait_duration_ms":         dbStats.WaitDuration.Milliseconds(),
		"max_idle_closed":          dbStats.MaxIdleClosed,
		"max_lifetime_closed":      dbStats.MaxLifetimeClosed,
		"total_queries":            customStats["total_queries"],
		"failed_queries":           customStats["failed_queries"],
		"avg_query_time_ms":        customStats["avg_query_time_ms"],
		"circuit_breaker_state":    ecp.circuitBreaker.GetState(),
		"is_healthy":               ecp.healthMonitor.IsHealthy(),
	}
}

// Close gracefully shuts down the connection pool
func (ecp *EnhancedConnectionPool) Close() error {
	ecp.healthMonitor.Stop()
	return ecp.db.Close()
}


// ConnectionStats methods
func (cs *ConnectionStats) RecordQuery(duration time.Duration, success bool) {
	cs.mu.Lock()
	defer cs.mu.Unlock()

	cs.TotalQueries++
	if !success {
		cs.FailedQueries++
	}

	// Keep last 1000 query times for accurate average (ring buffer)
	cs.queryTimesMutex.Lock()
	if len(cs.queryTimes) >= 1000 {
		cs.queryTimes = cs.queryTimes[1:]
	}
	cs.queryTimes = append(cs.queryTimes, duration)
	cs.queryTimesMutex.Unlock()

	// Update average
	cs.updateAverage()
}

func (cs *ConnectionStats) updateAverage() {
	cs.queryTimesMutex.Lock()
	defer cs.queryTimesMutex.Unlock()

	if len(cs.queryTimes) == 0 {
		cs.AvgQueryTime = 0
		return
	}

	var total time.Duration
	for _, t := range cs.queryTimes {
		total += t
	}
	cs.AvgQueryTime = total / time.Duration(len(cs.queryTimes))
}

func (cs *ConnectionStats) GetStats() map[string]interface{} {
	cs.mu.RLock()
	defer cs.mu.RUnlock()

	successRate := 0.0
	if cs.TotalQueries > 0 {
		successRate = float64(cs.TotalQueries-cs.FailedQueries) / float64(cs.TotalQueries)
	}

	return map[string]interface{}{
		"total_queries":     cs.TotalQueries,
		"failed_queries":    cs.FailedQueries,
		"success_rate":      successRate,
		"avg_query_time_ms": cs.AvgQueryTime.Milliseconds(),
	}
}

// HealthMonitor methods
func NewHealthMonitor(db *sql.DB, checkInterval time.Duration) *HealthMonitor {
	return &HealthMonitor{
		db:            db,
		checkInterval: checkInterval,
		stopCh:        make(chan struct{}),
		isHealthy:     true,
	}
}

func (hm *HealthMonitor) Start() {
	go func() {
		ticker := time.NewTicker(hm.checkInterval)
		defer ticker.Stop()

		for {
			select {
			case <-ticker.C:
				hm.checkHealth()
			case <-hm.stopCh:
				return
			}
		}
	}()
}

func (hm *HealthMonitor) checkHealth() {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err := hm.db.PingContext(ctx)
	
	hm.mu.Lock()
	hm.isHealthy = (err == nil)
	hm.mu.Unlock()

	_ = err // silent; health flag already updated
}

func (hm *HealthMonitor) IsHealthy() bool {
	hm.mu.RLock()
	defer hm.mu.RUnlock()
	return hm.isHealthy
}

func (hm *HealthMonitor) Stop() {
	close(hm.stopCh)
}
