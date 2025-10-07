package database

import (
	"context"
	"database/sql"
	"fmt"
	"time"
)

// HealthStatus represents the health status of database
type HealthStatus struct {
	Healthy         bool
	PrimaryHealthy  bool
	ReplicasHealthy []bool
	Latency         time.Duration
	Message         string
	Timestamp       time.Time
}

// HealthChecker performs health checks on database
type HealthChecker struct {
	db              *Database
	checkInterval   time.Duration
	timeout         time.Duration
	lastHealthCheck *HealthStatus
	stopCh          chan struct{}
}

// NewHealthChecker creates a new health checker
func NewHealthChecker(db *Database, checkInterval, timeout time.Duration) *HealthChecker {
	if checkInterval == 0 {
		checkInterval = 30 * time.Second
	}
	if timeout == 0 {
		timeout = 5 * time.Second
	}

	return &HealthChecker{
		db:            db,
		checkInterval: checkInterval,
		timeout:       timeout,
		stopCh:        make(chan struct{}),
	}
}

// Start begins periodic health checks
func (hc *HealthChecker) Start() {
	ticker := time.NewTicker(hc.checkInterval)
	defer ticker.Stop()

	// Initial health check
	hc.Check()

	for {
		select {
		case <-ticker.C:
			hc.Check()
		case <-hc.stopCh:
			return
		}
	}
}

// Stop stops the health checker
func (hc *HealthChecker) Stop() {
	close(hc.stopCh)
}

// Check performs a health check
func (hc *HealthChecker) Check() *HealthStatus {
	ctx, cancel := context.WithTimeout(context.Background(), hc.timeout)
	defer cancel()

	start := time.Now()
	status := &HealthStatus{
		Timestamp: start,
		Healthy:   true,
	}

	// Check primary
	if err := hc.checkConnection(ctx, hc.db.Primary); err != nil {
		status.PrimaryHealthy = false
		status.Healthy = false
		status.Message = fmt.Sprintf("Primary database unhealthy: %v", err)
	} else {
		status.PrimaryHealthy = true
	}

	// Check replicas
	status.ReplicasHealthy = make([]bool, len(hc.db.Replicas))
	for i, replica := range hc.db.Replicas {
		if err := hc.checkConnection(ctx, replica); err != nil {
			status.ReplicasHealthy[i] = false
			if status.Message != "" {
				status.Message += "; "
			}
			status.Message += fmt.Sprintf("Replica %d unhealthy: %v", i, err)
		} else {
			status.ReplicasHealthy[i] = true
		}
	}

	status.Latency = time.Since(start)
	hc.lastHealthCheck = status

	return status
}

// checkConnection checks if a database connection is healthy
func (hc *HealthChecker) checkConnection(ctx context.Context, db *sql.DB) error {
	// Ping test
	if err := db.PingContext(ctx); err != nil {
		return fmt.Errorf("ping failed: %w", err)
	}

	// Simple query test
	var result int
	if err := db.QueryRowContext(ctx, "SELECT 1").Scan(&result); err != nil {
		return fmt.Errorf("query test failed: %w", err)
	}

	// Check connection pool stats
	stats := db.Stats()
	if stats.OpenConnections == 0 {
		return fmt.Errorf("no open connections")
	}

	// Check for high wait times (potential connection pool exhaustion)
	if stats.WaitCount > 0 && stats.WaitDuration > 10*time.Second {
		return fmt.Errorf("high connection wait time: %v", stats.WaitDuration)
	}

	return nil
}

// GetLastHealthCheck returns the last health check result
func (hc *HealthChecker) GetLastHealthCheck() *HealthStatus {
	return hc.lastHealthCheck
}

// IsHealthy returns whether the database is healthy
func (hc *HealthChecker) IsHealthy() bool {
	if hc.lastHealthCheck == nil {
		return false
	}
	return hc.lastHealthCheck.Healthy
}

// AutoReconnect attempts to reconnect unhealthy databases
func (hc *HealthChecker) AutoReconnect(ctx context.Context) error {
	status := hc.Check()

	if !status.PrimaryHealthy {
		// Critical: primary is down
		return fmt.Errorf("primary database is unhealthy, cannot auto-reconnect")
	}

	// Try to reconnect unhealthy replicas
	for i, healthy := range status.ReplicasHealthy {
		if !healthy && i < len(hc.db.Replicas) {
			replicaHost := hc.db.config.ReplicaHosts[i]

			// Close old connection
			hc.db.Replicas[i].Close()

			// Attempt reconnection
			newConn, err := hc.db.connectToDB(
				replicaHost,
				hc.db.config.Port,
				hc.db.config.DBName,
				hc.db.config.User,
				hc.db.config.Password,
			)

			if err != nil {
				return fmt.Errorf("failed to reconnect replica %d: %w", i, err)
			}

			hc.db.Replicas[i] = newConn
		}
	}

	return nil
}

// CheckReplicationLag checks replication lag for replicas
func (hc *HealthChecker) CheckReplicationLag(ctx context.Context) (map[int]time.Duration, error) {
	lags := make(map[int]time.Duration)

	// Get primary LSN
	var primaryLSN string
	err := hc.db.Primary.QueryRowContext(ctx,
		"SELECT pg_current_wal_lsn()::text",
	).Scan(&primaryLSN)
	if err != nil {
		return nil, fmt.Errorf("failed to get primary LSN: %w", err)
	}

	// Check each replica's lag
	for i, replica := range hc.db.Replicas {
		var replicaLSN string
		var lagBytes int64

		err := replica.QueryRowContext(ctx,
			`SELECT 
				pg_last_wal_replay_lsn()::text,
				pg_wal_lsn_diff($1::pg_lsn, pg_last_wal_replay_lsn())
			`,
			primaryLSN,
		).Scan(&replicaLSN, &lagBytes)

		if err != nil {
			return nil, fmt.Errorf("failed to check replica %d lag: %w", i, err)
		}

		// Estimate lag in time (rough estimation: 1MB = ~1 second)
		lagDuration := time.Duration(lagBytes/1024/1024) * time.Second
		lags[i] = lagDuration
	}

	return lags, nil
}
