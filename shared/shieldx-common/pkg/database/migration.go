package database

import (
	"context"
	"embed"
	"fmt"
	"time"

	"github.com/golang-migrate/migrate/v4"
	"github.com/golang-migrate/migrate/v4/database/postgres"
	"github.com/golang-migrate/migrate/v4/source/iofs"
)

// MigrationManager manages database migrations
type MigrationManager struct {
	db      *Database
	migrate *migrate.Migrate
}

// NewMigrationManager creates a new migration manager
func NewMigrationManager(db *Database, migrations embed.FS, migrationsPath string) (*MigrationManager, error) {
	// Create source driver from embedded filesystem
	sourceDriver, err := iofs.New(migrations, migrationsPath)
	if err != nil {
		return nil, fmt.Errorf("failed to create source driver: %w", err)
	}

	// Create database driver
	dbDriver, err := postgres.WithInstance(db.Primary, &postgres.Config{
		MigrationsTable: "schema_migrations",
		DatabaseName:    db.config.DBName,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create database driver: %w", err)
	}

	// Create migrate instance
	m, err := migrate.NewWithInstance("iofs", sourceDriver, db.config.DBName, dbDriver)
	if err != nil {
		return nil, fmt.Errorf("failed to create migrate instance: %w", err)
	}

	return &MigrationManager{
		db:      db,
		migrate: m,
	}, nil
}

// Up runs all pending migrations
func (mm *MigrationManager) Up() error {
	if err := mm.migrate.Up(); err != nil && err != migrate.ErrNoChange {
		return fmt.Errorf("failed to run migrations: %w", err)
	}
	return nil
}

// Down rolls back the last migration
func (mm *MigrationManager) Down() error {
	if err := mm.migrate.Down(); err != nil && err != migrate.ErrNoChange {
		return fmt.Errorf("failed to rollback migration: %w", err)
	}
	return nil
}

// Steps runs a specific number of migrations (positive for up, negative for down)
func (mm *MigrationManager) Steps(n int) error {
	if err := mm.migrate.Steps(n); err != nil && err != migrate.ErrNoChange {
		return fmt.Errorf("failed to run %d migration steps: %w", n, err)
	}
	return nil
}

// Migrate to a specific version
func (mm *MigrationManager) Migrate(version uint) error {
	if err := mm.migrate.Migrate(version); err != nil && err != migrate.ErrNoChange {
		return fmt.Errorf("failed to migrate to version %d: %w", version, err)
	}
	return nil
}

// Version returns the current migration version
func (mm *MigrationManager) Version() (uint, bool, error) {
	version, dirty, err := mm.migrate.Version()
	if err != nil && err != migrate.ErrNilVersion {
		return 0, false, fmt.Errorf("failed to get migration version: %w", err)
	}
	return version, dirty, nil
}

// Force sets the migration version without running migrations
func (mm *MigrationManager) Force(version int) error {
	if err := mm.migrate.Force(version); err != nil {
		return fmt.Errorf("failed to force version %d: %w", version, err)
	}
	return nil
}

// Drop drops everything in the database
func (mm *MigrationManager) Drop() error {
	if err := mm.migrate.Drop(); err != nil {
		return fmt.Errorf("failed to drop database: %w", err)
	}
	return nil
}

// Close closes the migration manager
func (mm *MigrationManager) Close() error {
	sourceErr, dbErr := mm.migrate.Close()
	if sourceErr != nil {
		return fmt.Errorf("failed to close migration source: %w", sourceErr)
	}
	if dbErr != nil {
		return fmt.Errorf("failed to close migration database: %w", dbErr)
	}
	return nil
}

// GetMigrationHistory returns the migration history
func (mm *MigrationManager) GetMigrationHistory(ctx context.Context) ([]MigrationRecord, error) {
	query := `
		SELECT version, dirty, created_at
		FROM schema_migrations
		ORDER BY version DESC
	`

	rows, err := mm.db.Query(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to query migration history: %w", err)
	}
	defer rows.Close()

	var records []MigrationRecord
	for rows.Next() {
		var record MigrationRecord
		if err := rows.Scan(&record.Version, &record.Dirty, &record.CreatedAt); err != nil {
			return nil, fmt.Errorf("failed to scan migration record: %w", err)
		}
		records = append(records, record)
	}

	return records, rows.Err()
}

// MigrationRecord represents a migration record
type MigrationRecord struct {
	Version   uint
	Dirty     bool
	CreatedAt time.Time
}

// ValidateMigrations checks if migrations are in a valid state
func (mm *MigrationManager) ValidateMigrations(ctx context.Context) error {
	version, dirty, err := mm.Version()
	if err != nil {
		return fmt.Errorf("failed to get current version: %w", err)
	}

	if dirty {
		return fmt.Errorf("database is in dirty state at version %d - manual intervention required", version)
	}

	return nil
}

// CreateMigrationLock creates a migration lock to prevent concurrent migrations
func (mm *MigrationManager) CreateMigrationLock(ctx context.Context) (*MigrationLock, error) {
	// Create advisory lock table if not exists
	_, err := mm.db.Exec(ctx, `
		CREATE TABLE IF NOT EXISTS migration_locks (
			id INTEGER PRIMARY KEY,
			locked_at TIMESTAMP NOT NULL,
			locked_by VARCHAR(255) NOT NULL
		)
	`)
	if err != nil {
		return nil, fmt.Errorf("failed to create lock table: %w", err)
	}

	lock := &MigrationLock{
		db:     mm.db,
		lockID: 1,
	}

	if err := lock.Lock(ctx); err != nil {
		return nil, err
	}

	return lock, nil
}

// MigrationLock prevents concurrent migrations
type MigrationLock struct {
	db     *Database
	lockID int
	locked bool
}

// Lock acquires the migration lock
func (ml *MigrationLock) Lock(ctx context.Context) error {
	// Use PostgreSQL advisory lock
	var locked bool
	err := ml.db.Primary.QueryRowContext(ctx,
		"SELECT pg_try_advisory_lock($1)",
		ml.lockID,
	).Scan(&locked)

	if err != nil {
		return fmt.Errorf("failed to acquire advisory lock: %w", err)
	}

	if !locked {
		return fmt.Errorf("migration lock is already held")
	}

	ml.locked = true
	return nil
}

// Unlock releases the migration lock
func (ml *MigrationLock) Unlock(ctx context.Context) error {
	if !ml.locked {
		return nil
	}

	var unlocked bool
	err := ml.db.Primary.QueryRowContext(ctx,
		"SELECT pg_advisory_unlock($1)",
		ml.lockID,
	).Scan(&unlocked)

	if err != nil {
		return fmt.Errorf("failed to release advisory lock: %w", err)
	}

	if !unlocked {
		return fmt.Errorf("failed to unlock migration lock")
	}

	ml.locked = false
	return nil
}

// AutoMigrate runs migrations automatically with lock protection
func AutoMigrate(ctx context.Context, db *Database, migrations embed.FS, migrationsPath string) error {
	mm, err := NewMigrationManager(db, migrations, migrationsPath)
	if err != nil {
		return err
	}
	defer mm.Close()

	// Acquire lock
	lock, err := mm.CreateMigrationLock(ctx)
	if err != nil {
		return fmt.Errorf("failed to acquire migration lock: %w", err)
	}
	defer lock.Unlock(ctx)

	// Validate current state
	if err := mm.ValidateMigrations(ctx); err != nil {
		return err
	}

	// Run migrations
	if err := mm.Up(); err != nil {
		return fmt.Errorf("failed to run migrations: %w", err)
	}

	return nil
}
