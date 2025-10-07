package database

import (
	"fmt"

	"github.com/golang-migrate/migrate/v4"
	"github.com/golang-migrate/migrate/v4/database/postgres"
	_ "github.com/golang-migrate/migrate/v4/source/file"
)

// NewMigrationManagerFromFile creates migration manager from file:// source
func NewMigrationManagerFromFile(db *Database, sourceURL string) (*MigrationManager, error) {
	// Create database driver
	dbDriver, err := postgres.WithInstance(db.Primary, &postgres.Config{
		MigrationsTable: "schema_migrations",
		DatabaseName:    db.config.DBName,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create database driver: %w", err)
	}

	// Create migrate instance with file source
	m, err := migrate.NewWithDatabaseInstance(sourceURL, db.config.DBName, dbDriver)
	if err != nil {
		return nil, fmt.Errorf("failed to create migrate instance: %w", err)
	}

	return &MigrationManager{
		db:      db,
		migrate: m,
	}, nil
}
