package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"shieldx/shared/shieldx-common/pkg/database"
)

func main() {
	ctx := context.Background()

	// Configure database connection from environment
	config := database.DBConfig{
		Host:     getEnv("DB_HOST", "localhost"),
		Port:     getEnvInt("DB_PORT", 5432),
		User:     getEnv("DB_USER", "postgres"),
		Password: getEnv("PGPASSWORD", ""),
		DBName:   getEnv("DB_NAME", "fortress"),
		SSLMode:  getEnv("DB_SSL_MODE", "disable"),

		MaxOpenConns:     getEnvInt("DB_MAX_OPEN_CONNS", 25),
		MaxIdleConns:     getEnvInt("DB_MAX_IDLE_CONNS", 5),
		ConnMaxLifetime:  getEnvDuration("DB_CONN_MAX_LIFETIME", 5*time.Minute),
		ConnMaxIdleTime:  getEnvDuration("DB_CONN_MAX_IDLE_TIME", 1*time.Minute),
		ConnectTimeout:   getEnvDuration("DB_CONNECT_TIMEOUT", 10*time.Second),
		StatementTimeout: getEnvDuration("DB_STATEMENT_TIMEOUT", 30*time.Second),
	}

	// Parse replica hosts
	replicaHosts := getEnvArray("DB_REPLICA_HOSTS")
	if len(replicaHosts) > 0 {
		config.ReplicaHosts = replicaHosts
	}

	// Connect to database
	log.Println("Connecting to database...")
	db, err := database.NewDatabase(config)
	if err != nil {
		log.Fatalf("Failed to connect to database: %v", err)
	}
	defer db.Close()

	// Verify connection
	if err := db.Ping(ctx); err != nil {
		log.Fatalf("Failed to ping database: %v", err)
	}
	log.Println("✓ Database connection established")

	// Get migration service from command line
	if len(os.Args) < 2 {
		log.Println("Usage: migrate-db <service> [command]")
		log.Println("Services: credits, shadow, cdefnet, all")
		log.Println("Commands: up (default), down, status, force <version>")
		os.Exit(1)
	}

	service := os.Args[1]
	command := "up"
	if len(os.Args) > 2 {
		command = os.Args[2]
	}

	// Run migrations
	log.Printf("Note: For production use, embed migration files in the binary")
	log.Printf("Currently using file-based migrations from: migrations/%s/", service)

	// Since we can't use embed.FS from command line tool easily,
	// we'll use the file-based approach
	migrationPath := fmt.Sprintf("file://migrations/%s", service)

	switch service {
	case "credits", "shadow", "cdefnet":
		runFileMigrations(ctx, db, service, migrationPath, command)
	case "all":
		for _, svc := range []string{"credits", "shadow", "cdefnet"} {
			path := fmt.Sprintf("file://migrations/%s", svc)
			runFileMigrations(ctx, db, svc, path, command)
		}
	default:
		log.Fatalf("Unknown service: %s", service)
	}

	log.Println("✓ Migrations completed successfully")
}

func runFileMigrations(ctx context.Context, db *database.Database, name string, sourceURL string, command string) {
	log.Printf("Running migrations for %s from %s...", name, sourceURL)

	// Use file-based migration source
	mm, err := database.NewMigrationManagerFromFile(db, sourceURL)
	if err != nil {
		log.Fatalf("Failed to create migration manager for %s: %v", name, err)
	}
	defer mm.Close()

	// Get current version
	version, dirty, err := mm.Version()
	if err != nil && err.Error() != "no migration" {
		log.Printf("  Warning: %v", err)
		version = 0
		dirty = false
	}

	log.Printf("  Current version: %d (dirty: %v)", version, dirty)

	// Execute command
	switch command {
	case "up":
		if err := mm.Up(); err != nil && err.Error() != "no change" {
			log.Fatalf("Failed to run migrations for %s: %v", name, err)
		}
		newVersion, _, _ := mm.Version()
		log.Printf("  ✓ Migrated to version %d", newVersion)

	case "down":
		if err := mm.Down(); err != nil && err.Error() != "no change" {
			log.Fatalf("Failed to rollback migration for %s: %v", name, err)
		}
		newVersion, _, _ := mm.Version()
		log.Printf("  ✓ Rolled back to version %d", newVersion)

	case "status":
		history, err := mm.GetMigrationHistory(ctx)
		if err != nil {
			log.Printf("  Warning: Failed to get history: %v", err)
			return
		}
		for _, record := range history {
			status := "✓"
			if record.Dirty {
				status = "✗ (dirty)"
			}
			log.Printf("  %s Version %d - %s", status, record.Version, record.CreatedAt.Format(time.RFC3339))
		}

	case "force":
		if len(os.Args) < 4 {
			log.Fatalf("Force command requires version argument")
		}
		var forceVersion int
		fmt.Sscanf(os.Args[3], "%d", &forceVersion)
		if err := mm.Force(forceVersion); err != nil {
			log.Fatalf("Failed to force version for %s: %v", name, err)
		}
		log.Printf("  ✓ Forced to version %d", forceVersion)

	default:
		log.Fatalf("Unknown command: %s", command)
	}
}

func runMigrations(ctx context.Context, db *database.Database, name string, migrations interface{}, path string, command string) {
	// Legacy function - kept for compatibility
	log.Printf("Note: Use runFileMigrations instead")
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		var i int
		if _, err := fmt.Sscanf(value, "%d", &i); err == nil {
			return i
		}
	}
	return defaultValue
}

func getEnvDuration(key string, defaultValue time.Duration) time.Duration {
	if value := os.Getenv(key); value != "" {
		if d, err := time.ParseDuration(value); err == nil {
			return d
		}
	}
	return defaultValue
}

func getEnvArray(key string) []string {
	if value := os.Getenv(key); value != "" {
		// Simple comma-separated parsing
		var result []string
		current := ""
		for _, c := range value {
			if c == ',' {
				if current != "" {
					result = append(result, current)
					current = ""
				}
			} else {
				current += string(c)
			}
		}
		if current != "" {
			result = append(result, current)
		}
		return result
	}
	return nil
}
