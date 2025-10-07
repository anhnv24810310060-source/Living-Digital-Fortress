package database

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// BackupConfig configuration for database backups
type BackupConfig struct {
	BackupDir     string
	RetentionDays int
	Compression   bool
	IncludeSchema bool
	IncludeData   bool
	ParallelJobs  int
	CustomFormat  bool // pg_dump custom format for faster restore
}

// BackupManager manages database backups
type BackupManager struct {
	db     *Database
	config BackupConfig
}

// NewBackupManager creates a new backup manager
func NewBackupManager(db *Database, config BackupConfig) (*BackupManager, error) {
	// Set defaults
	if config.BackupDir == "" {
		config.BackupDir = "/var/backups/postgres"
	}
	if config.RetentionDays == 0 {
		config.RetentionDays = 7
	}
	if config.ParallelJobs == 0 {
		config.ParallelJobs = 4
	}
	if !config.IncludeSchema && !config.IncludeData {
		config.IncludeSchema = true
		config.IncludeData = true
	}

	// Create backup directory if not exists
	if err := os.MkdirAll(config.BackupDir, 0700); err != nil {
		return nil, fmt.Errorf("failed to create backup directory: %w", err)
	}

	return &BackupManager{
		db:     db,
		config: config,
	}, nil
}

// Backup creates a new database backup
func (bm *BackupManager) Backup(ctx context.Context) (string, error) {
	timestamp := time.Now().Format("20060102_150405")
	fileName := fmt.Sprintf("%s_%s", bm.db.config.DBName, timestamp)

	var backupPath string
	if bm.config.CustomFormat {
		backupPath = filepath.Join(bm.config.BackupDir, fileName+".dump")
	} else if bm.config.Compression {
		backupPath = filepath.Join(bm.config.BackupDir, fileName+".sql.gz")
	} else {
		backupPath = filepath.Join(bm.config.BackupDir, fileName+".sql")
	}

	// Build pg_dump command
	args := []string{
		"-h", bm.db.config.Host,
		"-p", fmt.Sprintf("%d", bm.db.config.Port),
		"-U", bm.db.config.User,
		"-d", bm.db.config.DBName,
	}

	// Add options
	if bm.config.CustomFormat {
		args = append(args, "-Fc") // Custom format
	} else {
		args = append(args, "-Fp") // Plain SQL
	}

	if bm.config.ParallelJobs > 1 && bm.config.CustomFormat {
		args = append(args, "-j", fmt.Sprintf("%d", bm.config.ParallelJobs))
	}

	if !bm.config.IncludeSchema {
		args = append(args, "--data-only")
	}
	if !bm.config.IncludeData {
		args = append(args, "--schema-only")
	}

	args = append(args, "-f", backupPath)

	// Set password via environment variable
	cmd := exec.CommandContext(ctx, "pg_dump", args...)
	cmd.Env = append(os.Environ(), fmt.Sprintf("PGPASSWORD=%s", bm.db.config.Password))

	// Execute backup
	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("pg_dump failed: %w, output: %s", err, string(output))
	}

	// Compress if needed and not using custom format
	if bm.config.Compression && !bm.config.CustomFormat {
		if err := bm.compressFile(backupPath); err != nil {
			return "", fmt.Errorf("failed to compress backup: %w", err)
		}
	}

	// Verify backup
	info, err := os.Stat(backupPath)
	if err != nil {
		return "", fmt.Errorf("failed to verify backup: %w", err)
	}

	if info.Size() == 0 {
		return "", fmt.Errorf("backup file is empty")
	}

	return backupPath, nil
}

// compressFile compresses a file using gzip
func (bm *BackupManager) compressFile(filePath string) error {
	cmd := exec.Command("gzip", filePath)
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("gzip failed: %w", err)
	}
	return nil
}

// Restore restores database from a backup
func (bm *BackupManager) Restore(ctx context.Context, backupPath string) error {
	// Check if backup file exists
	if _, err := os.Stat(backupPath); err != nil {
		return fmt.Errorf("backup file not found: %w", err)
	}

	// Determine format
	isCustomFormat := strings.HasSuffix(backupPath, ".dump")
	isCompressed := strings.HasSuffix(backupPath, ".gz")

	var cmd *exec.Cmd

	if isCustomFormat {
		// Use pg_restore for custom format
		args := []string{
			"-h", bm.db.config.Host,
			"-p", fmt.Sprintf("%d", bm.db.config.Port),
			"-U", bm.db.config.User,
			"-d", bm.db.config.DBName,
			"--clean",         // Drop objects before recreating
			"--if-exists",     // Don't error if objects don't exist
			"--no-owner",      // Don't restore ownership
			"--no-privileges", // Don't restore privileges
		}

		if bm.config.ParallelJobs > 1 {
			args = append(args, "-j", fmt.Sprintf("%d", bm.config.ParallelJobs))
		}

		args = append(args, backupPath)
		cmd = exec.CommandContext(ctx, "pg_restore", args...)
	} else {
		// Use psql for SQL format
		args := []string{
			"-h", bm.db.config.Host,
			"-p", fmt.Sprintf("%d", bm.db.config.Port),
			"-U", bm.db.config.User,
			"-d", bm.db.config.DBName,
		}

		if isCompressed {
			// Decompress on the fly
			args = append(args, "-c", fmt.Sprintf("gunzip -c %s | psql %s", backupPath, strings.Join(args, " ")))
			cmd = exec.CommandContext(ctx, "sh", args...)
		} else {
			args = append(args, "-f", backupPath)
			cmd = exec.CommandContext(ctx, "psql", args...)
		}
	}

	cmd.Env = append(os.Environ(), fmt.Sprintf("PGPASSWORD=%s", bm.db.config.Password))

	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("restore failed: %w, output: %s", err, string(output))
	}

	return nil
}

// ListBackups lists all available backups
func (bm *BackupManager) ListBackups() ([]BackupInfo, error) {
	files, err := os.ReadDir(bm.config.BackupDir)
	if err != nil {
		return nil, fmt.Errorf("failed to read backup directory: %w", err)
	}

	var backups []BackupInfo
	for _, file := range files {
		if file.IsDir() {
			continue
		}

		name := file.Name()
		if !strings.HasPrefix(name, bm.db.config.DBName) {
			continue
		}

		info, err := file.Info()
		if err != nil {
			continue
		}

		backups = append(backups, BackupInfo{
			Name:      name,
			Path:      filepath.Join(bm.config.BackupDir, name),
			Size:      info.Size(),
			CreatedAt: info.ModTime(),
		})
	}

	// Sort by creation time (newest first)
	sort.Slice(backups, func(i, j int) bool {
		return backups[i].CreatedAt.After(backups[j].CreatedAt)
	})

	return backups, nil
}

// BackupInfo information about a backup
type BackupInfo struct {
	Name      string
	Path      string
	Size      int64
	CreatedAt time.Time
}

// CleanupOldBackups removes backups older than retention period
func (bm *BackupManager) CleanupOldBackups() (int, error) {
	backups, err := bm.ListBackups()
	if err != nil {
		return 0, err
	}

	cutoff := time.Now().AddDate(0, 0, -bm.config.RetentionDays)
	deleted := 0

	for _, backup := range backups {
		if backup.CreatedAt.Before(cutoff) {
			if err := os.Remove(backup.Path); err != nil {
				return deleted, fmt.Errorf("failed to delete backup %s: %w", backup.Name, err)
			}
			deleted++
		}
	}

	return deleted, nil
}

// VerifyBackup verifies a backup file integrity
func (bm *BackupManager) VerifyBackup(ctx context.Context, backupPath string) error {
	// Check file exists and is not empty
	info, err := os.Stat(backupPath)
	if err != nil {
		return fmt.Errorf("backup file not found: %w", err)
	}

	if info.Size() == 0 {
		return fmt.Errorf("backup file is empty")
	}

	// For custom format, use pg_restore --list to verify
	if strings.HasSuffix(backupPath, ".dump") {
		cmd := exec.CommandContext(ctx, "pg_restore", "--list", backupPath)
		if err := cmd.Run(); err != nil {
			return fmt.Errorf("backup verification failed: %w", err)
		}
	}

	return nil
}

// CreateBackupSchedule creates a cron-like backup schedule
type BackupSchedule struct {
	manager  *BackupManager
	interval time.Duration
	stopCh   chan struct{}
}

// NewBackupSchedule creates a new backup schedule
func NewBackupSchedule(manager *BackupManager, interval time.Duration) *BackupSchedule {
	return &BackupSchedule{
		manager:  manager,
		interval: interval,
		stopCh:   make(chan struct{}),
	}
}

// Start begins scheduled backups
func (bs *BackupSchedule) Start(ctx context.Context) {
	ticker := time.NewTicker(bs.interval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			if _, err := bs.manager.Backup(ctx); err != nil {
				fmt.Printf("Scheduled backup failed: %v\n", err)
			}

			// Cleanup old backups
			if deleted, err := bs.manager.CleanupOldBackups(); err != nil {
				fmt.Printf("Backup cleanup failed: %v\n", err)
			} else if deleted > 0 {
				fmt.Printf("Cleaned up %d old backups\n", deleted)
			}

		case <-bs.stopCh:
			return
		case <-ctx.Done():
			return
		}
	}
}

// Stop stops the backup schedule
func (bs *BackupSchedule) Stop() {
	close(bs.stopCh)
}
