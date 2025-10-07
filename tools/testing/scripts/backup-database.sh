#!/bin/bash
# Database Backup Script for Living Digital Fortress
# Supports automated backup, compression, and retention management

set -e

# Configuration
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_USER="${DB_USER:-postgres}"
DB_NAME="${DB_NAME:-fortress}"
BACKUP_DIR="${BACKUP_DIR:-/var/backups/postgres}"
RETENTION_DAYS="${RETENTION_DAYS:-7}"
COMPRESSION="${COMPRESSION:-true}"
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v pg_dump &> /dev/null; then
        log_error "pg_dump not found. Please install PostgreSQL client tools."
        exit 1
    fi
    
    if [ "$COMPRESSION" = "true" ] && ! command -v gzip &> /dev/null; then
        log_error "gzip not found. Please install gzip or disable compression."
        exit 1
    fi
    
    if [ ! -d "$BACKUP_DIR" ]; then
        log_info "Creating backup directory: $BACKUP_DIR"
        mkdir -p "$BACKUP_DIR"
    fi
}

# Perform backup
backup_database() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_file="${BACKUP_DIR}/${DB_NAME}_${timestamp}.dump"
    
    log_info "Starting backup of database: $DB_NAME"
    log_info "Backup file: $backup_file"
    
    # Perform backup using custom format for better performance
    PGPASSWORD="$PGPASSWORD" pg_dump \
        -h "$DB_HOST" \
        -p "$DB_PORT" \
        -U "$DB_USER" \
        -d "$DB_NAME" \
        -Fc \
        -j "$PARALLEL_JOBS" \
        -f "$backup_file"
    
    if [ $? -eq 0 ]; then
        log_info "Backup completed successfully"
        
        # Get file size
        local size=$(du -h "$backup_file" | cut -f1)
        log_info "Backup size: $size"
        
        # Optional compression (custom format is already compressed)
        if [ "$COMPRESSION" = "true" ]; then
            log_info "Compressing backup..."
            gzip "$backup_file"
            backup_file="${backup_file}.gz"
            local compressed_size=$(du -h "$backup_file" | cut -f1)
            log_info "Compressed size: $compressed_size"
        fi
        
        echo "$backup_file"
    else
        log_error "Backup failed"
        exit 1
    fi
}

# Verify backup
verify_backup() {
    local backup_file="$1"
    
    log_info "Verifying backup: $backup_file"
    
    # Check if file exists and is not empty
    if [ ! -f "$backup_file" ]; then
        log_error "Backup file not found: $backup_file"
        return 1
    fi
    
    if [ ! -s "$backup_file" ]; then
        log_error "Backup file is empty: $backup_file"
        return 1
    fi
    
    # For .dump files, use pg_restore --list
    if [[ "$backup_file" == *.dump ]] || [[ "$backup_file" == *.dump.gz ]]; then
        if [[ "$backup_file" == *.gz ]]; then
            gunzip -c "$backup_file" | pg_restore --list - > /dev/null 2>&1
        else
            pg_restore --list "$backup_file" > /dev/null 2>&1
        fi
        
        if [ $? -eq 0 ]; then
            log_info "Backup verification successful"
            return 0
        else
            log_error "Backup verification failed"
            return 1
        fi
    fi
    
    log_info "Backup file looks good"
    return 0
}

# Cleanup old backups
cleanup_old_backups() {
    log_info "Cleaning up backups older than $RETENTION_DAYS days..."
    
    local deleted_count=0
    local cutoff_date=$(date -d "$RETENTION_DAYS days ago" +%s)
    
    for backup in "$BACKUP_DIR"/${DB_NAME}_*.dump* ; do
        if [ -f "$backup" ]; then
            local file_date=$(stat -c %Y "$backup")
            
            if [ "$file_date" -lt "$cutoff_date" ]; then
                log_info "Deleting old backup: $(basename $backup)"
                rm -f "$backup"
                ((deleted_count++))
            fi
        fi
    done
    
    if [ $deleted_count -gt 0 ]; then
        log_info "Deleted $deleted_count old backup(s)"
    else
        log_info "No old backups to delete"
    fi
}

# List backups
list_backups() {
    log_info "Available backups in $BACKUP_DIR:"
    echo ""
    
    for backup in "$BACKUP_DIR"/${DB_NAME}_*.dump* ; do
        if [ -f "$backup" ]; then
            local size=$(du -h "$backup" | cut -f1)
            local date=$(stat -c %y "$backup" | cut -d'.' -f1)
            printf "  %-50s %10s  %s\n" "$(basename $backup)" "$size" "$date"
        fi
    done
    
    echo ""
}

# Restore from backup
restore_database() {
    local backup_file="$1"
    
    if [ -z "$backup_file" ]; then
        log_error "No backup file specified"
        exit 1
    fi
    
    if [ ! -f "$backup_file" ]; then
        log_error "Backup file not found: $backup_file"
        exit 1
    fi
    
    log_warn "WARNING: This will restore database $DB_NAME from backup"
    log_warn "All current data will be replaced!"
    read -p "Are you sure you want to continue? (yes/no): " confirm
    
    if [ "$confirm" != "yes" ]; then
        log_info "Restore cancelled"
        exit 0
    fi
    
    log_info "Starting restore from: $backup_file"
    
    # Drop and recreate database
    PGPASSWORD="$PGPASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -c "DROP DATABASE IF EXISTS ${DB_NAME};"
    PGPASSWORD="$PGPASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -c "CREATE DATABASE ${DB_NAME};"
    
    # Restore
    if [[ "$backup_file" == *.gz ]]; then
        gunzip -c "$backup_file" | PGPASSWORD="$PGPASSWORD" pg_restore \
            -h "$DB_HOST" \
            -p "$DB_PORT" \
            -U "$DB_USER" \
            -d "$DB_NAME" \
            -j "$PARALLEL_JOBS" \
            --clean \
            --if-exists \
            --no-owner \
            --no-privileges
    else
        PGPASSWORD="$PGPASSWORD" pg_restore \
            -h "$DB_HOST" \
            -p "$DB_PORT" \
            -U "$DB_USER" \
            -d "$DB_NAME" \
            -j "$PARALLEL_JOBS" \
            --clean \
            --if-exists \
            --no-owner \
            --no-privileges \
            "$backup_file"
    fi
    
    if [ $? -eq 0 ]; then
        log_info "Restore completed successfully"
    else
        log_error "Restore failed"
        exit 1
    fi
}

# Main
main() {
    local action="${1:-backup}"
    
    case "$action" in
        backup)
            check_prerequisites
            backup_file=$(backup_database)
            verify_backup "$backup_file"
            cleanup_old_backups
            ;;
        list)
            list_backups
            ;;
        restore)
            check_prerequisites
            restore_database "$2"
            ;;
        verify)
            verify_backup "$2"
            ;;
        cleanup)
            cleanup_old_backups
            ;;
        *)
            echo "Usage: $0 {backup|list|restore <file>|verify <file>|cleanup}"
            echo ""
            echo "Commands:"
            echo "  backup          - Create a new backup"
            echo "  list            - List all available backups"
            echo "  restore <file>  - Restore from a backup file"
            echo "  verify <file>   - Verify a backup file"
            echo "  cleanup         - Remove old backups"
            echo ""
            echo "Environment variables:"
            echo "  DB_HOST         - Database host (default: localhost)"
            echo "  DB_PORT         - Database port (default: 5432)"
            echo "  DB_USER         - Database user (default: postgres)"
            echo "  DB_NAME         - Database name (default: fortress)"
            echo "  PGPASSWORD      - Database password"
            echo "  BACKUP_DIR      - Backup directory (default: /var/backups/postgres)"
            echo "  RETENTION_DAYS  - Days to keep backups (default: 7)"
            echo "  COMPRESSION     - Enable compression (default: true)"
            echo "  PARALLEL_JOBS   - Parallel jobs for backup/restore (default: 4)"
            exit 1
            ;;
    esac
}

main "$@"
