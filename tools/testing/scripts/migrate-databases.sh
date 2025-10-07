#!/bin/bash
# Production Database Migration Script with Backup
# PERSON 3 - Business Logic & Infrastructure
# P0 Requirement: MUST backup before migrations

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
BACKUP_DIR="${BACKUP_DIR:-/backups}"
MIGRATION_DIR="${MIGRATION_DIR:-/workspaces/Living-Digital-Fortress/migrations}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DRY_RUN="${DRY_RUN:-false}"

# Services to migrate
SERVICES=("credits" "shadow" "cdefnet")

# Logging functions
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
    
    # Check if backup directory exists
    if [ ! -d "$BACKUP_DIR" ]; then
        log_info "Creating backup directory: $BACKUP_DIR"
        mkdir -p "$BACKUP_DIR"
    fi
    
    # Check required tools
    for tool in psql pg_dump pg_isready; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool is not installed"
            exit 1
        fi
    done
    
    log_info "Prerequisites check passed"
}

# Backup database
backup_database() {
    local service=$1
    local db_url=$2
    local backup_file="${BACKUP_DIR}/${service}_backup_${TIMESTAMP}.sql"
    
    log_info "Backing up $service database..."
    
    if [ "$DRY_RUN" = "true" ]; then
        log_warn "DRY RUN: Would backup to $backup_file"
        return 0
    fi
    
    # Extract connection parameters
    local db_host=$(echo "$db_url" | sed -n 's/.*@\([^:]*\):.*/\1/p')
    local db_port=$(echo "$db_url" | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')
    local db_name=$(echo "$db_url" | sed -n 's/.*\/\([^?]*\).*/\1/p')
    local db_user=$(echo "$db_url" | sed -n 's/.*:\/\/\([^:]*\):.*/\1/p')
    local db_pass=$(echo "$db_url" | sed -n 's/.*:\/\/[^:]*:\([^@]*\)@.*/\1/p')
    
    # Set password for pg_dump
    export PGPASSWORD="$db_pass"
    
    # Check database connectivity
    if ! pg_isready -h "$db_host" -p "$db_port" -U "$db_user" &> /dev/null; then
        log_error "Cannot connect to $service database"
        return 1
    fi
    
    # Create backup
    pg_dump -h "$db_host" -p "$db_port" -U "$db_user" -d "$db_name" \
        --no-owner --no-privileges --clean --if-exists \
        -f "$backup_file" 2>&1 | tee "${backup_file}.log"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        # Compress backup
        gzip "$backup_file"
        backup_file="${backup_file}.gz"
        
        local backup_size=$(du -h "$backup_file" | cut -f1)
        log_info "Backup completed: $backup_file (${backup_size})"
        
        # Calculate checksum
        local checksum=$(sha256sum "$backup_file" | cut -d' ' -f1)
        echo "$checksum  $backup_file" > "${backup_file}.sha256"
        log_info "Backup checksum: $checksum"
        
        # Store metadata
        cat > "${backup_file}.meta" <<EOF
{
  "service": "$service",
  "timestamp": "$TIMESTAMP",
  "database": "$db_name",
  "host": "$db_host",
  "size": "$backup_size",
  "checksum": "$checksum",
  "backup_file": "$backup_file"
}
EOF
        
        return 0
    else
        log_error "Backup failed for $service"
        return 1
    fi
}

# Verify backup integrity
verify_backup() {
    local backup_file=$1
    
    log_info "Verifying backup integrity..."
    
    if [ ! -f "$backup_file" ]; then
        log_error "Backup file not found: $backup_file"
        return 1
    fi
    
    if [ ! -f "${backup_file}.sha256" ]; then
        log_warn "Checksum file not found, skipping verification"
        return 0
    fi
    
    if sha256sum -c "${backup_file}.sha256" &> /dev/null; then
        log_info "Backup integrity verified"
        return 0
    else
        log_error "Backup integrity check failed"
        return 1
    fi
}

# Run migrations
run_migrations() {
    local service=$1
    local db_url=$2
    local migration_path="${MIGRATION_DIR}/${service}"
    
    log_info "Running migrations for $service..."
    
    if [ ! -d "$migration_path" ]; then
        log_warn "Migration directory not found: $migration_path"
        return 0
    fi
    
    if [ "$DRY_RUN" = "true" ]; then
        log_warn "DRY RUN: Would run migrations from $migration_path"
        return 0
    fi
    
    # Count pending migrations
    local pending_count=$(ls -1 "$migration_path"/*.up.sql 2>/dev/null | wc -l)
    log_info "Found $pending_count migration files"
    
    # Run migrations using golang-migrate
    if command -v migrate &> /dev/null; then
        migrate -path "$migration_path" -database "$db_url" up
        local exit_code=$?
        
        if [ $exit_code -eq 0 ]; then
            log_info "Migrations completed successfully for $service"
            return 0
        else
            log_error "Migration failed for $service (exit code: $exit_code)"
            return 1
        fi
    else
        log_error "migrate tool not found. Install golang-migrate"
        return 1
    fi
}

# Rollback database to backup
rollback_database() {
    local service=$1
    local db_url=$2
    local backup_file=$3
    
    log_warn "Rolling back $service database to backup..."
    
    if [ "$DRY_RUN" = "true" ]; then
        log_warn "DRY RUN: Would rollback from $backup_file"
        return 0
    fi
    
    # Verify backup before rollback
    if ! verify_backup "$backup_file"; then
        log_error "Backup verification failed, cannot rollback"
        return 1
    fi
    
    # Extract connection parameters
    local db_host=$(echo "$db_url" | sed -n 's/.*@\([^:]*\):.*/\1/p')
    local db_port=$(echo "$db_url" | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')
    local db_name=$(echo "$db_url" | sed -n 's/.*\/\([^?]*\).*/\1/p')
    local db_user=$(echo "$db_url" | sed -n 's/.*:\/\/\([^:]*\):.*/\1/p')
    local db_pass=$(echo "$db_url" | sed -n 's/.*:\/\/[^:]*:\([^@]*\)@.*/\1/p')
    
    export PGPASSWORD="$db_pass"
    
    # Restore from backup
    log_info "Restoring database from backup..."
    
    if [[ "$backup_file" == *.gz ]]; then
        gunzip -c "$backup_file" | psql -h "$db_host" -p "$db_port" -U "$db_user" -d "$db_name"
    else
        psql -h "$db_host" -p "$db_port" -U "$db_user" -d "$db_name" < "$backup_file"
    fi
    
    if [ $? -eq 0 ]; then
        log_info "Database rollback completed for $service"
        return 0
    else
        log_error "Database rollback failed for $service"
        return 1
    fi
}

# Cleanup old backups
cleanup_old_backups() {
    local retention_days="${BACKUP_RETENTION_DAYS:-30}"
    
    log_info "Cleaning up backups older than $retention_days days..."
    
    if [ "$DRY_RUN" = "true" ]; then
        log_warn "DRY RUN: Would delete backups older than $retention_days days"
        find "$BACKUP_DIR" -name "*.sql.gz" -mtime +$retention_days -type f
        return 0
    fi
    
    local deleted_count=$(find "$BACKUP_DIR" -name "*.sql.gz" -mtime +$retention_days -type f -delete -print | wc -l)
    log_info "Deleted $deleted_count old backup files"
}

# Main migration workflow
main() {
    log_info "=== ShieldX Production Database Migration ==="
    log_info "Timestamp: $TIMESTAMP"
    log_info "Backup Directory: $BACKUP_DIR"
    log_info "Migration Directory: $MIGRATION_DIR"
    
    if [ "$DRY_RUN" = "true" ]; then
        log_warn "Running in DRY RUN mode - no changes will be made"
    fi
    
    check_prerequisites
    
    # Track backup files for rollback
    declare -A backup_files
    local failed_services=()
    
    # Migrate each service
    for service in "${SERVICES[@]}"; do
        log_info "Processing service: $service"
        
        # Get database URL from environment
        local db_url_var="${service^^}_DATABASE_URL"
        local db_url="${!db_url_var:-}"
        
        if [ -z "$db_url" ]; then
            log_warn "Database URL not set for $service (expected: $db_url_var), skipping"
            continue
        fi
        
        # Backup database
        local backup_file="${BACKUP_DIR}/${service}_backup_${TIMESTAMP}.sql.gz"
        if backup_database "$service" "$db_url"; then
            backup_files[$service]="$backup_file"
        else
            log_error "Backup failed for $service, skipping migration"
            failed_services+=("$service")
            continue
        fi
        
        # Run migrations
        if ! run_migrations "$service" "$db_url"; then
            log_error "Migration failed for $service"
            failed_services+=("$service")
            
            # Ask for rollback
            if [ "$DRY_RUN" != "true" ]; then
                read -p "Rollback $service to backup? (y/n) " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    rollback_database "$service" "$db_url" "${backup_files[$service]}"
                fi
            fi
        fi
        
        log_info "---"
    done
    
    # Cleanup old backups
    cleanup_old_backups
    
    # Summary
    log_info "=== Migration Summary ==="
    log_info "Total services: ${#SERVICES[@]}"
    log_info "Failed services: ${#failed_services[@]}"
    
    if [ ${#failed_services[@]} -eq 0 ]; then
        log_info "All migrations completed successfully!"
        exit 0
    else
        log_error "Failed services: ${failed_services[*]}"
        exit 1
    fi
}

# Handle script arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --backup-dir)
            BACKUP_DIR="$2"
            shift 2
            ;;
        --help)
            cat <<EOF
Usage: $0 [OPTIONS]

Production Database Migration Script

OPTIONS:
    --dry-run           Run without making changes
    --backup-dir DIR    Specify backup directory (default: /backups)
    --help              Show this help message

ENVIRONMENT VARIABLES:
    CREDITS_DATABASE_URL    PostgreSQL URL for credits service
    SHADOW_DATABASE_URL     PostgreSQL URL for shadow service
    CDEFNET_DATABASE_URL    PostgreSQL URL for cdefnet service
    BACKUP_RETENTION_DAYS   Days to keep backups (default: 30)

EXAMPLE:
    # Dry run
    $0 --dry-run
    
    # Production migration
    export CREDITS_DATABASE_URL="postgres://user:pass@host:5432/credits"
    export SHADOW_DATABASE_URL="postgres://user:pass@host:5432/shadow"
    $0
EOF
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main function
main
