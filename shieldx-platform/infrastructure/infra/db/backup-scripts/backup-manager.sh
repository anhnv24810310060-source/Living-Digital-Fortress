#!/bin/sh
set -e

echo "🔄 ShieldX Backup Manager Started"
echo "📅 $(date)"
echo "🗄️  PostgreSQL Host: $POSTGRES_HOST"
echo "⏰ Backup Schedule: $BACKUP_SCHEDULE"
echo "📦 Retention: $BACKUP_RETENTION_DAYS days"

# Ensure backup directory exists
mkdir -p /backups

# Backup function
backup_database() {
    DB_NAME=$1
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BACKUP_FILE="/backups/${DB_NAME}_${TIMESTAMP}.sql.gz"
    
    echo "📦 Backing up $DB_NAME..."
    
    PGPASSWORD="$POSTGRES_PASSWORD" pg_dump \
        -h "$POSTGRES_HOST" \
        -U "$POSTGRES_USER" \
        -d "$DB_NAME" \
        --format=custom \
        --compress=9 \
        | gzip > "$BACKUP_FILE"
    
    if [ $? -eq 0 ]; then
        echo "✅ Backup completed: $BACKUP_FILE"
        
        # Calculate size
        SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
        echo "   Size: $SIZE"
    else
        echo "❌ Backup failed for $DB_NAME"
        return 1
    fi
}

# Cleanup old backups
cleanup_old_backups() {
    echo "🧹 Cleaning up old backups (retention: ${BACKUP_RETENTION_DAYS} days)..."
    find /backups -name "*.sql.gz" -mtime +${BACKUP_RETENTION_DAYS} -delete
    echo "✅ Cleanup completed"
}

# Main backup loop
while true; do
    echo ""
    echo "⏰ Starting backup cycle at $(date)"
    echo "=========================================="
    
    backup_database "credits_db"
    backup_database "contauth_db"
    backup_database "shadow_db"
    backup_database "guardian_db"
    
    cleanup_old_backups
    
    echo "=========================================="
    echo "💤 Sleeping until next backup cycle..."
    # Sleep for 24 hours
    sleep 86400
done
