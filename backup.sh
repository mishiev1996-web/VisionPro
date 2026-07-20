#!/bin/bash
# Daily backup script for VisionPro
# Run via cron: 0 2 * * * /opt/VisionPro/backup.sh

BACKUP_DIR="/opt/VisionPro/backups"
DATE=$(date +%Y-%m-%d)
KEEP_DAYS=7

# Create today's backup dir
mkdir -p "$BACKUP_DIR/$DATE"

# Copy files
cp /opt/VisionPro/data/football.db "$BACKUP_DIR/$DATE/"
cp /opt/VisionPro/data/tennis.db "$BACKUP_DIR/$DATE/"
cp /opt/VisionPro/model.pkl "$BACKUP_DIR/$DATE/"
cp /opt/VisionPro/tennis/tennis_model.pkl "$BACKUP_DIR/$DATE/"

# Verify SQLite integrity
for db in "$BACKUP_DIR/$DATE/football.db" "$BACKUP_DIR/$DATE/tennis.db"; do
    result=$(sqlite3 "$db" "PRAGMA integrity_check;" 2>/dev/null)
    if [ "$result" = "ok" ]; then
        echo "[$(date)] $db: integrity OK"
    else
        echo "[$(date)] $db: INTEGRITY FAILED - $result"
    fi
done

# Log sizes
echo "[$(date)] Backup completed: $DATE"
ls -lh "$BACKUP_DIR/$DATE/"

# Clean old backups
find "$BACKUP_DIR" -maxdepth 1 -type d -mtime +$KEEP_DAYS -exec rm -rf {} \; 2>/dev/null
echo "[$(date)] Old backups cleaned (keeping $KEEP_DAYS days)"
