#!/usr/bin/env bash
# Database backup script for HappyTorch
#
# Usage: ./scripts/backup.sh
#
# Cron job (daily at 2:00 AM):
#   0 2 * * * /path/to/HappyTorch/scripts/backup.sh >> /var/log/happytorch-backup.log 2>&1
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Source env
set -a
source .env
set +a

BACKUP_DIR="$PROJECT_ROOT/backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="$BACKUP_DIR/happytorch_${TIMESTAMP}.sql.gz"
RETAIN_DAYS=7

mkdir -p "$BACKUP_DIR"

echo "[$(date)] Starting backup..."

docker compose -f docker-compose.prod.yml exec -T postgres \
    pg_dump -U "${DB_USER:-happytorch}" "${DB_NAME:-happytorch}" \
    | gzip > "$BACKUP_FILE"

echo "[$(date)] Backup saved to $BACKUP_FILE ($(du -h "$BACKUP_FILE" | cut -f1))"

# Remove backups older than RETAIN_DAYS
find "$BACKUP_DIR" -name "happytorch_*.sql.gz" -mtime +$RETAIN_DAYS -delete
echo "[$(date)] Cleaned up backups older than ${RETAIN_DAYS} days"
