#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== HappyTorch Production Deploy ==="

# Check .env file
if [ ! -f .env ]; then
    echo "Error: .env file not found. Copy .env.prod.example to .env and configure it."
    exit 1
fi

# Source env for domain variable
set -a
source .env
set +a

# Build frontend
echo "[1/4] Building frontend..."
cd frontend
if [ -f pnpm-lock.yaml ]; then
    pnpm install --frozen-lockfile
elif [ -f package-lock.json ]; then
    npm ci
else
    npm install
fi
npm run build
cd "$PROJECT_ROOT"

# Build and start services
echo "[2/4] Building Docker images..."
docker compose -f docker-compose.prod.yml build

echo "[3/4] Starting services..."
docker compose -f docker-compose.prod.yml up -d

# Wait for postgres to be healthy
echo "[4/4] Waiting for services to be healthy..."
timeout=60
elapsed=0
until docker compose -f docker-compose.prod.yml exec -T postgres pg_isready -U "${DB_USER:-happytorch}" > /dev/null 2>&1; do
    if [ "$elapsed" -ge "$timeout" ]; then
        echo "Error: PostgreSQL did not become ready within ${timeout}s"
        exit 1
    fi
    sleep 2
    elapsed=$((elapsed + 2))
done

# Run seed if requested
if [ "${1:-}" = "--seed" ]; then
    echo "Running seed script..."
    docker compose -f docker-compose.prod.yml exec -T backend /usr/local/bin/app seed 2>/dev/null || \
        echo "Note: seed command not available in backend binary, run manually if needed."
fi

echo ""
echo "=== Deploy complete ==="
echo "Services running:"
docker compose -f docker-compose.prod.yml ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"
echo ""

# SSL setup hint
if [ ! -d certbot/conf/live ]; then
    echo "=== SSL Setup Required ==="
    echo "Run the following to obtain an SSL certificate:"
    echo ""
    echo "  # First, temporarily comment out the HTTPS server block in nginx/conf.d/default.conf"
    echo "  # Then restart nginx, and run:"
    echo "  docker compose -f docker-compose.prod.yml run --rm certbot certonly \\"
    echo "    --webroot --webroot-path=/var/www/certbot \\"
    echo "    -d ${DOMAIN:-yourdomain.com} --email your@email.com --agree-tos --no-eff-email"
    echo ""
    echo "  # After obtaining the cert, restore the HTTPS block and restart nginx:"
    echo "  docker compose -f docker-compose.prod.yml restart nginx"
fi
