#!/usr/bin/env bash
# =============================================================================
# atlas-trader — Linux production deployment script
# Usage: bash scripts/deploy.sh
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'  # no colour

info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
die()     { error "$*"; exit 1; }

# ---------------------------------------------------------------------------
# Resolve project root (directory containing this script's parent)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

info "Project root: ${PROJECT_ROOT}"
cd "${PROJECT_ROOT}"

# ---------------------------------------------------------------------------
# 1. Prerequisite checks
# ---------------------------------------------------------------------------
info "Checking prerequisites..."

command -v docker >/dev/null 2>&1 \
    || die "docker not found. Install Docker Engine: https://docs.docker.com/engine/install/"

# Accept both 'docker-compose' (v1) and 'docker compose' (v2 plugin)
if command -v docker-compose >/dev/null 2>&1; then
    DC="docker-compose"
elif docker compose version >/dev/null 2>&1; then
    DC="docker compose"
else
    die "docker-compose not found. Install it or upgrade Docker Desktop."
fi
info "Using compose command: ${DC}"

[[ -f ".env" ]] \
    || die ".env file not found. Copy .env.example to .env and fill in your values:\n  cp .env.example .env"

# ---------------------------------------------------------------------------
# 2. Required environment variable validation
# ---------------------------------------------------------------------------
info "Validating required environment variables..."

# shellcheck source=/dev/null
set -a; source .env; set +a

REQUIRED_VARS=(ANTHROPIC_API_KEY ALPACA_API_KEY)
MISSING=()

for var in "${REQUIRED_VARS[@]}"; do
    val="${!var:-}"
    if [[ -z "${val}" || "${val}" == your_* || "${val}" == sk-ant-...* ]]; then
        MISSING+=("${var}")
    fi
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
    error "The following required variables are not set in .env:"
    for v in "${MISSING[@]}"; do
        error "  - ${v}"
    done
    die "Fix .env before deploying."
fi

info "All required environment variables are present."

# ---------------------------------------------------------------------------
# 3. Pull latest code
# ---------------------------------------------------------------------------
info "Pulling latest code from remote..."
if git pull --ff-only; then
    info "Code updated successfully."
else
    warn "git pull failed (maybe no remote, or local changes). Continuing with current code."
fi

# ---------------------------------------------------------------------------
# 4. Build Docker images (no cache for reproducible production builds)
# ---------------------------------------------------------------------------
info "Building Docker images (--no-cache)..."
${DC} build --no-cache
info "Build complete."

# ---------------------------------------------------------------------------
# 5. Stop existing containers gracefully
# ---------------------------------------------------------------------------
info "Stopping existing containers (if any)..."
${DC} down --remove-orphans || true
info "Existing containers stopped."

# ---------------------------------------------------------------------------
# 6. Start in production mode (detached)
# ---------------------------------------------------------------------------
info "Starting atlas-trader in production mode..."
${DC} up -d
info "Containers started."

# ---------------------------------------------------------------------------
# 7. Health check — wait up to 30 s
# ---------------------------------------------------------------------------
HEALTH_URL="http://localhost:8080/health"
MAX_WAIT=30
INTERVAL=3
elapsed=0
healthy=false

info "Waiting for health check at ${HEALTH_URL} (timeout: ${MAX_WAIT}s)..."

while [[ ${elapsed} -lt ${MAX_WAIT} ]]; do
    if curl -sf "${HEALTH_URL}" >/dev/null 2>&1; then
        healthy=true
        break
    fi
    sleep ${INTERVAL}
    elapsed=$((elapsed + INTERVAL))
    info "  Still waiting... (${elapsed}s elapsed)"
done

# ---------------------------------------------------------------------------
# 8. Show recent logs regardless of health outcome
# ---------------------------------------------------------------------------
info "Recent container logs:"
echo "----------------------------------------------------------------------"
${DC} logs --tail=20
echo "----------------------------------------------------------------------"

# ---------------------------------------------------------------------------
# 9. Final status
# ---------------------------------------------------------------------------
if [[ "${healthy}" == true ]]; then
    info "Health check PASSED after ~${elapsed}s."
    echo ""
    echo -e "${GREEN}============================================================${NC}"
    echo -e "${GREEN}  atlas-trader deployed successfully!${NC}"
    echo -e "${GREEN}  Trader API : http://localhost:8080${NC}"
    echo -e "${GREEN}  Dashboard  : http://localhost:8501${NC}"
    echo -e "${GREEN}============================================================${NC}"
    exit 0
else
    error "Health check FAILED after ${MAX_WAIT}s."
    echo ""
    echo -e "${RED}============================================================${NC}"
    echo -e "${RED}  Deployment may have issues. Check the logs above.${NC}"
    echo -e "${RED}  Run: ${DC} logs -f trader${NC}"
    echo -e "${RED}============================================================${NC}"
    exit 1
fi
