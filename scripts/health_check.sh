#!/usr/bin/env bash
# =============================================================================
# atlas-trader — Standalone health check
# Exit 0 if the trader container is healthy, 1 otherwise.
# Usage: bash scripts/health_check.sh [host] [port]
# =============================================================================
HOST="${1:-localhost}"
PORT="${2:-8080}"
URL="http://${HOST}:${PORT}/health"

curl -sf "${URL}" | python3 -m json.tool
