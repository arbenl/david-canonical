#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# DAVID / M0.1 — stop script
# Kills FastAPI + Next.js; leaves Postgres running.
# Use --all to also stop Postgres.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/.david_pids"

G='\033[0;32m'; Y='\033[1;33m'; R='\033[0;31m'; N='\033[0m'
info()  { echo -e "${G}▸ DAVID${N}  $*"; }
warn()  { echo -e "${Y}▸ DAVID${N}  $*"; }

ALL=false
for arg in "$@"; do
  [ "$arg" = "--all" ] && ALL=true
done

# Kill from PID file (if start.sh is still running, this is a no-op — Ctrl+C handles it)
if [ -f "$PID_FILE" ]; then
  read -r FASTAPI_PID NEXTJS_PID < "$PID_FILE" 2>/dev/null || true
  [ -n "${FASTAPI_PID:-}" ] && kill "$FASTAPI_PID" 2>/dev/null && info "FastAPI stopped (PID $FASTAPI_PID)" || true
  [ -n "${NEXTJS_PID:-}"  ] && kill "$NEXTJS_PID"  2>/dev/null && info "Next.js stopped  (PID $NEXTJS_PID)"  || true
  rm -f "$PID_FILE"
fi

# Fallback: kill by port if PID file was absent (e.g. after a crash)
for PORT in 8080 3001; do
  PIDS=$(lsof -ti ":$PORT" 2>/dev/null || true)
  if [ -n "$PIDS" ]; then
    echo "$PIDS" | xargs kill 2>/dev/null || true
    info "Killed process on :$PORT"
  fi
done

if $ALL; then
  info "Stopping Postgres container..."
  cd "$SCRIPT_DIR" && docker compose stop postgres
  info "Postgres stopped."
else
  warn "Postgres is still running. Use './stop.sh --all' or 'docker compose stop' to halt it."
fi

info "Done."
