#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# DAVID / M0.1 — unified startup script
#
# Starts: Postgres (Docker) → FastAPI (:8080) → Next.js UI (:3001)
# Usage:  ./start.sh
#         ./start.sh --no-ui      (skip Next.js — API only)
#         ./start.sh --reset      (drop DB volume first — fresh start)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/.david_pids"
LOG_DIR="$SCRIPT_DIR/.david_logs"
mkdir -p "$LOG_DIR"

# ── flags ────────────────────────────────────────────────────────────────────
NO_UI=false
RESET=false
for arg in "$@"; do
  case $arg in
    --no-ui)  NO_UI=true  ;;
    --reset)  RESET=true  ;;
    --help)
      echo "Usage: $0 [--no-ui] [--reset]"
      echo "  --no-ui   Start Postgres + FastAPI only (skip Next.js)"
      echo "  --reset   Drop the Postgres volume before starting (fresh DB)"
      exit 0 ;;
  esac
done

# ── colours ──────────────────────────────────────────────────────────────────
G='\033[0;32m'; Y='\033[1;33m'; B='\033[0;34m'; R='\033[0;31m'; N='\033[0m'
info()  { echo -e "${G}▸ DAVID${N}  $*"; }
warn()  { echo -e "${Y}▸ DAVID${N}  $*"; }
error() { echo -e "${R}▸ DAVID${N}  $*" >&2; }
step()  { echo -e "\n${B}── $* ──${N}"; }

# ── cleanup on exit ───────────────────────────────────────────────────────────
FASTAPI_PID=""
NEXTJS_PID=""

cleanup() {
  echo ""
  info "Shutting down..."
  [ -n "$FASTAPI_PID" ] && kill "$FASTAPI_PID" 2>/dev/null && info "FastAPI stopped"
  [ -n "$NEXTJS_PID"  ] && kill "$NEXTJS_PID"  2>/dev/null && info "Next.js stopped"
  rm -f "$PID_FILE"
  info "Done. Postgres keeps running (docker compose stop to halt it)."
}
trap cleanup INT TERM EXIT

# ─────────────────────────────────────────────────────────────────────────────
step "1 / 4  Postgres"
# ─────────────────────────────────────────────────────────────────────────────

cd "$SCRIPT_DIR"

if $RESET; then
  warn "Reset requested — removing pgdata volume..."
  docker compose down -v 2>/dev/null || true
fi

if ! docker info >/dev/null 2>&1; then
  error "Docker is not running. Start Docker Desktop first."
  exit 1
fi

info "Starting Postgres container..."
docker compose up -d postgres

info "Waiting for Postgres to accept connections..."
TRIES=0
until docker compose exec -T postgres pg_isready -U david -q 2>/dev/null; do
  TRIES=$((TRIES+1))
  if [ $TRIES -ge 30 ]; then
    error "Postgres did not become ready after 30s."
    exit 1
  fi
  sleep 1
done
info "Postgres ready (port 5432)"

# ─────────────────────────────────────────────────────────────────────────────
step "2 / 4  DB schema"
# ─────────────────────────────────────────────────────────────────────────────

if uv run david db init 2>&1 | tee -a "$LOG_DIR/db_init.log" | grep -q "Schema created"; then
  info "Schema created"
else
  info "Schema already exists (idempotent)"
fi

# ─────────────────────────────────────────────────────────────────────────────
step "3 / 4  FastAPI  :8080"
# ─────────────────────────────────────────────────────────────────────────────

info "Starting FastAPI..."
uv run uvicorn david.api.server:api \
  --port 8080 \
  --reload \
  --log-level warning \
  > "$LOG_DIR/fastapi.log" 2>&1 &
FASTAPI_PID=$!

# Wait until /healthz responds
info "Waiting for FastAPI to be ready..."
TRIES=0
until curl -sf http://localhost:8080/healthz >/dev/null 2>&1; do
  TRIES=$((TRIES+1))
  if [ $TRIES -ge 30 ]; then
    error "FastAPI did not start. Check: $LOG_DIR/fastapi.log"
    cat "$LOG_DIR/fastapi.log" | tail -20
    exit 1
  fi
  # Check it didn't crash
  if ! kill -0 "$FASTAPI_PID" 2>/dev/null; then
    error "FastAPI process died. Check: $LOG_DIR/fastapi.log"
    cat "$LOG_DIR/fastapi.log" | tail -20
    exit 1
  fi
  sleep 1
done
info "FastAPI ready  →  http://localhost:8080/docs"

# ─────────────────────────────────────────────────────────────────────────────
step "4 / 4  Next.js UI  :3001"
# ─────────────────────────────────────────────────────────────────────────────

if $NO_UI; then
  warn "Skipping Next.js (--no-ui)"
else
  UI_DIR="$SCRIPT_DIR/david-ui"
  if [ ! -d "$UI_DIR/node_modules" ]; then
    info "Installing Next.js dependencies (first run)..."
    cd "$UI_DIR" && npm install --silent && cd "$SCRIPT_DIR"
  fi

  info "Starting Next.js..."
  cd "$UI_DIR"
  npm run dev \
    > "$LOG_DIR/nextjs.log" 2>&1 &
  NEXTJS_PID=$!
  cd "$SCRIPT_DIR"

  # Wait for Next.js compiler
  info "Waiting for Next.js to compile..."
  TRIES=0
  until curl -sf http://localhost:3001 >/dev/null 2>&1; do
    TRIES=$((TRIES+1))
    if [ $TRIES -ge 60 ]; then
      warn "Next.js is taking longer than usual. It may still be compiling."
      warn "Check: $LOG_DIR/nextjs.log"
      break
    fi
    if ! kill -0 "$NEXTJS_PID" 2>/dev/null; then
      error "Next.js process died. Check: $LOG_DIR/nextjs.log"
      cat "$LOG_DIR/nextjs.log" | tail -20
      break
    fi
    sleep 1
  done
  info "Next.js ready  →  http://localhost:3001"
fi

# Save PIDs for stop.sh
echo "$FASTAPI_PID ${NEXTJS_PID:-}" > "$PID_FILE"

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${G}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${N}"
echo -e "  ${G}DAVID / M0.1 is running${N}"
echo ""
echo -e "  Dashboard   →  ${B}http://localhost:3001${N}"
echo -e "  FastAPI     →  ${B}http://localhost:8080/docs${N}"
echo -e "  Postgres    →  ${B}localhost:5432${N}  (user: david)"
echo ""
echo -e "  Logs        →  ${B}$LOG_DIR/${N}"
echo -e "${G}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${N}"
echo ""
echo "  Press Ctrl+C to stop FastAPI + Next.js"
echo "  (Postgres keeps running until: docker compose stop)"
echo ""
# ─────────────────────────────────────────────────────────────────────────────

# Block until a child exits
wait $FASTAPI_PID 2>/dev/null || true
