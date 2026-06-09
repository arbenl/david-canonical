#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# DAVID / M0.1 — local M4 development loop
#
# The research / SBC iteration loop runs best locally: MCMC fits are CPU+RAM
# bound and the Mac M4 is far faster (and free) versus Railway's shared compute.
# Railway stays the production deploy target that Vercel queries.
#
# Starts:  Postgres (Docker, :5544) → FastAPI (:8080) → Next.js UI (:3001)
#
# Usage:   ./scripts/dev-local.sh              full stack
#          ./scripts/dev-local.sh --no-ui      Postgres + FastAPI only
#          ./scripts/dev-local.sh --reset      recreate the local DB container
#          ./scripts/dev-local.sh --seed       seed a certified demo fit fixture
#
# API-URL wiring (do not hardcode in git): the frontend resolves the backend via
# DAVID_API_URL with a localhost fallback. This script exports it at RUNTIME only
# (http://localhost:8080). In production Vercel sets DAVID_API_URL → Railway, so
# nothing localhost ever lands in a committed file.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$ROOT/.david_pids"
LOG_DIR="$ROOT/.david_logs"
mkdir -p "$LOG_DIR"

# ── tunables (env-overridable) ───────────────────────────────────────────────
PG_CONTAINER="${DAVID_PG_CONTAINER:-david_pg_local}"
PG_PORT="${DAVID_PG_PORT:-5544}"
API_PORT="${DAVID_API_PORT:-8080}"
UI_PORT="${DAVID_UI_PORT:-3001}"

# ── runtime env (NEVER written to a committed file) ──────────────────────────
export DATABASE_URL="postgresql://david:david@localhost:${PG_PORT}/david"
export DAVID_API_URL="http://localhost:${API_PORT}"
export ALLOWED_ORIGINS="http://localhost:${UI_PORT},http://localhost:3000"

# ── flags ────────────────────────────────────────────────────────────────────
NO_UI=false; RESET=false; SEED=false
for arg in "$@"; do
  case $arg in
    --no-ui) NO_UI=true ;;
    --reset) RESET=true ;;
    --seed)  SEED=true ;;
    --help)
      sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "unknown flag: $arg (try --help)"; exit 1 ;;
  esac
done

# ── colours / logging ─────────────────────────────────────────────────────────
G='\033[0;32m'; Y='\033[1;33m'; B='\033[0;34m'; R='\033[0;31m'; N='\033[0m'
info()  { echo -e "${G}▸ DAVID${N}  $*"; }
warn()  { echo -e "${Y}▸ DAVID${N}  $*"; }
error() { echo -e "${R}▸ DAVID${N}  $*" >&2; }
step()  { echo -e "\n${B}── $* ──${N}"; }

# ── runner: prefer uv, fall back to the project venv ──────────────────────────
if command -v uv >/dev/null 2>&1; then
  RUN=(uv run)
elif [ -x "$ROOT/.venv/bin/python" ]; then
  RUN=("$ROOT/.venv/bin/python" -m)
else
  error "Neither 'uv' nor .venv/bin/python found. Install deps first (uv sync)."
  exit 1
fi

# ── cleanup ───────────────────────────────────────────────────────────────────
FASTAPI_PID=""; NEXTJS_PID=""
cleanup() {
  echo ""
  info "Shutting down..."
  [ -n "$FASTAPI_PID" ] && kill "$FASTAPI_PID" 2>/dev/null && info "FastAPI stopped" || true
  [ -n "$NEXTJS_PID"  ] && kill "$NEXTJS_PID"  2>/dev/null && info "Next.js stopped"  || true
  rm -f "$PID_FILE"
  local pg="${ACTIVE_PG:-$PG_CONTAINER}"
  info "Done. Postgres container '$pg' keeps running (docker stop $pg to halt)."
}
trap cleanup INT TERM EXIT

cd "$ROOT"

# ─────────────────────────────────────────────────────────────────────────────
step "1 / 4  Postgres  :$PG_PORT"
# ─────────────────────────────────────────────────────────────────────────────
if ! docker info >/dev/null 2>&1; then
  error "Docker is not running. Start Docker Desktop first."
  exit 1
fi

if $RESET; then
  warn "Reset requested — removing container '$PG_CONTAINER'..."
  docker rm -f "$PG_CONTAINER" >/dev/null 2>&1 || true
fi

ACTIVE_PG="$PG_CONTAINER"
if nc -z localhost "$PG_PORT" >/dev/null 2>&1; then
  EXISTING="$(docker ps --filter "publish=$PG_PORT" --format '{{.Names}}' | head -1)"
  ACTIVE_PG="${EXISTING:-external}"
  info "Postgres already serving on :$PG_PORT ($ACTIVE_PG) — reusing"
elif docker ps -a --format '{{.Names}}' | grep -qx "$PG_CONTAINER"; then
  info "Starting existing container '$PG_CONTAINER'..."
  docker start "$PG_CONTAINER" >/dev/null
elif docker ps --format '{{.Names}}' | grep -q . && \
     docker ps -a --format '{{.Ports}} {{.Names}}' | grep -q "$PG_PORT->"; then
  warn "Port $PG_PORT mapped by another container; reusing it."
else
  info "Creating Postgres container '$PG_CONTAINER' on :$PG_PORT..."
  docker run -d --name "$PG_CONTAINER" \
    -e POSTGRES_USER=david -e POSTGRES_PASSWORD=david -e POSTGRES_DB=david \
    -p "${PG_PORT}:5432" postgres:16 >/dev/null
fi

info "Waiting for Postgres to accept connections..."
TRIES=0
until nc -z localhost "$PG_PORT" >/dev/null 2>&1; do
  TRIES=$((TRIES+1)); [ $TRIES -ge 30 ] && { error "Postgres not ready on :$PG_PORT"; exit 1; }
  sleep 1
done
info "Postgres ready  →  localhost:$PG_PORT (db: david)"

# ─────────────────────────────────────────────────────────────────────────────
step "2 / 4  DB schema"
# ─────────────────────────────────────────────────────────────────────────────
if "${RUN[@]}" david db init 2>&1 | tee -a "$LOG_DIR/db_init.log" | grep -q "Schema created"; then
  info "Schema created"
else
  info "Schema already present (idempotent)"
fi

if $SEED; then
  info "Seeding certified demo fit fixture..."
  "${RUN[@]}" python scripts/seed_verify.py certified >>"$LOG_DIR/db_init.log" 2>&1 \
    && info "Seed complete (run: fit_verify_certified)" \
    || warn "Seed failed (non-fatal) — see $LOG_DIR/db_init.log"
fi

# ─────────────────────────────────────────────────────────────────────────────
step "3 / 4  FastAPI  :$API_PORT"
# ─────────────────────────────────────────────────────────────────────────────
if lsof -ti ":$API_PORT" >/dev/null 2>&1; then
  error "Port $API_PORT is already in use. Stop the other process (make stop) and retry."
  exit 1
fi

info "Starting FastAPI (DATABASE_URL → :$PG_PORT)..."
"${RUN[@]}" uvicorn david.api.server:api \
  --port "$API_PORT" --reload --log-level warning \
  > "$LOG_DIR/fastapi.log" 2>&1 &
FASTAPI_PID=$!

info "Waiting for FastAPI /healthz..."
TRIES=0
until curl -sf "http://localhost:$API_PORT/healthz" >/dev/null 2>&1; do
  TRIES=$((TRIES+1))
  if ! kill -0 "$FASTAPI_PID" 2>/dev/null; then
    error "FastAPI died. Last log lines:"; tail -20 "$LOG_DIR/fastapi.log"; exit 1
  fi
  [ $TRIES -ge 30 ] && { error "FastAPI not ready. See $LOG_DIR/fastapi.log"; tail -20 "$LOG_DIR/fastapi.log"; exit 1; }
  sleep 1
done
info "FastAPI ready  →  http://localhost:$API_PORT/docs"

# ─────────────────────────────────────────────────────────────────────────────
step "4 / 4  Next.js UI  :$UI_PORT"
# ─────────────────────────────────────────────────────────────────────────────
if $NO_UI; then
  warn "Skipping Next.js (--no-ui)"
else
  UI_DIR="$ROOT/david-ui"
  [ -d "$UI_DIR/node_modules" ] || { info "Installing UI deps (first run)..."; (cd "$UI_DIR" && npm install --silent); }
  info "Starting Next.js (DAVID_API_URL → $DAVID_API_URL)..."
  # DAVID_API_URL is already exported above → inherited by next.config rewrites
  # and server components. The browser stays same-origin via the /api rewrite.
  (cd "$UI_DIR" && npm run dev > "$LOG_DIR/nextjs.log" 2>&1) &
  NEXTJS_PID=$!
  info "Waiting for Next.js to compile..."
  TRIES=0
  until curl -sf "http://localhost:$UI_PORT" >/dev/null 2>&1; do
    TRIES=$((TRIES+1))
    if ! kill -0 "$NEXTJS_PID" 2>/dev/null; then
      error "Next.js died. Last log lines:"; tail -20 "$LOG_DIR/nextjs.log"; break
    fi
    [ $TRIES -ge 60 ] && { warn "Next.js still compiling — check $LOG_DIR/nextjs.log"; break; }
    sleep 1
  done
  info "Next.js ready  →  http://localhost:$UI_PORT"
fi

echo "$FASTAPI_PID ${NEXTJS_PID:-}" > "$PID_FILE"

echo ""
echo -e "${G}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${N}"
echo -e "  ${G}DAVID / M0.1 — local M4 dev loop${N}"
echo ""
echo -e "  Dashboard   →  ${B}http://localhost:$UI_PORT/router${N}"
echo -e "  FastAPI     →  ${B}http://localhost:$API_PORT/docs${N}"
echo -e "  Postgres    →  ${B}localhost:$PG_PORT${N}  (container: $ACTIVE_PG)"
echo -e "  Logs        →  ${B}$LOG_DIR/${N}"
echo -e "${G}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${N}"
echo ""
echo "  Ctrl+C stops FastAPI + Next.js (Postgres keeps running)."
echo ""

wait "$FASTAPI_PID" 2>/dev/null || true
