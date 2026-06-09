# ─────────────────────────────────────────────────────────────────────────────
# DAVID / M0.1  —  Makefile
#
# Commands:
#   make dev         Start everything (Postgres + FastAPI + UI)
#   make api         Start Postgres + FastAPI only (no UI)
#   make stop        Kill FastAPI + Next.js (keeps Postgres)
#   make stop-all    Kill everything including Postgres
#   make reset       Wipe DB + restart from scratch
#   make logs        Tail all service logs
#   make status      Show what's running on each port
#   make ingest      Run one ingest cycle (scrape → normalize → code → adjudicate)
#   make fit         Run HSMM fit + forecast
#   make sbc         Run measurement SBC
#   make falsify     Run F1–F15 falsification battery
#   make test        Run test suite
#   make ui-install  Install Next.js dependencies
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: dev dev-local api stop stop-all reset logs status ingest fit sbc falsify test ui-install schedule unschedule help

SHELL := /bin/bash

LOG_DIR   := $(CURDIR)/.david_logs
SINCE     ?= 2024-01-01
UNTIL     ?= $(shell date +%Y-%m-%d)
HORIZONS  ?= 6

# ── Startup ───────────────────────────────────────────────────────────────────

dev:
	@chmod +x start.sh stop.sh
	@./start.sh

# Local M4 research loop: dedicated Postgres on :5544 + FastAPI + UI.
# Pass flags through, e.g.  make dev-local ARGS="--seed"  or  make dev-local ARGS="--no-ui"
dev-local:
	@chmod +x scripts/dev-local.sh
	@./scripts/dev-local.sh $(ARGS)

api:
	@chmod +x start.sh stop.sh
	@./start.sh --no-ui

reset:
	@chmod +x start.sh stop.sh
	@./start.sh --reset

# ── Shutdown ──────────────────────────────────────────────────────────────────

stop:
	@chmod +x stop.sh
	@./stop.sh

stop-all:
	@chmod +x stop.sh
	@./stop.sh --all

# ── Logs ──────────────────────────────────────────────────────────────────────

logs:
	@mkdir -p $(LOG_DIR)
	@echo "=== FastAPI ==============================================" && \
	 tail -n 30 $(LOG_DIR)/fastapi.log 2>/dev/null || echo "(no log yet)" && \
	 echo "" && \
	 echo "=== Next.js =============================================" && \
	 tail -n 30 $(LOG_DIR)/nextjs.log  2>/dev/null || echo "(no log yet)" && \
	 echo "" && \
	 echo "=== DB init =============================================" && \
	 tail -n 10 $(LOG_DIR)/db_init.log 2>/dev/null || echo "(no log yet)"

# ── Status ────────────────────────────────────────────────────────────────────

status:
	@echo "=== Port status ========================================="
	@lsof -i :5544 -s TCP:LISTEN 2>/dev/null | grep -q LISTEN && echo "✓ Postgres  :5544" || echo "✗ Postgres  :5544  (not running)"
	@lsof -i :8080 -s TCP:LISTEN 2>/dev/null | grep -q LISTEN && echo "✓ FastAPI   :8080" || echo "✗ FastAPI   :8080  (not running)"
	@lsof -i :3001 -s TCP:LISTEN 2>/dev/null | grep -q LISTEN && echo "✓ Next.js   :3001" || echo "✗ Next.js   :3001  (not running)"
	@lsof -i :3000 -s TCP:LISTEN 2>/dev/null | grep -q LISTEN && echo "✓ Next.js   :3000" || true
	@echo ""
	@echo "=== Docker containers ==================================="
	@docker compose ps 2>/dev/null || echo "(Docker not running)"

# ── Pipeline ──────────────────────────────────────────────────────────────────

ingest:
	@echo "==> Ingesting evidence ($(SINCE) – $(UNTIL))"
	uv run david ingest --since $(SINCE) --until $(UNTIL)

fit:
	@echo "==> Running HSMM fit + forecast (horizon=$(HORIZONS)m)"
	uv run david fit
	uv run david forecast --horizon $(HORIZONS)

sbc:
	@echo "==> Running SBC (measurement)"
	uv run david sbc
	@echo "==> Running SBC (forecast)"
	uv run david sbc --forecast

falsify:
	@echo "==> Running falsification battery F1–F15"
	uv run david falsify

# ── Development ───────────────────────────────────────────────────────────────

test:
	@echo "==> Running test suite"
	uv run pytest tests/ -v --tb=short

ui-install:
	@cd david-ui && npm install

# ── DB helpers ────────────────────────────────────────────────────────────────

# ── Scheduling (macOS launchd — replaces Railway cron) ───────────────────────

schedule:
	@chmod +x scripts/launchd/install.sh
	@./scripts/launchd/install.sh

unschedule:
	@chmod +x scripts/launchd/install.sh
	@./scripts/launchd/install.sh uninstall

# ── DB helpers ────────────────────────────────────────────────────────────────

db-init:
	uv run david db init

db-shell:
	docker compose exec postgres psql -U david -d david

# ── Help ──────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "  DAVID / M0.1  —  available targets"
	@echo ""
	@echo "  make dev          Start all services (Postgres + FastAPI :8080 + UI :3001)"
	@echo "  make api          Start Postgres + FastAPI only"
	@echo "  make reset        Wipe Postgres volume + restart"
	@echo "  make stop         Stop FastAPI + Next.js  (Postgres keeps running)"
	@echo "  make stop-all     Stop everything including Postgres"
	@echo ""
	@echo "  make status       Show which ports are listening"
	@echo "  make logs         Tail all service logs"
	@echo ""
	@echo "  make ingest       Scrape → normalize → LLM code → adjudicate"
	@echo "  make fit          HSMM fit + forecast"
	@echo "  make sbc          Simulation-Based Calibration"
	@echo "  make falsify      F1–F15 falsification battery"
	@echo ""
	@echo "  make test         Run pytest"
	@echo "  make db-shell     Open psql shell"
	@echo ""
	@echo "  Variables (override on command line):"
	@echo "    SINCE=2024-01-01   UNTIL=today   HORIZONS=6"
	@echo ""
