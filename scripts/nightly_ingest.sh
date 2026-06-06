#!/usr/bin/env bash
# nightly_ingest.sh — automated evidence pull + LLM coding + queue rebuild
#
# Cron: 0 3 * * *  /path/to/david/canonical/scripts/nightly_ingest.sh
#
# Human interaction: NONE during the run. Operator reviews the rebuilt
# adjudicator_queue.json the next morning.

set -euo pipefail

cd "$(dirname "$0")/.."

LOG_DIR="${DAVID_LOG_DIR:-./data/logs}"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$LOG_DIR/nightly_ingest_$STAMP.log"

{
    echo "=== nightly_ingest $STAMP ==="
    david ingest
    david calibrate-coders
    echo "=== done ==="
} 2>&1 | tee "$LOG"
