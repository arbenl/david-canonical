#!/usr/bin/env bash
# weekly_fit.sh — weekly fit, SBC, falsification, forecast, routing.
#
# Cron: 0 4 * * 1  /path/to/david/canonical/scripts/weekly_fit.sh
#
# B-8 strict order: sbc → fit → forecast_sbc → falsify → forecast → route
# SBC validates the inference algorithm first; a failed SBC exits immediately
# so no real-data fit is wasted. After a successful SBC run, `david falsify`
# records sbc_block in the falsification ledger, and `david forecast` blocks
# fail-closed when that ledger explicitly records sbc_passed=False.
#
# Fails closed at every step. Exit code 2 from any step short-circuits the rest.

set -euo pipefail

cd "$(dirname "$0")/.."

LOG_DIR="${DAVID_LOG_DIR:-./data/logs}"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$LOG_DIR/weekly_fit_$STAMP.log"

run_step() {
    local step_name=$1; shift
    echo "--- $step_name ---"
    if ! "$@"; then
        echo "STEP FAILED: $step_name"
        exit 2
    fi
}

{
    echo "=== weekly_fit $STAMP ==="
    run_step sbc            david sbc
    run_step fit            david fit
    run_step forecast_sbc   david sbc --forecast
    run_step falsify        david falsify
    for h in 3 6 9 12; do
        run_step "forecast_h$h" david forecast --horizon "$h"
    done
    run_step route          david route
    echo "=== done ==="
} 2>&1 | tee "$LOG"
