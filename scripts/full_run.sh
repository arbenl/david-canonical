#!/usr/bin/env bash
# full_run.sh — single end-to-end pass (smoke test / first install)

set -euo pipefail
cd "$(dirname "$0")/.."

./scripts/nightly_ingest.sh
./scripts/weekly_fit.sh
