#!/usr/bin/env bash
# Install DAVID launchd jobs on macOS (replaces Railway cron).
# Run once; jobs persist across reboots.
#
# Usage:
#   ./scripts/launchd/install.sh          # install both jobs
#   ./scripts/launchd/install.sh uninstall  # remove both jobs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"

JOBS=(
  "com.david.nightly-ingest"
  "com.david.weekly-fit"
)

if [[ "${1:-}" == "uninstall" ]]; then
  for job in "${JOBS[@]}"; do
    launchctl unload "$LAUNCH_AGENTS_DIR/$job.plist" 2>/dev/null || true
    rm -f "$LAUNCH_AGENTS_DIR/$job.plist"
    echo "Uninstalled $job"
  done
  echo "Done."
  exit 0
fi

mkdir -p "$LAUNCH_AGENTS_DIR"
mkdir -p "$(cd "$SCRIPT_DIR/../.." && pwd)/data/logs"

for job in "${JOBS[@]}"; do
  src="$SCRIPT_DIR/$job.plist"
  dst="$LAUNCH_AGENTS_DIR/$job.plist"
  cp "$src" "$dst"
  launchctl unload "$dst" 2>/dev/null || true
  launchctl load "$dst"
  echo "Installed $job"
done

echo ""
echo "Scheduled jobs:"
echo "  com.david.nightly-ingest  — daily 03:00 (scrape + code + adjudicate)"
echo "  com.david.weekly-fit      — Monday 04:00 (fit + SBC + forecast + route)"
echo ""
echo "Logs: data/logs/launchd_*.log"
echo "To uninstall: ./scripts/launchd/install.sh uninstall"
