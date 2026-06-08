#!/bin/bash
# Railway build script.
# CmdStan is installed to /app/.cmdstan/ so Railpack's final
# "copy /app" step includes it in the runtime image.
set -euo pipefail

CMDSTAN_VERSION="2.39.0"
CMDSTAN_URL="https://github.com/stan-dev/cmdstan/releases/download/v${CMDSTAN_VERSION}/cmdstan-${CMDSTAN_VERSION}.tar.gz"
CMDSTAN_DIR="/app/.cmdstan/cmdstan-${CMDSTAN_VERSION}"

echo "=== [1/3] Python dependencies ==="
uv sync --locked --no-editable

echo "=== [2/3] Install CmdStan ${CMDSTAN_VERSION} → /app/.cmdstan/ ==="
if [ -d "${CMDSTAN_DIR}" ] && [ -f "${CMDSTAN_DIR}/bin/stanc" ]; then
    echo "[david] CmdStan already at ${CMDSTAN_DIR} — skipping"
else
    # Ensure make + g++ are available (apt-get update first — empty cache otherwise)
    if ! command -v make &>/dev/null; then
        echo "[david] Installing build tools via apt…"
        apt-get update -qq && apt-get install -y make g++ libstdc++-12-dev
    fi

    echo "[david] Downloading ${CMDSTAN_URL}…"
    curl -fsSL "${CMDSTAN_URL}" -o /tmp/cmdstan.tar.gz

    mkdir -p "/app/.cmdstan"
    tar -xzf /tmp/cmdstan.tar.gz -C "/app/.cmdstan"
    rm -f /tmp/cmdstan.tar.gz

    echo "[david] Building CmdStan (make build — ~8 min)…"
    make -C "${CMDSTAN_DIR}" build
    echo "[david] CmdStan built at ${CMDSTAN_DIR}"
fi

# Register path with cmdstanpy
uv run python -c "
import cmdstanpy
cmdstanpy.set_cmdstan_path('${CMDSTAN_DIR}')
print('[david] CmdStan path set:', '${CMDSTAN_DIR}')
"

echo "=== [3/3] Pre-compile Stan model ==="
CMDSTAN="${CMDSTAN_DIR}" uv run python -c "
from cmdstanpy import CmdStanModel
import pathlib
stan_file = pathlib.Path('/app/stan/m01_forward.stan')
if not stan_file.exists():
    raise FileNotFoundError(f'Stan file not found: {stan_file}')
model = CmdStanModel(stan_file=str(stan_file))
print(f'[david] Stan binary compiled: {model.exe_file}')
"

echo "=== Build complete ==="
