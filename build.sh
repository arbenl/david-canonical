#!/bin/bash
# Railway build script.
# aptPkgs in nixpacks.toml ensures make + g++ are already present.
set -euo pipefail

CMDSTAN_VERSION="2.39.0"
CMDSTAN_URL="https://github.com/stan-dev/cmdstan/releases/download/v${CMDSTAN_VERSION}/cmdstan-${CMDSTAN_VERSION}.tar.gz"
CMDSTAN_DIR="${HOME}/.cmdstan/cmdstan-${CMDSTAN_VERSION}"

echo "=== [1/3] Python dependencies ==="
uv sync --locked --no-editable

echo "=== [2/3] Install CmdStan ${CMDSTAN_VERSION} (direct tarball — no GitHub API) ==="
if [ -d "${CMDSTAN_DIR}" ]; then
    echo "[david] CmdStan already at ${CMDSTAN_DIR} — skipping download"
else
    echo "[david] Downloading ${CMDSTAN_URL}…"
    curl -fsSL "${CMDSTAN_URL}" -o /tmp/cmdstan.tar.gz
    mkdir -p "${HOME}/.cmdstan"
    tar -xzf /tmp/cmdstan.tar.gz -C "${HOME}/.cmdstan"
    echo "[david] Building CmdStan (make build)…"
    make -C "${CMDSTAN_DIR}" build
    rm -f /tmp/cmdstan.tar.gz
fi
# Register path with cmdstanpy
uv run python -c "import cmdstanpy; cmdstanpy.set_cmdstan_path('${CMDSTAN_DIR}')"

echo "=== [3/3] Pre-compile Stan model → bakes binary into image ==="
uv run python -c "
from cmdstanpy import CmdStanModel
import pathlib
stan_file = pathlib.Path('/app/stan/m01_forward.stan')
if not stan_file.exists():
    raise FileNotFoundError(f'Stan file not found: {stan_file}')
model = CmdStanModel(stan_file=str(stan_file))
print(f'Binary compiled: {model.exe_file}')
" || echo "[david] WARNING: Stan pre-compile failed — will compile at runtime"

echo "=== Build complete ==="
