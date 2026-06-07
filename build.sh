#!/bin/bash
# Railway build script.
# aptPkgs in nixpacks.toml ensures make + g++ are already present.
set -euo pipefail

CMDSTAN_VERSION="2.39.0"
CMDSTAN_URL="https://github.com/stan-dev/cmdstan/releases/download/v${CMDSTAN_VERSION}/cmdstan-${CMDSTAN_VERSION}.tar.gz"

echo "=== [1/3] Python dependencies ==="
uv sync --locked --no-editable

echo "=== [2/3] Install CmdStan ${CMDSTAN_VERSION} (direct URL — no GitHub API) ==="
uv run python -c "
import cmdstanpy, sys
version = '${CMDSTAN_VERSION}'
url = '${CMDSTAN_URL}'
try:
    cmdstanpy.install_cmdstan(url=url, version=version)
    print('[david] CmdStan installed OK')
except Exception as e:
    print(f'[david] WARNING: CmdStan install failed: {e}', file=sys.stderr)
    print('[david] Will attempt install at runtime via _ensure_cmdstan()', file=sys.stderr)
"

echo "=== [3/3] Pre-compile Stan model → bakes binary into image ==="
uv run python -c "
from cmdstanpy.utils.cmdstan import cmdstan_path
import pathlib, sys
try:
    cmdstan_path()  # verify CmdStan is present before trying to compile
except ValueError:
    print('[david] CmdStan not installed — skipping pre-compile (runtime will compile)', file=sys.stderr)
    sys.exit(0)

from cmdstanpy import CmdStanModel
stan_file = pathlib.Path('/app/stan/m01_forward.stan')
if not stan_file.exists():
    raise FileNotFoundError(f'Stan file not found: {stan_file}')
model = CmdStanModel(stan_file=str(stan_file))
print(f'Binary compiled: {model.exe_file}')
"

echo "=== Build complete ==="
