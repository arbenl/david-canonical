#!/bin/bash
# Railway build script.
# aptPkgs in nixpacks.toml ensures make + g++ are already present.
set -euo pipefail

echo "=== [1/3] Python dependencies ==="
uv sync --locked --no-editable

echo "=== [2/3] Install CmdStan ==="
uv run python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"

echo "=== [3/3] Pre-compile Stan model → bakes binary into image ==="
# CmdStanModel looks for /app/stan/m01_forward (binary) first.
# If found, it skips compilation at runtime — no CmdStan needed then.
uv run python -c "
from cmdstanpy import CmdStanModel
import pathlib
stan_file = pathlib.Path('/app/stan/m01_forward.stan')
if not stan_file.exists():
    raise FileNotFoundError(f'Stan file not found: {stan_file}')
model = CmdStanModel(stan_file=str(stan_file))
print(f'Binary compiled: {model.exe_file}')
"

echo "=== Build complete ==="
