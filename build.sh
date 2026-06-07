#!/bin/bash
# Railway build script — runs during image build, output baked in.
set -euo pipefail

echo "=== [1/3] Installing Python dependencies ==="
uv sync --locked --no-editable

echo "=== [2/3] Installing CmdStan (Stan C++ toolchain) ==="
# Install to a fixed absolute path so it's independent of $HOME resolution
uv run python -c "
import cmdstanpy, pathlib, os
install_dir = pathlib.Path('/opt/cmdstan')
install_dir.mkdir(parents=True, exist_ok=True)
cmdstanpy.install_cmdstan(dir=str(install_dir))
# Find the versioned subdirectory and record the path
dirs = sorted(install_dir.glob('cmdstan-*'))
if dirs:
    print(f'CmdStan installed to: {dirs[-1]}')
else:
    raise RuntimeError('install_cmdstan succeeded but no cmdstan-* dir found')
"

echo "=== [3/3] Pre-compiling Stan model binary ==="
# Compile m01_forward.stan → /app/stan/m01_forward (native binary).
# CmdStanModel checks for this binary first; if found it skips compilation
# entirely, so runtime containers never need CmdStan installed.
uv run python -c "
import pathlib, os
from cmdstanpy import CmdStanModel

# Point cmdstanpy at the fixed install dir
install_dir = pathlib.Path('/opt/cmdstan')
dirs = sorted(install_dir.glob('cmdstan-*'))
if not dirs:
    raise RuntimeError('CmdStan not found in /opt/cmdstan')
os.environ['CMDSTAN'] = str(dirs[-1])

stan_file = pathlib.Path('/app/stan/m01_forward.stan')
if not stan_file.exists():
    raise FileNotFoundError(f'Stan file not found: {stan_file}')

model = CmdStanModel(stan_file=str(stan_file))
print(f'Stan binary compiled: {model.exe_file}')
"

echo "=== Build complete — Stan binary baked into image ==="
