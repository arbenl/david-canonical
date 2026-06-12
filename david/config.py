"""Canonical paths and constants for DAVID/M0.1."""

from __future__ import annotations

import os
import json
from pathlib import Path

# Roots
def _find_root() -> Path:
    """Locate project root regardless of install mode (editable vs non-editable).

    When installed non-editable (e.g. Railway `uv sync --no-editable`),
    __file__ lives inside site-packages, not the source tree. Walk up until
    pyproject.toml is found; fall back to /app (Railway convention).
    """
    candidate = Path(__file__).resolve().parent
    for _ in range(8):
        if (candidate / "pyproject.toml").exists():
            return candidate
        candidate = candidate.parent
    return Path("/app")

ROOT = Path(os.environ.get("DAVID_ROOT", "")) or _find_root()
DATA_ROOT = Path(os.environ.get("DAVID_DATA_ROOT", ROOT / "data"))
STAN_ROOT = ROOT / "stan"
CONFIG_ROOT = ROOT / "config"

# Data subdirs (append-only)
RAW_DIR = DATA_ROOT / "raw"
CODED_DIR = DATA_ROOT / "coded"
ADJUDICATED_DIR = DATA_ROOT / "adjudicated"
GOLD_DIR = DATA_ROOT / "gold"
FITS_DIR = DATA_ROOT / "fits"
FORECASTS_DIR = DATA_ROOT / "forecasts"

# Stan models
M01_FORWARD_STAN = STAN_ROOT / "m01_forward.stan"
CODER_CALIBRATION_STAN = STAN_ROOT / "coder_calibration.stan"
SYNTHETIC_GENERATOR_STAN = STAN_ROOT / "synthetic_generator.stan"

# Config artifacts
SOURCE_REGISTRY = CONFIG_ROOT / "source_registry.json"
SOURCE_INDEPENDENCE_LEDGER = CONFIG_ROOT / "source_independence_ledger.json"
DOMAIN_TAXONOMY_REGISTRY = CONFIG_ROOT / "domain_taxonomy.json"
PRE_REGISTRATION = CONFIG_ROOT / "m01_preregistration_v3.json"

# Theorem floors (pre-registered; reviewed quarterly)
ID_DISTANCE_FLOOR = 0.05            # Theorem A' practical identifiability
INFORMATIVENESS_FLOOR_LOWER_95 = 0.10  # Theorem B'.2 lower 95% CI on I(O)
N_EFF_I2_FLOOR = 3.0                   # Theorem B'.2 N_eff × I² floor (weak-channel guard)
POSTERIOR_FDP_DEFAULT_Q = 0.10      # Theorem C posterior expected FDP
HORIZON_PRIOR_DRIFT_TAU = 0.50      # Theorem D-forecast horizon-validity threshold
LAMBDA_ENDOG_INTERVAL_MAX_WIDTH = 0.20  # Gap-5 endogenous-observability sensitivity bound

# SBC thresholds
SBC_KS_ALPHA = 0.05
SBC_BONFERRONI = True   # apply Bonferroni correction: effective α = SBC_KS_ALPHA / n_params
FORECAST_SBC_NOMINAL_80_BAND = (0.75, 0.85)
FORECAST_SBC_NOMINAL_95_BAND = (0.90, 0.98)

# Fit thresholds (mirror council_m01 fit_contract)
R_HAT_MAX = 1.05
BULK_ESS_MIN = 400
TAIL_ESS_MIN = 400
DIVERGENCES_ALLOWED = 0
MIN_CHAINS = 4
MIN_POSTERIOR_DRAWS = 4000

# Human-loop budget
ADJUDICATOR_HOURS_PER_CYCLE = 4.0
# Minority-share threshold for escalation to human.
# minority_share = min(ones, zeros) / n_coders
# 0.0 = all coders agree (auto-adjudicated); 0.5 = perfect split.
# Items with minority_share > ADJUDICATOR_DISAGREEMENT_THRESHOLD go to queue.
ADJUDICATOR_DISAGREEMENT_THRESHOLD = 0.30
GOLD_STANDARD_SAMPLE_RATE = 0.05

# Pre-registered tactic classes (must match K in m01_forward.stan).
# Bump MODEL_VERSION if you add or remove tactics.
def _load_tactic_classes() -> tuple[str, ...]:
    if not DOMAIN_TAXONOMY_REGISTRY.exists():
        return ("SIO", "MIO", "CSIO")
    try:
        with open(DOMAIN_TAXONOMY_REGISTRY) as f:
            data = json.load(f)
        tactics = data.get("tactics", [])
        if not tactics:
            return ("SIO", "MIO", "CSIO")
        return tuple(t["id"] for t in tactics)
    except Exception:
        return ("SIO", "MIO", "CSIO")

TACTIC_CLASSES: tuple[str, ...] = _load_tactic_classes()

# LLM pool config file (list of LlmCoderConfig dicts)
LLM_POOL_REGISTRY = CONFIG_ROOT / "llm_pool.json"

# Forecast horizons (months) emitted by `david forecast`
FORECAST_HORIZONS_MONTHS = (3, 6, 9, 12)

# Model version (bump on Stan or theorem change)
MODEL_VERSION = "m01_forward_v0.1.1"

# Database
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://david:david@localhost:5544/david",
)
