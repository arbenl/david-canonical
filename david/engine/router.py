"""Forecast routing: FG1..FG6 + posterior expected FDP threshold.

Applies the gate stack to each forecast cell and writes route_ledger.json.

Order of operations:
  FG1  measurement battery F1, F3, F4, F5 passed on fit run
  FG2  Theorem A' identification distance >= floor
  FG3  Theorem B' I(O) lower 95% CI >= floor
  FG4  Forecast SBC F14 passed for current model version
  FG5  Horizon h <= h*(g)
  FG6  Endogenous-observability lambda interval width <= max
Then apply posterior expected FDP threshold across cells that passed FG1..FG6.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..config import (
    FORECASTS_DIR, FITS_DIR, POSTERIOR_FDP_DEFAULT_Q,
    ID_DISTANCE_FLOOR, INFORMATIVENESS_FLOOR_LOWER_95,
    LAMBDA_ENDOG_INTERVAL_MAX_WIDTH,
)
from ..theorems.C_renamed import compute_posterior_fdp_threshold


ROUTES = (
    "headline",
    "monitor_only",
    "aggregate_only",
    "evidence_gap",
    "withhold",
    "horizon_prior_dominated",
    "prior_dominated",
)


def latest_forecast_dir() -> Path:
    candidates = sorted(p for p in FORECASTS_DIR.iterdir() if p.is_dir())
    if not candidates:
        raise FileNotFoundError("no forecasts to route; run `david forecast` first")
    return candidates[-1]


def latest_fit_dir() -> Path:
    candidates = sorted(p for p in FITS_DIR.iterdir() if p.is_dir())
    if not candidates:
        raise FileNotFoundError("no fit found")
    return candidates[-1]


def _measurement_gates_pass() -> tuple[bool, list[str]]:
    """Read fit_summary.json and confirm F1, F3, F4, F5 passed."""
    fit_dir = latest_fit_dir()
    summary_path = fit_dir / "fit_summary.json"
    if not summary_path.exists():
        return False, ["fit_summary_missing"]
    data = json.loads(summary_path.read_text())
    failed = [f for f in ("F1", "F3", "F4", "F5")
              if data.get("gates", {}).get(f, {}).get("gate_status") != "pass"]
    return (not failed), failed


def _forecast_sbc_pass() -> tuple[bool, str]:
    summary = FITS_DIR / "forecast_sbc" / "forecast_sbc_summary.json"
    if not summary.exists():
        return False, "forecast_sbc_summary_missing"
    data = json.loads(summary.read_text())
    return data.get("gate_status") == "pass", data.get("reason", "")


def apply_forecast_routing(
    q: float = POSTERIOR_FDP_DEFAULT_Q,
    horizon_files_glob: str = "cells_h*.json",
    out_dir: Path | None = None,
) -> dict[str, Any]:
    forecast_dir = latest_forecast_dir()
    out_dir = out_dir or forecast_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    meas_ok, meas_failed = _measurement_gates_pass()
    sbc_ok, sbc_reason = _forecast_sbc_pass()

    route_counts: dict[str, int] = {r: 0 for r in ROUTES}
    routed_cells = []

    for cells_path in sorted(forecast_dir.glob(horizon_files_glob)):
        records = json.loads(cells_path.read_text())
        for cell in records:
            reasons: list[str] = []
            route = "headline"

            # FG1
            if not meas_ok:
                route = "withhold"; reasons.append(f"FG1_fail:{','.join(meas_failed)}")

            # FG2 identification distance
            if (
                cell["identification_distance_posterior_median"] < ID_DISTANCE_FLOOR
                and route == "headline"
            ):
                route = "evidence_gap"; reasons.append("FG2_d_theta_below_floor")

            # FG3 informativeness
            if (
                cell["informativeness_I_O_lower_95"] < INFORMATIVENESS_FLOOR_LOWER_95
                and route == "headline"
            ):
                route = "prior_dominated"; reasons.append("FG3_I_O_lower95_below_floor")

            # FG4 forecast SBC
            if not sbc_ok and route == "headline":
                route = "withhold"; reasons.append(f"FG4_forecast_sbc:{sbc_reason}")

            # FG5 horizon validity
            if not cell["horizon_validity"]["below_h_star"] and route == "headline":
                route = "horizon_prior_dominated"
                reasons.append("FG5_above_h_star")

            # FG6 endogenous observability
            lo, hi = cell["lambda_endogenous_bounds"]
            if (hi - lo) > LAMBDA_ENDOG_INTERVAL_MAX_WIDTH and route == "headline":
                route = "monitor_only"
                reasons.append(f"FG6_lambda_width_{hi-lo:.3f}_above_max")

            cell["forecast_route"] = route
            cell["route_reasons"] = reasons
            route_counts[route] = route_counts.get(route, 0) + 1
            routed_cells.append(cell)

    # Apply posterior-FDP threshold across "headline" cells only
    headline_cells = [c for c in routed_cells if c["forecast_route"] == "headline"]
    if headline_cells:
        p_hat = np.array([c["p_active"] for c in headline_cells])
        fdp_result = compute_posterior_fdp_threshold(p_hat, q=q)
        for i, c in enumerate(headline_cells):
            c["headline_flagged_by_posterior_fdp"] = (
                i in set(fdp_result.flagged_indices)
            )
        fdp_summary = fdp_result.__dict__
    else:
        fdp_summary = {"q_target": q, "n_cells": 0, "n_flagged": 0}

    ledger = {
        "route_counts": route_counts,
        "posterior_fdp": fdp_summary,
        "n_cells_total": len(routed_cells),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    ledger_path = out_dir / "route_ledger.json"
    ledger_path.write_text(json.dumps(ledger, indent=2))
    return {
        "ledger_path": str(ledger_path),
        "route_counts": route_counts,
    }
