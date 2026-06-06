"""Forecast emission with horizon-validity gating (Theorem D-forecast).

emit_forecasts(horizon_months) takes the latest fit and writes the canonical
per-cell forecast object documented in ARCHITECTURE.md §7.1.

The horizon-validity gate uses the h_star pre-computed in run_fit (via
theorem D') rather than recomputing it here. Cells with horizon_months > h_star
are flagged horizon_prior_dominated in the output record.

Flow:
  1. load_posterior(fit_dir) → {column: (D,) array} from draws.parquet
  2. Parse a_future[r,h,k] columns for the requested horizon
  3. For each (r, k) cell: compute median, 80/95 CIs, horizon-validity flag
  4. Write cells_h{horizon:02d}.json

When draws.parquet is absent or a_future columns are not present (e.g. fit
was run with H_forecast = 0), the function returns gate_status='skip' with a
clear reason rather than raising.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..config import (
    FITS_DIR, FORECASTS_DIR, FORECAST_HORIZONS_MONTHS, MODEL_VERSION,
)


def latest_fit_dir() -> Path:
    """Return the most recent directory in FITS_DIR that has a fit_summary.json."""
    if not FITS_DIR.exists():
        raise FileNotFoundError(f"{FITS_DIR} does not exist; run `david fit` first.")
    candidates = sorted(
        p for p in FITS_DIR.iterdir()
        if p.is_dir() and (p / "fit_summary.json").exists()
    )
    if not candidates:
        raise FileNotFoundError("No completed fit found; run `david fit` first.")
    return candidates[-1]


def load_posterior(fit_dir: Path) -> dict[str, np.ndarray]:
    """Load posterior draws from draws.parquet → {column_name: (D,) array}.

    Raises FileNotFoundError when draws.parquet is absent.
    """
    draws_path = fit_dir / "draws.parquet"
    if not draws_path.exists():
        raise FileNotFoundError(
            f"draws.parquet not found in {fit_dir}. Run `david fit` first."
        )
    try:
        import pandas as pd
        df = pd.read_parquet(draws_path)
    except Exception as exc:
        raise FileNotFoundError(
            f"Could not read {draws_path}: {exc}"
        ) from exc
    return {col: df[col].to_numpy() for col in df.columns}


def _parse_a_future(posterior: dict[str, np.ndarray]) -> dict[tuple[int, int, int], np.ndarray]:
    """Parse a_future[r,h,k] columns from the posterior dict.

    Stan 1-indexed: a_future[1,1,1] is series=1, horizon=1, tactic=1.
    Returns {(r, h, k): draws_array} with 0-based series/tactic for convenience.
    """
    pattern = re.compile(r'^a_future\[(\d+),(\d+),(\d+)\]$')
    result: dict[tuple[int, int, int], np.ndarray] = {}
    for col, vals in posterior.items():
        m = pattern.match(col)
        if m:
            r, h, k = int(m.group(1)), int(m.group(2)), int(m.group(3))
            result[(r, h, k)] = vals.astype(float)
    return result


def emit_forecasts(
    horizon_months: int = 6,
    cell_filter: str | None = None,
    out_dir: Path | None = None,
    fit_dir: Path | None = None,
) -> dict[str, Any]:
    """Emit per-cell forecast records for a given horizon.

    Parameters
    ----------
    horizon_months : int
        Forecast horizon in months. Must be in FORECAST_HORIZONS_MONTHS.
    cell_filter : str | None
        Optional filter string; if supplied, only cells whose key contains
        the string are emitted (e.g. 'r1k2' for series 1 tactic 2).
    out_dir : Path | None
        Output directory. Defaults to FORECASTS_DIR / run_id.
    fit_dir : Path | None
        Fit directory to use. Defaults to latest_fit_dir() if None.

    Returns
    -------
    dict
        Typed result with gate_status, n_cells, cells_path, horizon, fit_run_id.
    """
    if horizon_months not in FORECAST_HORIZONS_MONTHS:
        raise ValueError(
            f"horizon_months={horizon_months} not in pre-registered "
            f"FORECAST_HORIZONS_MONTHS={FORECAST_HORIZONS_MONTHS}"
        )

    try:
        fit_dir = fit_dir or latest_fit_dir()
    except FileNotFoundError as exc:
        return {"gate_status": "fail", "reason": str(exc), "n_cells": 0}

    run_id = fit_dir.name
    out_dir = out_dir or FORECASTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load posterior draws
    try:
        posterior = load_posterior(fit_dir)
    except FileNotFoundError as exc:
        return {"gate_status": "fail", "reason": str(exc), "n_cells": 0,
                "horizon": horizon_months, "fit_run_id": run_id}

    # Parse a_future columns
    a_future = _parse_a_future(posterior)
    if not a_future:
        return {
            "gate_status": "skip",
            "reason": (
                "a_future not present in draws.parquet; "
                "re-run `david fit` with H_forecast>0 "
                f"(set to max(FORECAST_HORIZONS_MONTHS)={max(FORECAST_HORIZONS_MONTHS)})"
            ),
            "n_cells": 0,
            "cells_path": None,
            "horizon": horizon_months,
            "fit_run_id": run_id,
        }

    # Load theorem D' h_star from fit_summary (pre-computed in run_fit)
    h_star: int = 0
    fit_summary: dict = {}
    summary_path = fit_dir / "fit_summary.json"
    if summary_path.exists():
        fit_summary = json.loads(summary_path.read_text())
        h_star = (
            fit_summary.get("theorems", {})
                       .get("D_prime", {})
                       .get("h_star_months", 0)
        )

    below_h_star = horizon_months <= h_star

    # Build cell records for the requested horizon
    cells_records: list[dict[str, Any]] = []
    timestamp = datetime.utcnow().isoformat() + "Z"

    for (r, h, k), draws in sorted(a_future.items()):
        if h != horizon_months:
            continue
        cell_key = f"r{r}k{k}"
        if cell_filter and cell_filter not in cell_key:
            continue

        median   = float(np.median(draws))
        lo80, hi80 = (float(x) for x in np.quantile(draws, [0.10, 0.90]))
        lo95, hi95 = (float(x) for x in np.quantile(draws, [0.025, 0.975]))

        # p_marginal: mean over all draws and horizon steps (rough stationary approx)
        # A more precise value comes from theorem D'.stationary_marginal_time per stratum.
        marginal_keys = [(r, hh, k) for hh in range(1, max(FORECAST_HORIZONS_MONTHS) + 1)
                         if (r, hh, k) in a_future]
        if marginal_keys:
            marginal_pool = np.concatenate([a_future[key] for key in marginal_keys])
            p_marginal = float(marginal_pool.mean())
        else:
            p_marginal = median  # fallback

        record: dict[str, Any] = {
            "cell": {
                "series": r,
                "tactic": k,
                # Country / policy / tactic labels populated by ingest pipeline;
                # stubs here until source_registry is wired.
                "country": f"series_{r}",
                "policy": "unknown",
                "tactic_label": f"tactic_{k}",
            },
            "horizon_months": horizon_months,
            "p_active": median,
            "p_marginal": p_marginal,
            "credible_interval_80": [lo80, hi80],
            "credible_interval_95": [lo95, hi95],
            "horizon_validity": {
                "h_star_months": h_star,
                "below_h_star": below_h_star,
                "forecast_route": (
                    "conditional_forecast" if below_h_star
                    else "horizon_prior_dominated"
                ),
            },
            # Identification and informativeness from theorem gates (global level)
            "theorem_A_prime": fit_summary.get("theorems", {}).get("A_prime", {}),
            "theorem_B_prime": fit_summary.get("theorems", {}).get("B_prime", {}),
            "model_version": MODEL_VERSION,
            "fit_run_id": run_id,
            "timestamp": timestamp,
        }
        cells_records.append(record)

    cells_path = out_dir / f"cells_h{horizon_months:02d}.json"
    cells_path.write_text(json.dumps(cells_records, indent=2))

    return {
        "gate_status": "pass",
        "n_cells": len(cells_records),
        "cells_path": str(cells_path),
        "horizon": horizon_months,
        "fit_run_id": run_id,
        "timestamp": timestamp,
    }


def emit_all_horizons(
    cell_filter: str | None = None,
    out_dir: Path | None = None,
    fit_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Emit cells_h*.json for all pre-registered forecast horizons."""
    try:
        resolved_fit = fit_dir or latest_fit_dir()
    except FileNotFoundError as exc:
        return [{"gate_status": "fail", "reason": str(exc), "n_cells": 0}]
    return [
        emit_forecasts(h, cell_filter=cell_filter, out_dir=out_dir, fit_dir=resolved_fit)
        for h in FORECAST_HORIZONS_MONTHS
    ]
