"""Forecast emission with horizon-validity gating (Theorem D-forecast).

emit_forecasts(horizon_months, cell_filter) takes the latest fit and writes
the canonical per-cell forecast object documented in ARCHITECTURE.md §7.1.

The horizon-validity gate (FG5) is checked here, not in the router, because
above h* we substitute the marginal regime prediction for the conditional one.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..config import (
    FORECASTS_DIR, FITS_DIR, MODEL_VERSION,
    HORIZON_PRIOR_DRIFT_TAU, FORECAST_HORIZONS_MONTHS,
)
from ..theorems.D_forecast_horizon import (
    horizon_validity, stationary_marginal_time,
)


def latest_fit_dir() -> Path:
    candidates = sorted(p for p in FITS_DIR.iterdir() if p.is_dir() and (p / "fit_summary.json").exists())
    if not candidates:
        raise FileNotFoundError("no fit found; run `david fit` first")
    return candidates[-1]


def load_posterior(fit_dir: Path) -> dict[str, np.ndarray]:
    """Load posterior draws of (alpha_activity, jump, dwell_lambda, terminal regime posteriors)."""
    # TODO: read from draws.parquet (column-store of cmdstanpy draws)
    # For scaffolding, the loader returns the schema; the user wires the actual reader.
    raise NotImplementedError(
        "Wire to data/fits/{run_id}/draws.parquet reader. "
        "Schema: alpha_activity[L,K], jump[L,L], dwell_lambda[L], z_T[R,L] (terminal posterior)."
    )


def emit_forecasts(
    horizon_months: int = 6,
    cell_filter: str | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    if horizon_months not in FORECAST_HORIZONS_MONTHS:
        raise ValueError(f"horizon must be one of {FORECAST_HORIZONS_MONTHS}")
    fit_dir = latest_fit_dir()
    run_id = fit_dir.name
    out_dir = out_dir or FORECASTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    posterior = load_posterior(fit_dir)
    cells_records: list[dict[str, Any]] = []

    # Iterate strata; here `g` is (c, p, k). Cell index has time t_now = today.
    for g in iter_strata(cell_filter):
        h_validity = horizon_validity(
            cell_id=g.id,
            Pi_off_diag=g.posterior_Pi_off,           # mean transition
            dwell_mean=g.posterior_dwell_mean,
            z_t_distribution=g.terminal_regime_posterior,
            h_max=max(FORECAST_HORIZONS_MONTHS) + 6,
            tau=HORIZON_PRIOR_DRIFT_TAU,
        )

        below_h_star = horizon_months <= h_validity.h_star_months

        # Conditional vs marginal forecast
        if below_h_star:
            p_active_draws = g.forecast_p_active_draws(horizon_months)   # posterior, (D,)
        else:
            # Replace with marginal regime prediction: time-weighted stationary
            pi_inf = stationary_marginal_time(g.posterior_Pi_off, g.posterior_dwell_mean)
            p_active_draws = g.marginal_p_active_draws(pi_inf)

        median = float(np.median(p_active_draws))
        lo80, hi80 = (float(x) for x in np.quantile(p_active_draws, [0.10, 0.90]))
        lo95, hi95 = (float(x) for x in np.quantile(p_active_draws, [0.025, 0.975]))

        cell = {
            "cell": {"c": g.c, "t_now": date.today().isoformat(), "p": g.p, "k": g.k},
            "horizon_months": horizon_months,
            "p_active": median,
            "credible_interval_80": [lo80, hi80],
            "credible_interval_95": [lo95, hi95],
            "regime_distribution": g.regime_distribution_at(horizon_months),
            "horizon_validity": {
                "h_star_months": h_validity.h_star_months,
                "prior_drift_share": h_validity.horizon_validity_curve[-1][1],
                "below_h_star": below_h_star,
            },
            "identification_distance_posterior_median": g.id_distance_median,
            "informativeness_I_O_posterior_median": g.I_O_median,
            "informativeness_I_O_lower_95": g.I_O_lower_95,
            "lambda_endogenous_bounds": g.lambda_bounds,
            # Routing decision is filled in by router.apply_forecast_routing()
            "forecast_route": "pending_route",
            "route_reasons": [],
            "model_version": MODEL_VERSION,
            "fit_run_id": run_id,
            "evidence_cutoff": g.evidence_cutoff,
        }
        cells_records.append(cell)

    cells_path = out_dir / f"cells_h{horizon_months:02d}.json"
    cells_path.write_text(json.dumps(cells_records, indent=2))
    return {
        "n_cells": len(cells_records),
        "cells_path": str(cells_path),
        "horizon": horizon_months,
        "fit_run_id": run_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


def iter_strata(cell_filter: str | None):
    """TODO: iterate strata bound to the current fit.

    Each stratum exposes:
      .c, .p, .k, .id, .evidence_cutoff
      .posterior_Pi_off, .posterior_dwell_mean, .terminal_regime_posterior
      .forecast_p_active_draws(h), .marginal_p_active_draws(pi_inf)
      .regime_distribution_at(h)
      .id_distance_median, .I_O_median, .I_O_lower_95, .lambda_bounds
    """
    raise NotImplementedError(
        "Implement stratum iterator backed by the fit artifact directory "
        "and the source_registry + opportunity_units CSV."
    )
