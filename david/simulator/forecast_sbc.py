"""SBC for the forecast block.

For each of N synthetic worlds:
  1. Draw theta_true from prior.
  2. Generate history (T months) and continuation (H months).
  3. Fit m01_forward.stan on history only.
  4. Compute coverage of true (a_future, z_future) by posterior credible bands.

Pass criterion: nominal 80% and 95% credible intervals achieve coverage in
the pre-registered bands. This is F14 of the falsification battery.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..config import (
    FITS_DIR, MODEL_VERSION,
    FORECAST_SBC_NOMINAL_80_BAND, FORECAST_SBC_NOMINAL_95_BAND,
)
from .synthetic_world import HyperPrior, sample_world


def coverage_at_level(
    truth: np.ndarray,                    # (worlds, R, H, K) binary
    forecast_draws: np.ndarray,           # (worlds, R, H, K, D) probabilities
    level: float,
) -> float:
    """Empirical coverage of credible intervals at nominal level."""
    if not 0 < level < 1:
        raise ValueError("level must be in (0,1)")
    alpha = (1 - level) / 2
    lower = np.quantile(forecast_draws, alpha, axis=-1)
    upper = np.quantile(forecast_draws, 1 - alpha, axis=-1)
    covered = (truth >= lower) & (truth <= upper)
    return float(covered.mean())


def run_forecast_sbc(
    n_worlds: int = 200,
    prior: HyperPrior | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    out_dir = out_dir or FITS_DIR / "forecast_sbc"
    out_dir.mkdir(parents=True, exist_ok=True)
    prior = prior or HyperPrior(R=2, T=12, L=3, K=3, S=2, M=2, H=6)

    truths = []
    forecast_draws = []

    def _logistic(x):
        return 1.0 / (1.0 + np.exp(-x))

    for w in range(n_worlds):
        world = sample_world(prior, seed=w + 100_000)
        T = world.b.shape[1]
        H = prior.H
        R = world.selected.shape[0]
        K = world.y.shape[2]
        # True PROBABILITY at each (r, h, k): inv_logit(alpha[z_true, k]).
        # Coverage compares the posterior CI for a_future (probability) against
        # the true probability, not the binary realisation world.a.
        p_true = np.zeros((R, H, K))
        for r in range(R):
            for h in range(H):
                z_true = world.z[r, T + h]
                p_true[r, h, :] = _logistic(world.theta["alpha_activity"][z_true])
        truths.append(p_true)
        # Fit on history; pull a_future draws from generated quantities
        draws = fit_and_forecast(world, prior.H, seed=w + 100_000)
        forecast_draws.append(draws["a_future"])             # (R, H, K, D)

    truth_arr = np.stack(truths, axis=0)
    draws_arr = np.stack(forecast_draws, axis=0)

    cov80 = coverage_at_level(truth_arr, draws_arr, 0.80)
    cov95 = coverage_at_level(truth_arr, draws_arr, 0.95)

    lo80, hi80 = FORECAST_SBC_NOMINAL_80_BAND
    lo95, hi95 = FORECAST_SBC_NOMINAL_95_BAND
    pass_80 = lo80 <= cov80 <= hi80
    pass_95 = lo95 <= cov95 <= hi95

    summary = {
        "model_version": MODEL_VERSION,
        "n_worlds": n_worlds,
        "horizon": prior.H,
        "coverage_80": cov80,
        "coverage_95": cov95,
        "nominal_80_band": list(FORECAST_SBC_NOMINAL_80_BAND),
        "nominal_95_band": list(FORECAST_SBC_NOMINAL_95_BAND),
        "pass_80": pass_80,
        "pass_95": pass_95,
        "gate_status": "pass" if pass_80 and pass_95 else "fail",
        "reason": (
            "forecast_coverage_in_band" if pass_80 and pass_95
            else f"coverage_outside_band_80={cov80:.3f}_95={cov95:.3f}"
        ),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    path = out_dir / "forecast_sbc_summary.json"
    path.write_text(json.dumps(summary, indent=2))
    summary["summary_path"] = str(path)
    return summary


def fit_and_forecast(world, horizon: int, seed: int | None = None) -> dict[str, Any]:
    """Fit m01_forward.stan on history and return forecast draws.

    Returns dict with key `a_future` of shape (R, H, K, D) where D = n_draws.
    """
    from ..model.fit import assemble_fit_data_from_synthetic, _get_compiled_model

    data = assemble_fit_data_from_synthetic(world, horizon=horizon)
    model = _get_compiled_model()

    fit = model.sample(
        data=data,
        chains=2,
        iter_warmup=200,
        iter_sampling=200,
        seed=seed if seed is not None else 42,
        show_progress=False,
        show_console=False,
        adapt_delta=0.90,
    )

    # a_future shape from Stan: (n_draws, R, H, K)
    a_future_draws = fit.stan_variable("a_future")  # (D, R, H, K)
    D = a_future_draws.shape[0]
    R = a_future_draws.shape[1]
    H = a_future_draws.shape[2]
    K = a_future_draws.shape[3]
    # Reshape to (R, H, K, D)
    a_future = np.transpose(a_future_draws, (1, 2, 3, 0))

    return {"a_future": a_future, "n_draws": D}
