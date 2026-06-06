"""Simulation-Based Calibration for the measurement layer.

Talts, Betancourt, Simpson, Vehtari, Gelman (2018). Validating Bayesian
inference algorithms with simulation-based calibration.

For each of N synthetic worlds:
  1. Draw theta_true from the prior.
  2. Generate (B, Y, selected, observability) from synthetic_world.sample_world.
  3. Fit m01_forward.stan to the synthetic data.
  4. Compute rank statistic of theta_true within posterior draws.

Pass criterion: rank statistics uniform on [0, N_post]. Tested via
Kolmogorov-Smirnov against uniform at SBC_KS_ALPHA.

This is the proof anchor for the measurement layer. F2 of the falsification
battery is satisfied by passing this.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as sp_stats

from ..config import (
    FITS_DIR, MODEL_VERSION, SBC_BONFERRONI, SBC_KS_ALPHA,
)
from .synthetic_world import HyperPrior, sample_world


def rank_statistic(theta_true: float, posterior_draws: np.ndarray) -> int:
    """Rank of theta_true among posterior draws (0..len(posterior_draws))."""
    return int(np.sum(posterior_draws < theta_true))


def ks_uniformity_test(ranks: np.ndarray, n_draws_per_fit: int) -> dict:
    """KS test that ranks are uniform on [0, n_draws_per_fit]."""
    if ranks.size == 0:
        return {"statistic": float("nan"), "pvalue": float("nan"), "n": 0}
    cdf = lambda x: x / n_draws_per_fit
    statistic, pvalue = sp_stats.kstest(ranks, cdf)
    return {"statistic": float(statistic), "pvalue": float(pvalue), "n": int(ranks.size)}


def run_sbc(
    n_worlds: int = 200,
    prior: HyperPrior | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Run SBC for the measurement layer.

    Returns a typed result with gate_status.
    """
    out_dir = out_dir or FITS_DIR / "sbc"
    out_dir.mkdir(parents=True, exist_ok=True)
    prior = prior or HyperPrior(R=2, T=12, L=3, K=3, S=2, M=2, H=0)

    parameter_ranks: dict[str, list[int]] = {}
    n_draws_per_fit = 0

    for w in range(n_worlds):
        world = sample_world(prior, seed=w)
        posterior = fit_measurement_layer(world, seed=w)
        n_draws_per_fit = posterior["n_draws"]
        for name, true_value in flatten_params(world.theta).items():
            draws = posterior["draws"].get(name)
            if draws is None:
                continue
            parameter_ranks.setdefault(name, []).append(
                rank_statistic(true_value, draws)
            )

    summary = {
        "model_version": MODEL_VERSION,
        "n_worlds": n_worlds,
        "n_draws_per_fit": n_draws_per_fit,
        "ks_alpha": SBC_KS_ALPHA,
        "per_parameter_ks": {
            name: ks_uniformity_test(np.asarray(ranks), n_draws_per_fit)
            for name, ranks in parameter_ranks.items()
        },
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    n_params = len(summary["per_parameter_ks"])
    effective_alpha = SBC_KS_ALPHA / n_params if (SBC_BONFERRONI and n_params > 0) else SBC_KS_ALPHA
    summary["effective_alpha"] = effective_alpha
    summary["bonferroni_applied"] = SBC_BONFERRONI
    failed = [
        name for name, ks in summary["per_parameter_ks"].items()
        if ks["pvalue"] < effective_alpha
    ]
    summary["failed_parameters"] = failed
    summary["gate_status"] = "pass" if not failed else "fail"
    summary["reason"] = (
        "all_parameters_uniform" if not failed
        else f"ks_fail_on_{len(failed)}_parameters"
    )
    summary_path = out_dir / "sbc_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    summary["summary_path"] = str(summary_path)
    return summary


def flatten_params(theta: dict) -> dict[str, float]:
    """Flatten a parameter dict to scalar names for SBC ranking.

    The Pi diagonal (pi_ii = 0) is excluded: it's identically zero by
    construction in both the generative model and Stan, producing degenerate
    rank statistics (always 0) that trivially fail KS without conveying
    useful calibration information.
    """
    out: dict[str, float] = {}
    for name, val in theta.items():
        arr = np.atleast_1d(np.asarray(val)).ravel()
        if arr.size == 1:
            out[name] = float(arr[0])
        else:
            if name == "Pi":
                L = int(round(arr.size ** 0.5))
                for i in range(L):
                    for j in range(L):
                        if i != j:
                            out[f"Pi[{i * L + j}]"] = float(arr[i * L + j])
            else:
                for i, v in enumerate(arr):
                    out[f"{name}[{i}]"] = float(v)
    return out


def fit_measurement_layer(world, seed: int | None = None) -> dict[str, Any]:
    """Fit m01_forward.stan on a synthetic world; return flat posterior draws.

    Uses 2 chains x 200 warmup x 200 sampling (400 draws total). The compiled
    model is cached across calls to amortize compilation cost.
    """
    from ..model.fit import assemble_fit_data_from_synthetic, _get_compiled_model, extract_theta_space_draws

    data = assemble_fit_data_from_synthetic(world, horizon=1)
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

    draws = extract_theta_space_draws(fit)
    n_draws = 2 * 200
    return {"n_draws": n_draws, "draws": draws}
