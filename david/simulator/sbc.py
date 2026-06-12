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


_N_BINS = 10  # histogram bins for shape diagnostics


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


def histogram_diagnostics(
    ranks: np.ndarray,
    n_draws_per_fit: int,
    n_bins: int = _N_BINS,
) -> dict[str, object]:
    """B-6: Per-parameter rank-histogram shape diagnostics.

    Returns
    -------
    dict with:
      chi_squared       float   χ² statistic vs expected uniform bin counts
      chi_squared_p     float   p-value (χ² CDF, dof = n_bins - 1)
      histogram_shape   str     "uniform" | "U_shaped" | "center_clustered" | "skew"
      bin_counts        list[int]
    """
    if ranks.size < n_bins:
        return {
            "chi_squared": float("nan"),
            "chi_squared_p": float("nan"),
            "histogram_shape": "insufficient_data",
            "bin_counts": [],
        }

    counts, _ = np.histogram(ranks, bins=n_bins, range=(0, n_draws_per_fit))
    expected = ranks.size / n_bins

    # χ² against uniform
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    chi2_p = float(sp_stats.chi2.sf(chi2, df=n_bins - 1))

    # Split into left-edge, center, right-edge bands.
    # n_edge bins at each extreme (2 bins each for n_bins=10).
    n_edge = max(1, n_bins // 5)
    left_edge_count  = int(counts[:n_edge].sum())
    right_edge_count = int(counts[-n_edge:].sum())
    center_count     = int(counts[n_edge:-n_edge].sum())
    n_center_bins    = n_bins - 2 * n_edge

    left_rate   = left_edge_count  / (n_edge * expected)
    right_rate  = right_edge_count / (n_edge * expected)
    centre_rate = center_count     / (n_center_bins * expected)

    # Classification:
    # U_shaped         — BOTH edge bands are dense (prior/width mismatch)
    # center_clustered — centre is dense relative to both edges (likelihood overcounting)
    # skew             — ONE edge band is dense, not the other (posterior bias)
    # uniform          — none of the above
    EXCESS = 1.35  # density ratio threshold vs uniform expectation

    if left_rate >= EXCESS and right_rate >= EXCESS:
        shape = "U_shaped"
    elif centre_rate >= EXCESS and left_rate < EXCESS and right_rate < EXCESS:
        shape = "center_clustered"
    elif (left_rate >= EXCESS) != (right_rate >= EXCESS):
        # Exactly one edge is heavy
        shape = "skew"
    else:
        shape = "uniform"

    return {
        "chi_squared": chi2,
        "chi_squared_p": chi2_p,
        "histogram_shape": shape,
        "bin_counts": counts.tolist(),
    }


def run_sbc(
    n_worlds: int = 200,
    prior: HyperPrior | None = None,
    out_dir: Path | None = None,
    base_seed: int = 0,
) -> dict[str, Any]:
    """Run SBC for the measurement layer.

    Seeds are derived as ``base_seed + world_index`` so the full set is
    reproducible from (base_seed, n_worlds) alone.

    Returns a typed result with gate_status. B-6: every parameter entry in
    ``per_parameter_ks`` now carries ``chi_squared``, ``chi_squared_p``,
    ``histogram_shape``, and ``bin_counts``.
    """
    out_dir = out_dir or FITS_DIR / "sbc"
    out_dir.mkdir(parents=True, exist_ok=True)
    prior = prior or HyperPrior(R=2, T=12, L=3, K=3, S=2, M=2, H=0)

    parameter_ranks: dict[str, list[int]] = {}
    n_draws_per_fit = 0
    seeds_used = list(range(base_seed, base_seed + n_worlds))

    for w, seed in enumerate(seeds_used):
        world = sample_world(prior, seed=seed)
        posterior = fit_measurement_layer(world, seed=seed)
        n_draws_per_fit = posterior["n_draws"]
        for name, true_value in flatten_params(world.theta).items():
            draws = posterior["draws"].get(name)
            if draws is None:
                continue
            parameter_ranks.setdefault(name, []).append(
                rank_statistic(true_value, draws)
            )

    # Build per-parameter diagnostics: KS + B-6 histogram shape
    per_param: dict[str, dict] = {}
    for name, ranks in parameter_ranks.items():
        arr = np.asarray(ranks)
        ks  = ks_uniformity_test(arr, n_draws_per_fit)
        hd  = histogram_diagnostics(arr, n_draws_per_fit)
        per_param[name] = {**ks, **hd}

    summary: dict[str, Any] = {
        "model_version": MODEL_VERSION,
        "n_worlds": n_worlds,
        "n_draws_per_fit": n_draws_per_fit,
        "ks_alpha": SBC_KS_ALPHA,
        "base_seed": base_seed,
        "seeds": seeds_used,
        "per_parameter_ks": per_param,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    n_params = len(per_param)
    effective_alpha = SBC_KS_ALPHA / n_params if (SBC_BONFERRONI and n_params > 0) else SBC_KS_ALPHA
    summary["effective_alpha"] = effective_alpha
    summary["bonferroni_applied"] = SBC_BONFERRONI
    failed = [
        name for name, diag in per_param.items()
        if diag["pvalue"] < effective_alpha
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
