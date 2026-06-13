"""Simulation-Based Calibration for the measurement layer.

Talts, Betancourt, Simpson, Vehtari, Gelman (2018). Validating Bayesian
inference algorithms with simulation-based calibration.

For each of N synthetic worlds:
  1. Draw theta_true from the prior.
  2. Generate (B, Y, selected, observability) from synthetic_world.sample_world.
  3. Fit m01_forward.stan to the synthetic data.
  4. Compute rank statistic of theta_true within posterior draws.

Pass criterion: rank histograms pass a binned chi-square falsification stack
with Benjamini-Hochberg control and a global chi-square test. KS statistics are
reported as diagnostics but no longer the sole gate.

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
    FITS_DIR, MODEL_VERSION, SBC_BONFERRONI, SBC_CHI2_BH_ALPHA,
    SBC_BULK_ESS_MIN, SBC_DIVERGENCES_ALLOWED, SBC_HISTOGRAM_EXCESS,
    SBC_KS_ALPHA, SBC_MAX_DISCARD_FRACTION, SBC_R_HAT_MAX,
    SBC_RANK_DRAWS_TARGET,
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
    # U_shaped         — BOTH edge bands are dense: posterior under-dispersed /
    #                    overconfident, the expected signature of likelihood
    #                    double-counting or correlated-coder overcounting.
    # center_clustered — centre is dense: posterior over-dispersed.
    # skew             — ONE edge band is dense, not the other (posterior bias)
    # uniform          — none of the above
    EXCESS = SBC_HISTOGRAM_EXCESS

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


def _benjamini_hochberg_rejections(
    p_values: dict[str, float],
    alpha: float,
) -> list[str]:
    finite = sorted(
        (name, p) for name, p in p_values.items()
        if isinstance(p, float) and np.isfinite(p)
    )
    m = len(finite)
    if m == 0:
        return []
    cutoff_rank = 0
    for rank, (_name, p) in enumerate(finite, start=1):
        if p <= alpha * rank / m:
            cutoff_rank = rank
    if cutoff_rank == 0:
        return []
    cutoff = finite[cutoff_rank - 1][1]
    return [name for name, p in finite if p <= cutoff]


def _sbc_gate_failures_from_diagnostics(
    per_param: dict[str, dict],
    alpha: float = SBC_CHI2_BH_ALPHA,
) -> dict[str, Any]:
    chi2_p_values = {
        name: float(diag["chi_squared_p"])
        for name, diag in per_param.items()
        if isinstance(diag.get("chi_squared_p"), float)
        and np.isfinite(diag["chi_squared_p"])
    }
    bh_failed = _benjamini_hochberg_rejections(chi2_p_values, alpha=alpha)

    chi2_stats = [
        float(diag["chi_squared"])
        for diag in per_param.values()
        if isinstance(diag.get("chi_squared"), float)
        and np.isfinite(diag["chi_squared"])
    ]
    global_stat = float(np.sum(chi2_stats)) if chi2_stats else float("nan")
    global_df = len(chi2_stats) * (_N_BINS - 1)
    global_p = (
        float(sp_stats.chi2.sf(global_stat, df=global_df))
        if global_df > 0 else float("nan")
    )
    global_failed = bool(np.isfinite(global_p) and global_p < alpha)

    failed = sorted(set(bh_failed))
    if global_failed:
        failed = sorted(set(failed) | {"__global_rank_histogram__"})

    return {
        "failed_parameters": failed,
        "bh_failed_parameters": bh_failed,
        "global_chi_squared": global_stat,
        "global_chi_squared_df": global_df,
        "global_chi_squared_p": global_p,
        "global_chi_squared_failed": global_failed,
        "chi2_bh_alpha": alpha,
    }


def _posterior_diagnostics_acceptance(posterior: dict[str, Any]) -> tuple[bool, list[str]]:
    diagnostics = posterior.get("diagnostics") or {}
    reasons: list[str] = []

    rhat = diagnostics.get("rhat_max")
    if rhat is None or not np.isfinite(float(rhat)):
        reasons.append("R_hat_missing")
    elif float(rhat) > SBC_R_HAT_MAX:
        reasons.append(f"R_hat={float(rhat):.4f}>{SBC_R_HAT_MAX}")

    ess_bulk = diagnostics.get("ess_bulk_min")
    if ess_bulk is None or not np.isfinite(float(ess_bulk)):
        reasons.append("ESS_bulk_missing")
    elif float(ess_bulk) < SBC_BULK_ESS_MIN:
        reasons.append(f"ESS_bulk={float(ess_bulk):.0f}<{SBC_BULK_ESS_MIN}")

    divergences = diagnostics.get("divergences")
    if divergences is None:
        reasons.append("divergences_missing")
    elif int(divergences) > SBC_DIVERGENCES_ALLOWED:
        reasons.append(f"divergences={int(divergences)}>{SBC_DIVERGENCES_ALLOWED}")

    return not reasons, reasons


def _rank_draw_count_for_accepted_worlds(accepted: list[dict[str, Any]]) -> int:
    if not accepted:
        return 0
    candidates = [int(p.get("n_draws", 0)) for p in accepted]
    for posterior in accepted:
        ess = (posterior.get("diagnostics") or {}).get("ess_bulk_min")
        if ess is not None and np.isfinite(float(ess)):
            candidates.append(int(np.floor(float(ess))))
    return max(0, min(SBC_RANK_DRAWS_TARGET, *candidates))


def _thin_draws(draws: np.ndarray, n_target: int) -> np.ndarray:
    arr = np.asarray(draws)
    if n_target <= 0:
        raise ValueError("n_target must be positive")
    if arr.shape[0] < n_target:
        raise ValueError("cannot thin to more draws than are present")
    if arr.shape[0] == n_target:
        return arr
    idx = np.linspace(0, arr.shape[0] - 1, n_target, dtype=int)
    return arr[idx]


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
    prior = prior or HyperPrior(R=2, T=12, L=3, K=3, S=3, M=2, H=0)

    accepted_worlds: list[tuple[Any, dict[str, Any], int]] = []
    discarded_worlds: list[dict[str, Any]] = []
    seeds_used = list(range(base_seed, base_seed + n_worlds))

    for w, seed in enumerate(seeds_used):
        world = sample_world(prior, seed=seed)
        posterior = fit_measurement_layer(world, seed=seed)
        accepted, reasons = _posterior_diagnostics_acceptance(posterior)
        if not accepted:
            discarded_worlds.append({
                "world_index": w,
                "seed": seed,
                "reasons": reasons,
                "diagnostics": posterior.get("diagnostics") or {},
            })
            continue
        accepted_worlds.append((world, posterior, seed))

    n_accepted = len(accepted_worlds)
    discard_fraction = (len(discarded_worlds) / n_worlds) if n_worlds else 0.0
    n_draws_per_fit = _rank_draw_count_for_accepted_worlds(
        [posterior for _world, posterior, _seed in accepted_worlds]
    )

    parameter_ranks: dict[str, list[int]] = {}
    if n_draws_per_fit >= 4:
        for world, posterior, _seed in accepted_worlds:
            for name, true_value in flatten_params(world.theta).items():
                draws = posterior["draws"].get(name)
                if draws is None:
                    continue
                thinned = _thin_draws(np.asarray(draws), n_draws_per_fit)
                parameter_ranks.setdefault(name, []).append(
                    rank_statistic(true_value, thinned)
                )

    if n_draws_per_fit < 4:
        parameter_ranks = {}

    # Build per-parameter diagnostics: KS + B-6 histogram shape
    per_param: dict[str, dict] = {}
    for name, ranks in parameter_ranks.items():
        arr = np.asarray(ranks)
        ks  = ks_uniformity_test(arr, n_draws_per_fit)
        hd  = histogram_diagnostics(arr, n_draws_per_fit)
        per_param[name] = {**ks, **hd}

    convergence_failed = (
        n_accepted == 0
        or n_draws_per_fit < 4
        or discard_fraction > SBC_MAX_DISCARD_FRACTION
    )

    convergence_reason = "sbc_world_convergence_screen_passed"
    if n_accepted == 0:
        convergence_reason = "all_sbc_worlds_discarded_by_convergence_screen"
    elif n_draws_per_fit < 4:
        convergence_reason = f"sbc_rank_draws_{n_draws_per_fit}_below_minimum_4"
    elif discard_fraction > SBC_MAX_DISCARD_FRACTION:
        convergence_reason = (
            f"sbc_discard_fraction_{discard_fraction:.3f}_exceeds_"
            f"{SBC_MAX_DISCARD_FRACTION}"
        )

    summary: dict[str, Any] = {
        "model_version": MODEL_VERSION,
        "n_worlds": n_worlds,
        "n_worlds_accepted": n_accepted,
        "n_worlds_discarded": len(discarded_worlds),
        "discard_fraction": discard_fraction,
        "discarded_worlds": discarded_worlds,
        "sbc_convergence_thresholds": {
            "rhat_max": SBC_R_HAT_MAX,
            "ess_bulk_min": SBC_BULK_ESS_MIN,
            "divergences_allowed": SBC_DIVERGENCES_ALLOWED,
            "max_discard_fraction": SBC_MAX_DISCARD_FRACTION,
            "rank_draws_target": SBC_RANK_DRAWS_TARGET,
        },
        "sbc_convergence_screen_status": "fail" if convergence_failed else "pass",
        "sbc_convergence_screen_reason": convergence_reason,
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
    summary["ks_failed_parameters_bonferroni"] = [
        name for name, diag in per_param.items()
        if diag["pvalue"] < effective_alpha
    ]
    gate = _sbc_gate_failures_from_diagnostics(per_param)
    summary.update(gate)
    failed = list(gate["failed_parameters"])
    if convergence_failed:
        failed = sorted(set(failed) | {"__sbc_convergence_screen__"})
    summary["failed_parameters"] = failed
    summary["gate_status"] = "pass" if not failed else "fail"
    summary["reason"] = (
        "rank_histograms_uniform" if not failed
        else (
            convergence_reason if convergence_failed
            else f"chi2_rank_histogram_fail_on_{len(failed)}_tests"
        )
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
    from ..model.fit import (
        _safe_fit_summary,
        assemble_fit_data_from_synthetic,
        _get_compiled_model,
        extract_theta_space_draws,
    )

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
    summary = _safe_fit_summary(fit)
    diagnostics = {
        "rhat_max": float(summary["R_hat"].dropna().max()),
        "ess_bulk_min": float(summary["ESS_bulk"].dropna().min()),
        "ess_tail_min": float(summary["ESS_tail"].dropna().min()),
        "divergences": int(fit.method_variables()["divergent__"].sum()),
    }
    return {"n_draws": n_draws, "draws": draws, "diagnostics": diagnostics}
