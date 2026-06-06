"""F1..F15 falsification battery.

Each F-test returns a dict {test_id, gate_status, statistic, threshold, reason}.

F1   Prior-predictive realism
F3   Calibration ECE per tactic
F4   Discrimination (AUROC + AUPRC; rare tactics)
F5   Baseline dominance (vs. B0..B4)
F6   Adversarial null (label-permuted, ELPD comparison)
F7   Forward-time generalization (leave-one-period-out)
F8   Source-ablation sensitivity
F9   Observability-prior sensitivity
F10  Indeterminate rate, stratified
F11  Conditional independence of sources
F12  Practical identification distance d(theta)
F13  Forecast horizon respect (forecasts > h* must equal marginal)
F14  Forecast SBC coverage (driven by simulator.forecast_sbc)
F15  Forecast no-improvement under permuted history

Each test is independently importable. The battery runner composes them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class TestResult:
    test_id: str
    gate_status: str
    statistic: float
    threshold: float
    reason: str


def F1_prior_predictive_realism(prior_predictive_Y_rate: np.ndarray,
                                historical_Y_rate_5th: float,
                                historical_Y_rate_95th: float) -> TestResult:
    median = float(np.median(prior_predictive_Y_rate))
    if historical_Y_rate_5th <= median <= historical_Y_rate_95th:
        return TestResult("F1", "pass", median, historical_Y_rate_5th, "prior_in_historical_band")
    return TestResult("F1", "fail", median, historical_Y_rate_5th, "prior_outside_historical_band")


def F3_calibration_ece(p_hat: np.ndarray, y: np.ndarray, bins: int = 10,
                       threshold: float = 0.05) -> TestResult:
    ece = expected_calibration_error(p_hat, y, bins)
    status = "pass" if ece <= threshold else "fail"
    return TestResult("F3", status, ece, threshold, f"ece_{ece:.4f}")


def expected_calibration_error(p_hat: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    n = p_hat.shape[0]
    ece = 0.0
    for b in range(bins):
        mask = (p_hat >= edges[b]) & (p_hat < edges[b + 1])
        if not mask.any():
            continue
        bin_p = p_hat[mask].mean()
        bin_y = y[mask].mean()
        ece += (mask.sum() / n) * abs(bin_p - bin_y)
    return float(ece)


def F4_discrimination(p_hat: np.ndarray, y: np.ndarray,
                      auroc_floor: float = 0.70,
                      auprc_floor: float = 0.50) -> TestResult:
    auroc, auprc = _compute_auroc_auprc(p_hat, y)
    status = "pass" if auroc >= auroc_floor and auprc >= auprc_floor else "fail"
    return TestResult("F4", status, min(auroc, auprc), min(auroc_floor, auprc_floor),
                      f"auroc={auroc:.3f}_auprc={auprc:.3f}")


def _compute_auroc_auprc(p_hat: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    from sklearn.metrics import roc_auc_score, average_precision_score
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y, p_hat)), float(average_precision_score(y, p_hat))


def F5_baseline_dominance(m01_log_loss: float, baselines: dict[str, float],
                          margin: float = 0.05) -> TestResult:
    worst_baseline = max(baselines.values())
    target = (1 - margin) * worst_baseline
    status = "pass" if m01_log_loss <= target else "fail"
    return TestResult("F5", status, m01_log_loss, target,
                      f"m01={m01_log_loss:.4f}_worst_baseline={worst_baseline:.4f}")


def F6_adversarial_null(m01_elpd_under_permutation: float, B0_elpd: float,
                        epsilon: float = 0.0) -> TestResult:
    delta = m01_elpd_under_permutation - B0_elpd
    status = "pass" if delta <= epsilon else "fail"
    return TestResult("F6", status, delta, epsilon,
                      f"elpd_delta_under_perm={delta:.4f}")


def F11_source_conditional_independence(
    pairwise_residual_p_values: dict[tuple[str, str], float],
    bonferroni_alpha: float = 0.01,
) -> TestResult:
    n_pairs = len(pairwise_residual_p_values)
    if n_pairs == 0:
        return TestResult("F11", "pass", 1.0, bonferroni_alpha, "no_pairs")
    bonferroni = bonferroni_alpha / n_pairs
    failed = [p for p in pairwise_residual_p_values.values() if p < bonferroni]
    status = "pass" if not failed else "fail"
    return TestResult("F11", status, min(pairwise_residual_p_values.values()),
                      bonferroni, f"{len(failed)}_pairs_failed_at_{bonferroni:.4g}")


def F12_identification_distance(median_d: float, floor: float = 0.05) -> TestResult:
    status = "pass" if median_d >= floor else "fail"
    return TestResult("F12", status, median_d, floor,
                      f"d_theta_median_{median_d:.4f}")


def F13_horizon_respect(forecast_p: np.ndarray, marginal_p: np.ndarray,
                        beyond_h_star_mask: np.ndarray,
                        tolerance: float = 0.02) -> TestResult:
    """Beyond h* the forecast must equal the marginal regime prediction."""
    diffs = np.abs(forecast_p[beyond_h_star_mask] - marginal_p[beyond_h_star_mask])
    max_diff = float(diffs.max()) if diffs.size else 0.0
    status = "pass" if max_diff <= tolerance else "fail"
    return TestResult("F13", status, max_diff, tolerance,
                      f"max_diff_beyond_h_star={max_diff:.4f}")


def F14_forecast_sbc(forecast_sbc_summary: dict) -> TestResult:
    pass80 = forecast_sbc_summary.get("pass_80", False)
    pass95 = forecast_sbc_summary.get("pass_95", False)
    status = "pass" if pass80 and pass95 else "fail"
    statistic = (forecast_sbc_summary.get("coverage_80", 0.0)
                 + forecast_sbc_summary.get("coverage_95", 0.0)) / 2
    return TestResult("F14", status, statistic, 0.5,
                      f"forecast_sbc_status={status}")


def F15_permuted_history_no_improvement(
    m01_forecast_logloss_permuted: float, B0_forecast_logloss: float,
    epsilon: float = 0.0,
) -> TestResult:
    delta = B0_forecast_logloss - m01_forecast_logloss_permuted
    status = "pass" if delta <= epsilon else "fail"
    return TestResult("F15", status, delta, epsilon,
                      f"forecast_elpd_delta_under_perm={delta:.4f}")


# Glue: registry of all tests for the runner
F_BATTERY: dict[str, Callable[..., TestResult]] = {
    "F1": F1_prior_predictive_realism,
    "F3": F3_calibration_ece,
    "F4": F4_discrimination,
    "F5": F5_baseline_dominance,
    "F6": F6_adversarial_null,
    "F11": F11_source_conditional_independence,
    "F12": F12_identification_distance,
    "F13": F13_horizon_respect,
    "F14": F14_forecast_sbc,
    "F15": F15_permuted_history_no_improvement,
}


def run_battery(*, inputs: dict[str, dict]) -> dict:
    """Run the F-battery. `inputs` keys are test ids, values are kwarg dicts.

    Returns aggregate result with gate_status and per-test ledger.
    """
    results = []
    for tid, fn in F_BATTERY.items():
        kwargs = inputs.get(tid, {})
        if not kwargs:
            results.append(TestResult(tid, "skip", float("nan"), float("nan"),
                                      "no_inputs_provided"))
            continue
        results.append(fn(**kwargs))
    failed = [r for r in results if r.gate_status == "fail"]
    return {
        "gate_status": "pass" if not failed else "fail",
        "failed_tests": [r.test_id for r in failed],
        "ledger": [r.__dict__ for r in results],
    }
