"""Theorem B' channel informativeness I(O) and information scaling N * I^2.

B'.1 (classification):  Bayes error for recovering A from a single Y is
                        (1 - I) / 2 where I = |rho(O) - delta(O)|.

B'.2 (estimation):      Effective sample size for estimating phi from N
                        replicates scales as N * I^2. Cells with low I(O)
                        are prior-dominated.

The cell-level gate is on the lower 95% credible bound of I(O). If it falls
below the floor, the cell is flagged prior_dominated and routed accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import INFORMATIVENESS_FLOOR_LOWER_95


@dataclass(frozen=True)
class InformativenessResult:
    cell_id: str
    posterior_median_I: float
    lower_95_I: float
    upper_95_I: float
    n_eff_phi_estimate: float    # N_replicates * I^2 (per B'.2)
    gate_status: str             # "pass" | "fail"
    reason: str


def informativeness_draws(
    rho_draws: np.ndarray,         # (D,)
    delta_draws: np.ndarray,       # (D,)
) -> np.ndarray:
    """Per-draw I(O) = |rho - delta|. Returns shape (D,)."""
    if rho_draws.shape != delta_draws.shape:
        raise ValueError("rho and delta draws must share shape")
    return np.abs(rho_draws - delta_draws)


def bayes_classification_error(I: float) -> float:
    """B'.1 Bayes error rate under equal prior odds."""
    if not 0.0 <= I <= 1.0:
        raise ValueError("I must be in [0, 1]")
    return 0.5 * (1.0 - I)


def effective_sample_size(n_replicates: int, I: float) -> float:
    """B'.2 effective N for estimating phi from N detections at informativeness I."""
    return float(n_replicates) * (I ** 2)


def check_cell(
    cell_id: str,
    rho_draws: np.ndarray,
    delta_draws: np.ndarray,
    n_replicates: int,
    floor_lower_95: float = INFORMATIVENESS_FLOOR_LOWER_95,
) -> InformativenessResult:
    I = informativeness_draws(rho_draws, delta_draws)
    median = float(np.median(I))
    lower, upper = (float(x) for x in np.quantile(I, [0.025, 0.975]))
    n_eff = effective_sample_size(n_replicates, median)
    if lower >= floor_lower_95:
        return InformativenessResult(
            cell_id=cell_id,
            posterior_median_I=median,
            lower_95_I=lower,
            upper_95_I=upper,
            n_eff_phi_estimate=n_eff,
            gate_status="pass",
            reason="I_lower_95_above_floor",
        )
    return InformativenessResult(
        cell_id=cell_id,
        posterior_median_I=median,
        lower_95_I=lower,
        upper_95_I=upper,
        n_eff_phi_estimate=n_eff,
        gate_status="fail",
        reason=f"I_lower_95_{lower:.4f}_below_floor_{floor_lower_95:.4f}",
    )
