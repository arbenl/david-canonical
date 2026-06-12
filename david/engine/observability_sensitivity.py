"""Endogenous-observability sensitivity (Gap-5 / partial identification).

The DAG edge A -> O (active interference suppresses observability) breaks the
exogeneity assumption of the standard detection model. We do not point-identify
its size; we report a partial-identification interval.

Model:
    O(lambda) = O_baseline - lambda * H_t
where H_t is a hiddenness factor (e.g., indicator of high-concealment regime).

For lambda in [0, lambda_max], recompute the posterior P(A=1 | data, lambda)
and report the [min, max] interval across the lambda grid.

lambda_max is elicited externally (default 0.25) and reviewed quarterly.

Pre-registered Θ^meas grid
--------------------------
`theta_meas_grid()` returns the canonical sensitivity grid used by Theorem C
(Sensitivity-envelope FDP, C-3). The grid parameters are fixed at
pre-registration and must not be changed at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LambdaBoundsResult:
    cell_id: str
    p_active_lower: float
    p_active_upper: float
    interval_width: float
    lambda_grid: list[float]
    p_active_curve: list[float]


# Pre-registered Θ^meas grid parameters — immutable at runtime.
_THETA_MEAS_LAMBDA_MAX: float = 0.25
_THETA_MEAS_N_STEPS: int = 6


def lambda_grid(lambda_max: float = 0.25, n_steps: int = 6) -> np.ndarray:
    return np.linspace(0.0, lambda_max, n_steps)


def theta_meas_grid() -> np.ndarray:
    """Return the pre-registered Θ^meas sensitivity grid (lambda values).

    This is the grid over which p_i^- = min_θ p_i^θ is computed in Theorem C
    (Sensitivity-envelope FDP). Fixed at pre-registration; immutable at runtime.
    """
    return lambda_grid(_THETA_MEAS_LAMBDA_MAX, _THETA_MEAS_N_STEPS)


def evaluate_cell(
    cell_id: str,
    p_active_fn,                # callable: lambda -> p_active (posterior median)
    lambda_max: float = 0.25,
    n_steps: int = 6,
) -> LambdaBoundsResult:
    grid = lambda_grid(lambda_max, n_steps)
    p_curve = np.array([float(p_active_fn(l)) for l in grid])
    lo, hi = float(p_curve.min()), float(p_curve.max())
    return LambdaBoundsResult(
        cell_id=cell_id,
        p_active_lower=lo,
        p_active_upper=hi,
        interval_width=hi - lo,
        lambda_grid=grid.tolist(),
        p_active_curve=p_curve.tolist(),
    )
