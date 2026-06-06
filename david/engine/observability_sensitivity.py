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


def lambda_grid(lambda_max: float = 0.25, n_steps: int = 6) -> np.ndarray:
    return np.linspace(0.0, lambda_max, n_steps)


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
