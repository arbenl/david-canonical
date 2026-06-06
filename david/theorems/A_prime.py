"""Theorem A' practical identification distance.

For each stratum g, the identification distance is

    d(theta) = min over s of {
        min(|rho_s - delta_s|, |rho_s - (1 - delta_s)|),
        min(phi_g, 1 - phi_g)
    }

Strata with posterior median d below `floor` are flagged
practically_non_identified. They are generically identifiable per Allman-Matias-
Rhodes 2009 but operationally too close to the singular set to support
cell-level claims.

Inputs are posterior draws as numpy arrays. No Stan dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..config import ID_DISTANCE_FLOOR


@dataclass(frozen=True)
class IdentificationDistanceResult:
    stratum_id: str
    posterior_median: float
    posterior_q05: float
    posterior_q95: float
    gate_status: str  # "pass" | "fail"
    reason: str


def identification_distance_draws(
    phi_draws: np.ndarray,         # shape (D,)
    rho_draws: np.ndarray,         # shape (D, S)
    delta_draws: np.ndarray,       # shape (D, S)
) -> np.ndarray:
    """Per-draw identification distance d(theta).

    Returns array shape (D,).
    """
    if phi_draws.ndim != 1 or rho_draws.ndim != 2 or delta_draws.ndim != 2:
        raise ValueError("phi_draws must be 1-D and rho/delta must be 2-D (D, S)")
    if rho_draws.shape != delta_draws.shape:
        raise ValueError("rho and delta draws must share shape")
    if phi_draws.shape[0] != rho_draws.shape[0]:
        raise ValueError("draw dimension mismatch between phi and rho")

    informativeness = np.abs(rho_draws - delta_draws)  # (D, S)
    label_flip_safety = np.abs(rho_draws - (1.0 - delta_draws))  # (D, S)
    source_min = np.minimum(informativeness, label_flip_safety)  # (D, S)
    per_draw_source = source_min.min(axis=1)  # (D,)
    phi_boundary = np.minimum(phi_draws, 1.0 - phi_draws)
    return np.minimum(per_draw_source, phi_boundary)


def check_stratum(
    stratum_id: str,
    phi_draws: np.ndarray,
    rho_draws: np.ndarray,
    delta_draws: np.ndarray,
    floor: float = ID_DISTANCE_FLOOR,
) -> IdentificationDistanceResult:
    d = identification_distance_draws(phi_draws, rho_draws, delta_draws)
    median = float(np.median(d))
    q05, q95 = (float(x) for x in np.quantile(d, [0.05, 0.95]))
    if median >= floor:
        return IdentificationDistanceResult(
            stratum_id=stratum_id,
            posterior_median=median,
            posterior_q05=q05,
            posterior_q95=q95,
            gate_status="pass",
            reason="d_theta_above_floor",
        )
    return IdentificationDistanceResult(
        stratum_id=stratum_id,
        posterior_median=median,
        posterior_q05=q05,
        posterior_q95=q95,
        gate_status="fail",
        reason=f"d_theta_median_{median:.4f}_below_floor_{floor:.4f}",
    )


def batch(
    posterior: Mapping[str, np.ndarray],
    stratum_ids: list[str],
    floor: float = ID_DISTANCE_FLOOR,
) -> list[IdentificationDistanceResult]:
    """Apply check_stratum across strata.

    Expects posterior[`phi_{g}`], posterior[`rho_{g}`], posterior[`delta_{g}`].
    """
    results = []
    for g in stratum_ids:
        results.append(
            check_stratum(
                stratum_id=g,
                phi_draws=posterior[f"phi_{g}"],
                rho_draws=posterior[f"rho_{g}"],
                delta_draws=posterior[f"delta_{g}"],
                floor=floor,
            )
        )
    return results
