"""Theorem A' practical identification distance.

For each stratum g, the identification distance is

    d(theta) = min(
        J_(3),                     # third-largest Youden J across source families
        min(phi_g, 1 - phi_g)      # activity margin
    )

where J_(3) = third-largest |rho_s - delta_s| across source families s.
Kruskal identifiability requires three informative independent source views, not
all S views. Using min_s J_s would penalise an identified stratum because one
nuisance source is weak. When S < 3 the stratum is under-sourced; the
conventional min_s is used as a conservative fallback (the gate will likely
fail on d_theta anyway since the rank condition is not met).

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

    J_draws = np.abs(rho_draws - delta_draws)  # (D, S); Youden J per draw per source
    S = J_draws.shape[1]
    if S >= 3:
        # Kruskal requires 3 informative views, not all S. Sort descending and
        # take the third-largest J so a single weak nuisance source cannot veto
        # an otherwise identified stratum.
        J_sorted = np.sort(J_draws, axis=1)[:, ::-1]  # (D, S) descending
        per_draw_source = J_sorted[:, 2]               # J_(3)
    else:
        per_draw_source = J_draws.min(axis=1)          # S<3: under-sourced fallback
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
