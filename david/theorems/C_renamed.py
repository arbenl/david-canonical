"""Theorem C (renamed) Bayesian posterior expected FDP control.

Given posterior probabilities p_i = P(A_i = 1 | data) for cells i = 1..M,
sort descending and choose the largest set whose mean (1 - p_i) is at most q.

  m_star = max { m : (1/m) sum_{j=1..m} (1 - p_(j)) <= q }

Flagging the top m_star cells guarantees that the posterior expected false
discovery proportion is at most q. This is NOT frequentist BH-FDR.

Sensitivity-envelope FDP (C-3)
-------------------------------
Under partial observability, p_i depends on the sensitivity parameter θ ∈ Θ^meas.
`compute_posterior_fdp_envelope` computes p_i^- = min_θ p_i^θ over the
pre-registered grid (defined in `david.engine.observability_sensitivity.theta_meas_grid`)
and runs the prefix scan on these conservative values.  Because p_i^- ≤ p_i^θ
for all θ, the FDP bound computed from p^- dominates the bound at every grid
point: FDP_env(R) ≤ q_d at the worst θ ∈ Θ^meas.  The scan also restores
monotonicity that would otherwise be broken by point-specific θ choices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import POSTERIOR_FDP_DEFAULT_Q


@dataclass(frozen=True)
class PosteriorFdpResult:
    q_target: float
    n_cells: int
    n_flagged: int
    threshold_p: float
    posterior_expected_fdp_at_threshold: float
    flagged_indices: list[int]


@dataclass(frozen=True)
class SensitivityEnvelopeResult:
    q_target: float
    n_cells: int
    n_flagged: int
    threshold_p: float
    posterior_expected_fdp_at_threshold: float
    flagged_indices: list[int]
    # Index into Θ^meas of the most conservative grid point (smallest sum(p_i)).
    worst_theta_index: int


def compute_posterior_fdp_threshold(
    p_hat: np.ndarray,
    q: float = POSTERIOR_FDP_DEFAULT_Q,
) -> PosteriorFdpResult:
    if p_hat.ndim != 1:
        raise ValueError("p_hat must be 1-D over cells")
    M = p_hat.shape[0]
    order = np.argsort(-p_hat)               # descending by p_i
    sorted_p = p_hat[order]
    cumulative_fp = np.cumsum(1.0 - sorted_p)
    # posterior expected FDP at flagging the top m cells
    posterior_fdp_at_m = cumulative_fp / np.arange(1, M + 1)
    eligible = posterior_fdp_at_m <= q
    if not eligible.any():
        return PosteriorFdpResult(
            q_target=q,
            n_cells=M,
            n_flagged=0,
            threshold_p=1.0,
            posterior_expected_fdp_at_threshold=0.0,
            flagged_indices=[],
        )
    m_star = int(np.max(np.where(eligible)[0]) + 1)
    threshold = float(sorted_p[m_star - 1])
    fdp_at_threshold = float(posterior_fdp_at_m[m_star - 1])
    flagged = order[:m_star].tolist()
    return PosteriorFdpResult(
        q_target=q,
        n_cells=M,
        n_flagged=m_star,
        threshold_p=threshold,
        posterior_expected_fdp_at_threshold=fdp_at_threshold,
        flagged_indices=flagged,
    )


def compute_posterior_fdp_envelope(
    p_hat_per_theta: np.ndarray,
    q: float = POSTERIOR_FDP_DEFAULT_Q,
) -> SensitivityEnvelopeResult:
    """Sensitivity-envelope FDP control (Theorem C, C-3).

    Parameters
    ----------
    p_hat_per_theta:
        Shape (n_theta, n_cells).  Row i is the posterior p_i^θ evaluated at
        the i-th point of the pre-registered grid Θ^meas (see
        `david.engine.observability_sensitivity.theta_meas_grid`).
    q:
        Target FDP level (default POSTERIOR_FDP_DEFAULT_Q = 0.10).

    Returns
    -------
    SensitivityEnvelopeResult with the conservative flag set and the index of
    the most conservative grid point.

    Mathematical guarantee
    ----------------------
    p_i^- = min_θ p_i^θ  ⟹  FDP_env(R) ≤ q_d at every θ ∈ Θ^meas.
    The σ(D)-measurability contract is preserved: this function consumes only
    marginal posteriors, never joint draws.
    """
    if p_hat_per_theta.ndim != 2:
        raise ValueError("p_hat_per_theta must be 2-D: (n_theta, n_cells)")

    # p_i^- = element-wise minimum over the sensitivity grid.
    p_conservative = p_hat_per_theta.min(axis=0)  # shape (n_cells,)

    # Most conservative grid point: θ that minimises the total probability
    # mass (highest aggregate FDP pressure if applied naively).
    worst_theta_index = int(np.argmin(p_hat_per_theta.sum(axis=1)))

    inner = compute_posterior_fdp_threshold(p_conservative, q=q)
    return SensitivityEnvelopeResult(
        q_target=inner.q_target,
        n_cells=inner.n_cells,
        n_flagged=inner.n_flagged,
        threshold_p=inner.threshold_p,
        posterior_expected_fdp_at_threshold=inner.posterior_expected_fdp_at_threshold,
        flagged_indices=inner.flagged_indices,
        worst_theta_index=worst_theta_index,
    )
