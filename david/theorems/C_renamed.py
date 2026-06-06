"""Theorem C (renamed) Bayesian posterior expected FDP control.

Given posterior probabilities p_i = P(A_i = 1 | data) for cells i = 1..M,
sort descending and choose the largest set whose mean (1 - p_i) is at most q.

  m_star = max { m : (1/m) sum_{j=1..m} (1 - p_(j)) <= q }

Flagging the top m_star cells guarantees that the posterior expected false
discovery proportion is at most q. This is NOT frequentist BH-FDR.
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
