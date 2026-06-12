"""Theorem C (renamed) Bayesian posterior expected FDP control.

Given posterior probabilities p_i = P(A_i = 1 | data) for cells i = 1..M,
sort descending and choose the largest set whose mean (1 - p_i) is at most q.

  m_star = max { m : (1/m) sum_{j=1..m} (1 - p_(j)) <= q }

Flagging the top m_star cells guarantees that the posterior expected false
discovery proportion is at most q. This is NOT frequentist BH-FDR.

σ(D)-measurability contract
----------------------------
The expectation gate (`compute_posterior_fdp_threshold`) consumes ONLY marginal
posteriors p_i.  The exceedance gate (`compute_fdp_exceedance_gate`) consumes
ONLY joint posterior draws A_i^(s).  These are separate functions by design:
the Theorem C proof depends on the measurability of the expectation gate with
respect to the marginal σ-algebra σ(D); merging them would destroy that property.

FG6 exceedance gate (C-4)
--------------------------
For prefix m (D = m flags), the empirical FDP in posterior draw s is:

  FDP^(s) = sum_{i=1}^{m} (1 - A_i^(s)) / max(1, m)

Accept the largest prefix m ≤ m* such that the empirical exceedance fraction
P̂(FDP^(s) > γ) ≤ α.  Exceedance is not monotone in m (dilution by adding a
high-confidence cell can push the empirical FDP below γ), so the scan runs
downward from m* — not upward.

Markov fallback: by Markov's inequality, P(FDP > γ) ≤ E[FDP]/γ ≤ q/γ.
If q ≤ γα the expectation rule alone guarantees the exceedance condition and
the downward scan is skipped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import POSTERIOR_FDP_DEFAULT_Q, FDP_EXCEEDANCE_GAMMA, FDP_EXCEEDANCE_ALPHA


@dataclass(frozen=True)
class PosteriorFdpResult:
    q_target: float
    n_cells: int
    n_flagged: int
    threshold_p: float
    posterior_expected_fdp_at_threshold: float
    flagged_indices: list[int]


@dataclass(frozen=True)
class ExceedanceGateResult:
    """Result of the FG6 exceedance gate from joint posterior draws (C-4).

    Fields
    ------
    m_star_expectation : m* from the marginal expectation rule (upper bound).
    m_accepted         : largest prefix accepted by the downward scan (≤ m*).
    exceedance_fraction_at_accepted : P̂(FDP^(s) > γ) at m_accepted (0.0 when
        m_accepted == 0, because no claims ⇒ FDP = 0 trivially).
    markov_fallback_used : True iff q ≤ γα triggered the Markov shortcut.
    """
    m_star_expectation: int
    m_accepted: int
    exceedance_fraction_at_accepted: float
    gamma: float
    alpha: float
    markov_fallback_used: bool


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


def compute_fdp_exceedance_gate(
    joint_draws: np.ndarray,
    m_star: int,
    gamma: float = FDP_EXCEEDANCE_GAMMA,
    alpha: float = FDP_EXCEEDANCE_ALPHA,
    q: float | None = None,
) -> ExceedanceGateResult:
    """FG6 exceedance gate from joint posterior draws (Theorem C, C-4).

    σ(D)-MEASURABILITY GUARD — this function is intentionally separate from
    ``compute_posterior_fdp_threshold``.  The expectation gate takes ONLY
    marginals; this gate takes ONLY joint draws.  Do not merge them.

    Parameters
    ----------
    joint_draws:
        Shape ``(n_draws, M)``.  Boolean/0-1 array where ``joint_draws[s, i]``
        is the posterior draw A_i^(s), ordered identically to the sorted
        marginals (i.e. column 0 = cell with the highest p_i, column M-1 =
        lowest).  The caller is responsible for reordering columns to match
        the descending-p_i sort used by ``compute_posterior_fdp_threshold``.
    m_star:
        Maximum candidate prefix from the marginal expectation rule
        (= ``PosteriorFdpResult.n_flagged``).
    gamma:
        FDP exceedance threshold (default ``FDP_EXCEEDANCE_GAMMA = 0.15``).
        A draw s "exceeds" when ``FDP^(s) > gamma``.
    alpha:
        Maximum tolerated exceedance fraction (default
        ``FDP_EXCEEDANCE_ALPHA = 0.05``).
    q:
        If provided, apply the Markov fallback: ``P(FDP > γ) ≤ E[FDP]/γ ≤
        q/γ``.  When ``q ≤ γ·α`` the expectation rule alone guarantees the
        exceedance condition, so ``m_star`` is accepted without scanning.

    Returns
    -------
    ExceedanceGateResult
        ``.m_accepted`` is the shrunken prefix size (≤ m_star).  When no
        prefix passes the scan returns 0 (no claims).
    """
    if joint_draws.ndim != 2:
        raise ValueError("joint_draws must be 2-D: (n_draws, M)")
    n_draws, M = joint_draws.shape
    if n_draws == 0:
        raise ValueError("joint_draws must contain at least one draw")
    if m_star < 0:
        raise ValueError("m_star must be non-negative")
    if m_star == 0:
        return ExceedanceGateResult(
            m_star_expectation=0, m_accepted=0,
            exceedance_fraction_at_accepted=0.0,
            gamma=gamma, alpha=alpha, markov_fallback_used=False,
        )
    if m_star > M:
        raise ValueError(f"m_star={m_star} exceeds joint_draws.shape[1]={M}")

    # Markov fallback: P(FDP > γ) ≤ E[FDP]/γ ≤ q/γ.
    # When q ≤ γ·α the expectation rule already guarantees exceedance ≤ α.
    if q is not None and q <= gamma * alpha:
        return ExceedanceGateResult(
            m_star_expectation=m_star, m_accepted=m_star,
            exceedance_fraction_at_accepted=0.0,
            gamma=gamma, alpha=alpha, markov_fallback_used=True,
        )

    draws = joint_draws.astype(float)

    # Downward scan from m_star.
    # Exceedance is NOT monotone in m (adding a high-confidence TP dilutes the
    # FDP ratio below γ, potentially lowering the exceedance fraction even as
    # m grows).  We therefore scan downward and return the FIRST (= LARGEST)
    # m whose empirical exceedance fraction is ≤ α.
    for m in range(m_star, 0, -1):
        fp_per_draw = (1.0 - draws[:, :m]).sum(axis=1) / m
        exceedance_frac = float((fp_per_draw > gamma).mean())
        if exceedance_frac <= alpha:
            return ExceedanceGateResult(
                m_star_expectation=m_star, m_accepted=m,
                exceedance_fraction_at_accepted=exceedance_frac,
                gamma=gamma, alpha=alpha, markov_fallback_used=False,
            )

    # No prefix passes: zero claims (FDP = 0 trivially at m = 0).
    return ExceedanceGateResult(
        m_star_expectation=m_star, m_accepted=0,
        exceedance_fraction_at_accepted=0.0,
        gamma=gamma, alpha=alpha, markov_fallback_used=False,
    )
