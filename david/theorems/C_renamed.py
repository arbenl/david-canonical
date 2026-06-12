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

from ..config import (
    POSTERIOR_FDP_DEFAULT_Q,
    FDP_EXCEEDANCE_GAMMA,
    FDP_EXCEEDANCE_ALPHA,
    FDP_MCSE_MARGIN_RATIO,
)


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


@dataclass(frozen=True)
class McseFloorResult:
    """Per-cell MCSE/ESS floor check (Theorem C, C-6).

    Fields
    ------
    n_cells_checked   : number of cells inspected (= columns of joint_draws).
    n_cells_excluded  : cells whose MCSE(p_i) ≥ mcse_threshold.
    excluded_indices  : column indices (original ordering) that failed.
    bulk_ess          : estimated bulk-ESS per cell (parallel to input columns).
    mcse              : MCSE(p_i) per cell (std / sqrt(bulk_ESS)).
    mcse_threshold    : the floor applied: FDP_MCSE_MARGIN_RATIO × q.
    binding_reason    : typed string for excluded cells ("fdp_mcse_exceeds_gate_margin").
    """
    n_cells_checked: int
    n_cells_excluded: int
    excluded_indices: list[int]
    bulk_ess: list[float]
    mcse: list[float]
    mcse_threshold: float
    binding_reason: str = "fdp_mcse_exceeds_gate_margin"


def _bulk_ess_1d(chain: np.ndarray) -> float:
    """Geyer initial positive-sequence bulk-ESS for a 1-D MCMC chain."""
    n = len(chain)
    if n < 4:
        return float(n)
    v = chain - chain.mean()
    # Circular autocorrelation via FFT (O(n log n))
    f = np.fft.rfft(v, n=2 * n)
    acov = np.fft.irfft(f * f.conj())[:n].real / n
    if acov[0] == 0.0:
        return float(n)
    rho = acov / acov[0]
    # Sum consecutive pairs (Geyer 1992) until the running sum goes negative.
    s = 1.0
    k = 1
    while k + 1 < n:
        pair_sum = rho[k] + rho[k + 1]
        if pair_sum < 0.0:
            break
        s += 2.0 * pair_sum
        k += 2
    return float(n) / max(s, 1.0)


def check_mcse_floor(
    joint_draws: np.ndarray,
    q: float = POSTERIOR_FDP_DEFAULT_Q,
    mcse_margin_ratio: float = FDP_MCSE_MARGIN_RATIO,
) -> McseFloorResult:
    """C-6: MCSE/ESS floor check on p_i from joint posterior draws.

    For each cell i, computes MCSE(p_i) = std(A_i^(s)) / sqrt(bulk_ESS_i)
    and excludes the cell if MCSE(p_i) ≥ mcse_margin_ratio × q.

    σ(D)-MEASURABILITY NOTE — this function uses joint draws ONLY to estimate
    the Monte Carlo error of p_i, not to compute the FDP gate itself.  The
    exclusion decision is based on the scalar MCSE statistic per cell; it does
    not depend on cross-cell joint structure.

    Parameters
    ----------
    joint_draws:
        Shape (n_draws, n_cells).  Column i is the draw sequence A_i^(s).
    q:
        Target FDP level (same q used in the expectation gate).
    mcse_margin_ratio:
        Pre-registered constant (default FDP_MCSE_MARGIN_RATIO = 0.10).
        Cells are excluded when MCSE(p_i) ≥ mcse_margin_ratio × q.

    Returns
    -------
    McseFloorResult.  ``excluded_indices`` lists the column indices whose
    MCSE exceeds the threshold; the caller is responsible for removing those
    cells from the claim-eligible family before calling
    ``compute_posterior_fdp_threshold``.
    """
    if joint_draws.ndim != 2:
        raise ValueError("joint_draws must be 2-D: (n_draws, n_cells)")
    n_draws, n_cells = joint_draws.shape
    if n_draws < 4:
        raise ValueError("joint_draws must contain at least 4 draws for ESS estimation")

    threshold = mcse_margin_ratio * q
    draws = joint_draws.astype(float)

    bulk_ess_vals: list[float] = []
    mcse_vals: list[float] = []
    excluded: list[int] = []

    for i in range(n_cells):
        col = draws[:, i]
        ess = _bulk_ess_1d(col)
        std_i = float(np.std(col))
        mcse_i = std_i / max(np.sqrt(ess), 1.0)
        bulk_ess_vals.append(ess)
        mcse_vals.append(mcse_i)
        if mcse_i >= threshold:
            excluded.append(i)

    return McseFloorResult(
        n_cells_checked=n_cells,
        n_cells_excluded=len(excluded),
        excluded_indices=excluded,
        bulk_ess=bulk_ess_vals,
        mcse=mcse_vals,
        mcse_threshold=threshold,
    )


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
