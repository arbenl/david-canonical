"""Tests for Theorem C posterior expected FDP threshold and exceedance gate (C-4)."""

from __future__ import annotations

import numpy as np
import pytest

from david.theorems.C_renamed import (
    compute_posterior_fdp_threshold,
    compute_fdp_exceedance_gate,
)


# ---------------------------------------------------------------------------
# Marginal expectation rule (pre-existing)
# ---------------------------------------------------------------------------

def test_no_flag_when_all_low():
    p = np.full(100, 0.05)
    res = compute_posterior_fdp_threshold(p, q=0.10)
    assert res.n_flagged == 0


def test_flag_only_high_confidence():
    p = np.concatenate([np.full(90, 0.05), np.full(10, 0.95)])
    res = compute_posterior_fdp_threshold(p, q=0.10)
    # Top 10 cells have (1 - 0.95) = 0.05 expected FDP, below q
    assert res.n_flagged == 10
    assert res.posterior_expected_fdp_at_threshold <= 0.10


def test_q_strict_blocks_lower_confidence():
    p = np.array([0.99, 0.95, 0.90, 0.80, 0.50])
    res = compute_posterior_fdp_threshold(p, q=0.05)
    # 0.99 alone gives FDP 0.01 <= 0.05 (pass);
    # 0.99 and 0.95 give FDP 0.03 (pass);
    # 0.99, 0.95, 0.90 give FDP ~0.053 (fail)
    assert res.n_flagged == 2


# ---------------------------------------------------------------------------
# C-4: FG6 exceedance gate — input validation
# ---------------------------------------------------------------------------

def test_exceedance_gate_rejects_1d_input():
    with pytest.raises(ValueError, match="2-D"):
        compute_fdp_exceedance_gate(np.array([0.9, 0.8, 0.7]), m_star=2)


def test_exceedance_gate_rejects_zero_draws():
    with pytest.raises(ValueError, match="at least one draw"):
        compute_fdp_exceedance_gate(np.empty((0, 3)), m_star=2)


def test_exceedance_gate_rejects_m_star_exceeds_columns():
    draws = np.ones((100, 3))
    with pytest.raises(ValueError, match="m_star=5 exceeds"):
        compute_fdp_exceedance_gate(draws, m_star=5)


def test_exceedance_gate_m_star_zero_returns_immediately():
    draws = np.ones((100, 5))
    res = compute_fdp_exceedance_gate(draws, m_star=0)
    assert res.m_accepted == 0
    assert res.m_star_expectation == 0
    assert not res.markov_fallback_used


# ---------------------------------------------------------------------------
# C-4: Markov fallback
# ---------------------------------------------------------------------------

def test_exceedance_gate_markov_fallback_triggered():
    """When q ≤ γ·α the Markov inequality guarantees exceedance ≤ α.

    q=0.005, γ=0.15, α=0.05 → γ·α = 0.0075 ≥ q → fallback fires.
    """
    draws = np.zeros((1000, 5))  # all false positives — worst case draws
    res = compute_fdp_exceedance_gate(
        draws, m_star=5, gamma=0.15, alpha=0.05, q=0.005
    )
    assert res.markov_fallback_used
    assert res.m_accepted == res.m_star_expectation == 5


def test_exceedance_gate_markov_fallback_not_triggered_with_default_q():
    """Default q=0.10, γ=0.15, α=0.05 → γ·α=0.0075 < q → fallback does NOT fire."""
    # All draws are true positives so exceedance is 0 regardless
    draws = np.ones((100, 3))
    res = compute_fdp_exceedance_gate(draws, m_star=3, q=0.10)
    assert not res.markov_fallback_used


# ---------------------------------------------------------------------------
# C-4: Core acceptance criterion — exceedance shrinks the naive flag set
# ---------------------------------------------------------------------------

def test_exceedance_gate_shrinks_set_correlated_draws():
    """C-4 acceptance criterion.

    5 headline cells (p sorted descending).  The marginal expectation rule
    (q = 0.10) flags all 5 (m* = 5).

    Joint draws: 1000 total.
    - 940 "good" draws: all 5 cells are truly active (A_i = 1), FDP = 0.
    - 60 "bad" draws: cells 4 and 5 (0-indexed 3,4) are false positives.
      FDP at m=5: 2/5 = 0.40 > γ=0.15 → these 60 draws trigger exceedance.
      FDP at m=4: 1/4 = 0.25 > γ=0.15 → still triggered.
      FDP at m=3: 0/3 = 0.00 ≤ γ      → safe.

    Exceedance fractions:
      m=5: 60/1000 = 0.06 > α=0.05 → FAIL (scan down)
      m=4: 60/1000 = 0.06 > α=0.05 → FAIL (scan down)
      m=3: 0/1000  = 0.00 ≤ α=0.05 → PASS → accept m=3

    The expectation rule (using marginals only) correctly flags 5; the
    exceedance gate (using joint draws) correctly shrinks the set to 3.
    Both results are recorded — σ(D)-measurability is preserved because the
    two paths use separate functions with disjoint inputs.
    """
    q, gamma, alpha = 0.10, 0.15, 0.05

    # Marginals: p already in descending order
    p = np.array([0.95, 0.94, 0.92, 0.91, 0.90])
    exp_result = compute_posterior_fdp_threshold(p, q=q)

    # E[FDP] at m=5: (0.05+0.06+0.08+0.09+0.10)/5 ≈ 0.076 ≤ q → flags 5
    assert exp_result.n_flagged == 5, (
        f"Expectation rule should flag 5 but flagged {exp_result.n_flagged}"
    )
    assert exp_result.posterior_expected_fdp_at_threshold <= q

    # Joint draws: correlated false discoveries at cells 3 and 4 (0-indexed)
    n_draws, M = 1000, 5
    draws = np.ones((n_draws, M), dtype=np.int32)
    draws[940:, 3] = 0  # cell at rank-3 is FP in last 60 draws
    draws[940:, 4] = 0  # cell at rank-4 is FP in last 60 draws

    exc_result = compute_fdp_exceedance_gate(
        draws, m_star=exp_result.n_flagged, gamma=gamma, alpha=alpha, q=q
    )

    # Exceedance gate must shrink the set
    assert exc_result.m_accepted < exp_result.n_flagged, (
        f"Exceedance gate ({exc_result.m_accepted}) should flag fewer "
        f"cells than expectation rule ({exp_result.n_flagged})"
    )
    assert exc_result.m_accepted == 3
    assert exc_result.exceedance_fraction_at_accepted == pytest.approx(0.0)
    assert not exc_result.markov_fallback_used

    # Both results are returned independently — σ(D)-measurability preserved
    assert exp_result.n_flagged == 5
    assert exc_result.m_star_expectation == 5


def test_exceedance_gate_no_shrink_when_draws_safe():
    """When all draws have FDP ≤ γ the gate accepts m_star unchanged."""
    draws = np.ones((500, 4))  # all TPs: FDP = 0 at every prefix
    res = compute_fdp_exceedance_gate(draws, m_star=4, gamma=0.15, alpha=0.05)
    assert res.m_accepted == 4
    assert res.exceedance_fraction_at_accepted == pytest.approx(0.0)


def test_exceedance_gate_no_prefix_passes_returns_zero():
    """When no prefix passes (all have exceedance > α), m_accepted = 0."""
    # Every draw has all cells as FP → FDP = 1.0 at every prefix
    draws = np.zeros((1000, 3))
    res = compute_fdp_exceedance_gate(draws, m_star=3, gamma=0.05, alpha=0.01)
    assert res.m_accepted == 0
    assert res.exceedance_fraction_at_accepted == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# C-4: Non-monotone exceedance profile — downward scan is required
# ---------------------------------------------------------------------------

def test_exceedance_gate_downward_scan_non_monotone():
    """Verify correctness when the exceedance profile is non-monotone in m.

    Construction (γ=0.40, α=0.05):
    - 30 draws: only A_1 = 0 (rank-0 cell FP).
        FDP at m=1: 1/1 = 1.0 > 0.40 → exceeds.
        FDP at m=2: 1/2 = 0.50 > 0.40 → exceeds.
        FDP at m≥3: 1/m ≤ 0.33 ≤ 0.40 → safe.
    - 30 draws: only A_2 = 0 (rank-1 cell FP).
        FDP at m=1: 0.       safe.
        FDP at m=2: 1/2 = 0.50 > 0.40 → exceeds.
        FDP at m≥3: 1/m ≤ 0.33 ≤ 0.40 → safe.
    - 60 draws: A_4 = 0 AND A_5 = 0 (rank-3,4 FP).
        FDP at m=4: 1/4 = 0.25 ≤ 0.40 → safe.
        FDP at m=5: 2/5 = 0.40 NOT > 0.40 → safe (not strictly greater).

    Resulting exceedance fractions (n_draws=1000):
      m=1: 30/1000 = 0.030 ≤ 0.05 → PASS
      m=2: 60/1000 = 0.060 > 0.05 → FAIL   (NON-MONOTONE: worse than m=1)
      m=3: 0/1000  = 0.000 ≤ 0.05 → PASS   (NON-MONOTONE: better than m=2)
      m=4: 0/1000  = 0.000 ≤ 0.05 → PASS
      m=5: 0/1000  = 0.000 ≤ 0.05 → PASS   (m_star = 5)

    Downward scan from m_star=5: m=5 → PASS → accept m=5.
    This is correct: m=5 is the LARGEST passing prefix.

    An upward scan stopping at first pass would return m=1, which is
    strictly smaller — the downward scan finds the globally optimal answer.
    """
    gamma, alpha = 0.40, 0.05
    n_draws, M = 1000, 5
    draws = np.ones((n_draws, M), dtype=np.int32)

    # 30 draws: rank-0 cell FP
    draws[0:30, 0] = 0
    # 30 draws: rank-1 cell FP
    draws[30:60, 1] = 0
    # 60 draws: rank-3 and rank-4 both FP
    draws[60:120, 3] = 0
    draws[60:120, 4] = 0

    # Verify the non-monotone profile manually
    def _exceedance(m: int) -> float:
        fp = (1.0 - draws[:, :m].astype(float)).sum(axis=1) / m
        return float((fp > gamma).mean())

    assert _exceedance(1) == pytest.approx(0.030)
    assert _exceedance(2) == pytest.approx(0.060)  # worse than m=1 → non-monotone
    assert _exceedance(3) == pytest.approx(0.000)  # better than m=2 → non-monotone
    assert _exceedance(5) == pytest.approx(0.000)

    # m_star = 5; downward scan starts at m=5 (PASS) → accepts 5
    res = compute_fdp_exceedance_gate(draws, m_star=5, gamma=gamma, alpha=alpha)
    assert res.m_accepted == 5
    assert res.exceedance_fraction_at_accepted == pytest.approx(0.0)

    # Also check that with m_star forced to 4 (skipping the non-monotone dip)
    # the gate still returns 4 directly
    res4 = compute_fdp_exceedance_gate(draws, m_star=4, gamma=gamma, alpha=alpha)
    assert res4.m_accepted == 4
