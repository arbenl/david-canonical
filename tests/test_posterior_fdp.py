"""Tests for Theorem C posterior expected FDP threshold and exceedance gate (C-4)."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from david.theorems.C_renamed import (
    compute_posterior_fdp_threshold,
    compute_fdp_exceedance_gate,
    compute_posterior_fdp_envelope,
    check_mcse_floor,
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


def test_expectation_gate_api_accepts_only_marginals():
    """C-5 guard: expectation FDP path has no joint-draw input surface."""
    sig = inspect.signature(compute_posterior_fdp_threshold)
    assert list(sig.parameters) == ["p_hat", "q"]
    assert sig.parameters["p_hat"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert sig.parameters["q"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD

    assert "joint_draws" not in sig.parameters


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


def test_exceedance_gate_uses_upper_confidence_limit():
    # Raw exceedance is 1/20 = 0.05, but the one-sided 95% UCL is > 0.05.
    # The gate must not certify this prefix on the raw fraction alone.
    draws = np.ones((20, 1), dtype=int)
    draws[0, 0] = 0

    res = compute_fdp_exceedance_gate(draws, m_star=1, gamma=0.15, alpha=0.05)

    assert res.m_accepted == 0
    assert res.exceedance_fraction_at_accepted == pytest.approx(0.0)


def test_exceedance_gate_accepts_chain_draw_shape_and_reports_split_ess():
    draws = np.ones((2, 500, 3), dtype=int)

    res = compute_fdp_exceedance_gate(draws, m_star=3, gamma=0.15, alpha=0.05)

    assert res.m_accepted == 3
    assert res.exceedance_ess_at_accepted == pytest.approx(1000.0)


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
# C-3: Sensitivity-envelope FDP tests
# ---------------------------------------------------------------------------

def test_envelope_rejects_invalid_shape():
    with pytest.raises(ValueError, match="2-D"):
        compute_posterior_fdp_envelope(np.array([0.9, 0.8, 0.7]))


def test_envelope_iid_grid_matches_single_theta():
    """When all θ points give the same p, envelope = single-θ rule."""
    p = np.array([0.95, 0.92, 0.87])
    p_grid = np.vstack([p, p, p])  # 3 identical rows
    single = compute_posterior_fdp_threshold(p, q=0.10)
    env = compute_posterior_fdp_envelope(p_grid, q=0.10)
    assert env.n_flagged == single.n_flagged
    assert env.posterior_expected_fdp_at_threshold == pytest.approx(
        single.posterior_expected_fdp_at_threshold
    )


def test_envelope_shrinks_naive_flag_set():
    """Core C-3 acceptance criterion.

    3-cell, 3-point Θ^meas grid constructed so that:
    - Naive single-θ rule (θ0) flags all 3 cells with FDP ≤ q.
    - At θ2 the third cell's p drops to 0.80, making p^- = [0.93, 0.90, 0.80].
    - Prefix scan on p^- finds FDP at m=3 = (0.07+0.10+0.20)/3 ≈ 0.123 > q.
    - Conservative (envelope) rule therefore flags only 2 cells.

    This verifies that the naive rule would admit a set whose FDP EXCEEDS q
    at the worst grid point, and that the envelope correctly shrinks the set.
    """
    q = 0.10

    # θ0 (nominal): all three cells look safe
    p_theta0 = np.array([0.95, 0.92, 0.87])
    # θ1: slight perturbation
    p_theta1 = np.array([0.94, 0.91, 0.86])
    # θ2: worst case — third cell's p drops significantly
    p_theta2 = np.array([0.93, 0.90, 0.80])

    p_grid = np.vstack([p_theta0, p_theta1, p_theta2])  # shape (3, 3)

    # --- Naive single-θ rule (θ0) ---
    naive = compute_posterior_fdp_threshold(p_theta0, q=q)
    # FDP at m=3: (0.05 + 0.08 + 0.13)/3 ≈ 0.087 ≤ 0.10  → flags 3
    assert naive.n_flagged == 3, (
        f"Naive rule should flag 3 cells but flagged {naive.n_flagged}"
    )
    assert naive.posterior_expected_fdp_at_threshold <= q

    # --- Verify the naive set VIOLATES the envelope guarantee ---
    # p^- = [0.93, 0.90, 0.80].  FDP at m=3 using p^-:
    p_conservative = p_grid.min(axis=0)
    fdp_at_3 = (
        (1 - p_conservative[np.argsort(-p_conservative)]).cumsum()
        / np.arange(1, 4)
    )[-1]
    assert fdp_at_3 > q, (
        f"FDP at m=3 under p^- should exceed q={q}, got {fdp_at_3:.4f}"
    )

    # --- Envelope rule ---
    env = compute_posterior_fdp_envelope(p_grid, q=q)

    # Must flag strictly fewer cells than the naive rule
    assert env.n_flagged < naive.n_flagged, (
        f"Envelope ({env.n_flagged}) should flag fewer cells than naive ({naive.n_flagged})"
    )
    # The envelope result itself must satisfy FDP ≤ q
    assert env.posterior_expected_fdp_at_threshold <= q
    # Worst-θ index must point to θ2 (row 2, smallest sum(p))
    assert env.worst_theta_index == 2


def test_envelope_worst_theta_index():
    """worst_theta_index identifies the θ with smallest total p mass."""
    p_grid = np.array([
        [0.90, 0.85, 0.80],   # θ0: sum = 2.55
        [0.70, 0.65, 0.60],   # θ1: sum = 1.95  ← most conservative
        [0.80, 0.75, 0.70],   # θ2: sum = 2.25
    ])
    env = compute_posterior_fdp_envelope(p_grid, q=0.10)
    assert env.worst_theta_index == 1


# ---------------------------------------------------------------------------
# C-6: MCSE/ESS floor — check_mcse_floor unit tests
# ---------------------------------------------------------------------------

def test_mcse_floor_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D"):
        check_mcse_floor(np.ones(100))


def test_mcse_floor_rejects_too_few_draws():
    with pytest.raises(ValueError, match="at least 4 draws"):
        check_mcse_floor(np.ones((3, 2)))


def test_mcse_floor_iid_high_ess_passes():
    """iid Bernoulli draws give ESS ≈ n_draws → MCSE ≈ 0 → all cells pass."""
    rng = np.random.default_rng(42)
    # 2000 iid draws, p ≈ 0.9 per cell: std ≈ sqrt(0.09) ≈ 0.30; ESS ≈ 2000
    # MCSE ≈ 0.30 / sqrt(2000) ≈ 0.0067; threshold = 0.10 * 0.10 = 0.01 → PASS
    draws = rng.binomial(1, 0.9, size=(2000, 4)).astype(float)
    result = check_mcse_floor(draws, q=0.10)
    assert result.n_cells_excluded == 0
    assert result.excluded_indices == []
    assert result.mcse_threshold == pytest.approx(0.01)


def test_mcse_floor_low_ess_excludes_cell():
    """C-6 acceptance criterion: low-ESS draws ⇒ cell excluded with typed reason.

    Construction:
    - Cell 0: iid Bernoulli(0.9), 2000 draws → ESS ≈ 2000, MCSE ≈ 0.007 < 0.01 → PASS.
    - Cell 1: highly autocorrelated — alternates blocks of 1s and 0s every 50 draws.
      Effective ESS is << 2000; MCSE >> threshold.

    The Geyer ESS estimator detects the strong positive autocorrelation and
    returns a small effective sample size, causing MCSE(p_1) ≥ 0.01 = 0.10 × q.
    Cell 1 must be in excluded_indices with binding_reason = "fdp_mcse_exceeds_gate_margin".
    """
    n = 2000
    # Cell 0: iid high-p → passes
    rng = np.random.default_rng(0)
    col0 = rng.binomial(1, 0.9, size=n).astype(float)

    # Cell 1: block-autocorrelated (50-draw blocks alternating 1 and 0)
    # ESS ≈ n / block_size = 2000 / 50 = 40
    # std ≈ 0.5 (p ≈ 0.5); MCSE ≈ 0.5 / sqrt(40) ≈ 0.079 >> 0.01 → FAIL
    col1 = np.repeat(np.tile([1.0, 0.0], n // (2 * 50)), 50)[:n]

    draws = np.column_stack([col0, col1])
    result = check_mcse_floor(draws, q=0.10)

    assert 1 in result.excluded_indices, (
        f"Cell 1 (low ESS, high MCSE) must be excluded; got excluded={result.excluded_indices}, "
        f"mcse={result.mcse}"
    )
    assert result.binding_reason == "fdp_mcse_exceeds_gate_margin"
    assert result.n_cells_excluded >= 1
    # Cell 0 (iid, high ESS) must not be excluded
    assert 0 not in result.excluded_indices, (
        f"Cell 0 (iid, good ESS) must pass; mcse[0]={result.mcse[0]:.6f}"
    )


def test_mcse_floor_uses_chain_axis_for_split_chain_ess():
    n_chains, n_draws = 2, 1000
    rng = np.random.default_rng(123)
    draws = rng.binomial(1, 0.9, size=(n_chains, n_draws, 2)).astype(float)
    block = np.repeat(np.tile([1.0, 0.0], n_draws // 100 + 1), 100)[:n_draws]
    draws[:, :, 1] = block

    result = check_mcse_floor(draws, q=0.10)

    assert 1 in result.excluded_indices
    assert 0 not in result.excluded_indices
    assert result.bulk_ess[1] < n_chains * n_draws


def test_mcse_floor_all_identical_draws_pass():
    """When all draws for a cell are identical (std = 0) MCSE = 0 → always passes."""
    draws = np.ones((1000, 3))
    result = check_mcse_floor(draws, q=0.10)
    assert result.n_cells_excluded == 0


def test_mcse_floor_threshold_scales_with_q():
    """mcse_threshold = mcse_margin_ratio × q; stricter q → stricter threshold."""
    rng = np.random.default_rng(7)
    # iid Bernoulli(0.6), 200 draws: std ≈ 0.49, ESS ≈ 200, MCSE ≈ 0.035
    draws = rng.binomial(1, 0.6, size=(200, 1)).astype(float)

    # q=0.10 → threshold=0.01 → 0.035 >> 0.01 → cell excluded
    res_strict = check_mcse_floor(draws, q=0.10)
    # q=0.50 → threshold=0.05 → 0.035 < 0.05 → cell passes
    res_loose = check_mcse_floor(draws, q=0.50)

    assert res_strict.mcse_threshold == pytest.approx(0.01)
    assert res_loose.mcse_threshold == pytest.approx(0.05)
    # Cell should be excluded under strict q but pass under loose q
    assert len(res_strict.excluded_indices) >= len(res_loose.excluded_indices)
