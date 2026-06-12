"""Tests for Theorem C posterior expected FDP threshold."""

from __future__ import annotations

import numpy as np
import pytest

from david.theorems.C_renamed import (
    compute_posterior_fdp_threshold,
    compute_posterior_fdp_envelope,
)


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
