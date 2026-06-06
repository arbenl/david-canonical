"""Tests for Theorem C posterior expected FDP threshold."""

from __future__ import annotations

import numpy as np

from david.theorems.C_renamed import compute_posterior_fdp_threshold


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
