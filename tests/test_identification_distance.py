"""Tests for Theorem A' practical-identification kernel."""

from __future__ import annotations

import numpy as np
import pytest

from david.theorems.A_prime import (
    identification_distance_draws,
    check_stratum,
)


def test_d_zero_when_sources_uninformative():
    """If rho == delta, d should be zero regardless of phi."""
    phi = np.full(1000, 0.5)
    rho = np.full((1000, 3), 0.3)
    delta = np.full((1000, 3), 0.3)
    d = identification_distance_draws(phi, rho, delta)
    assert np.allclose(d, 0.0)


def test_d_zero_when_phi_at_boundary():
    phi = np.full(1000, 1e-3)
    rho = np.full((1000, 3), 0.8)
    delta = np.full((1000, 3), 0.1)
    d = identification_distance_draws(phi, rho, delta)
    assert d.max() < 0.05


def test_d_high_when_well_separated():
    # With rho=0.85, delta=0.05: label-flip safety = |rho-(1-delta)| = 0.10
    # which caps d at 0.10; expect mean to be near this value.
    phi = np.full(1000, 0.5)
    rho = np.full((1000, 3), 0.85)
    delta = np.full((1000, 3), 0.05)
    d = identification_distance_draws(phi, rho, delta)
    assert d.mean() >= 0.08


def test_check_stratum_pass_fail():
    rng = np.random.default_rng(0)
    phi = rng.beta(8, 8, size=2000)
    # rho ~ 0.55, delta ~ 0.06: |rho-(1-delta)| ~ 0.39 >> 0.05
    rho = rng.beta(5, 4, size=(2000, 4))
    delta = rng.beta(2, 30, size=(2000, 4))
    result = check_stratum("g_test", phi, rho, delta, floor=0.05)
    assert result.gate_status == "pass"
    assert result.posterior_median >= 0.05
