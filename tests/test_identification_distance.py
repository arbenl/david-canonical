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
    # rho=0.85, delta=0.05: Youden J=0.80; phi_boundary=0.5 → d(θ)=0.50.
    phi = np.full(1000, 0.5)
    rho = np.full((1000, 3), 0.85)
    delta = np.full((1000, 3), 0.05)
    d = identification_distance_draws(phi, rho, delta)
    assert d.mean() >= 0.40


def test_d_uses_third_largest_J_not_min():
    # S=4: three informative sources (J≈0.70) and one weak nuisance (J≈0.05).
    # min_s J would give ≈0.05; J_(3) should give ≈0.70, allowing gate to pass.
    phi = np.full(500, 0.5)
    rho_strong = np.full((500, 3), 0.80)
    delta_strong = np.full((500, 3), 0.10)
    rho_weak = np.full((500, 1), 0.50)
    delta_weak = np.full((500, 1), 0.47)
    rho = np.concatenate([rho_strong, rho_weak], axis=1)
    delta = np.concatenate([delta_strong, delta_weak], axis=1)
    d = identification_distance_draws(phi, rho, delta)
    assert d.mean() > 0.40, "J_(3) should ignore the weak source; min_s would give ~0.03"


def test_check_stratum_pass_fail():
    rng = np.random.default_rng(0)
    phi = rng.beta(8, 8, size=2000)
    # rho ~ 0.55, delta ~ 0.06: J_(3) ≈ E[Beta(5,4)] - E[Beta(2,30)] ≈ 0.49 >> 0.05
    rho = rng.beta(5, 4, size=(2000, 4))
    delta = rng.beta(2, 30, size=(2000, 4))
    result = check_stratum("g_test", phi, rho, delta, floor=0.05)
    assert result.gate_status == "pass"
    assert result.posterior_median >= 0.05


def test_symmetric_channel_informative_anti_regression():
    # Symmetric channel: rho=0.9, delta=0.1
    # Under old |rho-(1-delta)| = |0.9 - 0.9| = 0.0 withdrawn form, d(theta) would be 0 or small.
    # Under J_(3) form, J = 0.8, phi=0.5 -> min(J, phi_margin) = min(0.8, 0.5) = 0.5.
    phi = np.full(1000, 0.5)
    rho = np.full((1000, 3), 0.9)
    delta = np.full((1000, 3), 0.1)
    d = identification_distance_draws(phi, rho, delta)
    # The J_(3) form correctly evaluates the distance to 0.5, NOT ~0.0
    assert np.allclose(d, 0.5)

