"""Tests for Theorem D-forecast horizon-validity bound."""

from __future__ import annotations

import numpy as np

from david.theorems.D_forecast_horizon import (
    stationary_marginal_embedded,
    stationary_marginal_time,
    horizon_validity,
)


def test_stationary_embedded_sums_to_one():
    Pi = np.array([[0.0, 0.5, 0.5],
                   [0.6, 0.0, 0.4],
                   [0.3, 0.7, 0.0]])
    nu = stationary_marginal_embedded(Pi)
    assert nu.sum() == nu.sum()
    assert np.isclose(nu.sum(), 1.0)
    assert (nu >= 0).all()


def test_stationary_time_weighted():
    Pi = np.array([[0.0, 0.5, 0.5],
                   [0.6, 0.0, 0.4],
                   [0.3, 0.7, 0.0]])
    mu = np.array([2.0, 4.0, 1.0])
    pi_inf = stationary_marginal_time(Pi, mu)
    assert np.isclose(pi_inf.sum(), 1.0)
    # State with longest dwell should have larger marginal share
    nu = stationary_marginal_embedded(Pi)
    expected_relative = nu * mu
    expected_relative /= expected_relative.sum()
    assert np.allclose(pi_inf, expected_relative)


def test_horizon_validity_returns_curve():
    Pi = np.array([[0.0, 0.5, 0.5],
                   [0.6, 0.0, 0.4],
                   [0.3, 0.7, 0.0]])
    mu = np.array([3.0, 3.0, 3.0])
    z_t = np.array([1.0, 0.0, 0.0])    # known terminal regime
    hv = horizon_validity("g_test", Pi, mu, z_t, h_max=6, tau=0.5, n_mc=200)
    assert len(hv.horizon_validity_curve) == 6
    assert hv.h_star_months >= 0
