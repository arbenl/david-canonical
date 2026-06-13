"""Tests for Theorem D-forecast horizon-validity bound."""

from __future__ import annotations

import numpy as np

from david.theorems.D_forecast_horizon import (
    first_crossing_h_star,
    forecast_regime_distribution,
    horizon_validity_from_z_future_draws,
    stationary_marginal_embedded,
    stationary_marginal_time,
    horizon_validity,
)


def test_stationary_embedded_sums_to_one():
    pi = np.array([[0.0, 0.5, 0.5],
                   [0.6, 0.0, 0.4],
                   [0.3, 0.7, 0.0]])
    nu = stationary_marginal_embedded(pi)
    assert nu.sum() == nu.sum()
    assert np.isclose(nu.sum(), 1.0)
    assert (nu >= 0).all()


def test_stationary_time_weighted():
    pi = np.array([[0.0, 0.5, 0.5],
                   [0.6, 0.0, 0.4],
                   [0.3, 0.7, 0.0]])
    mu = np.array([2.0, 4.0, 1.0])
    pi_inf = stationary_marginal_time(pi, mu)
    assert np.isclose(pi_inf.sum(), 1.0)
    # State with longest dwell should have larger marginal share
    nu = stationary_marginal_embedded(pi)
    expected_relative = nu * mu
    expected_relative /= expected_relative.sum()
    assert np.allclose(pi_inf, expected_relative)


def _withdrawn_max_form(drift_curve, tau, h_max):
    """The withdrawn max-form h* = max{h : drift(h) < tau} (audit, June 2026).

    Reference implementation kept ONLY as the adversary in the
    anti-regression test below. Never use this in kernel code.
    """
    h_star = 0
    for h, drift in drift_curve:
        if drift < tau:
            h_star = h
    return h_star


def test_first_crossing_stops_at_first_crossing_on_non_monotone_curve():
    # Drift dips back below tau = 0.5 after the first crossing at h = 4.
    curve = [(1, 0.10), (2, 0.30), (3, 0.45), (4, 0.55),
             (5, 0.40), (6, 0.35), (7, 0.60), (8, 0.70)]
    tau, h_max = 0.5, 8
    h_first = first_crossing_h_star(curve, tau=tau, h_max=h_max)
    h_max_form = _withdrawn_max_form(curve, tau=tau, h_max=h_max)
    # First crossing is at h = 4, so h* = 3.
    assert h_first == 3
    # The withdrawn max-form extends past the crossing (to h = 6 here):
    # the first-crossing form must be strictly smaller on this curve.
    assert h_max_form == 6
    assert h_first < h_max_form


def test_first_crossing_no_crossing_yields_h_max():
    curve = [(h, 0.1 + 0.02 * h) for h in range(1, 7)]  # never reaches 0.5
    assert first_crossing_h_star(curve, tau=0.5, h_max=6) == 6


def test_first_crossing_immediate_crossing_yields_zero():
    curve = [(1, 0.9), (2, 0.95), (3, 0.99)]
    assert first_crossing_h_star(curve, tau=0.5, h_max=3) == 0


def test_forecast_regime_distribution_uses_stationary_residual_at_origin():
    # lambda = 0 => dwell mean mu = 1, so the stationary residual is always 1.
    # Under the Stan GQ semantics, horizon 1 therefore transitions immediately.
    pi = np.array([[0.0, 1.0], [1.0, 0.0]])
    mu = np.array([1.0, 1.0])
    p = forecast_regime_distribution(pi, mu, z_t=0, horizon=1, n_mc=200)
    assert np.allclose(p, np.array([0.0, 1.0]))


def test_horizon_validity_h_star_matches_first_crossing_of_emitted_curve():
    # End-to-end: the h* reported by horizon_validity must equal the
    # first-crossing functional applied to its own emitted drift curve.
    pi = np.array([[0.0, 0.5, 0.5],
                   [0.6, 0.0, 0.4],
                   [0.3, 0.7, 0.0]])
    mu = np.array([3.0, 3.0, 3.0])
    z_t = np.array([1.0, 0.0, 0.0])
    hv = horizon_validity("g_fc", pi, mu, z_t, h_max=8, tau=0.5, n_mc=300)
    expected = first_crossing_h_star(hv.horizon_validity_curve, tau=0.5, h_max=8)
    assert hv.h_star_months == expected


def test_horizon_validity_returns_curve():
    pi = np.array([[0.0, 0.5, 0.5],
                   [0.6, 0.0, 0.4],
                   [0.3, 0.7, 0.0]])
    mu = np.array([3.0, 3.0, 3.0])
    z_t = np.array([1.0, 0.0, 0.0])    # known terminal regime
    hv = horizon_validity("g_test", pi, mu, z_t, h_max=6, tau=0.5, n_mc=200)
    assert len(hv.horizon_validity_curve) == 6
    assert hv.h_star_months >= 0


def test_horizon_validity_accepts_per_draw_terminal_posterior():
    pi = np.array([
        [[0.0, 1.0], [1.0, 0.0]],
        [[0.0, 1.0], [1.0, 0.0]],
    ])
    mu = np.array([[1.0, 1.0], [1.0, 1.0]])
    z_t_draws = np.array([[1.0, 0.0], [0.0, 1.0]])

    hv = horizon_validity(
        "series_terminal",
        pi,
        mu,
        z_t_draws,
        h_max=2,
        tau=0.5,
        n_mc=20,
    )

    assert len(hv.horizon_validity_curve) == 2
    assert hv.h_star_q05 <= hv.h_star_q95


def test_horizon_validity_from_z_future_uses_emitted_stan_paths():
    pi = np.array([
        [[0.0, 1.0], [1.0, 0.0]],
        [[0.0, 1.0], [1.0, 0.0]],
        [[0.0, 1.0], [1.0, 0.0]],
        [[0.0, 1.0], [1.0, 0.0]],
    ])
    mu = np.ones((4, 2))
    z_t_draws = np.tile(np.array([[1.0, 0.0]]), (4, 1))
    # Stan 1-indexed paths: all draws move to regime 2 at h=1, then regime 1.
    z_future = np.array([[2, 1], [2, 1], [2, 1], [2, 1]])

    hv = horizon_validity_from_z_future_draws(
        "z_future_series",
        pi,
        mu,
        z_t_draws,
        z_future,
        h_max=2,
        tau=0.5,
        n_bootstrap=20,
    )

    assert hv.horizon_validity_curve[0][0] == 1
    assert hv.horizon_validity_curve[0][1] > 0.5
    assert hv.h_star_months == 0
    assert hv.h_star_q05 == 0
