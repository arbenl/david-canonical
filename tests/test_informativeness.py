"""Tests for Theorem B' channel-informativeness kernel.

Closes the coverage gap noted in docs/THESIS_MATHEMATICAL_BASIS.md §2.4:
B'.1 (Bayes classification error) and B'.2 (N * I^2 information scaling) were
previously exercised only indirectly via the forecast engine. These tests pin
the algebraic identities and the cell-level gate directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from david.theorems.B_prime import (
    informativeness_draws,
    bayes_classification_error,
    effective_sample_size,
    dependence_adjusted_n_eff,
    check_cell,
)
from david.config import INFORMATIVENESS_FLOOR_LOWER_95, OBSERVABILITY_GRID


# ── informativeness I(O) = |rho - delta| ──────────────────────────────────────

def test_informativeness_is_absolute_difference():
    rho = np.array([0.9, 0.5, 0.2])
    delta = np.array([0.1, 0.5, 0.6])
    I = informativeness_draws(rho, delta)
    assert np.allclose(I, [0.8, 0.0, 0.4])


def test_informativeness_zero_when_channel_uninformative():
    rho = np.full(500, 0.4)
    delta = np.full(500, 0.4)
    assert np.allclose(informativeness_draws(rho, delta), 0.0)


def test_informativeness_shape_mismatch_raises():
    with pytest.raises(ValueError):
        informativeness_draws(np.zeros(5), np.zeros(4))


# ── B'.1 Bayes classification error = (1 - I) / 2 ─────────────────────────────

@pytest.mark.parametrize("info,expected", [
    (0.0, 0.5),    # uninformative channel → chance-level error
    (1.0, 0.0),    # perfect channel → zero error
    (0.5, 0.25),
    (0.8, 0.10),
])
def test_bayes_error_identity(info, expected):
    assert bayes_classification_error(info) == pytest.approx(expected)


def test_bayes_error_monotone_decreasing():
    grid = np.linspace(0.0, 1.0, 11)
    errs = [bayes_classification_error(float(x)) for x in grid]
    assert all(a >= b for a, b in zip(errs, errs[1:]))


@pytest.mark.parametrize("bad", [-0.01, 1.01, 2.0])
def test_bayes_error_rejects_out_of_range(bad):
    with pytest.raises(ValueError):
        bayes_classification_error(bad)


# ── B'.2 effective sample size = N * I^2 (quadratic collapse) ──────────────────

def test_effective_sample_size_identity():
    assert effective_sample_size(100, 0.5) == pytest.approx(25.0)
    assert effective_sample_size(100, 0.0) == pytest.approx(0.0)
    assert effective_sample_size(40, 1.0) == pytest.approx(40.0)


def test_effective_sample_size_scales_quadratically():
    # Halving informativeness must quarter the effective sample size.
    n_full = effective_sample_size(1000, 0.4)
    n_half = effective_sample_size(1000, 0.2)
    assert n_half == pytest.approx(n_full / 4.0)


# ── B'.3 Godambe Dependence-Adjusted Sample Size ──────────────────────────────

def test_dependence_adjusted_n_eff_iid_returns_raw_N():
    # If Corr(A_0, A_h) = 0, sum(gamma_h) = 0, denom = 1 -> N_eff = N
    corr_zero = np.zeros(5)
    n_eff = dependence_adjusted_n_eff(100, 0.5, 0.5, 0.5, corr_zero)
    assert n_eff == 100.0

def test_dependence_adjusted_n_eff_positive_serial_corr_reduces_N():
    # Positive correlation inflates variance, shrinking effective N
    corr_pos = np.array([0.5, 0.25, 0.125])
    # c = 0.5^2 * 0.5 * 0.5 / (0.5 * 0.5) = 0.25
    # sum_gamma_ratio = 0.25 * 0.875 = 0.21875
    # denom = 1.0 + 2 * 0.21875 = 1.4375
    # N_eff = 100 / 1.4375 = 69.565
    n_eff = dependence_adjusted_n_eff(100, 0.5, 0.5, 0.5, corr_pos)
    assert n_eff == pytest.approx(100.0 / 1.4375)

def test_dependence_adjusted_n_eff_negative_serial_corr_capped_at_N():
    # Negative correlation deflates variance, but the rule mandates capping at N
    corr_neg = np.array([-0.5, -0.25, -0.125])
    n_eff = dependence_adjusted_n_eff(100, 0.5, 0.5, 0.5, corr_neg)
    assert n_eff == 100.0


# ── cell-level gate on lower-95% credible bound of I(O) ───────────────────────

def test_check_cell_passes_when_lower95_above_floor():
    # Tight, well-separated channel: lower 95% bound comfortably above floor.
    rng = np.random.default_rng(0)
    rho = np.clip(rng.normal(0.85, 0.02, 4000), 0, 1)
    delta = np.clip(rng.normal(0.10, 0.02, 4000), 0, 1)
    res = check_cell("g1", rho, delta, n_replicates=50)
    assert res.gate_status == "pass"
    assert res.lower_95_I >= INFORMATIVENESS_FLOOR_LOWER_95
    assert res.reason == "I_lower_95_and_N_eff_I2_above_floor"


def test_check_cell_fails_when_lower95_below_floor():
    # Barely-separated channel: lower 95% bound dips under the 0.10 floor.
    rng = np.random.default_rng(1)
    rho = np.clip(rng.normal(0.50, 0.03, 4000), 0, 1)
    delta = np.clip(rng.normal(0.47, 0.03, 4000), 0, 1)
    res = check_cell("g2", rho, delta, n_replicates=50)
    assert res.gate_status == "fail"
    assert res.lower_95_I < INFORMATIVENESS_FLOOR_LOWER_95
    assert "below_floor" in res.reason


def test_check_cell_reports_n_eff_from_median():
    rho = np.full(2000, 0.8)
    delta = np.full(2000, 0.3)   # I = 0.5 deterministically
    res = check_cell("g3", rho, delta, n_replicates=200)
    assert res.posterior_median_I == pytest.approx(0.5)
    assert res.n_eff_adjusted == pytest.approx(200 * 0.5 ** 2)  # = 50


def test_check_cell_fails_gate2_when_n_too_small():
    # Gate 1 passes (I_lower95 = 0.20 >> 0.10) but N_eff × I² = 2 × 0.20² = 0.08 < 3.0.
    rho = np.full(2000, 0.60)
    delta = np.full(2000, 0.40)   # I = 0.20 deterministically; lower_95 = 0.20
    res = check_cell("g4", rho, delta, n_replicates=2)
    assert res.gate_status == "fail", "small-N cell should fail gate 2 even with adequate I"
    assert "N_eff_I2" in res.reason


# ── Production-gate acceptance test ───────────────────────────────────────────
# Verifies that _run_theorem_gates in fit.py enforces the N_eff × I² floor
# (gate 2) in addition to the I_lower95 floor (gate 1).  This was a known gap:
# the pre-registered N_EFF_I2_FLOOR = 3.0 was imported but never applied.

class _MockFit:
    """Minimal fit mock: returns preset draws for each stan_variable name."""
    def __init__(self, variables: dict):
        self._vars = variables

    def stan_variable(self, name: str):
        if name not in self._vars:
            raise KeyError(f"mock has no variable {name!r}")
        return self._vars[name]


def _make_mock_fit(D: int, S: int, L: int, K: int, I_val: float):
    """Build a MockFit where all sources have informativeness I_val."""
    rng = np.random.default_rng(42)
    # delta ~ 0.10, rho = delta + I so that |rho - delta| = I_val
    delta_raw = np.full((D, S), -2.2)         # sigmoid(-2.2) ≈ 0.10
    j_raw     = np.full((D, S),  np.log(I_val / (1.0 - I_val)))  # sigmoid(j_raw) = I_val
    return _MockFit({
        "delta_raw":          delta_raw,
        "j_raw":              j_raw,
        "delta_observability": np.zeros((D, S)),
        "j_observability":    np.zeros((D, S)),
        "alpha_activity":     np.zeros((D, L, K)),   # sigmoid(0) = 0.5
        "dwell_lambda":       np.ones((D, L)),
        "log_jump":           np.full((D, L, L), -np.log(L)),
        "terminal_regime_posterior_draw": np.full((D, 1, L), 1.0 / L),
        "z_future":           np.ones((D, 1, 12), dtype=int),
    })


def test_production_gate_enforces_n_eff_i2_floor():
    """Gate 2 (N_eff × I² ≥ 3.0) must fire in _run_theorem_gates.

    Scenario: I = 0.15 (passes gate 1: 0.15 > 0.10), but N_units = 3,
    so N_eff × I² = 3 × 0.15² = 0.0675 < 3.0 → gate 2 should fail.
    """
    from david.model.fit import _run_theorem_gates

    D, S, L, K = 400, 3, 2, 3
    I_val = 0.15          # just above gate-1 floor of 0.10
    n_units = 3           # tiny stratum → N_eff × I² << 3.0

    mock_fit = _make_mock_fit(D, S, L, K, I_val)
    data = {"U": n_units, "R": 1, "T": n_units, "K": K, "S": S, "M": 1, "L": L,
            "H_forecast": 1}

    gates = _run_theorem_gates(mock_fit, data)
    b = gates["B_prime"]
    assert b["gate_status"] == "fail", (
        f"expected gate 2 failure (N_eff×I²={b.get('n_eff_i2', '?'):.3f} < 3.0) "
        f"but got: {b}"
    )
    assert "N_eff_I2" in b["reason"]


def test_production_a_prime_fails_on_one_near_boundary_activity_cell():
    from david.model.fit import _run_theorem_gates

    D, S, L, K = 200, 3, 2, 3
    mock_fit = _make_mock_fit(D, S, L, K, I_val=0.50)
    alpha = np.zeros((D, L, K))
    alpha[:, 0, 0] = -10.0
    mock_fit._vars["alpha_activity"] = alpha
    terminal = np.zeros((D, 1, L))
    terminal[:, 0, 0] = 1.0
    mock_fit._vars["terminal_regime_posterior_draw"] = terminal

    gates = _run_theorem_gates(
        mock_fit,
        {"U": 60, "R": 1, "T": 20, "K": K, "S": S, "M": 1, "L": L, "H_forecast": 1},
    )

    a = gates["A_prime"]
    assert a["gate_status"] == "fail"
    assert a["aggregation"] == "min_over_terminal_weighted_series_tactic_cells_and_observability_grid"
    assert a["median_d_theta"] < 0.05


def test_production_a_prime_uses_terminal_regime_weighted_prevalence():
    """A low-prevalence latent regime does not bind if current posterior excludes it."""
    from david.model.fit import _run_theorem_gates

    D, S, L, K = 200, 3, 2, 3
    mock_fit = _make_mock_fit(D, S, L, K, I_val=0.50)
    alpha = np.zeros((D, L, K))
    alpha[:, 0, 0] = -10.0
    mock_fit._vars["alpha_activity"] = alpha
    terminal = np.zeros((D, 1, L))
    terminal[:, 0, 1] = 1.0
    mock_fit._vars["terminal_regime_posterior_draw"] = terminal

    gates = _run_theorem_gates(
        mock_fit,
        {"U": 60, "R": 1, "T": 20, "K": K, "S": S, "M": 1, "L": L, "H_forecast": 1},
    )

    a = gates["A_prime"]
    assert a["gate_status"] == "pass"
    assert a["aggregation"] == "min_over_terminal_weighted_series_tactic_cells_and_observability_grid"


def test_production_a_prime_uses_worst_observability_grid(monkeypatch):
    """A' must fail when any pre-registered O grid point is near-singular.

    The midpoint O=0.5 has strong Youden signal here; O=1.0 is weak. The old
    plug-in midpoint call would pass, while the grid-reduced gate must fail.
    """
    from types import SimpleNamespace

    from david.model.fit import _run_theorem_gates

    D, S, L, K = 20, 3, 2, 2
    mock_fit = _make_mock_fit(D, S, L, K, I_val=0.50)
    low_logit = np.log(0.02 / 0.98)
    mock_fit._vars["j_observability"] = np.full((D, S), 2.0 * low_logit)
    mock_fit._vars["j_raw"] = np.full((D, S), -low_logit)

    def fake_autocorrelation(Pi_off_diag, dwell_mean, phi_k, max_lag, n_mc=500):
        return np.zeros(max_lag)

    def fake_horizon_validity(
        cell_id,
        Pi_off_diag_draws,
        dwell_mean_draws,
        z_t_distribution,
        z_future_draws,
        h_max=18,
        tau=0.5,
        n_bootstrap=200,
    ):
        return SimpleNamespace(
            h_star_months=3,
            h_star_q05=3,
            h_star_q95=3,
            tau=tau,
            prior_drift_share_at_h_max=0.1,
        )

    monkeypatch.setattr(
        "david.theorems.B_prime.compute_activity_autocorrelation",
        fake_autocorrelation,
    )
    monkeypatch.setattr(
        "david.theorems.D_forecast_horizon.horizon_validity_from_z_future_draws",
        fake_horizon_validity,
    )

    gates = _run_theorem_gates(
        mock_fit,
        {"U": 60, "R": 1, "T": 20, "K": K, "S": S, "M": 1, "L": L, "H_forecast": 1},
    )

    a = gates["A_prime"]
    assert a["gate_status"] == "fail"
    assert a["observability_grid"] == [float(x) for x in OBSERVABILITY_GRID]
    assert a["observability_aggregation"] == "min_over_pre_registered_grid"
    assert a["worst_observability"] == 1.0
    assert a["median_d_theta_by_observability"]["0.5"] > 0.05
    assert a["median_d_theta_by_observability"]["1"] < 0.05


def test_production_b_prime_uses_third_largest_source_not_mean():
    from david.model.fit import _run_theorem_gates

    D, S, L, K = 200, 4, 2, 2
    mock_fit = _make_mock_fit(D, S, L, K, I_val=0.50)
    # Three weak sources and one strong source. Mean I would pass the 0.10 floor;
    # third-largest I is weak and must fail the B' source-informativeness floor.
    weak_i = 0.03
    strong_i = 0.50
    j_raw = np.column_stack(
        [
            np.full(D, np.log(weak_i / (1.0 - weak_i))),
            np.full(D, np.log(weak_i / (1.0 - weak_i))),
            np.full(D, np.log(weak_i / (1.0 - weak_i))),
            np.full(D, np.log(strong_i / (1.0 - strong_i))),
        ]
    )
    mock_fit._vars["j_raw"] = j_raw

    gates = _run_theorem_gates(
        mock_fit,
        {"U": 80, "R": 1, "T": 40, "K": K, "S": S, "M": 1, "L": L, "H_forecast": 1},
    )

    b = gates["B_prime"]
    assert b["gate_status"] == "fail"
    assert b["source_aggregation"] == "third_largest_source"
    assert b["lower_95_I_worst_source"] < 0.10


def test_production_gate_passes_when_both_gates_met():
    """Both gate 1 (I_lower95 ≥ 0.10) and gate 2 (N_eff×I² ≥ 3.0) must pass."""
    from david.model.fit import _run_theorem_gates

    D, S, L, K = 400, 3, 2, 3
    I_val = 0.50          # well above both thresholds
    n_units = 50          # N_eff × I² = 50 × 0.25 = 12.5 >> 3.0

    mock_fit = _make_mock_fit(D, S, L, K, I_val)
    data = {"U": n_units, "R": 1, "T": n_units, "K": K, "S": S, "M": 1, "L": L,
            "H_forecast": 1}

    gates = _run_theorem_gates(mock_fit, data)
    b = gates["B_prime"]
    assert b["gate_status"] == "pass", f"expected pass but got: {b}"
    assert b["reason"] == "I_lower_95_and_N_eff_I2_above_floor"


def test_production_gates_pass_shifted_poisson_mean_to_renewal_kernels(monkeypatch):
    """Renewal kernels consume dwell mean μ, not shifted-Poisson rate λ."""
    from types import SimpleNamespace

    from david.model.fit import _run_theorem_gates

    D, S, L, K = 20, 3, 2, 3
    lambda_val = 2.0
    mock_fit = _make_mock_fit(D, S, L, K, I_val=0.50)
    mock_fit._vars["dwell_lambda"] = np.full((D, L), lambda_val)

    captured: dict[str, list[np.ndarray] | np.ndarray] = {
        "autocorr_dwell": [],
        "horizon_dwell": [],
    }

    def fake_autocorrelation(Pi_off_diag, dwell_mean, phi_k, max_lag, n_mc=500):
        captured["autocorr_dwell"].append(np.asarray(dwell_mean).copy())
        return np.zeros(max_lag)

    def fake_horizon_validity(
        cell_id,
        Pi_off_diag_draws,
        dwell_mean_draws,
        z_t_distribution,
        z_future_draws,
        h_max=18,
        tau=0.5,
        n_bootstrap=200,
    ):
        captured["horizon_dwell"].append(np.asarray(dwell_mean_draws).copy())
        return SimpleNamespace(
            h_star_months=3,
            h_star_q05=3,
            h_star_q95=3,
            tau=tau,
            prior_drift_share_at_h_max=0.1,
        )

    monkeypatch.setattr(
        "david.theorems.B_prime.compute_activity_autocorrelation",
        fake_autocorrelation,
    )
    monkeypatch.setattr(
        "david.theorems.D_forecast_horizon.horizon_validity_from_z_future_draws",
        fake_horizon_validity,
    )

    gates = _run_theorem_gates(
        mock_fit,
        {"U": 50, "R": 1, "T": 12, "K": K, "S": S, "M": 1, "L": L, "H_forecast": 1},
    )

    assert gates["B_prime"]["gate_status"] == "pass"
    assert captured["autocorr_dwell"]
    assert captured["horizon_dwell"]
    assert np.allclose(captured["autocorr_dwell"][0], lambda_val + 1.0)
    assert np.allclose(captured["horizon_dwell"][0], lambda_val + 1.0)


def test_production_d_prime_gates_on_minimum_regime_q05(monkeypatch):
    from types import SimpleNamespace

    from david.model.fit import _run_theorem_gates

    D, S, L, K = 20, 3, 3, 3
    mock_fit = _make_mock_fit(D, S, L, K, I_val=0.50)
    terminal = np.zeros((D, 3, L))
    terminal[:, 0, 0] = 1.0
    terminal[:, 1, 1] = 1.0
    terminal[:, 2, :] = np.array([0.25, 0.25, 0.50])
    mock_fit._vars["terminal_regime_posterior_draw"] = terminal
    mock_fit._vars["z_future"] = np.ones((D, 3, 12), dtype=int)
    q05_by_series = {"series_0": 6, "series_1": 2, "series_2": 8}
    captured_z_t: dict[str, np.ndarray] = {}
    captured_z_future: dict[str, np.ndarray] = {}

    def fake_autocorrelation(Pi_off_diag, dwell_mean, phi_k, max_lag, n_mc=500):
        return np.zeros(max_lag)

    def fake_horizon_validity(
        cell_id,
        Pi_off_diag_draws,
        dwell_mean_draws,
        z_t_distribution,
        z_future_draws,
        h_max=18,
        tau=0.5,
        n_bootstrap=200,
    ):
        captured_z_t[cell_id] = np.asarray(z_t_distribution).copy()
        captured_z_future[cell_id] = np.asarray(z_future_draws).copy()
        q05 = q05_by_series[cell_id]
        return SimpleNamespace(
            h_star_months=12,
            h_star_q05=q05,
            h_star_q95=12,
            tau=tau,
            prior_drift_share_at_h_max=0.1,
        )

    monkeypatch.setattr(
        "david.theorems.B_prime.compute_activity_autocorrelation",
        fake_autocorrelation,
    )
    monkeypatch.setattr(
        "david.theorems.D_forecast_horizon.horizon_validity_from_z_future_draws",
        fake_horizon_validity,
    )

    gates = _run_theorem_gates(
        mock_fit,
        {"U": 90, "R": 3, "T": 30, "K": K, "S": S, "M": 1, "L": L, "H_forecast": 1},
    )

    d = gates["D_prime"]
    assert d["h_star_months"] == 2
    assert d["h_star_q05"] == 2
    assert d["aggregation"] == "min_series_z_future_q05_first_crossing"
    assert d["terminal_regime_posterior_source"] == "stan_generated_quantities"
    assert d["forecast_regime_source"] == "stan_z_future_generated_quantities"
    assert d["h_stars_q05_per_series"] == [6, 2, 8]
    assert set(captured_z_t) == {"series_0", "series_1", "series_2"}
    assert captured_z_t["series_2"].shape == (D, L)
    assert captured_z_future["series_2"].shape == (D, 12)
    assert np.allclose(captured_z_t["series_2"][0], np.array([0.25, 0.25, 0.50]))
    assert d["gate_status"] == "fail"


def test_production_d_prime_records_prior_sensitivity_route_changes(monkeypatch):
    from types import SimpleNamespace

    from david.model.fit import _run_theorem_gates

    D, S, L, K = 20, 3, 2, 3
    mock_fit = _make_mock_fit(D, S, L, K, I_val=0.50)
    mock_fit._vars["dwell_lambda"] = np.full((D, L), 2.0)

    def fake_autocorrelation(Pi_off_diag, dwell_mean, phi_k, max_lag, n_mc=500):
        return np.zeros(max_lag)

    def fake_horizon_validity(
        cell_id,
        Pi_off_diag_draws,
        dwell_mean_draws,
        z_t_distribution,
        z_future_draws,
        h_max=18,
        tau=0.5,
        n_bootstrap=200,
    ):
        mean_dwell = float(np.mean(dwell_mean_draws))
        q05 = 6 if mean_dwell > 3.5 else 3
        return SimpleNamespace(
            h_star_months=q05,
            h_star_q05=q05,
            h_star_q95=q05,
            tau=tau,
            prior_drift_share_at_h_max=0.1,
        )

    monkeypatch.setattr(
        "david.theorems.B_prime.compute_activity_autocorrelation",
        fake_autocorrelation,
    )
    monkeypatch.setattr(
        "david.theorems.D_forecast_horizon.horizon_validity_from_z_future_draws",
        fake_horizon_validity,
    )

    gates = _run_theorem_gates(
        mock_fit,
        {"U": 50, "R": 1, "T": 12, "K": K, "S": S, "M": 1, "L": L, "H_forecast": 1},
    )

    sensitivity = gates["D_prime"]["h_star_prior_sensitivity"]
    assert sensitivity["nominal_h_star"] == 3
    assert sensitivity["plus_1sd_h_star"] == 6
    assert sensitivity["sensitivity_changes_route"] is True
    assert 6 in sensitivity["route_change_horizons"]
