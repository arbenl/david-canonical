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
    check_cell,
)
from david.config import INFORMATIVENESS_FLOOR_LOWER_95


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
    assert res.n_eff_phi_estimate == pytest.approx(200 * 0.5 ** 2)  # = 50


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
