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

@pytest.mark.parametrize("I,expected", [
    (0.0, 0.5),    # uninformative channel → chance-level error
    (1.0, 0.0),    # perfect channel → zero error
    (0.5, 0.25),
    (0.8, 0.10),
])
def test_bayes_error_identity(I, expected):
    assert bayes_classification_error(I) == pytest.approx(expected)


def test_bayes_error_monotone_decreasing_in_I():
    Is = np.linspace(0.0, 1.0, 11)
    errs = [bayes_classification_error(float(i)) for i in Is]
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
    assert res.reason == "I_lower_95_above_floor"


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
