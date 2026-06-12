"""Tests for C-2: claim-eligible family definition.

The posterior expected FDP (Theorem C) must run ONLY over cells that
survived FG1–FG5, i.e. those assigned the ``headline`` route.  Cells
assigned ``evidence_gap``, ``withhold``, ``prior_dominated``,
``horizon_prior_dominated``, or ``monitor_only`` must never enter the
p_hat vector passed to the Theorem C kernel.

Gate ↔ route mapping under test:
  FG2 fail  → evidence_gap
  FG3 fail  → prior_dominated
  FG5 fail  → horizon_prior_dominated
  FG6 fail  → monitor_only
  all pass  → headline  (claim-eligible)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from david.engine.router import apply_forecast_routing
from david.theorems.C_renamed import compute_posterior_fdp_threshold


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cell(
    p_active: float,
    *,
    id_dist: float = 0.10,        # FG2 floor = 0.05; values < 0.05 fail
    info_lower95: float = 0.20,   # FG3 I-floor = 0.10; values < 0.10 fail
    n_eff_i2: float = 5.0,        # FG3 N_eff·I² floor = 3.0; values < 3.0 fail
    below_h_star: bool = True,    # FG5; False → horizon_prior_dominated
    lambda_lo: float = 0.40,      # FG6 width = hi-lo; > 0.20 fails
    lambda_hi: float = 0.60,
    cell_id: str = "cell",
) -> dict:
    return {
        "cell_id": cell_id,
        "p_active": p_active,
        "identification_distance_posterior_median": id_dist,
        "informativeness_I_O_lower_95": info_lower95,
        "informativeness_n_eff_i2": n_eff_i2,
        "horizon_validity": {"below_h_star": below_h_star},
        "lambda_endogenous_bounds": [lambda_lo, lambda_hi],
    }


def _run_router(
    cells: list[dict], tmp_path: Path
) -> tuple[dict, list[np.ndarray]]:
    """Write cells to a synthetic forecast dir, run the router.

    Returns (route_ledger dict, list of p_hat arrays seen by the Theorem C kernel).
    The Theorem C kernel is patched with a side-effect that captures every
    p_hat passed to it and delegates to the real implementation so the
    router does not error.
    """
    forecast_dir = tmp_path / "forecast_run"
    forecast_dir.mkdir()
    (forecast_dir / "cells_h3.json").write_text(json.dumps(cells))

    captured: list[np.ndarray] = []

    def _capture(p_hat: np.ndarray, q: float = 0.10):
        captured.append(p_hat.copy())
        # Delegate to the real kernel so the router completes normally.
        return compute_posterior_fdp_threshold(p_hat, q=q)

    with (
        patch("david.engine.router.latest_forecast_dir", return_value=forecast_dir),
        patch("david.engine.router._measurement_gates_pass", return_value=(True, [])),
        patch("david.engine.router._forecast_sbc_pass", return_value=(True, "")),
        patch(
            "david.engine.router.compute_posterior_fdp_threshold",
            side_effect=_capture,
        ),
    ):
        apply_forecast_routing(out_dir=tmp_path)

    ledger = json.loads((tmp_path / "route_ledger.json").read_text())
    return ledger, captured


# ---------------------------------------------------------------------------
# C-2 acceptance criterion: evidence_gap cell absent from p_hat
# ---------------------------------------------------------------------------

def test_evidence_gap_cell_absent_from_p_hat(tmp_path):
    """Cell failing FG2 (routed evidence_gap) must not enter the Theorem C p_hat.

    This is the primary acceptance criterion for tracker item C-2.
    """
    # id_dist 0.03 < 0.05 floor → evidence_gap
    cell_gap = _cell(p_active=0.99, id_dist=0.03, cell_id="A_evidence_gap")
    # all gates pass → headline
    cell_headline = _cell(p_active=0.85, cell_id="B_headline")

    ledger, captured = _run_router([cell_gap, cell_headline], tmp_path)

    assert captured, "compute_posterior_fdp_threshold was never called"
    p_hat = captured[0]

    assert cell_gap["p_active"] not in p_hat, (
        f"evidence_gap p_active={cell_gap['p_active']} incorrectly entered "
        f"Theorem C kernel; p_hat received = {p_hat}"
    )
    assert cell_headline["p_active"] in p_hat, (
        f"headline p_active={cell_headline['p_active']} missing from p_hat = {p_hat}"
    )

    assert ledger["route_counts"]["evidence_gap"] == 1
    assert ledger["route_counts"]["headline"] == 1
    assert ledger["m_claim_eligible"] == 1


# ---------------------------------------------------------------------------
# Comprehensive: all non-headline routes excluded
# ---------------------------------------------------------------------------

def test_all_non_headline_routes_excluded_from_p_hat(tmp_path):
    """Cells assigned any non-headline route must not appear in p_hat.

    Covers evidence_gap (FG2), prior_dominated (FG3 I-floor and N_eff·I²),
    horizon_prior_dominated (FG5), and monitor_only (FG6).
    """
    # FG2 fail → evidence_gap
    c_fg2 = _cell(p_active=0.91, id_dist=0.03, cell_id="C_evidence_gap")

    # FG3 I-lower-95 fail → prior_dominated
    c_fg3_i = _cell(p_active=0.92, info_lower95=0.05, cell_id="C_prior_dom_I")

    # FG3 N_eff·I² fail → prior_dominated  (id_dist ok so FG2 passes)
    c_fg3_n = _cell(p_active=0.93, n_eff_i2=1.0, cell_id="C_prior_dom_N")

    # FG5 fail → horizon_prior_dominated
    c_fg5 = _cell(p_active=0.94, below_h_star=False, cell_id="C_horizon")

    # FG6 fail → monitor_only (width 0.30 > 0.20 floor)
    c_fg6 = _cell(p_active=0.95, lambda_lo=0.10, lambda_hi=0.40, cell_id="C_monitor")

    # All gates pass → headline (sole member of claim-eligible family)
    c_headline = _cell(p_active=0.75, cell_id="C_headline")

    excluded_p = {c["p_active"] for c in [c_fg2, c_fg3_i, c_fg3_n, c_fg5, c_fg6]}
    ledger, captured = _run_router(
        [c_fg2, c_fg3_i, c_fg3_n, c_fg5, c_fg6, c_headline], tmp_path
    )

    assert captured, "compute_posterior_fdp_threshold was never called"
    p_hat = captured[0]

    for p_val in excluded_p:
        assert p_val not in p_hat, (
            f"Non-headline p_active={p_val} entered Theorem C kernel; "
            f"p_hat received = {p_hat}"
        )

    assert c_headline["p_active"] in p_hat
    assert len(p_hat) == 1, f"Expected exactly 1 headline cell in p_hat, got {len(p_hat)}"

    assert ledger["m_claim_eligible"] == 1
    assert ledger["route_counts"]["evidence_gap"] == 1
    assert ledger["route_counts"]["prior_dominated"] == 2
    assert ledger["route_counts"]["horizon_prior_dominated"] == 1
    assert ledger["route_counts"]["monitor_only"] == 1
    assert ledger["route_counts"]["headline"] == 1


# ---------------------------------------------------------------------------
# m_claim_eligible accuracy
# ---------------------------------------------------------------------------

def test_m_claim_eligible_matches_headline_count(tmp_path):
    """route_ledger.m_claim_eligible must equal the number of headline-routed cells."""
    cells = [
        _cell(p_active=0.80, cell_id="h1"),
        _cell(p_active=0.82, cell_id="h2"),
        _cell(p_active=0.84, cell_id="h3"),
        _cell(p_active=0.99, id_dist=0.02, cell_id="eg1"),         # evidence_gap
        _cell(p_active=0.97, info_lower95=0.04, cell_id="pd1"),    # prior_dominated
    ]

    ledger, _ = _run_router(cells, tmp_path)

    assert ledger["m_claim_eligible"] == 3
    assert ledger["route_counts"]["headline"] == 3
    assert ledger["m_claim_eligible"] == ledger["route_counts"]["headline"]


def test_m_claim_eligible_zero_when_all_excluded(tmp_path):
    """When no cell passes FG1–FG5, M must be 0 and the Theorem C kernel not called."""
    cells = [
        _cell(p_active=0.99, id_dist=0.02, cell_id="eg1"),   # evidence_gap
        _cell(p_active=0.98, id_dist=0.01, cell_id="eg2"),   # evidence_gap
    ]

    ledger, captured = _run_router(cells, tmp_path)

    assert captured == [], (
        "Theorem C kernel must not be called when M=0 (no claim-eligible cells)"
    )
    assert ledger["m_claim_eligible"] == 0
    assert ledger["route_counts"]["headline"] == 0
