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
from david.theorems.C_renamed import (
    compute_posterior_fdp_threshold,
    compute_fdp_exceedance_gate,
)


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
    prior_sensitive: bool = False,
    lambda_lo: float = 0.40,      # FG6 width = hi-lo; > 0.20 fails
    lambda_hi: float = 0.60,
    cell_id: str = "cell",
) -> dict:
    return {
        "cell_id": cell_id,
        "p_active": p_active,
        "source_independence": {
            "gate_status": "pass",
            "reason": "structural_source_independence_above_floor",
            "structural_s_eff": 3.4,
            "floor": 3.0,
            "missing_pair_keys": [],
        },
        "identification_distance_posterior_median": id_dist,
        "informativeness_I_O_lower_95": info_lower95,
        "informativeness_n_eff_i2": n_eff_i2,
        "horizon_validity": {
            "below_h_star": below_h_star,
            "prior_sensitivity": {
                "route_changes_at_horizon": prior_sensitive,
            },
        },
        "lambda_endogenous_bounds": [lambda_lo, lambda_hi],
    }


def _run_router(
    cells: list[dict],
    tmp_path: Path,
    headline_draws: np.ndarray | None = None,
) -> tuple[dict, list[np.ndarray]]:
    """Write cells to a synthetic forecast dir, run the router.

    Returns (route_ledger dict, list of p_hat arrays seen by the Theorem C kernel).
    The Theorem C kernel is patched with a side-effect that captures every
    p_hat passed to it and delegates to the real implementation so the
    router does not error.

    Parameters
    ----------
    headline_draws:
        Optional (n_draws, n_headline_cells) int array.  When provided it is
        saved as ``headline_draws.npy`` in the forecast dir, activating the
        C-4 exceedance gate path in the router.  Columns must be in the same
        order as the headline cells in ``cells``.
    """
    forecast_dir = tmp_path / "forecast_run"
    forecast_dir.mkdir()
    (forecast_dir / "cells_h3.json").write_text(json.dumps(cells))
    if headline_draws is not None:
        np.save(str(forecast_dir / "headline_draws.npy"), headline_draws)

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


def test_missing_n_eff_i2_routes_prior_dominated(tmp_path):
    """FG3 condition 2 is fail-closed: missing N_eff·I² cannot pass."""
    c_missing = _cell(p_active=0.99, cell_id="missing_n_eff")
    c_missing.pop("informativeness_n_eff_i2")
    c_headline = _cell(p_active=0.80, cell_id="headline")

    ledger, captured = _run_router([c_missing, c_headline], tmp_path)

    missing_in_ledger = next(c for c in ledger["cells"] if c["cell_id"] == "missing_n_eff")
    assert missing_in_ledger["forecast_route"] == "prior_dominated"
    assert "FG3_N_eff_I2_missing" in missing_in_ledger["route_reasons"]
    assert captured
    assert c_missing["p_active"] not in captured[0]


def test_missing_source_independence_routes_evidence_gap(tmp_path):
    """FG2 must fail closed when structural source-independence evidence is absent."""
    c_missing = _cell(p_active=0.99, cell_id="missing_source_independence")
    c_missing.pop("source_independence")
    c_headline = _cell(p_active=0.80, cell_id="headline")

    ledger, captured = _run_router([c_missing, c_headline], tmp_path)

    missing_in_ledger = next(
        c for c in ledger["cells"] if c["cell_id"] == "missing_source_independence"
    )
    assert missing_in_ledger["forecast_route"] == "evidence_gap"
    assert "FG2_source_independence_missing" in missing_in_ledger["route_reasons"]
    assert captured
    assert c_missing["p_active"] not in captured[0]


def test_low_structural_s_eff_routes_evidence_gap(tmp_path):
    """FG2 includes the structural S_eff/Kruskal source floor, not only d(theta)."""
    c_low = _cell(p_active=0.99, cell_id="low_s_eff")
    c_low["source_independence"] = {
        "gate_status": "fail",
        "reason": "structural_s_eff_2.400_below_floor_3.000",
        "structural_s_eff": 2.4,
        "floor": 3.0,
        "missing_pair_keys": [],
    }
    c_headline = _cell(p_active=0.80, cell_id="headline")

    ledger, captured = _run_router([c_low, c_headline], tmp_path)

    low_in_ledger = next(c for c in ledger["cells"] if c["cell_id"] == "low_s_eff")
    assert low_in_ledger["forecast_route"] == "evidence_gap"
    assert (
        "FG2_structural_source_independence:structural_s_eff_2.400_below_floor_3.000"
        in low_in_ledger["route_reasons"]
    )
    assert captured
    assert c_low["p_active"] not in captured[0]


def test_h_star_prior_sensitivity_routes_monitor_only(tmp_path):
    """P6: if ±1 SD dwell-prior perturbation changes h* route, no headline."""
    c_sensitive = _cell(
        p_active=0.96,
        cell_id="prior_sensitive_horizon",
        prior_sensitive=True,
    )
    c_headline = _cell(p_active=0.82, cell_id="headline")

    ledger, captured = _run_router([c_sensitive, c_headline], tmp_path)

    sensitive_in_ledger = next(
        c for c in ledger["cells"] if c["cell_id"] == "prior_sensitive_horizon"
    )
    assert sensitive_in_ledger["forecast_route"] == "monitor_only"
    assert "FG5_h_star_prior_sensitive" in sensitive_in_ledger["route_reasons"]
    assert captured
    assert c_sensitive["p_active"] not in captured[0]
    assert ledger["route_counts"]["monitor_only"] == 1
    assert ledger["h_star_prior_sensitivity"]["n_cells_prior_sensitive"] == 1
    assert "prior_sensitive_horizon" in ledger["h_star_prior_sensitivity"]["prior_sensitive_cells"]


def test_missing_h_star_prior_sensitivity_routes_monitor_only(tmp_path):
    """P6 sensitivity evidence is mandatory for headline eligibility."""
    c_missing = _cell(p_active=0.96, cell_id="missing_prior_sensitivity")
    c_missing["horizon_validity"].pop("prior_sensitivity")
    c_headline = _cell(p_active=0.82, cell_id="headline")

    ledger, captured = _run_router([c_missing, c_headline], tmp_path)

    missing_in_ledger = next(
        c for c in ledger["cells"] if c["cell_id"] == "missing_prior_sensitivity"
    )
    assert missing_in_ledger["forecast_route"] == "monitor_only"
    assert "FG5_h_star_prior_sensitivity_missing" in missing_in_ledger["route_reasons"]
    assert captured
    assert c_missing["p_active"] not in captured[0]
    assert ledger["h_star_prior_sensitivity"]["n_cells_missing_sensitivity"] == 1


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


# ---------------------------------------------------------------------------
# C-4: FG6 exceedance gate wiring in the router
# ---------------------------------------------------------------------------

def test_exceedance_gate_skipped_when_no_draws_file(tmp_path):
    """Without headline_draws.npy the exceedance gate records skipped=True.

    Existing C-2 tests have no draws file; this confirms backward-compatibility.
    """
    cells = [
        _cell(p_active=0.95, cell_id="h1"),
        _cell(p_active=0.90, cell_id="h2"),
    ]
    ledger, _ = _run_router(cells, tmp_path)

    exc = ledger["exceedance_gate"]
    assert exc.get("skipped") is True
    assert "no_joint_draws_file" in exc.get("reason", "")


def test_exceedance_gate_shrinks_flagged_set_in_ledger(tmp_path):
    """C-4 acceptance criterion: both expectation and exceedance results are
    recorded in the ledger, and the exceedance gate shrinks the flagged set.

    Scenario:
    - 5 headline cells with p = [0.95, 0.94, 0.92, 0.91, 0.90].
    - Expectation rule: m* = 5 (E[FDP] ≈ 0.076 ≤ q=0.10).
    - Joint draws: 940 good (all TP) + 60 bad (rank-3 and rank-4 FP).
      Exceedance at m=5 and m=4: 6% > α=5% → FAIL.
      Exceedance at m=3: 0% → PASS → m_accepted = 3.

    Bad draws are SPREAD UNIFORMLY (every 16 rows, not block-structured) so
    the per-cell chains are approximately iid → bulk-ESS ≈ n_draws → MCSE
    well below the C-6 threshold (0.01).  This isolates the exceedance gate
    from the MCSE floor check.
    """
    cells = [
        _cell(p_active=0.95, cell_id="h1"),
        _cell(p_active=0.94, cell_id="h2"),
        _cell(p_active=0.92, cell_id="h3"),
        _cell(p_active=0.91, cell_id="h4"),
        _cell(p_active=0.90, cell_id="h5"),
    ]

    # Joint draws ordered identically to headline_cells (p already descending).
    # Spread the 60 "bad" draws uniformly so per-cell ESS stays high (iid-like).
    n_draws, M = 1000, 5
    draws = np.ones((n_draws, M), dtype=np.int32)
    bad_rows = np.arange(0, 960, 16)  # 60 evenly-spaced rows
    draws[bad_rows, 3] = 0  # rank-3 cell FP in 60 spread draws
    draws[bad_rows, 4] = 0  # rank-4 cell FP in 60 spread draws

    ledger, captured = _run_router(cells, tmp_path, headline_draws=draws)

    # --- Expectation gate (posterior_fdp block) ---
    assert captured, "compute_posterior_fdp_threshold was never called"
    fdp = ledger["posterior_fdp"]
    assert fdp["n_flagged"] == 5, (
        f"Expectation rule should flag 5 cells, got {fdp['n_flagged']}"
    )

    # --- Exceedance gate (exceedance_gate block) ---
    assert "exceedance_gate" in ledger, "exceedance_gate block missing from ledger"
    exc = ledger["exceedance_gate"]
    assert not exc.get("skipped", False), f"Exceedance gate unexpectedly skipped: {exc}"
    assert exc["m_star_expectation"] == 5
    assert exc["m_accepted"] == 3
    assert not exc["markov_fallback_used"]

    # --- Per-cell flags: both keys recorded on headline cells ---
    headline_in_ledger = [
        c for c in ledger["cells"] if c["forecast_route"] == "headline"
    ]
    assert len(headline_in_ledger) == 5

    n_flagged_by_exp = sum(
        1 for c in headline_in_ledger if c.get("headline_flagged_by_posterior_fdp")
    )
    n_flagged_by_exc = sum(
        1 for c in headline_in_ledger if c.get("headline_flagged_by_exceedance_gate")
    )
    assert n_flagged_by_exp == 5, (
        f"Expected 5 cells flagged by expectation rule, got {n_flagged_by_exp}"
    )
    assert n_flagged_by_exc == 3, (
        f"Expected 3 cells flagged by exceedance gate, got {n_flagged_by_exc}"
    )
    # The exceedance gate never flags more than the expectation gate
    assert n_flagged_by_exc <= n_flagged_by_exp


def test_exceedance_gate_skipped_when_m_star_is_zero(tmp_path):
    """When m* = 0 (no cells pass FDP threshold) the exceedance gate is skipped."""
    # Single cell with very low p_active → E[FDP] = 0.99 >> q → n_flagged = 0
    cells = [_cell(p_active=0.01, cell_id="h1")]
    draws = np.ones((100, 1), dtype=np.int32)

    ledger, _ = _run_router(cells, tmp_path, headline_draws=draws)

    exc = ledger["exceedance_gate"]
    # m_star = 0 because E[FDP] for a single cell with p=0.01 is 0.99 > q
    assert exc.get("skipped") is True
    assert exc.get("reason") in ("m_star_is_zero", "no_joint_draws_file")


# ---------------------------------------------------------------------------
# C-6: MCSE/ESS floor wiring in the router
# ---------------------------------------------------------------------------

def _make_low_ess_draws(n_draws: int, n_cells: int, bad_cell: int) -> np.ndarray:
    """Return draws where `bad_cell` is block-autocorrelated (low ESS).

    All other cells are iid Bernoulli(0.9) from a fixed seed so they have
    high ESS and pass the MCSE floor.  The bad cell alternates 0/1 in 50-draw
    blocks, giving ESS ≈ n_draws / 50 and MCSE >> 0.01.
    """
    rng = np.random.default_rng(99)
    draws = rng.binomial(1, 0.9, size=(n_draws, n_cells)).astype(np.int32)
    # Overwrite bad_cell with a block-alternating chain.
    block = np.repeat(np.tile([1, 0], n_draws // (2 * 50) + 1), 50)[:n_draws]
    draws[:, bad_cell] = block
    return draws


def test_low_ess_cell_excluded_from_fdp_family(tmp_path):
    """C-6 acceptance criterion in the router.

    Three headline cells.  Cell 1 gets block-autocorrelated draws → low ESS →
    MCSE ≥ threshold → demoted to evidence_gap with reason
    'fdp_mcse_exceeds_gate_margin'.  Cells 0 and 2 are iid → high ESS → stay
    in the claim-eligible family.

    Uses a real (un-patched) router run so that check_mcse_floor and
    compute_posterior_fdp_threshold both execute.
    """
    cells = [
        _cell(p_active=0.95, cell_id="c0"),
        _cell(p_active=0.90, cell_id="c1"),  # will be excluded by MCSE
        _cell(p_active=0.85, cell_id="c2"),
    ]
    draws = _make_low_ess_draws(n_draws=2000, n_cells=3, bad_cell=1)

    forecast_dir = tmp_path / "forecast_run"
    forecast_dir.mkdir()
    (forecast_dir / "cells_h3.json").write_text(json.dumps(cells))
    np.save(str(forecast_dir / "headline_draws.npy"), draws)

    with (
        patch("david.engine.router.latest_forecast_dir", return_value=forecast_dir),
        patch("david.engine.router._measurement_gates_pass", return_value=(True, [])),
        patch("david.engine.router._forecast_sbc_pass", return_value=(True, "")),
    ):
        apply_forecast_routing(out_dir=tmp_path)

    ledger = json.loads((tmp_path / "route_ledger.json").read_text())

    # Cell c1 must be demoted to evidence_gap with the typed reason.
    c1 = next(c for c in ledger["cells"] if c["cell_id"] == "c1")
    assert c1["forecast_route"] == "evidence_gap", (
        f"Expected c1 to be demoted to evidence_gap; got {c1['forecast_route']}"
    )
    assert "fdp_mcse_exceeds_gate_margin" in c1["route_reasons"], (
        f"Expected binding reason in route_reasons; got {c1['route_reasons']}"
    )

    # m_claim_eligible must reflect the exclusion (c0 and c2 only).
    assert ledger["m_claim_eligible"] == 2, (
        f"Expected 2 claim-eligible cells after MCSE exclusion; "
        f"got {ledger['m_claim_eligible']}"
    )

    # MCSE floor block must be present and record the exclusion.
    mcse = ledger["mcse_floor"]
    assert not mcse.get("skipped", False)
    assert mcse["n_cells_excluded"] >= 1
    assert mcse["binding_reason"] == "fdp_mcse_exceeds_gate_margin"


def test_mcse_floor_skipped_when_no_draws_file(tmp_path):
    """Without headline_draws.npy the MCSE floor records skipped=True."""
    cells = [_cell(p_active=0.90, cell_id="h1")]
    ledger, _ = _run_router(cells, tmp_path)
    mcse = ledger["mcse_floor"]
    assert mcse.get("skipped") is True
    assert "no_joint_draws_file" in mcse.get("reason", "")


# ---------------------------------------------------------------------------
# C-7: Ledger fields completeness
# ---------------------------------------------------------------------------

def test_ledger_fdp_fields_complete(tmp_path):
    """C-7: route_ledger.json carries t*(q), m*, exceedance_fraction, and
    per-cell fdp_binding_reason; no silent headline cells.

    Uses spread bad draws (as in test_exceedance_gate_shrinks_flagged_set_in_ledger)
    so that all cells pass the MCSE floor and the full FDP block is populated.
    """
    cells = [
        _cell(p_active=0.95, cell_id="h1"),
        _cell(p_active=0.94, cell_id="h2"),
        _cell(p_active=0.92, cell_id="h3"),
        _cell(p_active=0.91, cell_id="h4"),
        _cell(p_active=0.90, cell_id="h5"),
    ]
    n_draws, M = 1000, 5
    draws = np.ones((n_draws, M), dtype=np.int32)
    bad_rows = np.arange(0, 960, 16)
    draws[bad_rows, 3] = 0
    draws[bad_rows, 4] = 0

    forecast_dir = tmp_path / "forecast_run"
    forecast_dir.mkdir()
    (forecast_dir / "cells_h3.json").write_text(json.dumps(cells))
    np.save(str(forecast_dir / "headline_draws.npy"), draws)

    with (
        patch("david.engine.router.latest_forecast_dir", return_value=forecast_dir),
        patch("david.engine.router._measurement_gates_pass", return_value=(True, [])),
        patch("david.engine.router._forecast_sbc_pass", return_value=(True, "")),
    ):
        apply_forecast_routing(out_dir=tmp_path)

    ledger = json.loads((tmp_path / "route_ledger.json").read_text())

    # --- C-7: t*(q) and m* must be present in posterior_fdp block ---
    fdp = ledger["posterior_fdp"]
    assert "t_star_q" in fdp, "t*(q) alias missing from posterior_fdp block"
    assert "m_star" in fdp, "m* alias missing from posterior_fdp block"
    assert isinstance(fdp["t_star_q"], float)
    assert isinstance(fdp["m_star"], int)

    # --- C-7: exceedance_fraction must be present in exceedance_gate block ---
    exc = ledger["exceedance_gate"]
    assert not exc.get("skipped", False), f"Exceedance gate unexpectedly skipped: {exc}"
    assert "exceedance_fraction" in exc, (
        "exceedance_fraction alias missing from exceedance_gate block"
    )

    # --- C-7: per-cell fdp_binding_reason — no silent headline cells ---
    headline_cells_in_ledger = [
        c for c in ledger["cells"] if c.get("forecast_route") == "headline"
    ]
    for c in headline_cells_in_ledger:
        assert "fdp_binding_reason" in c, (
            f"Cell {c.get('cell_id')} has no fdp_binding_reason (silent cell)"
        )
        assert c["fdp_binding_reason"] in (
            "flagged_fdp_expectation_and_exceedance",
            "flagged_fdp_expectation_only",
            "unflagged_below_fdp_threshold",
            "unflagged_exceedance_shrinkage",
        ), f"Unrecognised fdp_binding_reason: {c['fdp_binding_reason']}"

    # --- C-7: sensitivity_envelope block present (skipped when no theta grid) ---
    assert "sensitivity_envelope" in ledger
    env = ledger["sensitivity_envelope"]
    assert env.get("skipped") is True  # no theta_grid file → skipped is expected
    assert "worst_theta_index" in env  # key present even when skipped

    # --- mcse_floor block present ---
    assert "mcse_floor" in ledger
