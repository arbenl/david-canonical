from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from david.engine.router import _measurement_gates_pass
from david.simulator.adversarial_battery import F11_source_conditional_independence, run_battery


def _passing_required_inputs() -> dict[str, dict]:
    return {
        "F1": {
            "prior_predictive_Y_rate": np.array([0.2]),
            "historical_Y_rate_5th": 0.1,
            "historical_Y_rate_95th": 0.3,
        },
        "F11": {
            "pairwise_residual_p_values": {
                ("s1", "s2"): 0.8,
                ("s1", "s3"): 0.7,
                ("s2", "s3"): 0.6,
            }
        },
        "F12": {"median_d": 0.10},
        "F13": {
            "forecast_p": np.array([0.4]),
            "marginal_p": np.array([0.4]),
            "beyond_h_star_mask": np.array([True]),
        },
        "F14": {"forecast_sbc_summary": {"pass_80": True, "pass_95": True}},
    }


def test_run_battery_fails_when_required_inputs_missing():
    result = run_battery(inputs={})

    assert result["gate_status"] == "fail"
    assert set(result["failed_tests"]) >= {"F1", "F11", "F12", "F13", "F14"}
    f11 = next(r for r in result["ledger"] if r["test_id"] == "F11")
    assert f11["gate_status"] == "fail"
    assert f11["reason"] == "required_inputs_missing"


def test_run_battery_allows_conditional_missing_but_requires_required():
    result = run_battery(inputs=_passing_required_inputs())

    assert result["gate_status"] == "pass"
    skipped = {r["test_id"]: r for r in result["ledger"] if r["gate_status"] == "skip"}
    assert {"F3", "F4", "F5", "F6", "F15"}.issubset(skipped)
    assert all(r["reason"] == "conditional_inputs_missing" for r in skipped.values())


def test_f11_no_pairs_fails_closed():
    result = F11_source_conditional_independence({})

    assert result.gate_status == "fail"
    assert result.reason == "no_pairs_required_independence_evidence_missing"


def test_f11_structural_s_eff_fails_closed_below_floor():
    result = F11_source_conditional_independence(
        {("s1", "s2"): 0.9},
        structural_s_eff=2.5,
        structural_floor=3.0,
    )

    assert result.gate_status == "fail"
    assert result.reason == "structural_s_eff_2.500_below_floor_3.000"


def test_assemble_f11_inputs_from_adjudicated_rows_detects_source_dependence():
    from david.validation.falsification import _assemble_f11_inputs

    evidence_rows = []
    label_rows = []
    labels = [0] * 6 + [1] * 6
    for idx, label in enumerate(labels):
        for src in ("s1", "s2"):
            eid = f"ev_{idx}_{src}"
            evidence_rows.append(
                {
                    "evidence_id": eid,
                    "stratum_g": "g1",
                    "source_id": src,
                    "evidence_date": f"2026-01-{idx + 1:02d}",
                }
            )
            label_rows.append(
                {
                    "evidence_id": eid,
                    "coder_id": "c1",
                    "label": str(label),
                    "tactic_k": "SIO",
                }
            )

    inputs = _assemble_f11_inputs(
        {
            "evidence_rows": evidence_rows,
            "source_rows": [{"source_id": "s1"}, {"source_id": "s2"}],
            "label_rows": label_rows,
            "strata_rows": [],
        }
    )

    assert ("s1", "s2") in inputs["pairwise_residual_p_values"]
    structural_result = F11_source_conditional_independence(**inputs)
    assert structural_result.gate_status == "fail"
    assert structural_result.reason == "structural_s_eff_1.000_below_floor_3.000"

    inputs["structural_s_eff"] = 3.0
    result = F11_source_conditional_independence(**inputs)
    assert result.gate_status == "fail"
    assert result.reason.startswith("1_pairs_failed_at_")


def test_router_measurement_gate_allows_conditional_fit_skips_with_passing_ledger(tmp_path: Path):
    fit_dir = tmp_path / "fit_run"
    forecast_dir = tmp_path / "forecast_run"
    fit_dir.mkdir()
    forecast_dir.mkdir()
    (fit_dir / "fit_summary.json").write_text(
        json.dumps(
            {
                "gate_status": "pass",
                "gates": {
                    "F1": {"gate_status": "pass"},
                    "F3": {"gate_status": "skip"},
                    "F4": {"gate_status": "pass"},
                    "F5": {"gate_status": "pass"},
                },
            }
        )
    )
    (forecast_dir / "falsification_ledger.json").write_text(
        json.dumps(
            {
                "battery_result": {
                    "gate_status": "pass",
                    "failed_tests": [],
                }
            }
        )
    )

    with patch("david.engine.router.latest_fit_dir", return_value=fit_dir):
        ok, failed = _measurement_gates_pass(forecast_dir)

    assert ok is True
    assert failed == []


def test_router_measurement_gate_fails_closed_without_falsification_ledger(tmp_path: Path):
    fit_dir = tmp_path / "fit_run"
    forecast_dir = tmp_path / "forecast_run"
    fit_dir.mkdir()
    forecast_dir.mkdir()
    (fit_dir / "fit_summary.json").write_text(
        json.dumps(
            {
                "gate_status": "pass",
                "gates": {
                    "F1": {"gate_status": "pass"},
                    "F3": {"gate_status": "skip"},
                    "F4": {"gate_status": "skip"},
                    "F5": {"gate_status": "skip"},
                },
            }
        )
    )

    with patch("david.engine.router.latest_fit_dir", return_value=fit_dir):
        ok, failed = _measurement_gates_pass(forecast_dir)

    assert ok is False
    assert failed == ["falsification_ledger_missing"]
