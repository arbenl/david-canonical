from __future__ import annotations

import json

import pytest


def _fit_data() -> dict:
    return {
        "R": 1,
        "T": 2,
        "L": 3,
        "K": 3,
        "S": 3,
        "M": 2,
        "U": 2,
        "unit_time": [1, 2],
        "label_unit": [1, 2],
        "y": [0, 1],
    }


def test_f1_gate_fails_closed_without_frozen_reference(tmp_path, monkeypatch):
    from david.model.fit import _run_f1_gate

    monkeypatch.setattr("david.model.fit.PRIOR_PREDICTIVE_REFERENCE", tmp_path / "missing.json")

    result = _run_f1_gate(_fit_data(), n_prior_worlds=2)

    assert result["gate_status"] == "fail"
    assert result["n_prior_worlds"] == 0
    assert result["reason"].startswith("frozen_f1_reference_missing_or_invalid:")


def test_f1_gate_uses_frozen_reference_band(tmp_path, monkeypatch):
    from david.model.fit import _run_f1_gate

    ref = tmp_path / "prior_predictive_reference.json"
    ref.write_text(json.dumps({
        "version": "test",
        "reference_window": "2020-01-01:2021-01-01",
        "historical_band_5th": 0.0,
        "historical_band_95th": 1.0,
    }))
    monkeypatch.setattr("david.model.fit.PRIOR_PREDICTIVE_REFERENCE", ref)

    result = _run_f1_gate(_fit_data(), n_prior_worlds=2)

    assert result["gate_status"] == "pass"
    assert result["historical_band_5th"] == pytest.approx(0.0)
    assert result["historical_band_95th"] == pytest.approx(1.0)
    assert result["reference_version"] == "test"
