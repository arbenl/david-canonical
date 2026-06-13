"""Tests for the forecast engine (emit_forecasts, apply_forecast_routing).

These tests run offline — no Stan sampling needed. They inject synthetic
draws into temporary directories so the engine's full parse-emit-route
pipeline is exercised without hitting disk or the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from david.config import FORECAST_HORIZONS_MONTHS, LAMBDA_ENDOG_INTERVAL_MAX_WIDTH
from david.engine.forecast import (
    _parse_a_future,
    emit_forecasts,
    emit_all_horizons,
    load_posterior,
)


# ─── fixtures ────────────────────────────────────────────────────────────────

def _make_synthetic_draws(tmp_path: Path, R: int = 2, H_max: int = 6, K: int = 3,
                          D: int = 40) -> Path:
    """Write a minimal draws.parquet with a_future[r,h,k] columns."""
    rng = np.random.default_rng(42)
    cols: dict[str, np.ndarray] = {}
    for r in range(1, R + 1):
        for h in range(1, H_max + 1):
            for k in range(1, K + 1):
                cols[f"a_future[{r},{h},{k}]"] = rng.binomial(1, 0.4, size=D).astype(float)
    # Add one non-a_future column (should be ignored by _parse_a_future)
    cols["alpha_activity[1,1]"] = rng.normal(0, 1, size=D)
    df = pd.DataFrame(cols)
    draws_path = tmp_path / "draws.parquet"
    df.to_parquet(draws_path, index=False)
    return draws_path


def _make_fit_summary(tmp_path: Path, h_star: int = 3) -> Path:
    """Write a minimal fit_summary.json with theorem D' h_star."""
    summary = {
        "model_version": "m01_forward_v0.1.0",
        "run_id": tmp_path.name,
        "gate_status": "pass",
        "theorems": {
            "A_prime": {
                "theorem": "A_prime",
                "median_d_theta": 0.15,
                "q05_d_theta": 0.08,
                "floor": 0.05,
                "gate_status": "pass",
            },
            "B_prime": {
                "theorem": "B_prime",
                "median_I_worst_source": 0.12,
                "lower_95_I_worst_source": 0.10,
                "floor": 0.10,
                "gate_status": "pass",
            },
            "C_prime": {"theorem": "C_prime", "gate_status": "pass"},
            "D_prime": {
                "theorem": "D_prime",
                "h_star_months": h_star,
                "h_star_prior_sensitivity": {
                    "dwell_log_lambda_prior_sd": 0.5,
                    "nominal_h_star": h_star,
                    "minus_1sd_h_star": max(0, h_star - 3),
                    "plus_1sd_h_star": h_star + 3,
                    "route_change_horizons": [h_star + 3],
                    "sensitivity_changes_route": True,
                },
                "gate_status": "pass" if h_star >= 3 else "fail",
            },
        },
        "gates": {
            "F1": {"gate_status": "pass"},
            "F3": {"gate_status": "skip"},
            "F4": {"gate_status": "skip"},
            "F5": {"gate_status": "skip"},
        },
    }
    p = tmp_path / "fit_summary.json"
    p.write_text(json.dumps(summary))
    return p


# ─── _parse_a_future ────────────────────────────────────────────────────────

def test_parse_a_future_keys():
    posterior = {
        "a_future[1,3,1]": np.array([0.1]),
        "a_future[2,6,2]": np.array([0.2]),
        "alpha_activity[1,1]": np.array([0.0]),   # should be ignored
    }
    parsed = _parse_a_future(posterior)
    assert set(parsed) == {(1, 3, 1), (2, 6, 2)}


def test_parse_a_future_empty():
    posterior = {"alpha_activity[1,1]": np.array([0.0])}
    assert _parse_a_future(posterior) == {}


# ─── load_posterior ──────────────────────────────────────────────────────────

def test_load_posterior_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="draws.parquet"):
        load_posterior(tmp_path)


def test_load_posterior_reads_columns(tmp_path: Path):
    _make_synthetic_draws(tmp_path, R=1, H_max=3, K=2, D=20)
    post = load_posterior(tmp_path)
    assert "a_future[1,1,1]" in post
    assert post["a_future[1,1,1]"].shape == (20,)


# ─── emit_forecasts ──────────────────────────────────────────────────────────

def test_emit_forecasts_no_fit(tmp_path: Path, monkeypatch):
    import david.engine.forecast as ef
    monkeypatch.setattr(ef, "latest_fit_dir",
                        lambda: (_ for _ in ()).throw(FileNotFoundError("no fit")))
    result = emit_forecasts(horizon_months=3)
    assert result["gate_status"] == "fail"


def test_emit_forecasts_no_draws(tmp_path: Path, monkeypatch):
    """fit_summary.json present but draws.parquet absent → gate_status=fail."""
    _make_fit_summary(tmp_path)
    import david.engine.forecast as ef
    monkeypatch.setattr(ef, "latest_fit_dir", lambda: tmp_path)
    result = emit_forecasts(horizon_months=3)
    assert result["gate_status"] == "fail"
    assert "draws.parquet" in result["reason"]


def test_emit_forecasts_no_a_future(tmp_path: Path, monkeypatch):
    """draws.parquet present but no a_future columns → gate_status=skip."""
    _make_fit_summary(tmp_path)
    df = pd.DataFrame({"alpha_activity[1,1]": np.zeros(10)})
    (tmp_path / "draws.parquet").write_bytes(df.to_parquet(index=False))
    import david.engine.forecast as ef
    monkeypatch.setattr(ef, "latest_fit_dir", lambda: tmp_path)
    result = emit_forecasts(horizon_months=3)
    assert result["gate_status"] == "skip"
    assert "a_future" in result["reason"]


def test_emit_forecasts_creates_cells(tmp_path: Path, monkeypatch):
    _make_synthetic_draws(tmp_path, R=2, H_max=6, K=3, D=40)
    _make_fit_summary(tmp_path, h_star=3)
    out_dir = tmp_path / "out"
    import david.engine.forecast as ef
    monkeypatch.setattr(ef, "latest_fit_dir", lambda: tmp_path)
    result = emit_forecasts(horizon_months=3, out_dir=out_dir, fit_dir=tmp_path)
    assert result["gate_status"] == "pass"
    assert result["n_cells"] == 6   # R=2 x K=3
    cells = json.loads(Path(result["cells_path"]).read_text())
    assert len(cells) == 6


def test_emit_forecasts_cell_has_required_router_fields(tmp_path: Path, monkeypatch):
    """Each cell must expose the flat fields read by apply_forecast_routing."""
    _make_synthetic_draws(tmp_path, R=1, H_max=3, K=2, D=20)
    _make_fit_summary(tmp_path, h_star=3)
    out_dir = tmp_path / "out"
    import david.engine.forecast as ef
    monkeypatch.setattr(ef, "latest_fit_dir", lambda: tmp_path)
    result = emit_forecasts(horizon_months=3, out_dir=out_dir, fit_dir=tmp_path)
    cells = json.loads(Path(result["cells_path"]).read_text())
    required = {
                "p_active", "credible_interval_80", "credible_interval_95",
                "horizon_validity", "identification_distance_posterior_median",
                "informativeness_I_O_lower_95", "lambda_endogenous_bounds",
                "horizon_prior_sensitivity", "source_independence",
                "model_version", "fit_run_id",
            }
    for cell in cells:
        missing = required - set(cell)
        assert not missing, f"cell missing fields: {missing}"
        assert "prior_sensitivity" in cell["horizon_validity"]
        assert cell["horizon_prior_sensitivity"]["horizon_months"] == 3


def test_emit_forecasts_horizon_validity_flag(tmp_path: Path, monkeypatch):
    """Cells at horizon > h_star should be flagged horizon_prior_dominated."""
    _make_synthetic_draws(tmp_path, R=1, H_max=6, K=1, D=20)
    _make_fit_summary(tmp_path, h_star=3)
    out_dir = tmp_path / "out"
    import david.engine.forecast as ef
    monkeypatch.setattr(ef, "latest_fit_dir", lambda: tmp_path)
    # horizon=3: at h_star → below_h_star=True
    r3 = emit_forecasts(horizon_months=3, out_dir=out_dir, fit_dir=tmp_path)
    c3 = json.loads(Path(r3["cells_path"]).read_text())[0]
    assert c3["horizon_validity"]["below_h_star"] is True
    assert c3["horizon_validity"]["forecast_route"] == "conditional_forecast"
    # horizon=6: above h_star → below_h_star=False
    r6 = emit_forecasts(horizon_months=6, out_dir=out_dir, fit_dir=tmp_path)
    c6 = json.loads(Path(r6["cells_path"]).read_text())[0]
    assert c6["horizon_validity"]["below_h_star"] is False
    assert c6["horizon_validity"]["forecast_route"] == "horizon_prior_dominated"


def test_emit_all_horizons_count(tmp_path: Path, monkeypatch):
    _make_synthetic_draws(tmp_path, R=1, H_max=max(FORECAST_HORIZONS_MONTHS), K=2, D=20)
    _make_fit_summary(tmp_path, h_star=6)
    out_dir = tmp_path / "out"
    import david.engine.forecast as ef
    monkeypatch.setattr(ef, "latest_fit_dir", lambda: tmp_path)
    results = emit_all_horizons(out_dir=out_dir, fit_dir=tmp_path)
    assert len(results) == len(FORECAST_HORIZONS_MONTHS)
    assert all(r["gate_status"] == "pass" for r in results)
