"""B-6/B-7/B-8 tests.

B-6  — rank-histogram shape diagnostics (chi_squared, histogram_shape)
B-7  — sbc_block in falsification ledger
B-8  — SbcFail raised by emit_forecasts when ledger.sbc_block.sbc_passed=False
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from david.simulator.sbc import (
    histogram_diagnostics,
    run_sbc,
    _N_BINS,
)
from david.engine.forecast import SbcFail, _sbc_guard


# ─── B-6: histogram_diagnostics ──────────────────────────────────────────────


def _uniform_ranks(n: int, n_draws: int = 400, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_draws, size=n)


def _u_shaped_ranks(n: int, n_draws: int = 400, seed: int = 0) -> np.ndarray:
    """Piles at both ends — probability mass at 0 and n_draws."""
    rng = np.random.default_rng(seed)
    half = n // 2
    low  = rng.integers(0, n_draws // 10, size=half)
    high = rng.integers(n_draws * 9 // 10, n_draws, size=n - half)
    return np.concatenate([low, high])


def _center_clustered_ranks(n: int, n_draws: int = 400, seed: int = 0) -> np.ndarray:
    """Tight cluster around the midpoint — overcounting likelihood."""
    rng = np.random.default_rng(seed)
    mu  = n_draws // 2
    raw = rng.normal(mu, n_draws * 0.05, size=n)
    return np.clip(raw, 0, n_draws - 1).astype(int)


def _skew_ranks(n: int, n_draws: int = 400, seed: int = 0) -> np.ndarray:
    """All ranks piled in the lower quarter."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_draws // 4, size=n)


def test_histogram_diagnostics_keys():
    ranks = _uniform_ranks(50)
    d = histogram_diagnostics(ranks, n_draws_per_fit=400)
    assert "chi_squared" in d
    assert "chi_squared_p" in d
    assert "histogram_shape" in d
    assert "bin_counts" in d


def test_uniform_shape_classified():
    ranks = _uniform_ranks(200)
    d = histogram_diagnostics(ranks, n_draws_per_fit=400)
    assert d["histogram_shape"] == "uniform"


def test_u_shaped_classified():
    ranks = _u_shaped_ranks(300)
    d = histogram_diagnostics(ranks, n_draws_per_fit=400)
    assert d["histogram_shape"] == "U_shaped"


def test_center_clustered_classified():
    ranks = _center_clustered_ranks(300)
    d = histogram_diagnostics(ranks, n_draws_per_fit=400)
    assert d["histogram_shape"] == "center_clustered"


def test_skew_classified():
    ranks = _skew_ranks(300)
    d = histogram_diagnostics(ranks, n_draws_per_fit=400)
    assert d["histogram_shape"] == "skew"


def test_chi_squared_nonnegative():
    ranks = _uniform_ranks(100)
    d = histogram_diagnostics(ranks, n_draws_per_fit=400)
    assert d["chi_squared"] >= 0.0


def test_chi_squared_p_in_unit_interval():
    ranks = _uniform_ranks(100)
    d = histogram_diagnostics(ranks, n_draws_per_fit=400)
    assert 0.0 <= d["chi_squared_p"] <= 1.0


def test_bin_counts_length():
    ranks = _uniform_ranks(100)
    d = histogram_diagnostics(ranks, n_draws_per_fit=400)
    assert len(d["bin_counts"]) == _N_BINS


def test_insufficient_data_flag():
    ranks = np.array([1, 2])  # fewer than _N_BINS
    d = histogram_diagnostics(ranks, n_draws_per_fit=400)
    assert d["histogram_shape"] == "insufficient_data"
    assert d["bin_counts"] == []


# ─── B-6: run_sbc includes shape diagnostics ─────────────────────────────────


@pytest.fixture(scope="module")
def small_sbc_result(tmp_path_factory):
    out = tmp_path_factory.mktemp("b6_sbc")
    from david.simulator.synthetic_world import HyperPrior
    prior = HyperPrior(R=2, T=8, L=3, K=3, S=2, M=2, H=0)
    return run_sbc(n_worlds=3, prior=prior, out_dir=out, base_seed=42)


def test_sbc_per_param_has_chi_squared(small_sbc_result):
    for name, diag in small_sbc_result["per_parameter_ks"].items():
        assert "chi_squared" in diag, f"{name} missing chi_squared"


def test_sbc_per_param_has_histogram_shape(small_sbc_result):
    for name, diag in small_sbc_result["per_parameter_ks"].items():
        assert "histogram_shape" in diag, f"{name} missing histogram_shape"
        assert diag["histogram_shape"] in (
            "uniform", "U_shaped", "center_clustered", "skew", "insufficient_data"
        ), f"{name} has unexpected shape {diag['histogram_shape']!r}"


def test_sbc_seeds_recorded(small_sbc_result):
    assert small_sbc_result["base_seed"] == 42
    assert small_sbc_result["seeds"] == [42, 43, 44]


def test_sbc_summary_written_with_shape(small_sbc_result, tmp_path):
    p = Path(small_sbc_result["summary_path"])
    assert p.exists()
    data = json.loads(p.read_text())
    first_param = next(iter(data["per_parameter_ks"].values()))
    assert "histogram_shape" in first_param
    assert "chi_squared" in first_param


# ─── B-7: sbc_block in falsification ledger ──────────────────────────────────


def _write_sbc_summary(sbc_dir: Path, n_worlds: int = 5, passed: bool = True) -> None:
    sbc_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(n_worlds))
    pvalue = 0.5 if passed else 0.001
    gate = "pass" if passed else "fail"
    failed = [] if passed else ["theta[0]"]
    summary = {
        "model_version": "test",
        "n_worlds": n_worlds,
        "n_draws_per_fit": 400,
        "ks_alpha": 0.05,
        "base_seed": 0,
        "seeds": seeds,
        "per_parameter_ks": {
            "theta[0]": {
                "statistic": 0.05, "pvalue": pvalue, "n": n_worlds,
                "chi_squared": 2.0, "chi_squared_p": 0.4,
                "histogram_shape": "uniform",
                "bin_counts": [5] * 10,
            }
        },
        "effective_alpha": 0.05,
        "bonferroni_applied": False,
        "failed_parameters": failed,
        "gate_status": gate,
        "reason": "all_parameters_uniform" if passed else "ks_fail_on_1_parameters",
        "timestamp": "2026-06-12T00:00:00Z",
    }
    (sbc_dir / "sbc_summary.json").write_text(json.dumps(summary))


def test_sbc_block_present_when_summary_exists(tmp_path):
    from david.validation.falsification import _build_sbc_block

    fits_dir = tmp_path / "fits"
    _write_sbc_summary(fits_dir / "sbc", n_worlds=5, passed=True)

    with patch("david.validation.falsification.FITS_DIR", fits_dir):
        block = _build_sbc_block()

    assert block["sbc_passed"] is True
    assert block["n_worlds"] == 5
    assert len(block["seeds"]) == 5
    assert block["failed_params"] == []
    assert isinstance(block["worst_p_value"], float)


def test_sbc_block_failed_recorded(tmp_path):
    from david.validation.falsification import _build_sbc_block

    fits_dir = tmp_path / "fits"
    _write_sbc_summary(fits_dir / "sbc", n_worlds=5, passed=False)

    with patch("david.validation.falsification.FITS_DIR", fits_dir):
        block = _build_sbc_block()

    assert block["sbc_passed"] is False
    assert block["failed_params"] == ["theta[0]"]


def test_sbc_block_missing_summary(tmp_path):
    from david.validation.falsification import _build_sbc_block

    fits_dir = tmp_path / "fits"  # no sbc subdir written
    with patch("david.validation.falsification.FITS_DIR", fits_dir):
        block = _build_sbc_block()

    assert block["sbc_passed"] is None
    assert "note" in block


def test_sbc_block_derives_missing_seeds_from_base_seed(tmp_path):
    from david.validation.falsification import _build_sbc_block

    fits_dir = tmp_path / "fits"
    _write_sbc_summary(fits_dir / "sbc", n_worlds=4, passed=True)
    summary_path = fits_dir / "sbc" / "sbc_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["base_seed"] = 42
    summary.pop("seeds")
    summary_path.write_text(json.dumps(summary))

    with patch("david.validation.falsification.FITS_DIR", fits_dir):
        block = _build_sbc_block()

    assert block["seeds"] == [42, 43, 44, 45]


def _make_falsification_ledger(
    forecast_dir: Path,
    fits_dir: Path,
    sbc_passed: bool = True,
) -> None:
    """Write a minimal falsification_ledger.json with an sbc_block."""
    _write_sbc_summary(fits_dir / "sbc", n_worlds=3, passed=sbc_passed)
    from david.validation.falsification import _build_sbc_block
    with patch("david.validation.falsification.FITS_DIR", fits_dir):
        sbc_block = _build_sbc_block()
    forecast_dir.mkdir(parents=True, exist_ok=True)
    ledger = {
        "battery_result": {"gate_status": "pass", "failed_tests": []},
        "sbc_block": sbc_block,
        "timestamp": "2026-06-12T00:00:00Z",
    }
    (forecast_dir / "falsification_ledger.json").write_text(json.dumps(ledger))


def test_ledger_sbc_block_keys(tmp_path):
    fits_dir = tmp_path / "fits"
    forecast_dir = tmp_path / "forecasts" / "run1"
    _make_falsification_ledger(forecast_dir, fits_dir, sbc_passed=True)
    ledger = json.loads((forecast_dir / "falsification_ledger.json").read_text())
    sb = ledger["sbc_block"]
    assert "sbc_passed" in sb
    assert "n_worlds" in sb
    assert "seeds" in sb
    assert "worst_p_value" in sb
    assert "failed_params" in sb


# ─── B-8: _sbc_guard / SbcFail ───────────────────────────────────────────────


def test_sbc_guard_passes_when_no_ledger(tmp_path):
    # No ledger file — should be a no-op (not a block)
    _sbc_guard(tmp_path)  # must not raise


def test_sbc_guard_passes_when_ledger_has_no_sbc_block(tmp_path):
    ledger = {"battery_result": {"gate_status": "pass", "failed_tests": []}}
    (tmp_path / "falsification_ledger.json").write_text(json.dumps(ledger))
    _sbc_guard(tmp_path)  # must not raise


def test_sbc_guard_passes_when_sbc_passed_true(tmp_path):
    ledger = {
        "sbc_block": {
            "sbc_passed": True,
            "failed_params": [],
            "worst_p_value": 0.42,
        }
    }
    (tmp_path / "falsification_ledger.json").write_text(json.dumps(ledger))
    _sbc_guard(tmp_path)  # must not raise


def test_sbc_guard_raises_when_sbc_passed_false(tmp_path):
    ledger = {
        "sbc_block": {
            "sbc_passed": False,
            "failed_params": ["theta[0]", "Pi[1]"],
            "worst_p_value": 0.001,
        }
    }
    (tmp_path / "falsification_ledger.json").write_text(json.dumps(ledger))
    with pytest.raises(SbcFail) as exc_info:
        _sbc_guard(tmp_path)
    assert "theta[0]" in exc_info.value.failed_params
    assert exc_info.value.worst_p == pytest.approx(0.001)


def test_sbc_guard_raises_when_fit_dir_ledger_failed(tmp_path):
    forecast_dir = tmp_path / "forecasts" / "run1"
    fit_dir = tmp_path / "fits" / "run1"
    fit_dir.mkdir(parents=True)
    ledger = {
        "sbc_block": {
            "sbc_passed": False,
            "failed_params": ["theta[0]"],
            "worst_p_value": 0.001,
        }
    }
    (fit_dir / "falsification_ledger.json").write_text(json.dumps(ledger))

    with pytest.raises(SbcFail):
        _sbc_guard(forecast_dir, fit_dir)


def test_sbc_fail_message_is_typed(tmp_path):
    """SbcFail str representation includes SBC_FAIL_CLOSED keyword."""
    ledger = {
        "sbc_block": {
            "sbc_passed": False,
            "failed_params": ["mu"],
            "worst_p_value": 0.002,
        }
    }
    (tmp_path / "falsification_ledger.json").write_text(json.dumps(ledger))
    with pytest.raises(SbcFail) as exc_info:
        _sbc_guard(tmp_path)
    assert "SBC_FAIL_CLOSED" in str(exc_info.value)


def test_sbc_guard_passes_when_sbc_passed_none(tmp_path):
    """sbc_passed=None means SBC hasn't run yet — not a block."""
    ledger = {
        "sbc_block": {
            "sbc_passed": None,
            "failed_params": [],
            "worst_p_value": None,
            "note": "sbc_summary.json not found",
        }
    }
    (tmp_path / "falsification_ledger.json").write_text(json.dumps(ledger))
    _sbc_guard(tmp_path)  # must not raise


@pytest.mark.parametrize("ledger_location", ["forecast_dir", "fit_dir"])
def test_emit_forecasts_raises_sbc_fail(tmp_path, ledger_location):
    """emit_forecasts checks both possible falsification ledger placements."""
    from david.engine.forecast import emit_forecasts

    # Build a falsification ledger with a failed SBC block. In weekly order,
    # falsify runs before forecast and writes this under fit_dir.
    forecast_run_dir = tmp_path / "forecasts" / "run_test"
    ledger = {
        "sbc_block": {
            "sbc_passed": False,
            "failed_params": ["theta[0]"],
            "worst_p_value": 0.001,
        }
    }
    forecast_run_dir.mkdir(parents=True, exist_ok=True)

    # Construct a minimal passing fit_dir with a fit_summary.json
    fit_dir = tmp_path / "fits" / "run_test"
    fit_dir.mkdir(parents=True, exist_ok=True)
    (fit_dir / "fit_summary.json").write_text(json.dumps({
        "run_id": "run_test",
        "gate_status": "pass",
        "timestamp": "2026-06-12T00:00:00Z",
    }))
    ledger_dir = forecast_run_dir if ledger_location == "forecast_dir" else fit_dir
    (ledger_dir / "falsification_ledger.json").write_text(json.dumps(ledger))

    with (
        patch("david.engine.forecast.FORECASTS_DIR", tmp_path / "forecasts"),
        patch("david.engine.forecast.latest_fit_dir", return_value=fit_dir),
    ):
        with pytest.raises(SbcFail):
            emit_forecasts(horizon_months=6, fit_dir=fit_dir)


def test_emit_forecasts_raises_sbc_fail_from_fit_dir_ledger(tmp_path):
    """emit_forecasts checks falsify-before-forecast ledger placement."""
    from david.engine.forecast import emit_forecasts

    fit_dir = tmp_path / "fits" / "run_test"
    fit_dir.mkdir(parents=True, exist_ok=True)
    (fit_dir / "fit_summary.json").write_text(json.dumps({
        "run_id": "run_test",
        "gate_status": "pass",
        "timestamp": "2026-06-12T00:00:00Z",
    }))
    (fit_dir / "falsification_ledger.json").write_text(json.dumps({
        "sbc_block": {
            "sbc_passed": False,
            "failed_params": ["theta[0]"],
            "worst_p_value": 0.001,
        }
    }))

    with (
        patch("david.engine.forecast.FORECASTS_DIR", tmp_path / "forecasts"),
        patch("david.engine.forecast.latest_fit_dir", return_value=fit_dir),
    ):
        with pytest.raises(SbcFail):
            emit_forecasts(horizon_months=6, fit_dir=fit_dir)
