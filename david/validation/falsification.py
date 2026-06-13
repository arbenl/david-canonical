"""Falsification harness: assembles F1..F15 inputs and runs the battery.

Reads:
  data/fits/{latest}/draws.parquet            posterior draws
  data/fits/{latest}/fit_summary.json         R-hat / ESS
  data/fits/forecast_sbc/forecast_sbc_summary.json
  data/forecasts/{latest}/cells_h*.json       per-cell predictions

Writes:
  data/forecasts/{latest}/falsification_ledger.json

Each F-test is implemented in simulator.adversarial_battery; this module is
the orchestrator that gathers inputs and dispatches.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..config import FITS_DIR, FORECASTS_DIR
from ..simulator.adversarial_battery import run_battery


def run_falsification() -> dict[str, Any]:
    """Top-level: collect inputs, run battery, write ledger.

    B-7: the ledger now contains an ``sbc_block`` assembled from the most
    recent measurement-SBC summary. This block is authoritative for the
    B-8 fail-closed guard in emit_forecasts().
    """
    fit_dir = _latest_fit_dir(FITS_DIR)
    forecast_dir = _latest_forecast_dir(FORECASTS_DIR, fit_dir)
    if fit_dir is None:
        return {"gate_status": "fail", "reason": "no_fit_to_falsify"}

    inputs = _assemble_inputs(fit_dir, forecast_dir)
    battery_result = run_battery(inputs=inputs)
    sbc_block = _build_sbc_block()

    ledger_path = (forecast_dir or fit_dir) / "falsification_ledger.json"
    ledger = {
        "fit_dir": str(fit_dir),
        "forecast_dir": str(forecast_dir) if forecast_dir else None,
        "battery_result": battery_result,
        "sbc_block": sbc_block,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    ledger_path.write_text(json.dumps(ledger, indent=2))
    return {
        "gate_status": battery_result["gate_status"],
        "failed_tests": battery_result["failed_tests"],
        "ledger_path": str(ledger_path),
        "sbc_passed": sbc_block["sbc_passed"],
    }


def _build_sbc_block() -> dict[str, Any]:
    """B-7: Read measurement-SBC summary and produce the typed sbc_block.

    Returns a dict with:
      sbc_passed       bool | None
      n_worlds         int | None
      seeds            list[int]   (recorded seeds, or base_seed + world_index)
      worst_p_value    float | None
      failed_params    list[str]
      sbc_summary_path str | None
    """
    sbc_path = FITS_DIR / "sbc" / "sbc_summary.json"
    if not sbc_path.exists():
        return {
            "sbc_passed": None,
            "n_worlds": None,
            "seeds": [],
            "worst_p_value": None,
            "failed_params": [],
            "sbc_summary_path": None,
            "note": "sbc_summary.json not found — run `david sbc` first",
        }

    try:
        sbc = json.loads(sbc_path.read_text())
    except Exception as exc:
        return {
            "sbc_passed": None,
            "n_worlds": None,
            "seeds": [],
            "worst_p_value": None,
            "failed_params": [],
            "sbc_summary_path": str(sbc_path),
            "note": f"failed to parse sbc_summary.json: {exc}",
        }

    per_param = sbc.get("per_parameter_ks", {})
    p_values = [
        v["pvalue"]
        for v in per_param.values()
        if isinstance(v.get("pvalue"), float) and not (v["pvalue"] != v["pvalue"])
    ]
    worst_p = min(p_values) if p_values else None
    failed  = sbc.get("failed_parameters", [])
    passed  = sbc.get("gate_status") == "pass"

    n_worlds = sbc.get("n_worlds")
    base_seed = sbc.get("base_seed", 0)
    if "seeds" in sbc:
        seeds = sbc["seeds"]
    elif isinstance(n_worlds, int) and isinstance(base_seed, int):
        seeds = list(range(base_seed, base_seed + n_worlds))
    else:
        seeds = []

    return {
        "sbc_passed": passed,
        "n_worlds": n_worlds,
        "seeds": seeds,
        "worst_p_value": worst_p,
        "failed_params": failed,
        "sbc_summary_path": str(sbc_path),
    }


def _latest_fit_dir(root: Path) -> Path | None:
    """Return the most recent *passing* fit dir, sorted by fit_summary timestamp.

    Mirrors forecast.py::latest_fit_dir — filters to dirs with fit_summary.json
    and gate_status == 'pass', then picks the latest by recorded timestamp.
    Falls back to alphabetical sort if timestamps are absent.
    """
    if not root.exists():
        return None

    def _ts(p: Path) -> str:
        try:
            return json.loads((p / "fit_summary.json").read_text()).get("timestamp", "")
        except Exception:
            return ""

    candidates = [
        p for p in root.iterdir()
        if p.is_dir()
        and (p / "fit_summary.json").exists()
        and json.loads((p / "fit_summary.json").read_text()).get("gate_status") == "pass"
    ]
    if not candidates:
        return None
    return max(candidates, key=_ts)


def _latest_forecast_dir(root: Path, fit_dir: Path | None) -> Path | None:
    """Return the forecast dir for the most recent passing fit.

    Prefers the dir whose name matches fit_dir.name (same run_id).  Falls back
    to the alphabetically latest dir that contains cells_h*.json files.
    """
    if not root.exists():
        return None
    # Best: exact match on run_id
    if fit_dir is not None:
        exact = root / fit_dir.name
        if exact.is_dir() and list(exact.glob("cells_h*.json")):
            return exact
    # Fallback: any dir containing forecast cells, sorted by fit_summary timestamp
    candidates = [
        p for p in root.iterdir()
        if p.is_dir() and list(p.glob("cells_h*.json"))
    ]
    if not candidates:
        return None

    def _ts(p: Path) -> str:
        fit_summary = FITS_DIR / p.name / "fit_summary.json"
        try:
            return json.loads(fit_summary.read_text()).get("timestamp", "")
        except Exception:
            return p.name

    return max(candidates, key=_ts)


def _assemble_inputs(fit_dir: Path, forecast_dir: Path | None) -> dict[str, dict]:
    """Read artifacts and build the inputs dict for adversarial_battery.run_battery().

    Reads from:
      fit_dir/fit_summary.json         — theorem results, R-hat, divergences
      fit_dir/draws.parquet            — full posterior draws
      forecast_dir/forecast_sbc*.json  — forecast SBC coverage (F14)

    Returns a dict keyed by test id. Missing artifacts → empty dict for that
    test → run_battery marks the test "skip" rather than "fail".
    """
    inputs: dict[str, dict] = {}

    # ── Load fit summary ──────────────────────────────────────────────────────
    fit_summary_path = fit_dir / "fit_summary.json"
    fit_summary: dict = {}
    if fit_summary_path.exists():
        fit_summary = json.loads(fit_summary_path.read_text())

    # ── Load posterior draws ──────────────────────────────────────────────────
    draws_path = fit_dir / "draws.parquet"
    draws_df = None
    if draws_path.exists():
        try:
            import pandas as pd
            draws_df = pd.read_parquet(draws_path)
        except Exception:
            draws_df = None

    # ── F12: identification distance from Theorem A' ──────────────────────────
    A_prime = fit_summary.get("theorems", {}).get("A_prime", {})
    if "median_d_theta" in A_prime:
        inputs["F12"] = {"median_d": A_prime["median_d_theta"]}

    # ── F14: forecast SBC coverage ────────────────────────────────────────────
    sbc_summary_path = FITS_DIR / "forecast_sbc" / "forecast_sbc_summary.json"
    if not sbc_summary_path.exists():
        sbc_summary_path = FITS_DIR / "sbc" / "forecast_sbc_summary.json"
    if sbc_summary_path.exists():
        sbc_summary = json.loads(sbc_summary_path.read_text())
        inputs["F14"] = {"forecast_sbc_summary": sbc_summary}

    # ── F1 (prior predictive) — already computed inside run_fit; pull from summary ──
    f1 = fit_summary.get("gates", {}).get("F1", {})
    if f1.get("gate_status") in ("pass", "fail"):
        # Re-supply as pre-computed arrays so adversarial_battery.F1 receives
        # a single-element array (median already computed); band from summary.
        med = f1.get("prior_predictive_Y_rate_median", float("nan"))
        b5  = f1.get("historical_band_5th", 0.0)
        b95 = f1.get("historical_band_95th", 1.0)
        inputs["F1"] = {
            "prior_predictive_Y_rate": np.array([med]),
            "historical_Y_rate_5th": b5,
            "historical_Y_rate_95th": b95,
        }

    # ── F13: horizon respect — need forecast cells and h_star from D' ─────────
    D_prime = fit_summary.get("theorems", {}).get("D_prime", {})
    if forecast_dir is not None and "h_star_months" in D_prime:
        h_star = D_prime["h_star_months"]
        # Try to load first forecast cell file for h > h_star
        beyond_paths = sorted(
            forecast_dir.glob(f"cells_h*.json")
        ) if forecast_dir else []
        beyond_p, marginal_p, mask = [], [], []
        for p in beyond_paths:
            try:
                h_val = int(p.stem.split("_h")[-1])
                cells = json.loads(p.read_text())
                for c in cells:
                    beyond_p.append(float(c.get("p_active", float("nan"))))
                    marginal_p.append(float(c.get("p_marginal", float("nan"))))
                    mask.append(h_val > h_star)
            except Exception:
                continue
        if beyond_p:
            inputs["F13"] = {
                "forecast_p": np.array(beyond_p),
                "marginal_p": np.array(marginal_p),
                "beyond_h_star_mask": np.array(mask, dtype=bool),
            }

    return inputs
