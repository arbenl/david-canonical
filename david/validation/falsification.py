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

from ..config import FITS_DIR, FORECASTS_DIR
from ..simulator.adversarial_battery import run_battery


def run_falsification() -> dict[str, Any]:
    """Top-level: collect inputs, run battery, write ledger."""
    fit_dir = _latest_dir(FITS_DIR)
    forecast_dir = _latest_dir(FORECASTS_DIR)
    if fit_dir is None:
        return {"gate_status": "fail", "reason": "no_fit_to_falsify"}

    inputs = _assemble_inputs(fit_dir, forecast_dir)
    battery_result = run_battery(inputs=inputs)

    ledger_path = (forecast_dir or fit_dir) / "falsification_ledger.json"
    ledger = {
        "fit_dir": str(fit_dir),
        "forecast_dir": str(forecast_dir) if forecast_dir else None,
        "battery_result": battery_result,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    ledger_path.write_text(json.dumps(ledger, indent=2))
    return {
        "gate_status": battery_result["gate_status"],
        "failed_tests": battery_result["failed_tests"],
        "ledger_path": str(ledger_path),
    }


def _latest_dir(root: Path) -> Path | None:
    if not root.exists():
        return None
    candidates = sorted(p for p in root.iterdir() if p.is_dir())
    return candidates[-1] if candidates else None


def _assemble_inputs(fit_dir: Path, forecast_dir: Path | None) -> dict[str, dict]:
    """Read artifacts and build the inputs dict for adversarial_battery.run_battery().

    TODO: populate each F-test's kwargs from artifacts.
    The skeleton returns an empty dict so all tests are marked "skip" until
    the user wires the readers, providing immediate visible signal of what
    is missing.
    """
    return {}
