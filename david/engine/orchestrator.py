"""End-to-end pipeline orchestrator.

`david run-all` calls into here; for now exposed as orchestrator.run_all() and
orchestrator.replay() for the CLI.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from ..config import FITS_DIR, FORECASTS_DIR, FORECAST_HORIZONS_MONTHS
from ..ingest.adjudicator_queue import build_queue
from ..ingest.llm_coder import calibrate_coders
from ..ingest.normalize import normalize_raw
from ..ingest.sources import run_scrapers
from ..model.fit import run_fit
from ..simulator.forecast_sbc import run_forecast_sbc
from ..simulator.sbc import run_sbc
from ..validation.falsification import run_falsification, _assemble_inputs
from ..simulator.adversarial_battery import run_battery
from .forecast import emit_forecasts, emit_all_horizons
from .router import apply_forecast_routing


def run_all(since: date | None = None) -> dict[str, Any]:
    """Run the full pipeline. Used by scripts/full_run.sh and CI smoke."""
    until = date.today()
    raw_paths = run_scrapers(since=since, until=until)
    normalized = normalize_raw(raw_paths)
    queue = build_queue(normalized)

    coder_result = calibrate_coders()
    if coder_result["gate_status"] != "pass":
        return {"status": "fail_at_coder_calibration", "detail": coder_result}

    fit_result = run_fit()
    if fit_result["gate_status"] != "pass":
        return {"status": "fail_at_fit", "detail": fit_result}

    sbc_result = run_sbc()
    if sbc_result["gate_status"] != "pass":
        return {"status": "fail_at_sbc", "detail": sbc_result}

    forecast_sbc_result = run_forecast_sbc()
    if forecast_sbc_result["gate_status"] != "pass":
        return {"status": "fail_at_forecast_sbc", "detail": forecast_sbc_result}

    falsify_result = run_falsification()
    if falsify_result["gate_status"] != "pass":
        return {"status": "fail_at_falsification", "detail": falsify_result}

    forecasts = emit_all_horizons()

    routing = apply_forecast_routing()
    return {
        "status": "pass",
        "n_ingested": len(normalized),
        "n_queued": len(queue),
        "fit_run_id": fit_result.get("run_id"),
        "forecasts": forecasts,
        "routing": routing,
    }


def replay(run_id: str) -> dict[str, Any]:
    """Reproduce a recorded run from artifacts.

    Reads fit_summary.json to verify the run exists, re-emits forecast cells
    for all horizons from the existing draws.parquet, then re-runs the
    falsification battery against those artifacts.

    This does NOT re-run the fit (which requires the original data cutoff and
    the same CmdStan binary); it only replays the post-fit stages. For a full
    reproducibility audit the caller should verify model_version matches the
    current code before invoking replay().
    """
    fit_dir = FITS_DIR / run_id
    if not fit_dir.exists():
        return {
            "gate_status": "fail",
            "reason": f"run_id '{run_id}' not found in {FITS_DIR}",
        }
    summary_path = fit_dir / "fit_summary.json"
    if not summary_path.exists():
        return {
            "gate_status": "fail",
            "reason": f"fit_summary.json missing in {fit_dir}",
        }
    fit_summary = json.loads(summary_path.read_text())

    # Re-emit forecast cells for all horizons from existing draws
    forecast_out = FORECASTS_DIR / run_id
    forecasts = emit_all_horizons(out_dir=forecast_out, fit_dir=fit_dir)

    # Re-run falsification battery against the (now re-emitted) artifacts
    forecast_dir = forecast_out if forecast_out.exists() else None
    falsify_inputs = _assemble_inputs(fit_dir, forecast_dir)
    battery_result = run_battery(inputs=falsify_inputs)

    return {
        "gate_status": battery_result["gate_status"],
        "run_id": run_id,
        "model_version": fit_summary.get("model_version"),
        "original_timestamp": fit_summary.get("timestamp"),
        "original_gate_status": fit_summary.get("gate_status"),
        "forecasts": forecasts,
        "falsification": battery_result,
    }
