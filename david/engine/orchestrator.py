"""End-to-end pipeline orchestrator.

`david run-all` calls into here; for now exposed as orchestrator.run_all() and
orchestrator.replay() for the CLI.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from ..ingest.adjudicator_queue import build_queue
from ..ingest.llm_coder import calibrate_coders
from ..ingest.normalize import normalize_raw
from ..ingest.sources import run_scrapers
from ..model.fit import run_fit
from ..simulator.forecast_sbc import run_forecast_sbc
from ..simulator.sbc import run_sbc
from ..validation.falsification import run_falsification
from .forecast import emit_forecasts
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

    forecasts = []
    for h in (6, 12):
        forecasts.append(emit_forecasts(horizon_months=h))

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

    TODO: read recorded versions (model SHA, code SHA, data cutoff) and rerun
    the pipeline with the same dependencies pinned.
    """
    raise NotImplementedError(
        "Replay must read run_id artifacts, restore evidence cutoff, "
        "and rerun pipeline with pinned model_version and code_version."
    )
