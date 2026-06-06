"""Human-loop budget enforcement.

Tracks adjudicator minutes spent per cycle and warns/blocks when over.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

from ..config import ADJUDICATOR_HOURS_PER_CYCLE, DATA_ROOT


BUDGET_LEDGER = DATA_ROOT / "human_loop_budget.json"


def _load() -> dict:
    if BUDGET_LEDGER.exists():
        return json.loads(BUDGET_LEDGER.read_text())
    return {"cycles": []}


def record_minutes(adjudicator_id: str, minutes: int, cycle_start: date) -> dict:
    led = _load()
    cycle = next((c for c in led["cycles"] if c["start"] == cycle_start.isoformat()), None)
    if cycle is None:
        cycle = {"start": cycle_start.isoformat(), "spent_minutes": {}}
        led["cycles"].append(cycle)
    cycle["spent_minutes"][adjudicator_id] = (
        cycle["spent_minutes"].get(adjudicator_id, 0) + minutes
    )
    BUDGET_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    BUDGET_LEDGER.write_text(json.dumps(led, indent=2))
    return cycle


def remaining_budget_minutes(cycle_start: date) -> int:
    led = _load()
    cycle = next((c for c in led["cycles"] if c["start"] == cycle_start.isoformat()), None)
    spent = 0 if cycle is None else sum(cycle["spent_minutes"].values())
    return int(ADJUDICATOR_HOURS_PER_CYCLE * 60) - spent
