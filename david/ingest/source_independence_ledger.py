"""Source structural-independence ledger.

The ledger records pairwise conditional-independence priors between source
families. Theorem A' requires S_eff >= 3 conditionally independent sources;
this module computes S_eff per stratum from the ledger and validates that
the ingest pipeline meets the floor.
"""

from __future__ import annotations

import itertools
import json
from datetime import date
from pathlib import Path

from ..config import SOURCE_INDEPENDENCE_LEDGER


def load_ledger() -> dict:
    if not SOURCE_INDEPENDENCE_LEDGER.exists():
        return {"reviewed_on": None, "pairs": {}}
    return json.loads(SOURCE_INDEPENDENCE_LEDGER.read_text())


def save_ledger(ledger: dict) -> None:
    SOURCE_INDEPENDENCE_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    SOURCE_INDEPENDENCE_LEDGER.write_text(json.dumps(ledger, indent=2))


def s_eff(active_sources: list[str], ledger: dict | None = None) -> float:
    """Effective number of independent sources.

    Naive approximation: sum of (1 - mean dependence) over each source
    relative to all other sources. A source paired with completely
    dependent partners contributes ~0; a source fully independent
    contributes ~1.
    """
    ledger = ledger or load_ledger()
    pairs = ledger.get("pairs", {})
    s = 0.0
    for src in active_sources:
        others = [o for o in active_sources if o != src]
        if not others:
            s += 1.0
            continue
        dep_scores = []
        for o in others:
            key = "::".join(sorted([src, o]))
            ind = pairs.get(key, {}).get("independence_score", 0.5)
            dep_scores.append(1.0 - ind)
        s += 1.0 - (sum(dep_scores) / len(dep_scores))
    return s


def is_above_floor(active_sources: list[str], floor: float = 3.0) -> bool:
    return s_eff(active_sources) >= floor
