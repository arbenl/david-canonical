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

from ..config import CONFIG_ROOT, SOURCE_INDEPENDENCE_LEDGER

SOURCE_INDEPENDENCE_FLOOR = 3.0


def load_ledger() -> dict:
    path = SOURCE_INDEPENDENCE_LEDGER
    if not path.exists():
        current_path = CONFIG_ROOT / "source_independence.json"
        path = current_path if current_path.exists() else path
    if not path.exists():
        return {"reviewed_on": None, "pairs": {}}
    return _normalise_ledger(json.loads(path.read_text()))


def _normalise_ledger(raw: dict) -> dict:
    """Normalize supported source-independence ledger schemas to pair entries."""
    if "pairs" in raw:
        return raw

    pairs: dict[str, dict] = {}
    for src in raw.get("sources", []):
        source_id = src.get("source_id")
        if not source_id:
            continue
        for rel in src.get("independence_scores", []):
            other = rel.get("vs") or rel.get("other_source_id")
            if not other:
                continue
            key = pair_key(source_id, other)
            score = float(rel.get("score", rel.get("independence_score", 0.5)))
            pairs[key] = {
                "independence_score": score,
                "rationale": rel.get("rationale", ""),
            }

    return {
        "reviewed_on": raw.get("reviewed_at") or raw.get("reviewed_on"),
        "pairs": pairs,
        "effective_independent_sources": raw.get("effective_independent_sources", {}),
    }


def save_ledger(ledger: dict) -> None:
    SOURCE_INDEPENDENCE_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    SOURCE_INDEPENDENCE_LEDGER.write_text(json.dumps(ledger, indent=2))


def pair_key(source_a: str, source_b: str) -> str:
    return "::".join(sorted([source_a, source_b]))


def missing_pair_keys(active_sources: list[str], ledger: dict | None = None) -> list[str]:
    """Return active-source pairs without reviewed structural evidence."""
    ledger = ledger or load_ledger()
    pairs = ledger.get("pairs", {})
    sources = sorted(set(active_sources))
    return [
        pair_key(a, b)
        for a, b in itertools.combinations(sources, 2)
        if pair_key(a, b) not in pairs
    ]


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
            key = pair_key(src, o)
            ind = pairs.get(key, {}).get("independence_score", 0.5)
            dep_scores.append(1.0 - ind)
        s += 1.0 - (sum(dep_scores) / len(dep_scores))
    return s


def is_above_floor(active_sources: list[str], floor: float = 3.0) -> bool:
    return s_eff(active_sources) >= floor


def structural_independence_summary(
    active_sources: list[str],
    ledger: dict | None = None,
    floor: float = SOURCE_INDEPENDENCE_FLOOR,
) -> dict:
    """Fail-closed structural source-independence summary for FG2/F11."""
    ledger = ledger or load_ledger()
    sources = sorted(set(active_sources))
    missing = missing_pair_keys(sources, ledger=ledger)
    value = float(s_eff(sources, ledger=ledger)) if sources else 0.0
    stale = _ledger_is_stale(ledger)
    status = "pass"
    reason = "structural_source_independence_above_floor"
    if len(sources) < 3:
        status = "fail"
        reason = "fewer_than_3_active_sources"
    elif missing:
        status = "fail"
        reason = "source_independence_pairs_missing"
    elif stale:
        status = "fail"
        reason = "source_independence_ledger_stale"
    elif value < floor:
        status = "fail"
        reason = f"structural_s_eff_{value:.3f}_below_floor_{floor:.3f}"
    return {
        "gate_status": status,
        "reason": reason,
        "active_sources": sources,
        "n_active_sources": len(sources),
        "structural_s_eff": value,
        "floor": float(floor),
        "missing_pair_keys": missing,
        "reviewed_on": ledger.get("reviewed_on") or ledger.get("reviewed_at"),
        "next_review_due": ledger.get("next_review_due"),
        "ledger_path": str(SOURCE_INDEPENDENCE_LEDGER),
    }


def _ledger_is_stale(ledger: dict) -> bool:
    due = ledger.get("next_review_due")
    if not due:
        return False
    try:
        return date.fromisoformat(str(due)) < date.today()
    except ValueError:
        return True
