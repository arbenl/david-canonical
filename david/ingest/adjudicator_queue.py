"""Adjudicator queue — auto-adjudication + human escalation.

Pipeline:
    code_evidence()         →  coder_labels table (Y[e, m, k])
    auto_adjudicate()       →  unanimous items: adjudicated=TRUE  (automatic)
                               split items:     adjudicator_queue.json (human)
    david fit               →  reads adjudicated=TRUE rows only

Human reviewers are shown only the queue items, ranked by Expected Information
Gain (EIG), capped at ADJUDICATOR_HOURS_PER_CYCLE hours per cycle.

Disagreement is measured as minority_share = min(ones, zeros) / n_coders:
    0.0  — unanimous (all agree)        → auto-adjudicated
    0.30 — threshold (default)          → borderline, auto-adjudicated
    0.31 — above threshold              → queued for human
    0.50 — perfect split                → definitely queued

The threshold is pre-registered in config.py and must not be changed after
the first real-data run without a new pre-registration version.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

from ..config import (
    ADJUDICATOR_DISAGREEMENT_THRESHOLD,
    ADJUDICATOR_HOURS_PER_CYCLE,
    DATA_ROOT,
    GOLD_STANDARD_SAMPLE_RATE,
)


QUEUE_PATH = DATA_ROOT / "adjudicator_queue.json"


@dataclass
class QueueItem:
    evidence_id: str
    reason: str               # disagree | gold_expand | active_sample
    eig_score: float          # higher = more identifying
    estimated_minutes: int    # adjudicator effort estimate
    worst_tactic: str = ""    # tactic class with highest disagreement
    minority_share: float = 0.0


# ─── disagreement metric ─────────────────────────────────────────────────────

def minority_share(labels: list[int]) -> float:
    """Fraction of coders in the minority.

    0.0 = unanimous (auto-adjudicated).
    0.5 = perfect split (maximum disagreement).

    Uses minority_share = min(ones, zeros) / n so that:
        [1, 1, 1]    → 0.0   unanimous
        [1, 1, 0]    → 0.33  slight disagreement
        [1, 0, 1, 0] → 0.5   perfect split
    """
    n = len(labels)
    if n == 0:
        return 0.0
    ones  = sum(1 for v in labels if v == 1)
    zeros = n - ones
    return min(ones, zeros) / n


# Keep the old function name for backward compatibility with existing code.
def coder_disagreement(Y_by_coder: dict[str, int]) -> float:
    """Legacy alias. Prefer minority_share(list(Y_by_coder.values()))."""
    return minority_share(list(Y_by_coder.values()))


def expected_information_gain(
    evidence_id: str,
    disagreement: float,
    stratum_id_distance: float = 0.10,
    stratum_S_eff: int = 3,
) -> float:
    """EIG approximation — higher = more valuable to adjudicate.

    Higher when:
      - coders disagree more (disagreement → minority_share)
      - stratum identification distance is low (every gold helps)
      - stratum S_eff is below the A' floor of 3 (active sampling)
    """
    disagree_term = disagreement                           # 0→low, 0.5→high
    id_term       = math.exp(-stratum_id_distance / 0.1)
    rep_term      = max(0, 3 - stratum_S_eff)
    return disagree_term + 0.5 * id_term + 0.75 * rep_term


# ─── auto-adjudication (core) ────────────────────────────────────────────────

def auto_adjudicate(
    queue_path: Path | None = None,
    threshold: float | None = None,
) -> dict:
    """Read coder_labels from DB, auto-adjudicate agreements, queue the rest.

    Algorithm
    ---------
    For each evidence_item not yet adjudicated:
      For each tactic_k:
        Compute minority_share across all coders.
      If max(minority_share over all tactics) <= threshold:
        → mark adjudicated=TRUE in DB  (unanimous or near-unanimous)
      Else:
        → add to adjudicator_queue.json  (human reviews this item)

    Queue items are ranked by EIG and capped at ADJUDICATOR_HOURS_PER_CYCLE.

    Returns
    -------
    {
        gate_status: "pass",
        auto_adjudicated: N,   # rows set adjudicated=TRUE
        queued_for_human: K,   # items in queue
        skipped_no_labels: J,  # evidence_items with 0 coder labels yet
        queue_path: str,
    }
    """
    queue_path = queue_path or QUEUE_PATH
    threshold  = threshold if threshold is not None else ADJUDICATOR_DISAGREEMENT_THRESHOLD

    # Pull unreviewed labels from DB
    try:
        from ..db.repositories import get_unreviewed_labels, mark_adjudicated
        label_rows = get_unreviewed_labels()
    except Exception as exc:
        return {
            "gate_status": "fail",
            "reason": "db_unavailable",
            "detail": str(exc),
        }

    if not label_rows:
        _write_queue([], queue_path, threshold)
        return {
            "gate_status": "pass",
            "auto_adjudicated": 0,
            "queued_for_human": 0,
            "skipped_no_labels": 0,
            "queue_path": str(queue_path),
        }

    # Group: evidence_id → tactic_k → [labels]
    tactic_labels: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    stratum_map: dict[str, str] = {}
    for row in label_rows:
        eid     = row["evidence_id"]
        tactic  = row["tactic_k"]
        label   = int(row["label"])
        stratum = row.get("stratum_id", "")
        tactic_labels[eid][tactic].append(label)
        stratum_map[eid] = stratum

    auto_ids:  list[str]     = []
    queue_items: list[QueueItem] = []

    for eid, tactics in tactic_labels.items():
        # Compute worst-case minority_share across all coded tactics
        worst_share  = 0.0
        worst_tactic = ""
        for tactic, labels in tactics.items():
            ms = minority_share(labels)
            if ms > worst_share:
                worst_share  = ms
                worst_tactic = tactic

        if worst_share <= threshold:
            auto_ids.append(eid)
        else:
            eig = expected_information_gain(eid, worst_share)
            reason = "disagree"
            if _gold_expand_sampled(eid):
                reason = "gold_expand"
            queue_items.append(QueueItem(
                evidence_id=eid,
                reason=reason,
                eig_score=eig,
                estimated_minutes=4,
                worst_tactic=worst_tactic,
                minority_share=worst_share,
            ))

    # Persist auto-adjudicated items
    n_auto = 0
    if auto_ids:
        try:
            n_auto = mark_adjudicated(auto_ids)
        except Exception as exc:
            return {
                "gate_status": "fail",
                "reason": "db_mark_failed",
                "detail": str(exc),
            }

    # Write queue (ranked by EIG, capped by budget)
    queue_items.sort(key=lambda x: -x.eig_score)
    capped = _cap_by_budget(queue_items, ADJUDICATOR_HOURS_PER_CYCLE)
    _write_queue(capped, queue_path, threshold)

    return {
        "gate_status": "pass",
        "auto_adjudicated": n_auto,
        "queued_for_human": len(capped),
        "skipped_no_labels": 0,
        "queue_path": str(queue_path),
        "threshold": threshold,
    }


# ─── legacy build_queue (kept for CLI backward-compat) ───────────────────────

def build_queue(items: Iterable, queue_path: Path | None = None) -> list[QueueItem]:
    """Build queue from CanonicalEvidence objects (legacy path).

    Prefer auto_adjudicate() which reads directly from DB.
    This function is kept for backward compatibility when the DB is unavailable.
    """
    queue_path = queue_path or QUEUE_PATH
    candidates: list[QueueItem] = []
    for it in items:
        Y_by_coder           = getattr(it, "Y_by_coder", {})
        stratum_id_distance  = getattr(it, "stratum_id_distance", 0.10)
        stratum_S_eff        = getattr(it, "stratum_S_eff", 3)

        ms = minority_share(list(Y_by_coder.values()))
        if ms <= ADJUDICATOR_DISAGREEMENT_THRESHOLD:
            if not _gold_expand_sampled(it.evidence_id):
                continue
            reason = "gold_expand"
        else:
            reason = "disagree"

        eig = expected_information_gain(
            it.evidence_id, ms, stratum_id_distance, stratum_S_eff,
        )
        candidates.append(QueueItem(
            evidence_id=it.evidence_id,
            reason=reason,
            eig_score=eig,
            estimated_minutes=4,
            minority_share=ms,
        ))

    candidates.sort(key=lambda x: -x.eig_score)
    capped = _cap_by_budget(candidates, ADJUDICATOR_HOURS_PER_CYCLE)
    _write_queue(capped, queue_path, ADJUDICATOR_DISAGREEMENT_THRESHOLD)
    return capped


# ─── helpers ─────────────────────────────────────────────────────────────────

def _write_queue(
    items: list[QueueItem],
    queue_path: Path,
    threshold: float,
) -> None:
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(json.dumps(
        {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "budget_hours": ADJUDICATOR_HOURS_PER_CYCLE,
            "disagreement_threshold": threshold,
            "n_in_queue": len(items),
            "items": [asdict(c) for c in items],
        },
        indent=2,
    ))


def _cap_by_budget(items: list[QueueItem], budget_hours: float) -> list[QueueItem]:
    budget_minutes = budget_hours * 60
    out, used = [], 0.0
    for it in items:
        if used + it.estimated_minutes > budget_minutes:
            break
        out.append(it)
        used += it.estimated_minutes
    return out


def _gold_expand_sampled(evidence_id: str) -> bool:
    """Deterministic sampling at GOLD_STANDARD_SAMPLE_RATE based on evidence_id."""
    import hashlib
    h = int(hashlib.sha256(evidence_id.encode()).hexdigest(), 16) / 2**256
    return h < GOLD_STANDARD_SAMPLE_RATE
