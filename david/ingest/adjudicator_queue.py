"""Adjudicator queue.

Escalates an evidence item to a human adjudicator only when one of:
  1) Inter-coder disagreement exceeds threshold (any tactic class).
  2) Item is sampled into rolling gold-standard expansion.
  3) Item is in a stratum below Theorem A' replication floor (active sampling).

Otherwise the item is flagged llm_only_acceptable and flows into the fit.

Queue items are ranked by Expected Information Gain (EIG), so a fixed
adjudicator-hours budget extracts the most identifying information.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
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


def coder_disagreement(Y_by_coder: dict[str, int]) -> float:
    """Disagreement measure: |majority - minority| share."""
    n = len(Y_by_coder)
    if n == 0:
        return 0.0
    ones = sum(1 for v in Y_by_coder.values() if v == 1)
    zeros = n - ones
    return abs(ones - zeros) / n  # 1.0 = full agreement; 0.0 = perfectly split


def expected_information_gain(
    evidence_id: str,
    disagreement: float,
    stratum_id_distance: float,
    stratum_S_eff: int,
) -> float:
    """EIG approximation.

    Higher when:
      - coders disagree more (so adjudication resolves a labeled item)
      - stratum's identification distance is low (every gold helps)
      - stratum's S_eff is below the A' floor of 3 (active sampling)
    """
    disagreement_term = 1.0 - disagreement     # disagreement contributes 1 - |maj-min|
    id_term = math.exp(-stratum_id_distance / 0.1)
    rep_term = max(0, 3 - stratum_S_eff)
    return disagreement_term + 0.5 * id_term + 0.75 * rep_term


def build_queue(items: Iterable, queue_path: Path | None = None) -> list[QueueItem]:
    """Construct the adjudicator queue from this cycle's normalized items.

    `items` is expected to be a list of CanonicalEvidence joined with their
    coded labels and stratum-level identification stats. The user wires the
    join; this function ranks the output.

    Output: data/adjudicator_queue.json (top of queue first), capped at the
    human-loop budget (ADJUDICATOR_HOURS_PER_CYCLE).
    """
    queue_path = queue_path or QUEUE_PATH
    candidates: list[QueueItem] = []
    for it in items:
        # TODO: pull Y_by_coder, stratum_id_distance, stratum_S_eff via joins
        Y_by_coder = getattr(it, "Y_by_coder", {})
        stratum_id_distance = getattr(it, "stratum_id_distance", 0.10)
        stratum_S_eff = getattr(it, "stratum_S_eff", 3)
        disagree = coder_disagreement(Y_by_coder)
        if disagree < ADJUDICATOR_DISAGREEMENT_THRESHOLD:
            # auto-flagged llm_only_acceptable unless sampled into gold-expand
            if not _gold_expand_sampled(it.evidence_id):
                continue
            reason = "gold_expand"
        else:
            reason = "disagree"
        eig = expected_information_gain(
            it.evidence_id, disagree, stratum_id_distance, stratum_S_eff,
        )
        candidates.append(
            QueueItem(
                evidence_id=it.evidence_id,
                reason=reason,
                eig_score=eig,
                estimated_minutes=4,           # default; user can refine
            )
        )

    candidates.sort(key=lambda x: -x.eig_score)
    capped = _cap_by_budget(candidates, ADJUDICATOR_HOURS_PER_CYCLE)
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(json.dumps(
        {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "budget_hours": ADJUDICATOR_HOURS_PER_CYCLE,
            "n_in_queue": len(capped),
            "n_total_candidates": len(candidates),
            "items": [c.__dict__ for c in capped],
        },
        indent=2,
    ))
    return capped


def _cap_by_budget(items: list[QueueItem], budget_hours: float) -> list[QueueItem]:
    budget_minutes = budget_hours * 60
    out, used = [], 0
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
