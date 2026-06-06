"""Multi-LLM coder pool with Dawid-Skene gold calibration.

The core automation move: LLM coders are imperfect coders with measurable
kappa_plus / kappa_minus reliability. They are NOT oracles. They flow into the
main fit through the same Dawid-Skene layer as human coders.

Calibration:
    1. Maintain gold/gold_b_calibration.csv with adjudicated Gold_B labels.
    2. For each LLM coder configuration (provider, model, prompt seed), run
       it on the gold set offline, collect Y_{e, m}, B_e.
    3. Fit coder_calibration.stan to get posterior kappa_plus[m], kappa_minus[m].
    4. Save posteriors to data/coded/coder_calibration_v{N}.json for the fit
       to consume.

Routine coding:
    For each new evidence item e and tactic class k:
        Y[e, m, k] = LLM_m(e, k)
    Items where M_llm coders disagree above threshold are escalated to the
    adjudicator queue. Items where M_llm coders agree are flagged llm_only.

This module does NOT call LLMs from inside CmdStan. LLMs are batched offline
into JSON-L; cmdstanpy reads pre-computed Y arrays.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

import numpy as np

from ..config import CODED_DIR, CONFIG_ROOT, GOLD_DIR


@dataclass(frozen=True)
class LlmCoderConfig:
    coder_id: str               # stable id: "anthropic_claude_opus_seed_001"
    provider: str
    model: str
    prompt_template_id: str
    seed: int
    temperature: float


class LlmBackend(Protocol):
    def code_item(self, text: str, tactic_class: str) -> int: ...


def get_backend(cfg: LlmCoderConfig) -> LlmBackend:
    """Resolve the backend; user wires actual provider clients."""
    raise NotImplementedError(
        f"LLM backend not wired for {cfg.provider}/{cfg.model}. "
        "Implement Anthropic/OpenAI clients in david/ingest/scrapers/ or a new module."
    )


def code_evidence(items, llm_pool: list[LlmCoderConfig], tactic_classes: list[str]) -> list[dict]:
    """Run each item through each LLM coder for each tactic class.

    Output schema:
        {
            "evidence_id": "...",
            "coder_id": "...",
            "tactic_class": "...",
            "Y": 0 | 1,
            "model_version": "...",
            "coded_at": "..."
        }
    """
    out_path = CODED_DIR / f"coded_{datetime.utcnow().date().isoformat()}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with out_path.open("w") as f:
        for cfg in llm_pool:
            backend = get_backend(cfg)
            for item in items:
                for k in tactic_classes:
                    Y = int(backend.code_item(item.text, k))
                    rec = {
                        "evidence_id": item.evidence_id,
                        "coder_id": cfg.coder_id,
                        "tactic_class": k,
                        "Y": Y,
                        "model_version": cfg.model,
                        "coded_at": datetime.utcnow().isoformat() + "Z",
                    }
                    records.append(rec)
                    f.write(json.dumps(rec) + "\n")
    return records


def calibrate_coders() -> dict:
    """Fit coder_calibration.stan on gold set.

    TODO: cmdstanpy.CmdStanModel(stan/coder_calibration.stan).sample(data=...)
    where data = gold_b_calibration.csv joined to coded_*.jsonl.
    """
    gold_path = GOLD_DIR / "gold_b_calibration.csv"
    if not gold_path.exists():
        return {
            "gate_status": "fail",
            "reason": "gold_b_calibration_missing",
            "expected_path": str(gold_path),
        }
    # TODO: build Stan data dict from gold + coded; fit; extract kappa posteriors.
    return {
        "gate_status": "fail",
        "reason": "coder_calibration_not_yet_implemented",
    }
