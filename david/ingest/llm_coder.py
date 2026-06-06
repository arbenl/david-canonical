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

import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

import numpy as np

from ..config import CODED_DIR, CODER_CALIBRATION_STAN, CONFIG_ROOT, GOLD_DIR

# ─── config types ────────────────────────────────────────────────────────────

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


# ─── Anthropic backend ───────────────────────────────────────────────────────

class AnthropicBackend:
    """Anthropic Claude backend for binary (0/1) evidence coding.

    Reads ANTHROPIC_API_KEY from the environment (required).  Each call to
    code_item() makes one Anthropic API call and returns 0 or 1.

    System prompt is minimal and stable: the actual tactic definitions live
    in config/prompt_templates/{template_id}.txt so they can be versioned.
    """

    _SYSTEM = (
        "You are a political-events coder. "
        "For each piece of text, respond with exactly 0 or 1 — "
        "1 if the text contains evidence of the stated tactic, 0 if not. "
        "Respond with a single digit only."
    )

    def __init__(self, cfg: LlmCoderConfig) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. "
                "Export it before running the coder: export ANTHROPIC_API_KEY=sk-ant-..."
            )
        import anthropic  # imported here so unit tests can patch without side-effects
        self._client = anthropic.Anthropic(api_key=api_key)
        self._cfg = cfg
        self._template = self._load_template(cfg.prompt_template_id)

    @staticmethod
    def _load_template(template_id: str) -> str:
        tpl_dir = CONFIG_ROOT / "prompt_templates"
        path = tpl_dir / f"{template_id}.txt"
        if path.exists():
            return path.read_text().strip()
        # Minimal built-in template if file is absent
        return (
            "Tactic to detect: {tactic_class}\n\n"
            "Evidence text:\n{text}\n\n"
            "Answer (0 or 1):"
        )

    def code_item(self, text: str, tactic_class: str) -> int:
        """Return 1 if `text` contains evidence of `tactic_class`, else 0."""
        user_msg = self._template.format(
            tactic_class=tactic_class,
            text=text[:4000],  # cap at 4 000 chars to stay within context limits
        )
        response = self._client.messages.create(
            model=self._cfg.model,
            max_tokens=4,
            temperature=self._cfg.temperature,
            system=self._SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = response.content[0].text.strip()
        # Fault-tolerant parse: accept "0", "1", "Yes"→1, "No"→0, else 0
        if raw.startswith("1") or raw.lower().startswith("yes"):
            return 1
        return 0


def get_backend(cfg: LlmCoderConfig) -> LlmBackend:
    """Resolve the backend from cfg.provider.

    Currently supported providers: 'anthropic'.
    To add a new provider (OpenAI, Vertex, etc.):
        1. Implement a class with code_item(text, tactic_class) -> int
        2. Add the provider key below.
    """
    _PROVIDER_DISPATCH = {
        "anthropic": AnthropicBackend,
    }
    cls = _PROVIDER_DISPATCH.get(cfg.provider.lower())
    if cls is None:
        supported = list(_PROVIDER_DISPATCH)
        raise NotImplementedError(
            f"LLM backend not wired for provider={cfg.provider!r}. "
            f"Supported providers: {supported}. "
            "Add a new class and register it in get_backend()."
        )
    return cls(cfg)


# ─── coding routine ──────────────────────────────────────────────────────────

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


# ─── calibration ─────────────────────────────────────────────────────────────

def _load_gold_labels(gold_path: Path) -> dict[str, int]:
    """Read gold_b_calibration.csv → {evidence_id: gold_label (0|1)}."""
    gold: dict[str, int] = {}
    with gold_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            eid = row.get("evidence_id", "").strip()
            lbl_raw = row.get("gold_label", row.get("label", "")).strip()
            if eid and lbl_raw in ("0", "1"):
                gold[eid] = int(lbl_raw)
    return gold


def _load_coded_labels(coded_dir: Path) -> list[dict]:
    """Read all coded_*.jsonl files → list of coding records."""
    records: list[dict] = []
    for p in sorted(coded_dir.glob("coded_*.jsonl")):
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except json.JSONDecodeError:
                continue
    return records


def _build_stan_data(
    gold: dict[str, int],
    coded: list[dict],
) -> dict | None:
    """Build the Dawid-Skene data dict for coder_calibration.stan.

    Returns None if there is insufficient data.

    Stan variable mapping:
        E_gold   — number of gold-annotated evidence items
        E_un     — number of un-annotated (routine-coded) evidence items
        M        — number of coder configurations
        Y_gold   — int[E_gold, M]: coder labels for gold items
        B_gold   — int[E_gold]: ground-truth binary labels for gold items
        Y_un     — int[E_un, M]: coder labels for un-annotated items

    For items where a coder has multiple labels (one per tactic class), we
    reduce to a single binary: 1 if ANY tactic class was coded 1, else 0.
    This follows the Dawid-Skene "event detected" convention.

    Items coded by fewer than M coders are excluded (requires complete matrix).
    """
    # Aggregate: (evidence_id, coder_id) → 1 if any Y=1, else 0
    label_map: dict[tuple[str, str], int] = defaultdict(int)
    coder_set: set[str] = set()
    for rec in coded:
        eid = rec.get("evidence_id", "").strip()
        cid = rec.get("coder_id", "").strip()
        y   = int(rec.get("Y", 0))
        if eid and cid:
            coder_set.add(cid)
            label_map[(eid, cid)] = max(label_map[(eid, cid)], y)

    coders = sorted(coder_set)
    M = len(coders)
    if M == 0:
        return None

    # Collect all evidence IDs that have ALL M coders labelled
    all_evidence = {eid for eid, _ in label_map}
    complete = [
        eid for eid in sorted(all_evidence)
        if all((eid, c) in label_map for c in coders)
    ]
    if not complete:
        return None

    gold_ids = [eid for eid in complete if eid in gold]
    un_ids   = [eid for eid in complete if eid not in gold]

    E_gold = len(gold_ids)
    E_un   = len(un_ids)
    if E_gold < 2:
        return None  # too few gold items; calibration is unreliable

    # Build matrices (Stan is 1-indexed but cmdstanpy takes Python lists/arrays)
    Y_gold = [[label_map[(eid, c)] for c in coders] for eid in gold_ids]
    B_gold = [gold[eid]            for eid in gold_ids]
    Y_un   = [[label_map[(eid, c)] for c in coders] for eid in un_ids] if un_ids else [[0]*M]
    E_un   = max(E_un, 1)  # Stan requires E_un >= 1

    return {
        "E_gold": E_gold,
        "E_un":   E_un,
        "M":      M,
        "Y_gold": Y_gold,
        "B_gold": B_gold,
        "Y_un":   Y_un,
        # Pass coders list as metadata (not used by Stan)
        "_coders": coders,
        "_gold_ids": gold_ids,
        "_un_ids":   un_ids,
    }


def calibrate_coders() -> dict:
    """Fit coder_calibration.stan on the gold standard set.

    Reads:
        data/gold/gold_b_calibration.csv   — adjudicated Gold_B labels
        data/coded/coded_*.jsonl            — LLM coder labels

    Fits:
        stan/coder_calibration.stan         — Dawid-Skene model

    Writes:
        data/coded/coder_calibration_v{N}.json   — kappa posterior summaries

    Returns typed result dict with gate_status='pass'|'fail'.
    """
    gold_path = GOLD_DIR / "gold_b_calibration.csv"
    if not gold_path.exists():
        return {
            "gate_status": "fail",
            "reason": "gold_b_calibration_missing",
            "expected_path": str(gold_path),
        }

    if not CODER_CALIBRATION_STAN.exists():
        return {
            "gate_status": "fail",
            "reason": "coder_calibration_stan_missing",
            "expected_path": str(CODER_CALIBRATION_STAN),
        }

    # Load gold labels
    gold = _load_gold_labels(gold_path)
    if not gold:
        return {"gate_status": "fail", "reason": "gold_b_calibration_empty"}

    # Load coded labels
    coded = _load_coded_labels(CODED_DIR)
    if not coded:
        return {
            "gate_status": "fail",
            "reason": "no_coded_jsonl_found",
            "detail": f"No coded_*.jsonl files in {CODED_DIR}",
        }

    # Build Stan data
    stan_data = _build_stan_data(gold, coded)
    if stan_data is None:
        return {
            "gate_status": "fail",
            "reason": "insufficient_overlap",
            "detail": (
                "Need ≥ 2 gold items coded by all M coders. "
                "Run code_evidence() on the gold set first."
            ),
        }

    coders: list[str] = stan_data.pop("_coders")
    stan_data.pop("_gold_ids", None)
    un_ids: list[str] = stan_data.pop("_un_ids", [])

    # Fit with cmdstanpy
    try:
        from cmdstanpy import CmdStanModel
    except ImportError:
        return {"gate_status": "fail", "reason": "cmdstanpy_not_installed"}

    try:
        model = CmdStanModel(stan_file=str(CODER_CALIBRATION_STAN))
        fit = model.sample(
            data=stan_data,
            chains=4,
            iter_warmup=500,
            iter_sampling=500,
            seed=20240101,
            show_progress=False,
        )
    except Exception as exc:
        return {"gate_status": "fail", "reason": "stan_sampling_failed", "detail": str(exc)}

    # Extract posterior summaries
    summary = fit.summary()

    def _extract(prefix: str, n: int) -> list[dict[str, float]]:
        out = []
        for i in range(1, n + 1):
            param = f"{prefix}[{i}]"
            if param in summary.index:
                row = summary.loc[param]
                out.append({
                    "mean":  float(row.get("Mean",   float("nan"))),
                    "sd":    float(row.get("StdDev", float("nan"))),
                    "q05":   float(row.get("5%",     float("nan"))),
                    "q50":   float(row.get("50%",    float("nan"))),
                    "q95":   float(row.get("95%",    float("nan"))),
                })
            else:
                out.append({"mean": float("nan")})
        return out

    M = stan_data["M"]
    kappa_plus  = _extract("kappa_plus",  M)
    kappa_minus = _extract("kappa_minus", M)

    # Save posteriors
    CODED_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(CODED_DIR.glob("coder_calibration_v*.json"))
    next_v = len(existing) + 1
    summary_path = CODED_DIR / f"coder_calibration_v{next_v:03d}.json"

    payload = {
        "version": next_v,
        "calibrated_at": datetime.utcnow().isoformat() + "Z",
        "n_gold": stan_data["E_gold"],
        "n_un":   stan_data["E_un"],
        "coders": coders,
        "kappa_plus":  kappa_plus,
        "kappa_minus": kappa_minus,
        "stan_file": str(CODER_CALIBRATION_STAN),
    }
    summary_path.write_text(json.dumps(payload, indent=2))

    return {
        "gate_status": "pass",
        "summary_path": str(summary_path),
        "n_coders": M,
        "n_gold":   stan_data["E_gold"],
        "version":  next_v,
    }
