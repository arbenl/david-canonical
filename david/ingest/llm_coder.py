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

from ..config import (
    CODED_DIR, CODER_CALIBRATION_STAN, CONFIG_ROOT, GOLD_DIR, LLM_POOL_REGISTRY,
)

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
            text=text[:4000],
        )
        try:
            response = self._client.messages.create(
                model=self._cfg.model,
                max_tokens=4,
                temperature=self._cfg.temperature,
                system=self._SYSTEM,
                messages=[
                    {"role": "user", "content": user_msg},
                ],
            )
            raw = response.content[0].text.strip()
            if raw.startswith("1") or raw.lower().startswith("yes"):
                return 1
            return 0
        except Exception:
            return 0


# ─── Groq backend ────────────────────────────────────────────────────────────

class GroqBackend:
    """Groq backend for binary (0/1) evidence coding — free tier, OpenAI-compatible.

    Reads GROQ_API_KEY from the environment.  Uses llama-3.3-70b-versatile by
    default (set via cfg.model in llm_pool.json).
    """

    _SYSTEM = (
        "You are a political-events coder. "
        "For each piece of text, respond with exactly 0 or 1 — "
        "1 if the text contains evidence of the stated tactic, 0 if not. "
        "Respond with a single digit only."
    )

    def __init__(self, cfg: LlmCoderConfig) -> None:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. "
                "Export it before running the coder: export GROQ_API_KEY=gsk_..."
            )
        from groq import Groq  # imported here so unit tests can patch without side-effects
        self._client = Groq(api_key=api_key)
        self._cfg = cfg
        self._template = AnthropicBackend._load_template(cfg.prompt_template_id)

    def code_item(self, text: str, tactic_class: str) -> int:
        """Return 1 if `text` contains evidence of `tactic_class`, else 0."""
        user_msg = self._template.format(
            tactic_class=tactic_class,
            text=text[:4000],
        )
        try:
            response = self._client.chat.completions.create(
                model=self._cfg.model,
                max_tokens=4,
                temperature=self._cfg.temperature,
                seed=self._cfg.seed,  # Groq supports seed for reproducibility
                messages=[
                    {"role": "system", "content": self._SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("1") or raw.lower().startswith("yes"):
                return 1
            return 0
        except Exception:
            return 0


class OllamaBackend:
    """Ollama backend for local binary (0/1) evidence coding.

    Communicates with local Ollama service running at http://localhost:11434.
    """

    _SYSTEM = (
        "You are a political-events coder. "
        "For each piece of text, respond with exactly 0 or 1 — "
        "1 if the text contains evidence of the stated tactic, 0 if not. "
        "Respond with a single digit only."
    )

    def __init__(self, cfg: LlmCoderConfig) -> None:
        self._cfg = cfg
        self._url = "http://localhost:11434/api/chat"
        self._template = AnthropicBackend._load_template(cfg.prompt_template_id)
        
        # Probe to check if Ollama is running
        import httpx
        try:
            resp = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
            if resp.status_code != 200:
                raise RuntimeError(f"Ollama returned status code {resp.status_code}")
        except Exception as e:
            raise RuntimeError(
                f"Ollama local service does not appear to be running. "
                f"Please start Ollama before executing. Error: {e}"
              )

    def code_item(self, text: str, tactic_class: str) -> int:
        """Return 1 if `text` contains evidence of `tactic_class`, else 0."""
        import httpx
        user_msg = self._template.format(
            tactic_class=tactic_class,
            text=text[:4000],
        )
        payload = {
            "model": self._cfg.model,
            "messages": [
                {"role": "system", "content": self._SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            "options": {
                "temperature": self._cfg.temperature,
                "seed": self._cfg.seed,
            },
            "stream": False
        }
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(self._url, json=payload)
                if resp.status_code == 200:
                    raw = resp.json().get("message", {}).get("content", "").strip()
                    if raw.startswith("1") or raw.lower().startswith("yes"):
                        return 1
                return 0
        except Exception:
            return 0


def get_backend(cfg: LlmCoderConfig) -> LlmBackend:
    """Resolve the backend from cfg.provider.

    Supported providers: 'anthropic', 'groq', 'ollama'.
    To add a new provider (Vertex, Together, etc.):
        1. Implement a class with code_item(text, tactic_class) -> int
        2. Add the provider key below.
    """
    _PROVIDER_DISPATCH = {
        "anthropic": AnthropicBackend,
        "groq": GroqBackend,
        "ollama": OllamaBackend,
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

def load_llm_pool() -> list[LlmCoderConfig]:
    """Load the LLM coder pool from config/llm_pool.json.

    Returns an empty list if the file doesn't exist (no LLM coders configured).
    Example llm_pool.json:
    [
      {
        "coder_id": "anthropic_claude_haiku_seed_001",
        "provider": "anthropic",
        "model": "claude-3-5-haiku-20241022",
        "prompt_template_id": "default",
        "seed": 1,
        "temperature": 0.0
      }
    ]
    """
    if not LLM_POOL_REGISTRY.exists():
        return []
    raw = json.loads(LLM_POOL_REGISTRY.read_text())
    return [LlmCoderConfig(**c) for c in raw]


def code_evidence(
    items,
    llm_pool: list[LlmCoderConfig],
    tactic_classes: list[str],
) -> list[dict]:
    """Run each item through each LLM coder for each tactic class.

    Writes coded labels to JSON-L (durable) and upserts each label to
    coder_labels in Postgres (non-fatal — JSON-L is the fallback).

    Output schema per record:
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
    records: list[dict] = []

    with out_path.open("a") as f:   # append so multiple runs accumulate
        for cfg in llm_pool:
            try:
                backend = get_backend(cfg)
            except RuntimeError as exc:
                # Missing API key for this provider — skip rather than abort
                import warnings
                warnings.warn(f"Skipping coder {cfg.coder_id!r}: {exc}")
                continue
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

                    # Upsert to Postgres (non-fatal)
                    try:
                        from ..db.repositories import upsert_coder_label
                        upsert_coder_label(
                            evidence_id=item.evidence_id,
                            coder_id=cfg.coder_id,
                            tactic_k=k,
                            label=Y,
                        )
                    except Exception:
                        pass  # JSON-L is the durable record

    return records


def code_uncoded_from_db(
    llm_pool: list,
    tactic_classes: list[str],
    limit: int = 300,
) -> list[dict]:
    """Fetch evidence_items that have no coder_labels yet and code them.

    Used by the pipeline to sweep up items that were scraped in a previous
    cycle but skipped coding (e.g., due to the now-removed policy_area filter).
    """
    from types import SimpleNamespace
    from ..db.connection import get_conn

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT e.evidence_id, e.text_content
                FROM   evidence_items e
                WHERE  e.adjudicated = FALSE
                  AND  e.text_content IS NOT NULL
                  AND  length(e.text_content) > 50
                  AND  e.stratum_id NOT LIKE 'gl\_%' ESCAPE '\'
                  AND  NOT EXISTS (
                           SELECT 1 FROM coder_labels cl
                           WHERE  cl.evidence_id = e.evidence_id
                       )
                ORDER  BY e.evidence_date DESC
                LIMIT  %s
                """,
                (limit,),
            )
            rows = cur.fetchall()

    if not rows:
        return []

    items = [SimpleNamespace(evidence_id=r[0], text=r[1]) for r in rows]
    return code_evidence(items, llm_pool, tactic_classes)


def code_missing_tactics_from_db(
    llm_pool: list,
    new_tactic_ids: list[str],
    limit: int = 2000,
) -> list[dict]:
    """Backfill coder_labels for tactics added after the initial coding run.

    Targets items that already have some labels (e.g. from a 3-tactic run) but
    have never been evaluated against any of new_tactic_ids. Safe to re-run —
    the upsert in code_evidence() is idempotent.
    """
    from types import SimpleNamespace
    from ..db.connection import get_conn

    if not new_tactic_ids:
        return []

    placeholders = ",".join(["%s"] * len(new_tactic_ids))
    n_new = len(new_tactic_ids)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT e.evidence_id, e.text_content
                FROM   evidence_items e
                WHERE  e.text_content IS NOT NULL
                  AND  length(e.text_content) > 50
                  AND  e.stratum_id NOT LIKE 'gl\\_%%' ESCAPE '\\'
                  AND  (
                           SELECT COUNT(DISTINCT cl.tactic_k)
                           FROM   coder_labels cl
                           WHERE  cl.evidence_id = e.evidence_id
                             AND  cl.tactic_k IN ({placeholders})
                       ) < %s
                ORDER  BY e.evidence_date DESC
                LIMIT  %s
                """,
                (*new_tactic_ids, n_new, limit),
            )
            rows = cur.fetchall()

    if not rows:
        return []

    items = [SimpleNamespace(evidence_id=r[0], text=r[1]) for r in rows]
    return code_evidence(items, llm_pool, new_tactic_ids)


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


def append_to_gold_calibration_csv(evidence_id: str, gold_label: int) -> None:
    """Append or update an entry in gold_b_calibration.csv."""
    from ..config import GOLD_DIR
    gold_path = GOLD_DIR / "gold_b_calibration.csv"
    gold_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    headers = ["evidence_id", "gold_label"]
    found = False

    if gold_path.exists():
        with gold_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            try:
                headers = next(reader)
            except StopIteration:
                pass
            for row in reader:
                if len(row) >= 2:
                    if row[0] == evidence_id:
                        row[1] = str(gold_label)
                        found = True
                    rows.append(row)
    if not found:
        rows.append([evidence_id, str(gold_label)])

    with gold_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)



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
    tactic_subset: set[str] | None = None,
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
        if tactic_subset and rec.get("tactic_class") not in tactic_subset:
            continue
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
    stan_data = _build_stan_data(gold, coded, tactic_subset={"SIO", "MIO", "CSIO"})
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
