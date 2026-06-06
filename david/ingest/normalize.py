"""Normalize raw evidence items into the canonical evidence schema.

Canonical schema enforces:
  - stable evidence_id (sha256 of (source_id, item_id, evidence_date))
  - country, policy_area, language normalized to ISO/registered values
  - mapping to opportunity unit (matches council_m01 opportunity_units_s0.csv)
  - structural-independence ledger lookup so downstream fit can compute S_eff

Output: data/raw_normalized/{YYYY-MM-DD}.jsonl
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..config import DATA_ROOT


class CanonicalEvidence(BaseModel):
    evidence_id: str
    source_id: str
    source_family: str
    evidence_date: str
    country: str
    policy_area: str
    language: str
    title: str
    url: str | None
    text: str
    opportunity_unit_id: str | None
    observability_tier: int = Field(ge=0, le=3)
    structural_independence_signature: str
    raw_metadata: dict[str, Any] = Field(default_factory=dict)


def hash_evidence_id(source_id: str, item_id: str, evidence_date: str) -> str:
    h = hashlib.sha256(f"{source_id}|{item_id}|{evidence_date}".encode("utf-8"))
    return h.hexdigest()[:16]


def normalize_raw(raw_paths: list[Path]) -> list[CanonicalEvidence]:
    """Read raw JSON-L files and emit canonical records.

    TODO: implement country / policy_area / opportunity_unit_id mapping using
    config/opportunity_units.csv (mirror of council_m01/program/opportunity_units_s0.csv).
    """
    today = date.today().isoformat()
    out_path = DATA_ROOT / "raw_normalized" / f"{today}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[CanonicalEvidence] = []
    with out_path.open("w") as f:
        for p in raw_paths:
            for line in p.read_text().splitlines():
                raw = json.loads(line)
                canon = _normalize_one(raw)
                records.append(canon)
                f.write(canon.model_dump_json() + "\n")
    return records


def _normalize_one(raw: dict[str, Any]) -> CanonicalEvidence:
    # TODO: country/policy mapping; observability tier from source family; etc.
    return CanonicalEvidence(
        evidence_id=hash_evidence_id(raw["source_id"], raw["item_id"], raw["evidence_date"]),
        source_id=raw["source_id"],
        source_family="unknown",
        evidence_date=raw["evidence_date"],
        country=raw["country"],
        policy_area=raw.get("policy_area", "unknown"),
        language=raw["language"],
        title=raw["title"],
        url=raw.get("url"),
        text=raw["text"],
        opportunity_unit_id=raw.get("opportunity_unit_id"),
        observability_tier=int(raw.get("observability_tier", 1)),
        structural_independence_signature=raw.get(
            "structural_independence_signature", raw["source_id"]
        ),
        raw_metadata=raw.get("metadata", {}),
    )
