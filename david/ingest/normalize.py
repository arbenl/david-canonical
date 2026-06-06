"""Normalize raw evidence items into the canonical evidence schema.

Canonical schema enforces:
  - stable evidence_id (sha256 of (source_id, item_id, evidence_date))
  - country, policy_area, language normalized to ISO/registered values
  - stratum_id derived as "{country}_{policy_area}" (slug)
  - observability_tier → observability score for the strata table
  - structural-independence signature from source registry

After normalization each record is upserted into Postgres (non-fatal —
JSON-L files remain the durable fallback).

Output: data/raw_normalized/{YYYY-MM-DD}.jsonl
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..config import DATA_ROOT

# ─── vocabulary tables ────────────────────────────────────────────────────────

# ISO-3166-1 alpha-2 normalisation.  Keys are lowercased raw values from
# scrapers; values are canonical ISO-3166-1 alpha-2 codes.
_COUNTRY_ALIASES: dict[str, str] = {
    # Albania
    "albania": "AL", "al": "AL", "alb": "AL",
    # Kosovo
    "kosovo": "XK", "xk": "XK", "kos": "XK",
    # Serbia
    "serbia": "RS", "rs": "RS", "srb": "RS",
    # Bosnia
    "bosnia": "BA", "ba": "BA", "bih": "BA", "bosnia and herzegovina": "BA",
    # North Macedonia
    "north macedonia": "MK", "mk": "MK", "mkd": "MK", "macedonia": "MK",
    # Montenegro
    "montenegro": "ME", "me": "ME", "mne": "ME",
    # Croatia
    "croatia": "HR", "hr": "HR", "hrv": "HR",
    # Slovenia
    "slovenia": "SI", "si": "SI", "svn": "SI",
    # EU-level
    "eu": "EU", "european union": "EU",
    # generic fallback
    "unknown": "ZZ", "": "ZZ",
}

# Canonical policy areas; map aliases to the registered vocabulary.
_POLICY_ALIASES: dict[str, str] = {
    "media freedom": "media_freedom",
    "media_freedom": "media_freedom",
    "press freedom": "media_freedom",
    "freedom of press": "media_freedom",
    "judicial independence": "judicial_independence",
    "judiciary": "judicial_independence",
    "rule of law": "rule_of_law",
    "roe": "rule_of_law",
    "corruption": "anti_corruption",
    "anti-corruption": "anti_corruption",
    "anti_corruption": "anti_corruption",
    "civil society": "civil_society_space",
    "civil_society": "civil_society_space",
    "civil_society_space": "civil_society_space",
    "ngos": "civil_society_space",
    "elections": "electoral_integrity",
    "electoral integrity": "electoral_integrity",
    "electoral_integrity": "electoral_integrity",
}
_POLICY_DEFAULT = "unknown"

# Observability score per tier (pre-registered in ARCHITECTURE.md §3.2)
_TIER_TO_I_O: dict[int, float] = {0: 0.30, 1: 0.50, 2: 0.70, 3: 0.90}


# ─── schema ──────────────────────────────────────────────────────────────────

class CanonicalEvidence(BaseModel):
    evidence_id: str
    source_id: str
    source_family: str
    evidence_date: str
    country: str           # ISO-3166-1 alpha-2
    policy_area: str       # canonical vocabulary term
    stratum_id: str        # "{country}_{policy_area}" slug
    language: str
    title: str
    url: str | None
    text: str
    opportunity_unit_id: str | None
    observability_tier: int = Field(ge=0, le=3)
    observability_score: float  # derived from tier
    structural_independence_signature: str
    raw_metadata: dict[str, Any] = Field(default_factory=dict)


def hash_evidence_id(source_id: str, item_id: str, evidence_date: str) -> str:
    h = hashlib.sha256(f"{source_id}|{item_id}|{evidence_date}".encode("utf-8"))
    return h.hexdigest()[:16]


def _slug(s: str) -> str:
    """Lower, strip, replace non-alnum with underscore."""
    return re.sub(r"[^a-z0-9]+", "_", s.lower().strip()).strip("_")


def _canonical_country(raw: str) -> str:
    return _COUNTRY_ALIASES.get(raw.strip().lower(), raw.strip().upper()[:2] or "ZZ")


def _canonical_policy(raw: str) -> str:
    return _POLICY_ALIASES.get(raw.strip().lower(), _POLICY_DEFAULT)


def _stratum_id(country_iso: str, policy: str) -> str:
    return f"{_slug(country_iso)}_{_slug(policy)}"


# ─── main normalization ───────────────────────────────────────────────────────

def _normalize_one(
    raw: dict[str, Any],
    source_family: str = "unknown",
    structural_independence_signature: str | None = None,
) -> CanonicalEvidence:
    country = _canonical_country(raw.get("country", ""))
    policy  = _canonical_policy(raw.get("policy_area", raw.get("metadata", {}).get("policy_area", "")))
    tier    = int(raw.get("observability_tier", 1))
    tier    = max(0, min(3, tier))

    return CanonicalEvidence(
        evidence_id=hash_evidence_id(raw["source_id"], raw["item_id"], raw["evidence_date"]),
        source_id=raw["source_id"],
        source_family=source_family,
        evidence_date=raw["evidence_date"],
        country=country,
        policy_area=policy,
        stratum_id=_stratum_id(country, policy),
        language=raw.get("language", "und"),
        title=raw["title"],
        url=raw.get("url"),
        text=raw["text"],
        opportunity_unit_id=raw.get("opportunity_unit_id"),
        observability_tier=tier,
        observability_score=_TIER_TO_I_O[tier],
        structural_independence_signature=(
            structural_independence_signature
            or raw.get("structural_independence_signature", raw["source_id"])
        ),
        raw_metadata=raw.get("metadata", {}),
    )


def _upsert_to_db(canon: CanonicalEvidence, source_family: str) -> None:
    """Persist one canonical record to Postgres.  Non-fatal — caller catches."""
    from ..db.repositories import (
        upsert_evidence_item,
        upsert_source,
        upsert_stratum,
    )

    # Ensure FK dependencies exist first
    upsert_source(
        source_id=canon.source_id,
        family=source_family,
    )
    upsert_stratum(
        stratum_id=canon.stratum_id,
        country=canon.country,
        policy=canon.policy_area,
        observability=canon.observability_score,
    )
    upsert_evidence_item(
        evidence_id=canon.evidence_id,
        stratum_id=canon.stratum_id,
        source_id=canon.source_id,
        evidence_date=canon.evidence_date,
        title=canon.title,
        url=canon.url,
        text_content=canon.text[:4096] if canon.text else None,  # cap at 4 KiB
        adjudicated=False,
    )


def normalize_raw(raw_paths: list[Path]) -> list[CanonicalEvidence]:
    """Read raw JSON-L files, normalize, write JSON-L, upsert to Postgres.

    Each raw item is normalized to CanonicalEvidence (stable evidence_id,
    ISO country, canonical policy_area, derived stratum_id).

    DB upserts are best-effort: failures are logged but do not abort the
    normalization run, so JSON-L files remain the durable fallback.
    """
    # Load source registry for family / independence lookup (once)
    try:
        from .sources import load_registry
        registry_by_id = {s.source_id: s for s in load_registry()}
    except Exception:
        registry_by_id = {}

    today = date.today().isoformat()
    out_path = DATA_ROOT / "raw_normalized" / f"{today}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[CanonicalEvidence] = []
    db_errors: list[str] = []

    with out_path.open("w") as f:
        for p in raw_paths:
            if not p.exists():
                continue
            for line in p.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue

                src = registry_by_id.get(raw.get("source_id", ""))
                family = src.family if src else "unknown"
                si_sig = src.source_id if src else raw.get("source_id", "")

                canon = _normalize_one(raw, source_family=family,
                                       structural_independence_signature=si_sig)
                records.append(canon)
                f.write(canon.model_dump_json() + "\n")

                # Upsert to Postgres (non-fatal)
                try:
                    _upsert_to_db(canon, source_family=family)
                except Exception as exc:
                    db_errors.append(f"{canon.evidence_id}: {exc}")

    if db_errors:
        # Surface as a warning in the JSON-L metadata sidecar rather than raising
        sidecar = out_path.with_suffix(".errors.jsonl")
        with sidecar.open("a") as ef:
            for msg in db_errors:
                ef.write(json.dumps({"error": msg}) + "\n")

    return records
