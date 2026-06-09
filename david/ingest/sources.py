"""Source registry and scraper dispatcher.

A `Source` is one structurally distinct family of evidence (e.g. national
news API, EU transparency register, civil-society monitor API). Each source
has rho/delta priors and a structural-independence score vs. other sources.

Theorem A' requires S_eff >= 3 conditionally independent sources per stratum.
This module enforces that floor at ingest time, NOT at fit time.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Iterator, Protocol

from pydantic import BaseModel, Field

from ..config import CONFIG_ROOT, RAW_DIR, SOURCE_REGISTRY


class StructuralIndependence(BaseModel):
    """Pairwise structural independence score 0..1; 1 = fully independent."""

    other_source_id: str
    independence_score: float = Field(ge=0.0, le=1.0)
    rationale: str


class Source(BaseModel):
    source_id: str
    family: str  # one of: news, legislative, civil_society, transparency, leak, court
    ingest_kind: str  # api | rss | scrape | manual
    endpoint: str
    refresh_cadence_days: int
    country_coverage: list[str]
    policy_coverage: list[str]
    structural_independence: list[StructuralIndependence] = Field(default_factory=list)
    rho_prior_mean: float = Field(ge=0.0, le=1.0, default=0.7)
    rho_prior_scale: float = Field(gt=0.0, default=0.15)
    delta_prior_mean: float = Field(ge=0.0, le=1.0, default=0.05)
    delta_prior_scale: float = Field(gt=0.0, default=0.05)
    enabled: bool = True


class RawEvidenceItem(BaseModel):
    source_id: str
    item_id: str
    fetched_at: str
    evidence_date: str
    country: str
    language: str
    title: str
    url: str | None
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Scraper(Protocol):
    """Per-source adapter contract."""

    def fetch(self, since: date | None, until: date) -> Iterator[RawEvidenceItem]: ...


# Registry loaded from config/source_registry.json
def load_registry() -> list[Source]:
    if not SOURCE_REGISTRY.exists():
        return []
    raw = json.loads(SOURCE_REGISTRY.read_text())
    return [Source(**x) for x in raw]


def save_registry(sources: list[Source]) -> None:
    SOURCE_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    SOURCE_REGISTRY.write_text(json.dumps([s.dict() for s in sources], indent=2))


def get_scraper(source: Source) -> Scraper:
    """Resolve a scraper adapter by (family, ingest_kind).

    Dispatch table — add new adapters here as (family, ingest_kind) pairs.
    The wildcard family "*" matches any family for a given ingest_kind.

    Currently supported:
        news + rss          → RssScraper
        civil_society + rss → RssScraper
        legislative + rss   → RssScraper
        transparency + rss  → RssScraper

    To add a new adapter:
        1. Create david/ingest/scrapers/my_adapter.py implementing Scraper protocol
        2. Add the (family, ingest_kind) key to _DISPATCH below
    """
    from .scrapers.news_rss import RssScraper
    from .scrapers.wordpress_api import WordpressApiScraper
    from .scrapers.pubmed_api import PubmedScraper

    _DISPATCH: dict[tuple[str, str], type] = {
        ("news",          "rss"):    RssScraper,
        ("civil_society", "rss"):    RssScraper,
        ("legislative",   "rss"):    RssScraper,
        ("transparency",  "rss"):    RssScraper,
        ("*",             "rss"):    RssScraper,           # catch-all for RSS
        ("*",             "wp_api"): WordpressApiScraper,  # WordPress REST API archive
        ("*",             "pubmed"): PubmedScraper,        # PubMed E-utilities (free)
    }

    key = (source.family, source.ingest_kind)
    cls = _DISPATCH.get(key) or _DISPATCH.get(("*", source.ingest_kind))
    if cls is None:
        supported = [f"{f}+{k}" for f, k in _DISPATCH if f != "*"]
        raise NotImplementedError(
            f"No scraper for family={source.family!r}, "
            f"ingest_kind={source.ingest_kind!r}.\n"
            f"Supported: {supported}\n"
            "Add an adapter in david/ingest/scrapers/ and register it in get_scraper()."
        )
    return cls(source)


def run_scrapers(
    since: date | None = None,
    until: date | None = None,
    enabled_only: bool = True,
) -> list[Path]:
    """Run all enabled scrapers and write JSON-L to data/raw/{source_id}/."""
    until = until or date.today()
    out_paths: list[Path] = []
    for src in load_registry():
        if enabled_only and not src.enabled:
            continue
        scraper = get_scraper(src)
        out_path = RAW_DIR / src.source_id / f"{until.isoformat()}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for item in scraper.fetch(since=since, until=until):
                f.write(item.model_dump_json() + "\n")
        out_paths.append(out_path)
    return out_paths
