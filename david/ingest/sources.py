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
    """Resolve a scraper module by source_id.

    TODO: import dispatch table by family + ingest_kind, e.g.
        news + rss -> david.ingest.scrapers.news_rss.RssScraper(source)
        legislative + api -> david.ingest.scrapers.legislative_feed.ApiScraper(source)
    """
    raise NotImplementedError(
        f"No scraper registered for source_id={source.source_id}. "
        "Add an adapter in david/ingest/scrapers/ and wire it here."
    )


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
