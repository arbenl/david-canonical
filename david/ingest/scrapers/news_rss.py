"""News RSS / Atom scraper adapter.

Skeleton example. Fill in the parsing and dedup logic per source.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Iterator

import feedparser

from ..sources import RawEvidenceItem, Source


class RssScraper:
    def __init__(self, source: Source):
        self.source = source

    def fetch(self, since: date | None, until: date) -> Iterator[RawEvidenceItem]:
        parsed = feedparser.parse(self.source.endpoint)
        for entry in parsed.entries:
            try:
                evidence_date = datetime(*entry.published_parsed[:6]).date()
            except Exception:
                continue
            if since is not None and evidence_date < since:
                continue
            if evidence_date > until:
                continue
            yield RawEvidenceItem(
                source_id=self.source.source_id,
                item_id=getattr(entry, "id", entry.link),
                fetched_at=datetime.utcnow().isoformat() + "Z",
                evidence_date=evidence_date.isoformat(),
                country=self._country_from_entry(entry),
                language=getattr(entry, "language", "en"),
                title=entry.title,
                url=entry.link,
                text=getattr(entry, "summary", entry.title),
                metadata={"raw_id": getattr(entry, "id", None)},
            )

    def _country_from_entry(self, entry) -> str:
        # TODO: implement country tagger; default to first country_coverage entry.
        return self.source.country_coverage[0] if self.source.country_coverage else "unknown"
