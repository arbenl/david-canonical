"""News RSS / Atom scraper adapter."""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Iterator

import feedparser

from ..sources import RawEvidenceItem, Source

# ── Country keyword tagger ─────────────────────────────────────────────────
# Maps ISO-2 country code → regex pattern matching country name/demonym variants.
# Ordered by specificity (longer/more-specific patterns first to avoid false matches).
_COUNTRY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Western Balkans — EU candidate countries (thesis focus)
    ("XK", re.compile(
        r"\b(Kosovo[av]?e?|Kosovar|Pristina|Prishtina|Prishtinë)\b", re.IGNORECASE
    )),
    ("MK", re.compile(
        r"\b(North\s+Macedonia|N\.?\s*Macedonia|Macedoni[ao]|Macedonian|Maqedoni[ae]|"
        r"Shkup|Skopje|VMRO)\b", re.IGNORECASE
    )),
    ("AL", re.compile(
        r"\b(Albania[n]?|Shqipëri[a]?|Shqiperi[a]?|Albanian|Tirana|Tiranë)\b",
        re.IGNORECASE
    )),
    ("RS", re.compile(
        r"\b(Serbia[n]?|Srbija|Belgrade|Beograd|Novi\s+Sad)\b", re.IGNORECASE
    )),
    ("BA", re.compile(
        r"\b(Bosnia[n]?|Herzegovina|Bosnian|Sarajevo|Mostar|Banja\s+Luka)\b",
        re.IGNORECASE
    )),
    ("ME", re.compile(
        r"\b(Montenegro[a-z]*|Montenegrin|Crna\s+Gora|Podgorica)\b", re.IGNORECASE
    )),
    # EU members — policy leaders/controls
    ("NL", re.compile(
        r"\b(Netherlands|Dutch|Holland|Nederland[s]?|Amsterdam|The\s+Hague|Den\s+Haag)\b",
        re.IGNORECASE
    )),
    ("IE", re.compile(
        r"\b(Ireland|Irish|Éire|Eire|Dublin)\b", re.IGNORECASE
    )),
    ("SE", re.compile(
        r"\b(Sweden|Swedish|Sverige|Svensk[a]?|Stockholm|snus)\b", re.IGNORECASE
    )),
    ("TR", re.compile(
        r"\b(Turkey|Turkish|Türkiye|Turkiye|Türk|Ankara|Istanbul|İstanbul)\b",
        re.IGNORECASE
    )),
    # Other relevant EU / candidate states
    ("HR", re.compile(r"\b(Croatia[n]?|Hrvatska|Zagreb)\b", re.IGNORECASE)),
    ("SI", re.compile(r"\b(Slovenia[n]?|Slovenija|Ljubljana)\b", re.IGNORECASE)),
    ("HU", re.compile(r"\b(Hungary|Hungarian|Magyarország|Budapest)\b", re.IGNORECASE)),
    ("PL", re.compile(r"\b(Poland|Polish|Polska|Warsaw|Warszawa)\b", re.IGNORECASE)),
    ("RO", re.compile(r"\b(Romania[n]?|România|Bucharest|București)\b", re.IGNORECASE)),
    ("BG", re.compile(r"\b(Bulgaria[n]?|България|Sofia|Sofija)\b", re.IGNORECASE)),
    ("DE", re.compile(r"\b(German[y]?|Deutschland|Berlin|Bundesrat|Bundestag)\b", re.IGNORECASE)),
    ("FR", re.compile(r"\b(France|French|Français|Paris|Élysée)\b", re.IGNORECASE)),
    ("GB", re.compile(r"\b(UK|United\s+Kingdom|Britain|British|England|London|Whitehall)\b", re.IGNORECASE)),
]

# ── Policy keyword tagger ──────────────────────────────────────────────────
# Maps policy slug → regex; first match wins.
def _load_policy_patterns() -> list[tuple[str, re.Pattern[str]]]:
    from ...config import DOMAIN_TAXONOMY_REGISTRY
    import json
    if not DOMAIN_TAXONOMY_REGISTRY.exists():
        return []
    with open(DOMAIN_TAXONOMY_REGISTRY) as f:
        data = json.load(f)
    patterns = []
    for policy in data.get("policies", []):
        regex_str = policy.get("regex")
        if regex_str:
            patterns.append((policy["id"], re.compile(regex_str, re.IGNORECASE)))
    return patterns

_POLICY_PATTERNS: list[tuple[str, re.Pattern[str]]] = _load_policy_patterns()


def _tag_country(text: str, coverage: list[str]) -> str:
    """Return ISO-2 country code from article text, or coverage[0] as fallback.

    Only tries patterns for countries explicitly listed in coverage (GLOBAL
    is a scope flag, not a wildcard — it does not enable patterns for countries
    not in coverage). This prevents articles mentioning incidental country
    names (e.g. "British American Tobacco") from being routed to strata the
    source does not monitor.
    """
    explicit = {c for c in coverage if c != "GLOBAL"}
    for iso, pattern in _COUNTRY_PATTERNS:
        if iso in explicit and pattern.search(text):
            return iso
    # fallback: first explicit non-GLOBAL entry in coverage
    for c in coverage:
        if c != "GLOBAL":
            return c
    return coverage[0] if coverage else "unknown"


def _tag_policy(text: str) -> str:
    """Return policy slug from article text, or 'general' if no match."""
    for slug, pattern in _POLICY_PATTERNS:
        if pattern.search(text):
            return slug
    return "general"


class RssScraper:
    def __init__(self, source: Source):
        self.source = source

    def fetch(self, since: date | None, until: date) -> Iterator[RawEvidenceItem]:
        parsed = feedparser.parse(self.source.endpoint)
        for entry in parsed.entries:
            try:
                evidence_date = datetime(*entry.published_parsed[:6]).date()
            except Exception:
                # Fallback 1: some feeds (e.g. BMJ Tobacco Control) use
                # `updated_parsed` instead of `published_parsed`.
                try:
                    evidence_date = datetime(*entry.updated_parsed[:6]).date()
                except Exception:
                    # Fallback 2: PRISM-standard publication date (ISO-8601 string)
                    prism_date = getattr(entry, "prism_publicationdate", None)
                    if prism_date:
                        try:
                            evidence_date = date.fromisoformat(prism_date[:10])
                        except Exception:
                            continue
                    else:
                        continue
            if since is not None and evidence_date < since:
                continue
            if evidence_date > until:
                continue
            text = getattr(entry, "summary", entry.title)
            combined = f"{entry.title} {text}"
            country = _tag_country(combined, self.source.country_coverage)
            policy = _tag_policy(combined)
            yield RawEvidenceItem(
                source_id=self.source.source_id,
                item_id=getattr(entry, "id", entry.link),
                fetched_at=datetime.utcnow().isoformat() + "Z",
                evidence_date=evidence_date.isoformat(),
                country=country,
                language=getattr(entry, "language", "en"),
                title=entry.title,
                url=entry.link,
                text=text,
                metadata={"raw_id": getattr(entry, "id", None), "policy_area": policy},
            )
