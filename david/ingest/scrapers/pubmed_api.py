"""PubMed E-utilities scraper adapter.

Uses NCBI's free Entrez API (no authentication required) to fetch
tobacco-relevant academic abstracts for Western Balkans countries.

Rate limit: 3 requests/second without API key, 10/sec with NCBI_API_KEY.
NCBI requests a User-Agent email identifier (NCBI_EMAIL env var or fallback).
"""

from __future__ import annotations

import calendar
import json
import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime
from typing import Iterator

from ..sources import RawEvidenceItem, Source
from .news_rss import _tag_country, _tag_policy

_NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_MAX_RESULTS = 10_000
_BATCH_SIZE = 100           # efetch batch size (NCBI limit)
_RATE_DELAY = 0.35          # seconds between requests (stay under 3/sec)
_NCBI_EMAIL = os.environ.get("NCBI_EMAIL", "arben.lila@gmail.com")
_USER_AGENT = f"DAVID-research/1.0 ({_NCBI_EMAIL})"

# Western Balkans country search terms (PubMed title/abstract)
_COUNTRY_TERMS = [
    "Albania", "Albanian", "Kosovo", "Kosovar",
    "North Macedonia", "Macedonian",
    "Serbia", "Serbian",
    "Bosnia", "Herzegovinian",
    "Montenegro", "Montenegrin",
]

# Core tobacco search terms
_TOBACCO_TERMS = [
    "tobacco", "cigarette", "smoking", "nicotine",
    "tobacco control", "smoke free", "smoking cessation",
    "tobacco industry",
]

_MONTH_ABBR = {v: k for k, v in enumerate(calendar.month_abbr) if v}


def _build_query(since: date | None, until: date) -> str:
    country_parts = [
        f'"{c}"[Title/Abstract]' if " " in c else f"{c}[Title/Abstract]"
        for c in _COUNTRY_TERMS
    ]
    tobacco_parts = [
        f'"{t}"[Title/Abstract]' if " " in t else f"{t}[Title/Abstract]"
        for t in _TOBACCO_TERMS
    ]
    country_q = " OR ".join(country_parts)
    tobacco_q = " OR ".join(tobacco_parts)
    since_str = since.strftime("%Y/%m/%d") if since else "1990/01/01"
    until_str = until.strftime("%Y/%m/%d")
    return (
        f"({tobacco_q}) AND ({country_q})"
        f' AND ("{since_str}"[PDAT] : "{until_str}"[PDAT])'
    )


def _esearch(query: str, retmax: int = _MAX_RESULTS) -> list[str]:
    url = (
        f"{_NCBI_BASE}/esearch.fcgi"
        f"?db=pubmed&term={urllib.parse.quote(query)}&retmax={retmax}&retmode=json"
    )
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.load(r)
    return data["esearchresult"]["idlist"]


def _parse_date(date_el: ET.Element | None) -> str:
    if date_el is None:
        return ""
    year = date_el.findtext("Year", default="")
    if not year:
        return ""
    month_raw = date_el.findtext("Month", default="1")
    day_raw = date_el.findtext("Day", default="1")
    try:
        month_num = int(month_raw)
    except ValueError:
        month_num = _MONTH_ABBR.get(month_raw[:3].capitalize(), 1)
    try:
        day_num = int(day_raw)
    except ValueError:
        day_num = 1
    return f"{year}-{month_num:02d}-{day_num:02d}"


def _efetch_batch(pmids: list[str]) -> list[dict]:
    ids_str = ",".join(pmids)
    url = (
        f"{_NCBI_BASE}/efetch.fcgi"
        f"?db=pubmed&id={ids_str}&retmode=xml&rettype=abstract"
    )
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=20) as r:
        xml_data = r.read().decode("utf-8")

    root = ET.fromstring(xml_data)
    articles = []
    for article_el in root.findall(".//PubmedArticle"):
        pmid = article_el.findtext(".//PMID", default="")
        title = article_el.findtext(".//ArticleTitle", default="")

        abstract_parts = article_el.findall(".//AbstractText")
        abstract = " ".join(
            ((el.get("Label", "") + ": ") if el.get("Label") else "") + (el.text or "")
            for el in abstract_parts
        ).strip()

        date_str = ""
        for path in [".//PubDate", ".//ArticleDate"]:
            date_str = _parse_date(article_el.find(path))
            if date_str:
                break

        journal = article_el.findtext(".//Journal/Title", default="")

        if pmid and title and date_str:
            articles.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "date": date_str,
                "journal": journal,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            })
    return articles


class PubmedScraper:
    def __init__(self, source: Source):
        self.source = source

    def fetch(self, since: date | None, until: date) -> Iterator[RawEvidenceItem]:
        query = _build_query(since, until)
        try:
            pmids = _esearch(query)
        except Exception:
            return
        time.sleep(_RATE_DELAY)

        for i in range(0, len(pmids), _BATCH_SIZE):
            batch = pmids[i : i + _BATCH_SIZE]
            try:
                articles = _efetch_batch(batch)
            except Exception:
                time.sleep(1.0)
                continue

            for art in articles:
                try:
                    evidence_date = datetime.strptime(art["date"], "%Y-%m-%d").date()
                except ValueError:
                    continue
                if since is not None and evidence_date < since:
                    continue
                if evidence_date > until:
                    continue

                combined = f"{art['title']} {art['abstract']}"
                policy = _tag_policy(combined)
                if policy == "general":
                    continue  # no tobacco policy signal; skip

                country = _tag_country(combined, self.source.country_coverage)

                yield RawEvidenceItem(
                    source_id=self.source.source_id,
                    item_id=f"pmid:{art['pmid']}",
                    fetched_at=datetime.utcnow().isoformat() + "Z",
                    evidence_date=evidence_date.isoformat(),
                    country=country,
                    language="en",
                    title=art["title"],
                    url=art["url"],
                    text=art["abstract"] or art["title"],
                    metadata={
                        "raw_id": art["pmid"],
                        "policy_area": policy,
                        "journal": art["journal"],
                    },
                )

            if i + _BATCH_SIZE < len(pmids):
                time.sleep(_RATE_DELAY)
