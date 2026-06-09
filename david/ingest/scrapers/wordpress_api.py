"""WordPress REST API scraper adapter.

Fetches all posts from a WordPress site's /wp-json/wp/v2/posts endpoint
with pagination, giving access to the full article archive rather than
just the 5-25 items that an RSS feed provides.

Requires source.ingest_kind == "wp_api".
"""

from __future__ import annotations

import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime
from typing import Iterator
import json

from ..sources import RawEvidenceItem, Source
from ..scrapers.news_rss import _tag_country, _tag_policy

_HEADERS = {
    "User-Agent": "DAVID-M01-research-scraper/0.1 (academic tobacco surveillance)",
}

_MAX_PAGES = 20    # safety cap — 20 × 100 = 2 000 posts max per source
_PER_PAGE  = 100
_TIMEOUT   = 15


def _fetch_page(base_url: str, page: int, since: date | None, until: date) -> tuple[list[dict], int]:
    """Fetch one page of WP posts. Returns (items, total_pages)."""
    params: dict[str, str] = {
        "per_page": str(_PER_PAGE),
        "page":     str(page),
        "orderby":  "date",
        "order":    "desc",
        "_fields":  "id,date,title,link,excerpt,content",
    }
    if until:
        params["before"] = until.isoformat() + "T23:59:59"
    if since:
        params["after"] = since.isoformat() + "T00:00:00"

    url = f"{base_url.rstrip('/')}?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            total_pages = int(resp.headers.get("X-WP-TotalPages", "1"))
            data = json.loads(resp.read())
        return data, total_pages
    except urllib.error.HTTPError as e:
        if e.code == 400:
            return [], 0   # page beyond last page
        raise


def _strip_html(html: str) -> str:
    """Minimal HTML-to-text: strip tags, collapse whitespace."""
    import re
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class WordpressApiScraper:
    def __init__(self, source: Source):
        self.source = source

    def fetch(self, since: date | None, until: date) -> Iterator[RawEvidenceItem]:
        api_base = self.source.endpoint   # e.g. https://example.org/wp-json/wp/v2/posts
        fetched_at = datetime.utcnow().isoformat() + "Z"

        for page in range(1, _MAX_PAGES + 1):
            try:
                items, total_pages = _fetch_page(api_base, page, since, until)
            except Exception:
                break

            for post in items:
                try:
                    evidence_date = datetime.fromisoformat(
                        post["date"].replace("Z", "+00:00")
                    ).date()
                except Exception:
                    continue

                title   = _strip_html(post.get("title", {}).get("rendered", ""))
                excerpt = _strip_html(post.get("excerpt", {}).get("rendered", ""))
                content = _strip_html(post.get("content", {}).get("rendered", ""))
                text    = excerpt or content[:400]
                combined = f"{title} {text}"

                country = _tag_country(combined, self.source.country_coverage)
                policy  = _tag_policy(combined)

                yield RawEvidenceItem(
                    source_id=self.source.source_id,
                    item_id=f"{self.source.source_id}_{post['id']}",
                    fetched_at=fetched_at,
                    evidence_date=evidence_date.isoformat(),
                    country=country,
                    language="en",
                    title=title,
                    url=post.get("link"),
                    text=text,
                    metadata={
                        "wp_id": post["id"],
                        "policy_area": policy,
                    },
                )

            if page >= total_pages:
                break
            time.sleep(0.5)   # polite crawl delay
