"""Per-source scraper adapters.

Each adapter implements the `Scraper` protocol from david.ingest.sources:

    def fetch(self, since: date | None, until: date) -> Iterator[RawEvidenceItem]: ...

Suggested modules to implement:

  news_rss.py              RSS / Atom for national newspapers + wire services
  legislative_feed.py      EU Transparency Register, OpenSecrets, national parl APIs
  civil_society_api.py     STOP, Tobacco Tactics, Tobacco Watch
  transparency_register.py EU TR, OECD lobby disclosures
  court_filings.py         Courtlistener, RECAP, RPC
  leak_archives.py         Truth Tobacco Industry Documents (UCSF)
"""
