"""Tests for the ingest layer — normalize, sources dispatch, llm_coder, API.

All tests are offline (no Postgres, no Anthropic API, no Stan).
Postgres calls are patched via monkeypatch; AnthropicBackend is patched
to avoid importing the `anthropic` package in CI.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  sources.py — dispatch table
# ═══════════════════════════════════════════════════════════════════════════════

class _FakeSource:
    """Minimal mock of david.ingest.sources.Source."""
    def __init__(self, family, ingest_kind, source_id="src_test"):
        self.family = family
        self.ingest_kind = ingest_kind
        self.source_id = source_id
        self.endpoint = "https://example.com/rss"
        self.country_coverage = ["AL"]
        self.policy_coverage = ["media_freedom"]


def test_get_scraper_rss_news():
    """news + rss → RssScraper instance returned (no error)."""
    from david.ingest.sources import get_scraper
    src = _FakeSource("news", "rss")
    scraper = get_scraper(src)
    # Just verify we get something with a fetch() attribute
    assert hasattr(scraper, "fetch")


def test_get_scraper_civil_society_rss():
    from david.ingest.sources import get_scraper
    src = _FakeSource("civil_society", "rss")
    scraper = get_scraper(src)
    assert hasattr(scraper, "fetch")


def test_get_scraper_wildcard_rss():
    """Unseen family with rss → wildcard catch-all."""
    from david.ingest.sources import get_scraper
    src = _FakeSource("court_filings", "rss")
    scraper = get_scraper(src)
    assert hasattr(scraper, "fetch")


def test_get_scraper_unsupported_raises():
    """Unknown ingest_kind → NotImplementedError with helpful message."""
    from david.ingest.sources import get_scraper
    src = _FakeSource("news", "soap_endpoint")
    with pytest.raises(NotImplementedError, match="No scraper for"):
        get_scraper(src)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  normalize.py — country/policy/stratum mapping
# ═══════════════════════════════════════════════════════════════════════════════

def _make_raw(
    source_id="src1",
    item_id="item1",
    evidence_date="2024-03-15",
    country="Albania",
    policy_area="media freedom",
    language="sq",
    title="Test title",
    text="Test body.",
    tier=1,
) -> dict:
    return {
        "source_id": source_id,
        "item_id": item_id,
        "evidence_date": evidence_date,
        "country": country,
        "policy_area": policy_area,
        "language": language,
        "title": title,
        "text": text,
        "observability_tier": tier,
        "url": None,
        "metadata": {},
    }


def test_normalize_one_country_iso():
    from david.ingest.normalize import _normalize_one
    raw = _make_raw(country="Albania")
    c = _normalize_one(raw)
    assert c.country == "AL"


def test_normalize_one_country_alias_lower():
    from david.ingest.normalize import _normalize_one
    c = _normalize_one(_make_raw(country="kosovo"))
    assert c.country == "XK"


def test_normalize_one_policy_canonical():
    from david.ingest.normalize import _normalize_one
    c = _normalize_one(_make_raw(policy_area="press freedom"))
    assert c.policy_area == "media_freedom"


def test_normalize_one_stratum_id():
    from david.ingest.normalize import _normalize_one
    c = _normalize_one(_make_raw(country="Albania", policy_area="judicial independence"))
    assert c.stratum_id == "al_judicial_independence"


def test_normalize_one_stable_evidence_id():
    """Same (source_id, item_id, date) → same hash every time."""
    from david.ingest.normalize import _normalize_one
    c1 = _normalize_one(_make_raw())
    c2 = _normalize_one(_make_raw())
    assert c1.evidence_id == c2.evidence_id
    assert len(c1.evidence_id) == 16


def test_normalize_one_tier_to_score():
    from david.ingest.normalize import _normalize_one
    c = _normalize_one(_make_raw(tier=2))
    assert c.observability_tier == 2
    assert abs(c.observability_score - 0.70) < 1e-9


def test_normalize_one_tier_clamped():
    from david.ingest.normalize import _normalize_one
    c = _normalize_one(_make_raw(tier=99))
    assert c.observability_tier == 3
    assert c.observability_score == 0.90


def test_normalize_raw_writes_jsonl(tmp_path, monkeypatch):
    """normalize_raw writes JSON-L; DB upserts are patched to no-ops."""
    monkeypatch.setattr("david.ingest.normalize.DATA_ROOT", tmp_path)
    # Patch away DB calls and source registry
    monkeypatch.setattr("david.ingest.normalize._upsert_to_db", lambda c, source_family: None)
    monkeypatch.setattr("david.ingest.sources.load_registry", lambda: [])
    # _make_raw() has no tobacco keywords; bypass the relevance filter for this unit test
    monkeypatch.setattr("david.ingest.normalize.is_tobacco_relevant", lambda title, text: True)

    raw_file = tmp_path / "raw.jsonl"
    raw_file.write_text(
        json.dumps(_make_raw()) + "\n" +
        json.dumps(_make_raw(item_id="item2")) + "\n"
    )

    from david.ingest.normalize import normalize_raw
    records = normalize_raw([raw_file])

    assert len(records) == 2
    out = tmp_path / "raw_normalized"
    jsonl_files = list(out.glob("*.jsonl"))
    assert jsonl_files
    lines = [ln for ln in jsonl_files[0].read_text().splitlines() if ln.strip()]
    assert len(lines) == 2


def test_normalize_raw_db_error_writes_sidecar(tmp_path, monkeypatch):
    """When DB upsert fails, error sidecar is written but function returns normally."""
    monkeypatch.setattr("david.ingest.normalize.DATA_ROOT", tmp_path)
    monkeypatch.setattr("david.ingest.normalize.is_tobacco_relevant", lambda title, text: True)

    def _failing_upsert(c, source_family):
        raise RuntimeError("connection refused")

    monkeypatch.setattr("david.ingest.normalize._upsert_to_db", _failing_upsert)

    raw_file = tmp_path / "r.jsonl"
    raw_file.write_text(json.dumps(_make_raw()) + "\n")

    from david.ingest.normalize import normalize_raw
    records = normalize_raw([raw_file])

    assert len(records) == 1  # normalization still worked
    # sidecar
    out = tmp_path / "raw_normalized"
    sidecars = list(out.glob("*.errors.jsonl"))
    assert sidecars
    err_content = sidecars[0].read_text()
    assert "connection refused" in err_content


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  llm_coder.py — backend dispatch + calibration helpers
# ═══════════════════════════════════════════════════════════════════════════════

def test_get_backend_unsupported_raises():
    from david.ingest.llm_coder import LlmCoderConfig, get_backend
    cfg = LlmCoderConfig(
        coder_id="gpt4_001", provider="openai", model="gpt-4",
        prompt_template_id="default", seed=1, temperature=0.0,
    )
    with pytest.raises(NotImplementedError, match="provider='openai'"):
        get_backend(cfg)


def test_get_backend_anthropic_no_key():
    """get_backend('anthropic') raises RuntimeError when API key is missing."""
    import os
    from david.ingest.llm_coder import LlmCoderConfig, get_backend
    cfg = LlmCoderConfig(
        coder_id="claude_001", provider="anthropic", model="claude-3-5-haiku-20241022",
        prompt_template_id="default", seed=1, temperature=0.0,
    )
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY not set"):
            get_backend(cfg)


def test_anthropic_backend_code_item_parses_0_and_1(monkeypatch):
    """AnthropicBackend.code_item() parses '0', '1', 'Yes', 'No' correctly."""
    import os
    from david.ingest.llm_coder import AnthropicBackend, LlmCoderConfig

    cfg = LlmCoderConfig(
        coder_id="claude_001", provider="anthropic", model="claude-3-5-haiku-20241022",
        prompt_template_id="default", seed=1, temperature=0.0,
    )

    # Patch anthropic.Anthropic so no real HTTP call is made
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="1")]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    mock_anthropic_module = MagicMock()
    mock_anthropic_module.Anthropic.return_value = mock_client

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
        with patch.dict("sys.modules", {"anthropic": mock_anthropic_module}):
            backend = AnthropicBackend(cfg)
            assert backend.code_item("test text", "suppression") == 1

            mock_response.content = [MagicMock(text="0")]
            assert backend.code_item("test text", "suppression") == 0

            mock_response.content = [MagicMock(text="Yes, definitely")]
            assert backend.code_item("text", "tactic") == 1

            mock_response.content = [MagicMock(text="garbage")]
            assert backend.code_item("text", "tactic") == 0


def test_get_backend_groq_no_key():
    """get_backend('groq') raises RuntimeError when GROQ_API_KEY is missing."""
    import os
    from david.ingest.llm_coder import LlmCoderConfig, get_backend
    cfg = LlmCoderConfig(
        coder_id="groq_001", provider="groq", model="llama-3.3-70b-versatile",
        prompt_template_id="default", seed=1, temperature=0.0,
    )
    with patch.dict(os.environ, {"GROQ_API_KEY": ""}):
        with pytest.raises(RuntimeError, match="GROQ_API_KEY not set"):
            get_backend(cfg)


def test_groq_backend_code_item_parses_0_and_1(monkeypatch):
    """GroqBackend.code_item() parses '0', '1', 'Yes', 'No' correctly."""
    import os
    from david.ingest.llm_coder import GroqBackend, LlmCoderConfig

    cfg = LlmCoderConfig(
        coder_id="groq_001", provider="groq", model="llama-3.3-70b-versatile",
        prompt_template_id="default", seed=1, temperature=0.0,
    )

    mock_choice = MagicMock()
    mock_choice.message.content = "1"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    mock_groq_module = MagicMock()
    mock_groq_module.Groq.return_value = mock_client

    with patch.dict(os.environ, {"GROQ_API_KEY": "gsk_test"}):
        with patch.dict("sys.modules", {"groq": mock_groq_module}):
            backend = GroqBackend(cfg)

            mock_choice.message.content = "1"
            assert backend.code_item("Philip Morris blocked the bill", "SIO") == 1

            mock_choice.message.content = "0"
            assert backend.code_item("Unrelated news article", "SIO") == 0

            mock_choice.message.content = "Yes, clearly"
            assert backend.code_item("text", "MIO") == 1

            mock_choice.message.content = "garbage output"
            assert backend.code_item("text", "CSIO") == 0


def test_groq_backend_seed_passed_to_api(monkeypatch):
    """GroqBackend passes cfg.seed to the Groq API for reproducibility."""
    import os
    from david.ingest.llm_coder import GroqBackend, LlmCoderConfig

    cfg = LlmCoderConfig(
        coder_id="groq_seed2", provider="groq", model="llama-3.3-70b-versatile",
        prompt_template_id="default", seed=42, temperature=0.0,
    )

    mock_choice = MagicMock()
    mock_choice.message.content = "1"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    mock_groq_module = MagicMock()
    mock_groq_module.Groq.return_value = mock_client

    with patch.dict(os.environ, {"GROQ_API_KEY": "gsk_test"}):
        with patch.dict("sys.modules", {"groq": mock_groq_module}):
            backend = GroqBackend(cfg)
            backend.code_item("text", "SIO")
            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert call_kwargs["seed"] == 42


def test_load_gold_labels(tmp_path):
    from david.ingest.llm_coder import _load_gold_labels
    csv_path = tmp_path / "gold.csv"
    csv_path.write_text("evidence_id,gold_label\nev1,1\nev2,0\ninvalid,2\n")
    gold = _load_gold_labels(csv_path)
    assert gold == {"ev1": 1, "ev2": 0}  # invalid label 2 excluded


def test_load_coded_labels(tmp_path):
    from david.ingest.llm_coder import _load_coded_labels
    p = tmp_path / "coded_2024-01-01.jsonl"
    recs = [
        {"evidence_id": "ev1", "coder_id": "c1", "tactic_class": "suppress", "Y": 1},
        {"evidence_id": "ev1", "coder_id": "c2", "tactic_class": "suppress", "Y": 0},
    ]
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    loaded = _load_coded_labels(tmp_path)
    assert len(loaded) == 2


def test_build_stan_data_happy_path():
    from david.ingest.llm_coder import _build_stan_data
    gold = {"ev1": 1, "ev2": 0, "ev3": 1}
    coded = [
        {"evidence_id": "ev1", "coder_id": "c1", "tactic_class": "t", "Y": 1},
        {"evidence_id": "ev1", "coder_id": "c2", "tactic_class": "t", "Y": 1},
        {"evidence_id": "ev2", "coder_id": "c1", "tactic_class": "t", "Y": 0},
        {"evidence_id": "ev2", "coder_id": "c2", "tactic_class": "t", "Y": 1},
        {"evidence_id": "ev3", "coder_id": "c1", "tactic_class": "t", "Y": 1},
        {"evidence_id": "ev3", "coder_id": "c2", "tactic_class": "t", "Y": 0},
        {"evidence_id": "ev4", "coder_id": "c1", "tactic_class": "t", "Y": 0},  # un-annotated
        {"evidence_id": "ev4", "coder_id": "c2", "tactic_class": "t", "Y": 0},
    ]
    data = _build_stan_data(gold, coded)
    assert data is not None
    assert data["E_gold"] == 3
    assert data["M"] == 2
    assert len(data["Y_gold"]) == 3
    assert len(data["B_gold"]) == 3
    assert set(data["B_gold"]) <= {0, 1}


def test_build_stan_data_insufficient_gold():
    from david.ingest.llm_coder import _build_stan_data
    # Only 1 gold item → should return None
    gold = {"ev1": 1}
    coded = [
        {"evidence_id": "ev1", "coder_id": "c1", "Y": 1, "tactic_class": "t"},
        {"evidence_id": "ev1", "coder_id": "c2", "Y": 0, "tactic_class": "t"},
    ]
    assert _build_stan_data(gold, coded) is None


def test_build_stan_data_no_coders():
    from david.ingest.llm_coder import _build_stan_data
    assert _build_stan_data({"ev1": 1, "ev2": 0}, []) is None


def test_calibrate_coders_missing_gold(tmp_path, monkeypatch):
    monkeypatch.setattr("david.ingest.llm_coder.GOLD_DIR", tmp_path / "gold")
    from david.ingest.llm_coder import calibrate_coders
    result = calibrate_coders()
    assert result["gate_status"] == "fail"
    assert "gold_b_calibration_missing" in result["reason"]


def test_calibrate_coders_missing_stan(tmp_path, monkeypatch):
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    (gold_dir / "gold_b_calibration.csv").write_text("evidence_id,gold_label\nev1,1\n")
    monkeypatch.setattr("david.ingest.llm_coder.GOLD_DIR", gold_dir)
    monkeypatch.setattr("david.ingest.llm_coder.CODER_CALIBRATION_STAN", tmp_path / "missing.stan")
    from david.ingest.llm_coder import calibrate_coders
    result = calibrate_coders()
    assert result["gate_status"] == "fail"
    assert "stan_missing" in result["reason"]


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  api/server.py — DB-backed endpoints (stubs + mock)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def test_client():
    from fastapi.testclient import TestClient
    from david.api.server import api
    return TestClient(api)


def test_healthz(test_client):
    r = test_client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def _make_mock_conn(rows, cols=None):
    cur = MagicMock()
    cur.__enter__ = lambda s: s
    cur.__exit__ = MagicMock(return_value=False)
    cur.fetchall.return_value = rows
    cur.fetchone.return_value = rows[0] if rows else None
    if cols:
        cur.description = [(c,) for c in cols]
    conn = MagicMock()
    conn.__enter__ = lambda s: s
    conn.__exit__ = MagicMock(return_value=False)
    conn.cursor.return_value = cur
    return conn, cur


def test_list_strata_db(test_client):
    rows = [("al_media_freedom", "AL", "media_freedom", 0.5, 10, 5)]
    cols = ["stratum_id", "country", "policy", "observability",
            "n_evidence", "n_adjudicated"]
    conn, _ = _make_mock_conn(rows, cols)

    @contextmanager
    def _fake_get_conn():
        yield conn

    # The /strata endpoint does `from ..db.connection import get_conn` inside
    # the request handler, so we patch at the connection module level.
    with patch("david.db.connection.get_conn", _fake_get_conn):
        r = test_client.get("/strata")

    assert r.status_code == 200
    data = r.json()
    assert "strata" in data
    assert data["n"] == 1


def test_list_evidence_no_db_returns_503(test_client):
    """When Postgres raises, /evidence returns 503 (not 500)."""
    def _fail(*a, **kw):
        raise RuntimeError("no db")

    with patch("david.db.connection.get_conn", side_effect=_fail):
        r = test_client.get("/evidence")

    assert r.status_code == 503
    assert "database_unavailable" in r.json()["error"]


def test_list_fit_runs_db(test_client):
    """GET /fit-runs returns the list from get_fit_runs()."""
    mock_runs = [{"run_id": "fit_001", "gate_status": "pass", "rhat_max": 1.01,
                  "ess_bulk_min": 450.0, "divergences": 0,
                  "model_version": "v0.1.0", "n_strata": 3, "n_labels": 120,
                  "started_at": "2024-01-01T00:00:00"}]
    with patch("david.db.repositories.get_fit_runs", return_value=mock_runs):
        r = test_client.get("/fit-runs?limit=5")
    assert r.status_code == 200
    assert r.json()["n"] == 1
    assert r.json()["fit_runs"][0]["run_id"] == "fit_001"


def test_list_forecast_cells_db(test_client):
    """GET /forecast-cells?run_id=x returns cells from get_forecast_cells()."""
    mock_cells = [{"series": 1, "tactic": 2, "horizon_months": 6,
                   "p_active": 0.42, "forecast_route": "conditional_forecast"}]
    with patch("david.db.repositories.get_forecast_cells", return_value=mock_cells):
        r = test_client.get("/forecast-cells?run_id=fit_001")
    assert r.status_code == 200
    assert r.json()["n"] == 1
    assert r.json()["forecast_cells"][0]["p_active"] == pytest.approx(0.42)


def test_get_evidence_not_found(test_client):
    """GET /evidence/{bad_id} → 404."""
    conn, cur = _make_mock_conn([])
    cur.fetchone.return_value = None

    @contextmanager
    def _fake_get_conn():
        yield conn

    with patch("david.db.connection.get_conn", _fake_get_conn):
        r = test_client.get("/evidence/doesnotexist")

    assert r.status_code == 404
