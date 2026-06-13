from __future__ import annotations

import json

import pytest

from david.ingest.source_independence_ledger import (
    load_ledger,
    missing_pair_keys,
    structural_independence_summary,
)


def test_canonical_source_independence_ledger_loads_reviewed_pairs():
    ledger = load_ledger()

    assert ledger["reviewed_on"] == "2026-06-09"
    assert "balkan_insight_rss::who_news_rss" in ledger["pairs"]
    assert ledger["pairs"]["balkan_insight_rss::who_news_rss"]["independence_score"] == pytest.approx(0.95)


def test_structural_summary_fails_closed_on_missing_pair():
    ledger = {
        "reviewed_on": "2026-06-13",
        "next_review_due": "2099-01-01",
        "pairs": {
            "s1::s2": {"independence_score": 1.0},
            "s1::s3": {"independence_score": 1.0},
        },
    }

    summary = structural_independence_summary(["s1", "s2", "s3"], ledger=ledger)

    assert summary["gate_status"] == "fail"
    assert summary["reason"] == "source_independence_pairs_missing"
    assert summary["missing_pair_keys"] == ["s2::s3"]


def test_structural_summary_passes_with_complete_above_floor_ledger():
    ledger = {
        "reviewed_on": "2026-06-13",
        "next_review_due": "2099-01-01",
        "pairs": {
            "s1::s2": {"independence_score": 1.0},
            "s1::s3": {"independence_score": 1.0},
            "s2::s3": {"independence_score": 1.0},
        },
    }

    summary = structural_independence_summary(["s1", "s2", "s3"], ledger=ledger)

    assert summary["gate_status"] == "pass"
    assert summary["structural_s_eff"] == pytest.approx(3.0)
    assert summary["missing_pair_keys"] == []


def test_committed_source_ledger_json_is_valid():
    with open("config/source_independence_ledger.json") as fh:
        payload = json.load(fh)

    assert payload["next_review_due"] == "2026-09-09"
    assert missing_pair_keys(
        ["balkan_insight_rss", "health_policy_watch_rss", "who_news_rss"],
        ledger=payload,
    ) == []
