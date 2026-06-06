"""Tests for auto_adjudicate() and the full automatic ingest pipeline.

All tests are offline — DB calls are mocked.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from david.ingest.adjudicator_queue import (
    auto_adjudicate,
    minority_share,
    expected_information_gain,
    _gold_expand_sampled,
    QUEUE_PATH,
)


# ─── minority_share ──────────────────────────────────────────────────────────

def test_minority_share_unanimous_ones():
    assert minority_share([1, 1, 1]) == 0.0

def test_minority_share_unanimous_zeros():
    assert minority_share([0, 0, 0]) == 0.0

def test_minority_share_perfect_split():
    assert minority_share([1, 0]) == pytest.approx(0.5)

def test_minority_share_slight_disagreement():
    # 2 say 1, 1 says 0 → minority = 1/3
    assert minority_share([1, 1, 0]) == pytest.approx(1/3)

def test_minority_share_empty():
    assert minority_share([]) == 0.0

def test_minority_share_single():
    assert minority_share([1]) == 0.0
    assert minority_share([0]) == 0.0


# ─── EIG ─────────────────────────────────────────────────────────────────────

def test_eig_higher_for_more_disagreement():
    eig_disagree = expected_information_gain("e1", disagreement=0.5)
    eig_agree    = expected_information_gain("e1", disagreement=0.0)
    assert eig_disagree > eig_agree

def test_eig_higher_for_low_s_eff():
    eig_low  = expected_information_gain("e1", 0.3, stratum_S_eff=1)
    eig_high = expected_information_gain("e1", 0.3, stratum_S_eff=5)
    assert eig_low > eig_high


# ─── auto_adjudicate — DB mocked ─────────────────────────────────────────────

def _make_label_rows(specs: list[dict]) -> list[dict]:
    """Build fake label rows from a list of {eid, tactic, coder_id, label} dicts."""
    return [
        {
            "evidence_id": s["eid"],
            "tactic_k": s.get("tactic", "SIO"),
            "coder_id": s.get("coder", "c1"),
            "label": s["label"],
            "stratum_id": s.get("stratum", "al_media_freedom"),
        }
        for s in specs
    ]


def test_auto_adjudicate_unanimous_marks_adjudicated(tmp_path):
    """Items where all coders agree → adjudicated=TRUE, not queued."""
    rows = _make_label_rows([
        {"eid": "ev1", "label": 1, "coder": "c1"},
        {"eid": "ev1", "label": 1, "coder": "c2"},
        {"eid": "ev1", "label": 1, "coder": "c3"},
    ])
    marked = []

    with patch("david.db.repositories.get_unreviewed_labels", return_value=rows):
        with patch("david.db.repositories.mark_adjudicated", side_effect=lambda ids: (marked.extend(ids) or len(ids))):
            result = auto_adjudicate(queue_path=tmp_path / "q.json", threshold=0.30)

    assert result["gate_status"] == "pass"
    assert "ev1" in marked
    assert result["auto_adjudicated"] == 1
    assert result["queued_for_human"] == 0


def test_auto_adjudicate_split_queued(tmp_path):
    """Items with high disagreement → queued for human, NOT auto-adjudicated."""
    rows = _make_label_rows([
        {"eid": "ev2", "label": 1, "coder": "c1"},
        {"eid": "ev2", "label": 0, "coder": "c2"},   # perfect split
    ])

    with patch("david.db.repositories.get_unreviewed_labels", return_value=rows):
        with patch("david.db.repositories.mark_adjudicated", return_value=0) as mock_mark:
            result = auto_adjudicate(queue_path=tmp_path / "q.json", threshold=0.30)

    assert result["gate_status"] == "pass"
    assert result["queued_for_human"] == 1
    assert result["auto_adjudicated"] == 0
    mock_mark.assert_not_called()


def test_auto_adjudicate_mixed_batch(tmp_path):
    """ev1 unanimous → auto; ev2 split → queued."""
    rows = _make_label_rows([
        {"eid": "ev1", "label": 1, "coder": "c1"},
        {"eid": "ev1", "label": 1, "coder": "c2"},
        {"eid": "ev2", "label": 1, "coder": "c1"},
        {"eid": "ev2", "label": 0, "coder": "c2"},
    ])
    auto_ids: list[str] = []

    with patch("david.db.repositories.get_unreviewed_labels", return_value=rows):
        with patch("david.db.repositories.mark_adjudicated",
                   side_effect=lambda ids: (auto_ids.extend(ids) or len(ids))):
            result = auto_adjudicate(queue_path=tmp_path / "q.json", threshold=0.30)

    assert result["auto_adjudicated"] == 1
    assert result["queued_for_human"] == 1
    assert "ev1" in auto_ids
    assert "ev2" not in auto_ids


def test_auto_adjudicate_threshold_boundary(tmp_path):
    """minority_share == threshold → still auto-adjudicated (≤ not <)."""
    # 2 coders: 1 agrees, 1 disagrees → minority_share = 0.5
    # With threshold = 0.5 (edge): should be auto-adjudicated
    rows = _make_label_rows([
        {"eid": "ev3", "label": 1, "coder": "c1"},
        {"eid": "ev3", "label": 0, "coder": "c2"},
    ])
    auto_ids: list[str] = []

    with patch("david.db.repositories.get_unreviewed_labels", return_value=rows):
        with patch("david.db.repositories.mark_adjudicated",
                   side_effect=lambda ids: (auto_ids.extend(ids) or len(ids))):
            result = auto_adjudicate(queue_path=tmp_path / "q.json", threshold=0.50)

    assert "ev3" in auto_ids   # exactly at threshold → auto


def test_auto_adjudicate_no_labels(tmp_path):
    """No unreviewed labels → pass with zeros."""
    with patch("david.db.repositories.get_unreviewed_labels", return_value=[]):
        result = auto_adjudicate(queue_path=tmp_path / "q.json")

    assert result["gate_status"] == "pass"
    assert result["auto_adjudicated"] == 0
    assert result["queued_for_human"] == 0


def test_auto_adjudicate_db_unavailable(tmp_path):
    """DB failure → gate_status=fail with clear reason."""
    with patch("david.db.repositories.get_unreviewed_labels",
               side_effect=RuntimeError("no db")):
        result = auto_adjudicate(queue_path=tmp_path / "q.json")

    assert result["gate_status"] == "fail"
    assert result["reason"] == "db_unavailable"
    assert "no db" in result["detail"]


def test_auto_adjudicate_writes_queue_json(tmp_path):
    """Queue JSON file is written correctly."""
    rows = _make_label_rows([
        {"eid": "ev10", "label": 1, "coder": "c1"},
        {"eid": "ev10", "label": 0, "coder": "c2"},
    ])

    with patch("david.db.repositories.get_unreviewed_labels", return_value=rows):
        with patch("david.db.repositories.mark_adjudicated", return_value=0):
            queue_path = tmp_path / "q.json"
            auto_adjudicate(queue_path=queue_path, threshold=0.30)

    assert queue_path.exists()
    q = json.loads(queue_path.read_text())
    assert "generated_at" in q
    assert q["n_in_queue"] == 1
    assert q["items"][0]["evidence_id"] == "ev10"
    assert "minority_share" in q["items"][0]


def test_auto_adjudicate_multi_tactic_worst_case(tmp_path):
    """Uses worst tactic for escalation decision."""
    # ev5: SIO unanimous (0.0), MIO split (0.5) → should be QUEUED
    rows = [
        {"evidence_id": "ev5", "tactic_k": "SIO", "coder_id": "c1", "label": 1, "stratum_id": "al"},
        {"evidence_id": "ev5", "tactic_k": "SIO", "coder_id": "c2", "label": 1, "stratum_id": "al"},
        {"evidence_id": "ev5", "tactic_k": "MIO", "coder_id": "c1", "label": 1, "stratum_id": "al"},
        {"evidence_id": "ev5", "tactic_k": "MIO", "coder_id": "c2", "label": 0, "stratum_id": "al"},
    ]

    with patch("david.db.repositories.get_unreviewed_labels", return_value=rows):
        with patch("david.db.repositories.mark_adjudicated", return_value=0) as mock_mark:
            result = auto_adjudicate(queue_path=tmp_path / "q.json", threshold=0.30)

    mock_mark.assert_not_called()
    assert result["queued_for_human"] == 1


# ─── load_llm_pool ───────────────────────────────────────────────────────────

def test_load_llm_pool_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr("david.ingest.llm_coder.LLM_POOL_REGISTRY",
                        tmp_path / "llm_pool.json")
    from david.ingest.llm_coder import load_llm_pool
    assert load_llm_pool() == []


def test_load_llm_pool_parses_json(tmp_path, monkeypatch):
    pool_path = tmp_path / "llm_pool.json"
    pool_path.write_text(json.dumps([{
        "coder_id": "anthropic_haiku_001",
        "provider": "anthropic",
        "model": "claude-3-5-haiku-20241022",
        "prompt_template_id": "default",
        "seed": 1,
        "temperature": 0.0,
    }]))
    monkeypatch.setattr("david.ingest.llm_coder.LLM_POOL_REGISTRY", pool_path)
    from david.ingest.llm_coder import load_llm_pool
    pool = load_llm_pool()
    assert len(pool) == 1
    assert pool[0].provider == "anthropic"
    assert pool[0].coder_id == "anthropic_haiku_001"


# ─── config ──────────────────────────────────────────────────────────────────

def test_tactic_classes_in_config():
    from david.config import TACTIC_CLASSES
    assert len(TACTIC_CLASSES) >= 1
    assert all(isinstance(t, str) for t in TACTIC_CLASSES)


def test_llm_pool_registry_path_in_config():
    from david.config import LLM_POOL_REGISTRY
    assert "llm_pool" in str(LLM_POOL_REGISTRY)
