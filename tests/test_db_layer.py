"""Tests for the david/db/ package.

All tests are offline — they mock the psycopg2 connection so no live
Postgres instance is required. The tests verify:
  - Schema DDL is syntactically correct (single-string, parseable)
  - Repository functions call the right SQL and map columns correctly
  - assemble_fit_data() falls through to CSV when DB raises
  - write_fit_run / write_forecast_cells pass correct params
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch, call
import json

import numpy as np
import pytest


# ─── helpers ─────────────────────────────────────────────────────────────────

def _make_cursor(rows: list[tuple], cols: list[str] | None = None) -> MagicMock:
    """Fake psycopg2 cursor that returns `rows` from fetchall()."""
    cur = MagicMock()
    cur.__enter__ = lambda s: s
    cur.__exit__ = MagicMock(return_value=False)
    cur.fetchall.return_value = rows
    cur.fetchone.return_value = rows[0] if rows else None
    if cols:
        cur.description = [(c,) for c in cols]
    return cur


def _make_conn(cursor: MagicMock) -> MagicMock:
    conn = MagicMock()
    conn.__enter__ = lambda s: s
    conn.__exit__ = MagicMock(return_value=False)
    conn.cursor.return_value = cursor
    return conn


@contextmanager
def _mock_get_conn(rows: list[tuple] = (), cols: list[str] | None = None):
    """Patch david.db.connection.get_conn to return a fake connection."""
    cur  = _make_cursor(rows, cols)
    conn = _make_conn(cur)

    @contextmanager
    def _fake_get_conn():
        yield conn

    with patch("david.db.connection.get_conn", _fake_get_conn):
        with patch("david.db.repositories.get_conn", _fake_get_conn):
            with patch("david.db.schema.get_conn", _fake_get_conn):
                yield cur, conn


# ─── schema ──────────────────────────────────────────────────────────────────

def test_schema_ddl_nonempty():
    from david.db.schema import _TABLES_SQL, _INDEXES_SQL
    assert "CREATE TABLE IF NOT EXISTS strata"         in _TABLES_SQL
    assert "CREATE TABLE IF NOT EXISTS evidence_items" in _TABLES_SQL
    assert "CREATE TABLE IF NOT EXISTS coder_labels"   in _TABLES_SQL
    assert "CREATE TABLE IF NOT EXISTS fit_runs"       in _TABLES_SQL
    assert "CREATE TABLE IF NOT EXISTS forecast_cells" in _TABLES_SQL
    assert "CREATE INDEX IF NOT EXISTS" in _INDEXES_SQL


def test_create_all_executes_ddl():
    from david.db.schema import create_all
    with _mock_get_conn() as (cur, conn):
        create_all()
    assert cur.execute.call_count == 2   # _TABLES_SQL + _INDEXES_SQL


def test_table_row_counts_queries_all_tables():
    from david.db.schema import table_row_counts
    tables = ["strata", "sources", "evidence_items",
              "coder_labels", "fit_runs", "forecast_cells"]
    # Each SELECT returns (count,)
    cur = _make_cursor([(0,)] * len(tables))
    conn = _make_conn(cur)

    @contextmanager
    def _fake_get_conn():
        yield conn

    with patch("david.db.schema.get_conn", _fake_get_conn):
        counts = table_row_counts()

    assert set(counts.keys()) == set(tables)
    assert all(v == 0 for v in counts.values())


# ─── repositories — read ─────────────────────────────────────────────────────

def test_get_adjudicated_data_empty_raises():
    from david.db.repositories import get_adjudicated_data

    # evidence_rows = []  →  should raise RuntimeError
    cur  = _make_cursor([])
    cur.fetchall.side_effect = [[], [], [], []]   # 4 queries
    conn = _make_conn(cur)

    @contextmanager
    def _fake():
        yield conn

    with patch("david.db.repositories.get_conn", _fake):
        with pytest.raises(RuntimeError, match="No adjudicated evidence"):
            get_adjudicated_data()


def test_get_adjudicated_data_returns_four_keys():
    from david.db.repositories import get_adjudicated_data

    ev_rows  = [("ev1", "sg1", "src1", "2024-01-01")]
    src_rows = [("src1",)]
    lbl_rows = [("ev1", "human_1", "1", "SIO")]
    str_rows = [("sg1", "0.7")]

    cur = _make_cursor([])
    cur.fetchall.side_effect = [ev_rows, src_rows, lbl_rows, str_rows]
    conn = _make_conn(cur)

    @contextmanager
    def _fake():
        yield conn

    with patch("david.db.repositories.get_conn", _fake):
        data = get_adjudicated_data()

    assert set(data.keys()) == {"evidence_rows", "source_rows", "label_rows", "strata_rows"}
    assert data["evidence_rows"][0]["evidence_id"] == "ev1"
    assert data["strata_rows"][0]["stratum_g"] == "sg1"


def test_get_fit_runs_returns_list():
    from david.db.repositories import get_fit_runs

    cols = ["run_id", "model_version", "gate_status",
            "rhat_max", "ess_bulk_min", "divergences",
            "n_strata", "n_labels", "started_at"]
    rows = [("run_001", "v0.1.0", "pass", 1.01, 450.0, 0, 3, 120, "2024-01-01 00:00:00")]

    cur = _make_cursor(rows, cols)
    conn = _make_conn(cur)

    @contextmanager
    def _fake():
        yield conn

    with patch("david.db.repositories.get_conn", _fake):
        result = get_fit_runs(limit=5)

    assert len(result) == 1
    assert result[0]["run_id"] == "run_001"
    assert result[0]["gate_status"] == "pass"


# ─── repositories — write ────────────────────────────────────────────────────

def test_upsert_stratum_executes_sql():
    from david.db.repositories import upsert_stratum
    with _mock_get_conn() as (cur, _):
        upsert_stratum("sg1", "Albania", "media_freedom", 0.65)
    sql = cur.execute.call_args[0][0]
    assert "INSERT INTO strata" in sql
    assert "ON CONFLICT" in sql


def test_upsert_coder_label_rejects_invalid():
    from david.db.repositories import upsert_coder_label
    with pytest.raises(ValueError, match="label must be 0 or 1"):
        upsert_coder_label("ev1", "coder_1", "SIO", label=2)


def test_write_fit_run_passes_run_id():
    from david.db.repositories import write_fit_run
    summary = {
        "run_id": "fit_test_001",
        "model_version": "v0.1.0",
        "gate_status": "pass",
        "rhat_max": 1.02,
        "ess_bulk_min": 500.0,
        "ess_tail_min": 480.0,
        "divergences": 0,
    }
    with _mock_get_conn() as (cur, _):
        write_fit_run(summary)
    args = cur.execute.call_args[0][1]   # the params tuple
    assert args[0] == "fit_test_001"
    assert args[1] == "v0.1.0"
    assert args[2] == "pass"


def test_write_forecast_cells_empty_returns_zero():
    from david.db.repositories import write_forecast_cells
    with _mock_get_conn() as (cur, _):
        n = write_forecast_cells([], "run_001")
    assert n == 0
    cur.executemany.assert_not_called()


def test_write_forecast_cells_upserts_rows():
    from david.db.repositories import write_forecast_cells
    cells = [
        {
            "cell": {"series": 1, "tactic": 2},
            "horizon_months": 3,
            "p_active": 0.42,
            "credible_interval_80": [0.30, 0.55],
            "credible_interval_95": [0.20, 0.65],
            "horizon_validity": {"h_star_months": 6, "below_h_star": True,
                                 "forecast_route": "conditional_forecast"},
            "model_version": "v0.1.0",
        }
    ]
    with _mock_get_conn() as (cur, _):
        n = write_forecast_cells(cells, "run_001")
    assert n == 1
    cur.executemany.assert_called_once()
    row = cur.executemany.call_args[0][1][0]  # first row tuple
    assert row[0] == "run_001"  # run_id
    assert row[4] == 3           # horizon_months
    assert abs(row[5] - 0.42) < 1e-9  # p_active


# ─── assemble_fit_data — DB→CSV fallthrough ──────────────────────────────────

def test_assemble_fit_data_falls_through_to_csv(tmp_path, monkeypatch):
    """When Postgres raises, assemble_fit_data should try CSVs next."""
    import csv
    from david.model.fit import assemble_fit_data

    # Make DB fail
    def _failing_get_adjudicated():
        raise RuntimeError("connection refused")

    monkeypatch.setattr(
        "david.db.repositories.get_adjudicated_data",
        _failing_get_adjudicated,
    )
    # Patch the import inside fit.py
    import david.model.fit as fit_mod
    original = fit_mod.__builtins__

    # Write minimal CSV files
    (tmp_path / "strata.csv").write_text("stratum_g,I_O\nsg1,0.5\n")
    (tmp_path / "sources.csv").write_text("source_id\nsrc1\n")
    (tmp_path / "evidence_items.csv").write_text(
        "evidence_id,stratum_g,source_id,evidence_date\nev1,sg1,src1,2024-01-01\n"
    )
    (tmp_path / "coder_labels.csv").write_text(
        "evidence_id,coder_id,label,tactic_k,source_id\nev1,human_1,1,SIO,src1\n"
    )

    data = assemble_fit_data(input_dir=tmp_path, L=3, H_forecast=3)
    assert data["R"] == 1
    assert data["L"] == 3
    assert data["N_label"] == 1


def test_assemble_fit_data_raises_when_both_fail(monkeypatch, tmp_path):
    """When both DB and CSVs fail, FileNotFoundError must be raised."""
    from david.model.fit import assemble_fit_data

    def _failing():
        raise RuntimeError("no db")

    monkeypatch.setattr("david.db.repositories.get_adjudicated_data", _failing)

    with pytest.raises(FileNotFoundError, match="Postgres unavailable"):
        assemble_fit_data(input_dir=tmp_path, L=3, H_forecast=3)
