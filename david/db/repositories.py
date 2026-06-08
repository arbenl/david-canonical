"""Data-access functions for the DAVID Postgres schema.

Each function operates through the get_conn() context manager and uses
plain psycopg2 — no ORM. All writes use ON CONFLICT DO UPDATE (upsert)
so they are idempotent and safe to retry.

Reading interface
-----------------
get_adjudicated_data()     → rows compatible with assemble_fit_data()
get_fit_runs(limit)        → recent fit runs for CLI status
get_forecast_cells(run_id) → forecast cells for a given run

Writing interface
-----------------
upsert_stratum(...)        → insert/update strata row
upsert_source(...)         → insert/update sources row
upsert_evidence_item(...)  → insert/update evidence_items row
upsert_coder_label(...)    → insert/update coder_labels row
write_fit_run(summary)     → write fit_summary dict to fit_runs
write_forecast_cells(cells, run_id) → bulk-upsert forecast_cells
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from .connection import get_conn

# ─── read ────────────────────────────────────────────────────────────────────

def get_adjudicated_data() -> dict[str, list[dict[str, str]]]:
    """Return adjudicated rows in the same shape that assemble_fit_data() expects.

    Returns a dict with four keys mirroring the four CSVs:
        evidence_rows  — [{evidence_id, stratum_g, source_id, evidence_date}, …]
        source_rows    — [{source_id}, …]
        label_rows     — [{evidence_id, coder_id, label, tactic_k}, …]
        strata_rows    — [{stratum_g, I_O}, …]

    Raises psycopg2.Error if the DB is unreachable or the schema is absent.
    Raises RuntimeError if no adjudicated evidence exists.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT evidence_id, stratum_id AS stratum_g,
                       source_id, evidence_date::text
                FROM   evidence_items
                WHERE  adjudicated = TRUE
                ORDER  BY stratum_id, evidence_date, evidence_id
            """)
            evidence_rows = [
                {"evidence_id": r[0], "stratum_g": r[1],
                 "source_id": r[2], "evidence_date": r[3]}
                for r in cur.fetchall()
            ]

            cur.execute("SELECT DISTINCT source_id FROM evidence_items ORDER BY source_id")
            source_rows = [{"source_id": r[0]} for r in cur.fetchall()]

            cur.execute("""
                SELECT cl.evidence_id, cl.coder_id,
                       cl.label::text, cl.tactic_k
                FROM   coder_labels cl
                JOIN   evidence_items ei USING (evidence_id)
                WHERE  ei.adjudicated = TRUE
                ORDER  BY cl.evidence_id, cl.coder_id, cl.tactic_k
            """)
            label_rows = [
                {"evidence_id": r[0], "coder_id": r[1],
                 "label": r[2], "tactic_k": r[3]}
                for r in cur.fetchall()
            ]

            cur.execute("""
                SELECT stratum_id AS stratum_g,
                       observability::text AS "I_O"
                FROM   strata
                ORDER  BY stratum_id
            """)
            strata_rows = [{"stratum_g": r[0], "I_O": r[1]} for r in cur.fetchall()]

    if not evidence_rows:
        raise RuntimeError(
            "No adjudicated evidence in the database. "
            "Mark evidence_items.adjudicated = TRUE to include rows in the fit."
        )

    return {
        "evidence_rows": evidence_rows,
        "source_rows": source_rows,
        "label_rows": label_rows,
        "strata_rows": strata_rows,
    }


def get_fit_runs(limit: int = 10) -> list[dict[str, Any]]:
    """Return the most recent fit runs (newest first)."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT run_id, model_version, gate_status,
                       rhat_max, ess_bulk_min, divergences,
                       n_strata, n_labels, started_at
                FROM   fit_runs
                ORDER  BY started_at DESC
                LIMIT  %s
            """, (limit,))
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


def get_forecast_cells(
    run_id: str,
    horizon_months: int | None = None,
) -> list[dict[str, Any]]:
    """Return forecast cells for a given run, optionally filtered by horizon."""
    sql = """
        SELECT series, tactic, horizon_months, p_active,
               ci_80_lo, ci_80_hi, ci_95_lo, ci_95_hi,
               h_star_months, below_h_star, forecast_route,
               stratum_id, emitted_at
        FROM   forecast_cells
        WHERE  run_id = %s
    """
    params: list[Any] = [run_id]
    if horizon_months is not None:
        sql += " AND horizon_months = %s"
        params.append(horizon_months)
    sql += " ORDER BY series, tactic, horizon_months"
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


# ─── write ───────────────────────────────────────────────────────────────────

def upsert_stratum(
    stratum_id: str,
    country: str,
    policy: str,
    observability: float = 0.5,
) -> None:
    """Insert or update a stratum record."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO strata (stratum_id, country, policy, observability)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (stratum_id) DO UPDATE
                    SET country = EXCLUDED.country,
                        policy  = EXCLUDED.policy,
                        observability = EXCLUDED.observability
            """, (stratum_id, country, policy, observability))


def upsert_source(
    source_id: str,
    family: str,
    ingest_kind: str = "manual",
    endpoint: str | None = None,
    enabled: bool = True,
    rho_prior_mean: float = 0.7,
    delta_prior_mean: float = 0.05,
) -> None:
    """Insert or update a source record."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO sources
                    (source_id, family, ingest_kind, endpoint, enabled,
                     rho_prior_mean, delta_prior_mean)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (source_id) DO UPDATE
                    SET family           = EXCLUDED.family,
                        ingest_kind      = EXCLUDED.ingest_kind,
                        endpoint         = EXCLUDED.endpoint,
                        enabled          = EXCLUDED.enabled,
                        rho_prior_mean   = EXCLUDED.rho_prior_mean,
                        delta_prior_mean = EXCLUDED.delta_prior_mean
            """, (source_id, family, ingest_kind, endpoint, enabled,
                  rho_prior_mean, delta_prior_mean))


def upsert_evidence_item(
    evidence_id: str,
    stratum_id: str,
    source_id: str,
    evidence_date: date | str,
    title: str | None = None,
    url: str | None = None,
    text_content: str | None = None,
    adjudicated: bool = False,
) -> None:
    """Insert or update an evidence item."""
    if isinstance(evidence_date, date):
        evidence_date = evidence_date.isoformat()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO evidence_items
                    (evidence_id, stratum_id, source_id, evidence_date,
                     title, url, text_content, adjudicated)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (evidence_id) DO UPDATE
                    SET stratum_id    = EXCLUDED.stratum_id,
                        source_id     = EXCLUDED.source_id,
                        evidence_date = EXCLUDED.evidence_date,
                        title         = EXCLUDED.title,
                        url           = EXCLUDED.url,
                        text_content  = EXCLUDED.text_content
                        -- adjudicated is intentionally NOT updated:
                        -- auto_adjudicate() is the only writer of TRUE values;
                        -- ingest must never reset items back to FALSE.
            """, (evidence_id, stratum_id, source_id, evidence_date,
                  title, url, text_content, adjudicated))


def upsert_coder_label(
    evidence_id: str,
    coder_id: str,
    tactic_k: str,
    label: int,
) -> None:
    """Insert or update a single coder label (Y_{e,m,k})."""
    if label not in (0, 1):
        raise ValueError(f"label must be 0 or 1, got {label!r}")
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO coder_labels (evidence_id, coder_id, tactic_k, label)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (evidence_id, coder_id, tactic_k) DO UPDATE
                    SET label    = EXCLUDED.label,
                        coded_at = NOW()
            """, (evidence_id, coder_id, tactic_k, label))


def write_fit_run(fit_summary: dict[str, Any]) -> None:
    """Write a fit_summary dict to fit_runs (upsert on run_id).

    Extracts scalar fields; stores the full summary in summary_json.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO fit_runs
                    (run_id, model_version, gate_status,
                     rhat_max, ess_bulk_min, ess_tail_min,
                     divergences, n_strata, n_labels, summary_json)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id) DO UPDATE
                    SET gate_status   = EXCLUDED.gate_status,
                        rhat_max      = EXCLUDED.rhat_max,
                        ess_bulk_min  = EXCLUDED.ess_bulk_min,
                        ess_tail_min  = EXCLUDED.ess_tail_min,
                        divergences   = EXCLUDED.divergences,
                        summary_json  = EXCLUDED.summary_json
            """, (
                fit_summary.get("run_id"),
                fit_summary.get("model_version"),
                fit_summary.get("gate_status"),
                fit_summary.get("rhat_max"),
                fit_summary.get("ess_bulk_min"),
                fit_summary.get("ess_tail_min"),
                fit_summary.get("divergences"),
                fit_summary.get("n_strata"),
                fit_summary.get("n_labels"),
                json.dumps(fit_summary),
            ))


def mark_adjudicated(evidence_ids: list[str]) -> int:
    """Set adjudicated=TRUE for a batch of evidence_ids. Returns row count updated."""
    if not evidence_ids:
        return 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE evidence_items
                SET    adjudicated = TRUE
                WHERE  evidence_id = ANY(%s)
                  AND  adjudicated = FALSE
            """, (evidence_ids,))
            return cur.rowcount


def get_unreviewed_labels() -> list[dict]:
    """Return coder labels for evidence items not yet adjudicated.

    Returns list of {evidence_id, tactic_k, coder_id, label} rows.
    Used by auto_adjudicate() to decide which items can be auto-resolved.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT cl.evidence_id, cl.tactic_k, cl.coder_id, cl.label,
                       ei.stratum_id
                FROM   coder_labels cl
                JOIN   evidence_items ei USING (evidence_id)
                WHERE  ei.adjudicated = FALSE
                ORDER  BY cl.evidence_id, cl.tactic_k, cl.coder_id
            """)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


def write_forecast_cells(
    cells: list[dict[str, Any]],
    run_id: str,
) -> int:
    """Bulk-upsert forecast cell records; returns number of rows written."""
    if not cells:
        return 0
    rows = []
    for c in cells:
        ci80 = c.get("credible_interval_80", [None, None])
        ci95 = c.get("credible_interval_95", [None, None])
        hv   = c.get("horizon_validity", {})
        cell = c.get("cell", {})
        rows.append((
            run_id,
            cell.get("stratum_id") or cell.get("country"),  # best-effort stratum_id
            cell.get("series"),
            cell.get("tactic"),
            c.get("horizon_months"),
            c.get("p_active"),
            ci80[0] if len(ci80) > 0 else None,
            ci80[1] if len(ci80) > 1 else None,
            ci95[0] if len(ci95) > 0 else None,
            ci95[1] if len(ci95) > 1 else None,
            hv.get("h_star_months"),
            hv.get("below_h_star"),
            hv.get("forecast_route") or c.get("forecast_route"),
            c.get("model_version"),
        ))
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany("""
                INSERT INTO forecast_cells
                    (run_id, stratum_id, series, tactic, horizon_months,
                     p_active, ci_80_lo, ci_80_hi, ci_95_lo, ci_95_hi,
                     h_star_months, below_h_star, forecast_route, model_version)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id, series, tactic, horizon_months) DO UPDATE
                    SET p_active       = EXCLUDED.p_active,
                        ci_80_lo       = EXCLUDED.ci_80_lo,
                        ci_80_hi       = EXCLUDED.ci_80_hi,
                        ci_95_lo       = EXCLUDED.ci_95_lo,
                        ci_95_hi       = EXCLUDED.ci_95_hi,
                        h_star_months  = EXCLUDED.h_star_months,
                        below_h_star   = EXCLUDED.below_h_star,
                        forecast_route = EXCLUDED.forecast_route,
                        model_version  = EXCLUDED.model_version,
                        emitted_at     = NOW()
            """, rows)
    return len(rows)
