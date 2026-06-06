"""DDL for the DAVID Postgres schema.

Run `david db init` (or call create_all() directly) to create all tables
and indexes. All statements use IF NOT EXISTS so the command is idempotent.

Table overview
--------------
strata          Country × policy pairs. Each row is one "series" in Stan.
sources         Source families (news API, legislative feed, …).
evidence_items  One row per piece of collected evidence.
coder_labels    Y_{e,m,k} — one row per (evidence, coder, tactic) annotation.
fit_runs        Metadata + gate results for each `david fit` run.
forecast_cells  Per-cell posterior summaries emitted by `david forecast`.
"""

from __future__ import annotations

from .connection import get_conn

# ─── tables ──────────────────────────────────────────────────────────────────

_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS strata (
    stratum_id    TEXT PRIMARY KEY,
    country       TEXT NOT NULL,
    policy        TEXT NOT NULL,
    observability REAL NOT NULL DEFAULT 0.5,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sources (
    source_id        TEXT PRIMARY KEY,
    family           TEXT NOT NULL,
    ingest_kind      TEXT NOT NULL DEFAULT 'manual',
    endpoint         TEXT,
    enabled          BOOLEAN NOT NULL DEFAULT TRUE,
    rho_prior_mean   REAL NOT NULL DEFAULT 0.7,
    delta_prior_mean REAL NOT NULL DEFAULT 0.05,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS evidence_items (
    evidence_id   TEXT PRIMARY KEY,
    stratum_id    TEXT NOT NULL REFERENCES strata(stratum_id),
    source_id     TEXT NOT NULL REFERENCES sources(source_id),
    evidence_date DATE NOT NULL,
    title         TEXT,
    url           TEXT,
    text_content  TEXT,
    fetched_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    adjudicated   BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS coder_labels (
    id          BIGSERIAL PRIMARY KEY,
    evidence_id TEXT    NOT NULL REFERENCES evidence_items(evidence_id),
    coder_id    TEXT    NOT NULL,
    tactic_k    TEXT    NOT NULL,
    label       SMALLINT NOT NULL CHECK (label IN (0, 1)),
    coded_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (evidence_id, coder_id, tactic_k)
);

CREATE TABLE IF NOT EXISTS fit_runs (
    run_id        TEXT PRIMARY KEY,
    model_version TEXT NOT NULL,
    gate_status   TEXT NOT NULL,
    rhat_max      REAL,
    ess_bulk_min  REAL,
    ess_tail_min  REAL,
    divergences   INTEGER,
    n_strata      INTEGER,
    n_labels      INTEGER,
    summary_json  JSONB,
    started_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS forecast_cells (
    id             BIGSERIAL PRIMARY KEY,
    run_id         TEXT    NOT NULL REFERENCES fit_runs(run_id),
    stratum_id     TEXT,
    series         INTEGER NOT NULL,
    tactic         INTEGER NOT NULL,
    horizon_months INTEGER NOT NULL,
    p_active       REAL    NOT NULL,
    ci_80_lo       REAL,
    ci_80_hi       REAL,
    ci_95_lo       REAL,
    ci_95_hi       REAL,
    h_star_months  INTEGER,
    below_h_star   BOOLEAN,
    forecast_route TEXT,
    model_version  TEXT,
    emitted_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (run_id, series, tactic, horizon_months)
);
"""

# ─── indexes ─────────────────────────────────────────────────────────────────

_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_evidence_stratum  ON evidence_items(stratum_id);
CREATE INDEX IF NOT EXISTS idx_evidence_source   ON evidence_items(source_id);
CREATE INDEX IF NOT EXISTS idx_evidence_date     ON evidence_items(evidence_date);
CREATE INDEX IF NOT EXISTS idx_evidence_adjud    ON evidence_items(adjudicated);
CREATE INDEX IF NOT EXISTS idx_labels_evidence   ON coder_labels(evidence_id);
CREATE INDEX IF NOT EXISTS idx_labels_tactic     ON coder_labels(tactic_k);
CREATE INDEX IF NOT EXISTS idx_labels_coder      ON coder_labels(coder_id);
CREATE INDEX IF NOT EXISTS idx_cells_run         ON forecast_cells(run_id);
CREATE INDEX IF NOT EXISTS idx_cells_horizon     ON forecast_cells(horizon_months);
CREATE INDEX IF NOT EXISTS idx_cells_route       ON forecast_cells(forecast_route);
CREATE INDEX IF NOT EXISTS idx_runs_status       ON fit_runs(gate_status);
CREATE INDEX IF NOT EXISTS idx_runs_started      ON fit_runs(started_at DESC);
"""

# ─── public API ──────────────────────────────────────────────────────────────

def create_all() -> None:
    """Create all tables and indexes (idempotent — safe to run repeatedly)."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(_TABLES_SQL)
            cur.execute(_INDEXES_SQL)


def table_row_counts() -> dict[str, int]:
    """Return {table_name: row_count} for all DAVID tables."""
    tables = ["strata", "sources", "evidence_items",
              "coder_labels", "fit_runs", "forecast_cells"]
    counts: dict[str, int] = {}
    with get_conn() as conn:
        with conn.cursor() as cur:
            for t in tables:
                cur.execute(f"SELECT COUNT(*) FROM {t}")
                row = cur.fetchone()
                counts[t] = int(row[0]) if row else 0
    return counts
