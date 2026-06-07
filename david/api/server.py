"""DAVID/M0.1 — Read-only forecast + evidence HTTP API.

File-based endpoints (existing):
    GET /healthz
    GET /forecasts/latest?horizon=6
    GET /forecasts/{run_id}/cells_h{horizon}.json
    GET /forecasts/{run_id}/route_ledger.json
    GET /forecasts/{run_id}/falsification_ledger.json

Database-backed endpoints (new):
    GET /strata
    GET /sources
    GET /evidence?stratum_id=&adjudicated=&limit=100
    GET /evidence/{evidence_id}
    GET /fit-runs?limit=10
    GET /forecast-cells?run_id=&horizon_months=

All endpoints return JSON. DB endpoints fall back to a 503 with a clear
message if Postgres is unreachable, so the file-based routes continue to work
without a live database.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..config import FITS_DIR, FORECASTS_DIR

api = FastAPI(
    title="DAVID/M0.1 forecast API",
    version="0.1.0",
    description="Read-only API for DAVID/M0.1 forecasts and evidence.",
)

# CORS — allow Vercel frontend + local dev
_ALLOWED_ORIGINS = [o.strip() for o in os.environ.get(
    "ALLOWED_ORIGINS",
    "https://david-x8lj.vercel.app,http://localhost:3001,http://localhost:3000",
).split(",") if o.strip()]

api.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─── helpers ─────────────────────────────────────────────────────────────────

@api.get("/queue")
def get_queue() -> Any:
    """Current adjudicator queue (items needing human review)."""
    from ..config import DATA_ROOT
    q_path = DATA_ROOT / "adjudicator_queue.json"
    if not q_path.exists():
        return {"n_in_queue": 0, "items": [], "generated_at": None}
    return json.loads(q_path.read_text())


@api.get("/fit-runs/{run_id}")
def get_fit_run(run_id: str) -> Any:
    """Full fit summary for a specific run (theorems, gates, MCMC diagnostics)."""
    summary_path = FITS_DIR / run_id / "fit_summary.json"
    if not summary_path.exists():
        raise HTTPException(status_code=404, detail=f"run_id={run_id!r} not found")
    return json.loads(summary_path.read_text())


@api.get("/sbc")
def get_sbc() -> Any:
    """Latest SBC results — measurement + forecast layers."""
    sbc_path          = FITS_DIR / "sbc" / "sbc_summary.json"
    forecast_sbc_path = FITS_DIR / "forecast_sbc" / "forecast_sbc_summary.json"
    result: dict[str, Any] = {}
    if sbc_path.exists():
        result["measurement"] = json.loads(sbc_path.read_text())
    if forecast_sbc_path.exists():
        result["forecast"] = json.loads(forecast_sbc_path.read_text())
    if not result:
        raise HTTPException(status_code=404,
                            detail="No SBC results yet. Run `david sbc` first.")
    return result


@api.get("/config")
def get_config() -> dict:
    """Pre-registered configuration values (thresholds, tactic classes, model version)."""
    from ..config import (
        ID_DISTANCE_FLOOR, INFORMATIVENESS_FLOOR_LOWER_95,
        POSTERIOR_FDP_DEFAULT_Q, HORIZON_PRIOR_DRIFT_TAU,
        ADJUDICATOR_DISAGREEMENT_THRESHOLD, GOLD_STANDARD_SAMPLE_RATE,
        R_HAT_MAX, BULK_ESS_MIN, TAIL_ESS_MIN, DIVERGENCES_ALLOWED,
        SBC_KS_ALPHA, SBC_BONFERRONI, FORECAST_HORIZONS_MONTHS,
        TACTIC_CLASSES, MODEL_VERSION,
    )
    return {
        "model_version": MODEL_VERSION,
        "tactic_classes": list(TACTIC_CLASSES),
        "forecast_horizons_months": list(FORECAST_HORIZONS_MONTHS),
        "theorem_floors": {
            "A_prime_id_distance": ID_DISTANCE_FLOOR,
            "B_prime_informativeness_lower_95": INFORMATIVENESS_FLOOR_LOWER_95,
            "C_prime_posterior_fdp": POSTERIOR_FDP_DEFAULT_Q,
            "D_prime_horizon_drift_tau": HORIZON_PRIOR_DRIFT_TAU,
        },
        "fit_thresholds": {
            "rhat_max": R_HAT_MAX,
            "bulk_ess_min": BULK_ESS_MIN,
            "tail_ess_min": TAIL_ESS_MIN,
            "divergences_allowed": DIVERGENCES_ALLOWED,
        },
        "sbc": {
            "ks_alpha": SBC_KS_ALPHA,
            "bonferroni": SBC_BONFERRONI,
        },
        "human_loop": {
            "disagreement_threshold": ADJUDICATOR_DISAGREEMENT_THRESHOLD,
            "gold_sample_rate": GOLD_STANDARD_SAMPLE_RATE,
        },
    }


def _latest_forecast_dir() -> Path:
    if not FORECASTS_DIR.exists():
        raise HTTPException(status_code=404, detail="No forecasts yet")
    candidates = sorted(p for p in FORECASTS_DIR.iterdir() if p.is_dir())
    if not candidates:
        raise HTTPException(status_code=404, detail="No forecasts yet")
    return candidates[-1]


def _db_error_503(exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "error": "database_unavailable",
            "detail": str(exc),
            "hint": "Start Postgres with `docker compose up -d` then `david db init`",
        },
    )


# ─── file-based routes (preserved) ───────────────────────────────────────────

@api.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@api.get("/forecasts/latest")
def latest(horizon: int = 6) -> dict:
    run = _latest_forecast_dir()
    path = run / f"cells_h{horizon:02d}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"no forecast for h={horizon}")
    return {"run_id": run.name, "horizon": horizon, "cells": json.loads(path.read_text())}


@api.get("/forecasts/{run_id}/cells_h{horizon}.json")
def cells(run_id: str, horizon: int) -> dict:
    path = FORECASTS_DIR / run_id / f"cells_h{horizon:02d}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="not found")
    return {"run_id": run_id, "horizon": horizon, "cells": json.loads(path.read_text())}


@api.get("/forecasts/{run_id}/route_ledger.json")
def route_ledger(run_id: str) -> dict:
    path = FORECASTS_DIR / run_id / "route_ledger.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="not found")
    return json.loads(path.read_text())


@api.get("/forecasts/{run_id}/falsification_ledger.json")
def falsification_ledger(run_id: str) -> dict:
    path = FORECASTS_DIR / run_id / "falsification_ledger.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="not found")
    return json.loads(path.read_text())


# ─── database-backed routes ───────────────────────────────────────────────────

@api.get("/strata")
def list_strata() -> Any:
    """All strata with their observability score and evidence counts."""
    try:
        from ..db.connection import get_conn
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT s.stratum_id, s.country, s.policy, s.observability,
                           COUNT(e.evidence_id)  AS n_evidence,
                           COUNT(e.evidence_id) FILTER (WHERE e.adjudicated)
                                                 AS n_adjudicated
                    FROM   strata s
                    LEFT JOIN evidence_items e USING (stratum_id)
                    GROUP BY s.stratum_id, s.country, s.policy, s.observability
                    ORDER BY s.stratum_id
                """)
                cols = [d[0] for d in cur.description]
                rows = [dict(zip(cols, row)) for row in cur.fetchall()]
        return {"strata": rows, "n": len(rows)}
    except Exception as exc:
        return _db_error_503(exc)


@api.get("/sources")
def list_sources() -> Any:
    """All sources in the database."""
    try:
        from ..db.connection import get_conn
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT source_id, family, ingest_kind, endpoint,
                           enabled, rho_prior_mean, delta_prior_mean
                    FROM   sources
                    ORDER  BY source_id
                """)
                cols = [d[0] for d in cur.description]
                rows = [dict(zip(cols, row)) for row in cur.fetchall()]
        return {"sources": rows, "n": len(rows)}
    except Exception as exc:
        return _db_error_503(exc)


@api.get("/evidence")
def list_evidence(
    stratum_id: Optional[str] = Query(None, description="Filter by stratum_id"),
    adjudicated: Optional[bool] = Query(None, description="Filter by adjudicated flag"),
    source_id: Optional[str] = Query(None, description="Filter by source_id"),
    limit: int = Query(100, ge=1, le=5000, description="Max rows to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
) -> Any:
    """List evidence items with optional filters."""
    try:
        from ..db.connection import get_conn
        conditions: list[str] = []
        params: list[Any] = []
        if stratum_id is not None:
            conditions.append("stratum_id = %s")
            params.append(stratum_id)
        if adjudicated is not None:
            conditions.append("adjudicated = %s")
            params.append(adjudicated)
        if source_id is not None:
            conditions.append("source_id = %s")
            params.append(source_id)
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        params += [limit, offset]

        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT evidence_id, stratum_id, source_id,
                           evidence_date::text, title, url, adjudicated
                    FROM   evidence_items
                    {where}
                    ORDER  BY evidence_date DESC, evidence_id
                    LIMIT  %s OFFSET %s
                """, params)
                cols = [d[0] for d in cur.description]
                rows = [dict(zip(cols, row)) for row in cur.fetchall()]

                # Total count (without pagination)
                cur.execute(
                    f"SELECT COUNT(*) FROM evidence_items {where}",
                    params[:-2],
                )
                total = cur.fetchone()[0]

        return {"evidence": rows, "n": len(rows), "total": total,
                "limit": limit, "offset": offset}
    except Exception as exc:
        return _db_error_503(exc)


@api.get("/evidence/{evidence_id}")
def get_evidence(evidence_id: str) -> Any:
    """Single evidence item with all coder labels."""
    try:
        from ..db.connection import get_conn
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT evidence_id, stratum_id, source_id,
                           evidence_date::text, title, url,
                           text_content, adjudicated
                    FROM   evidence_items
                    WHERE  evidence_id = %s
                """, (evidence_id,))
                row = cur.fetchone()
                if row is None:
                    raise HTTPException(status_code=404,
                                        detail=f"evidence_id={evidence_id!r} not found")
                cols = [d[0] for d in cur.description]
                item = dict(zip(cols, row))

                cur.execute("""
                    SELECT coder_id, tactic_k, label, coded_at::text
                    FROM   coder_labels
                    WHERE  evidence_id = %s
                    ORDER  BY coder_id, tactic_k
                """, (evidence_id,))
                lbl_cols = [d[0] for d in cur.description]
                labels = [dict(zip(lbl_cols, r)) for r in cur.fetchall()]

        return {"evidence": item, "labels": labels, "n_labels": len(labels)}
    except HTTPException:
        raise
    except Exception as exc:
        return _db_error_503(exc)


@api.get("/fit-runs")
def list_fit_runs(
    limit: int = Query(10, ge=1, le=200, description="Max rows to return"),
) -> Any:
    """Recent fit runs (newest first)."""
    try:
        from ..db.repositories import get_fit_runs
        runs = get_fit_runs(limit=limit)
        return {"fit_runs": runs, "n": len(runs)}
    except Exception as exc:
        return _db_error_503(exc)


@api.get("/forecast-cells")
def list_forecast_cells(
    run_id: Optional[str] = Query(None, description="Fit run ID; latest if omitted"),
    horizon_months: Optional[int] = Query(None, description="Filter by horizon"),
) -> Any:
    """Forecast cells from the database."""
    try:
        from ..db.repositories import get_forecast_cells

        if run_id is None:
            # Resolve latest run_id from fit_runs table
            from ..db.repositories import get_fit_runs
            runs = get_fit_runs(limit=1)
            if not runs:
                return {"forecast_cells": [], "n": 0,
                        "detail": "No fit runs in database. Run `david fit` first."}
            run_id = runs[0]["run_id"]

        cells = get_forecast_cells(run_id=run_id, horizon_months=horizon_months)
        return {"run_id": run_id, "forecast_cells": cells, "n": len(cells)}
    except Exception as exc:
        return _db_error_503(exc)


# ─── pipeline trigger endpoints ──────────────────────────────────────────────

_PIPELINE_SECRET = os.environ.get("PIPELINE_SECRET", "")
_pipeline_running = False   # simple in-process guard (single replica)
_pipeline_step: str = ""    # human-readable current step for UI
_pipeline_started_at: Optional[str] = None


def _auth_check(authorization: str) -> None:
    """Raise 401 if PIPELINE_SECRET is set and header doesn't match."""
    if _PIPELINE_SECRET and authorization != f"Bearer {_PIPELINE_SECRET}":
        raise HTTPException(status_code=401, detail="Unauthorized")


def _run_pipeline_bg(since_days: int = 7, fit_only: bool = False) -> None:
    """Run (ingest →) fit in background. Guarded against concurrent runs."""
    global _pipeline_running, _pipeline_step, _pipeline_started_at
    if _pipeline_running:
        return
    _pipeline_running = True
    import subprocess
    import datetime
    from datetime import date, timedelta
    _pipeline_started_at = datetime.datetime.utcnow().isoformat() + "Z"
    try:
        if not fit_only:
            _pipeline_step = "ingesting"
            since = (date.today() - timedelta(days=since_days)).isoformat()
            subprocess.run(["david", "ingest", "--since", since],
                           check=True, capture_output=True, text=True)
        _pipeline_step = "fitting"
        subprocess.run(["david", "fit"],
                       capture_output=True, text=True)   # non-fatal if fit fails
        _pipeline_step = "done"
    except Exception as exc:
        _pipeline_step = f"error: {exc}"
        print(f"[pipeline] error: {exc}", flush=True)
    finally:
        _pipeline_running = False


@api.post("/internal/pipeline/run")
async def trigger_pipeline(
    background_tasks: BackgroundTasks,
    since_days: int = Query(7, ge=1, le=365),
    fit_only: bool = Query(False, description="Skip ingest; run only `david fit`"),
    authorization: str = Header(default=""),
) -> dict:
    """Trigger ingest → fit pipeline in the background.

    Protected by PIPELINE_SECRET env var (Bearer token).
    If PIPELINE_SECRET is empty the endpoint is open (dev mode).
    """
    _auth_check(authorization)
    if _pipeline_running:
        return {"status": "already_running",
                "message": "Pipeline is already running, try again later.",
                "step": _pipeline_step}
    background_tasks.add_task(_run_pipeline_bg, since_days, fit_only)
    if fit_only:
        return {"status": "queued", "message": "Fit-only job started in background."}
    return {"status": "queued",
            "message": f"Ingest (last {since_days}d) + fit started in background."}


@api.get("/internal/pipeline/status")
async def pipeline_status() -> dict:
    """Live pipeline status + evidence counts + last fit result."""
    import datetime
    base: dict = {
        "running": _pipeline_running,
        "step": _pipeline_step,
        "started_at": _pipeline_started_at,
        "polled_at": datetime.datetime.utcnow().isoformat() + "Z",
    }
    try:
        from ..db.repositories import get_fit_runs
        from ..db.connection import get_conn
        runs = get_fit_runs(limit=1)
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM evidence_items")
                n_evidence = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM evidence_items WHERE adjudicated")
                n_adjudicated = cur.fetchone()[0]
        return {**base, "last_fit": runs[0] if runs else None,
                "n_evidence": n_evidence, "n_adjudicated": n_adjudicated}
    except Exception as exc:
        return {**base, "last_fit": None, "error": str(exc)}


# ─── server entry point ───────────────────────────────────────────────────────

def run(port: int | None = None) -> None:
    import uvicorn
    _port = port or int(os.environ.get("PORT", 8080))
    _host = "0.0.0.0" if os.environ.get("RAILWAY_ENVIRONMENT") else "127.0.0.1"
    uvicorn.run(api, host=_host, port=_port)
