"""Read-only forecast HTTP server."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException

from ..config import FITS_DIR, FORECASTS_DIR


api = FastAPI(title="DAVID/M0.1 forecast API", version="0.1.0")


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


def _latest_forecast_dir() -> Path:
    candidates = sorted(p for p in FORECASTS_DIR.iterdir() if p.is_dir())
    if not candidates:
        raise HTTPException(status_code=404, detail="no forecasts yet")
    return candidates[-1]


def run(port: int = 8080) -> None:
    import uvicorn
    uvicorn.run(api, host="127.0.0.1", port=port)
