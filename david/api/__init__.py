"""Read-only API for forecast consumption.

server.py exposes:
  GET  /healthz                                    health check
  GET  /forecasts/latest?horizon=6                 latest run
  GET  /forecasts/{run_id}/cells_h{h}.json         specific run
  GET  /forecasts/{run_id}/route_ledger.json
  GET  /forecasts/{run_id}/falsification_ledger.json

No writes. No human-action endpoints. Operators and external systems
consume forecasts read-only; the source of truth is the JSON ledger.
"""
