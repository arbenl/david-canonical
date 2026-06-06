"""DAVID/M0.1 canonical CLI.

Single entry point. Each subcommand writes a typed JSON result and exits with
0 on pass, 2 on fail-closed, 1 on system error.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import typer
from rich.console import Console

from . import config
from .engine import orchestrator
from .engine.forecast import emit_forecasts
from .engine.router import apply_forecast_routing
from .ingest.adjudicator_queue import build_queue
from .ingest.llm_coder import calibrate_coders
from .ingest.normalize import normalize_raw
from .ingest.sources import run_scrapers
from .model.fit import run_fit
from .simulator.adversarial_battery import run_battery
from .simulator.forecast_sbc import run_forecast_sbc
from .simulator.sbc import run_sbc
from .validation.falsification import run_falsification

app = typer.Typer(
    no_args_is_help=True,
    help="DAVID/M0.1 canonical CLI. See docs/ARCHITECTURE.md.",
)
db_app = typer.Typer(no_args_is_help=True, help="Postgres database commands.")
app.add_typer(db_app, name="db")
console = Console()


@app.command()
def ingest(
    since: str = typer.Option(None, help="ISO date; default = last cycle"),
    until: str = typer.Option(None, help="ISO date; default = today"),
) -> None:
    """Run scrapers, normalize, dispatch LLM coders, refresh adjudicator queue."""
    since_d = date.fromisoformat(since) if since else None
    until_d = date.fromisoformat(until) if until else date.today()
    raw_paths = run_scrapers(since=since_d, until=until_d)
    normalized = normalize_raw(raw_paths)
    queue = build_queue(normalized)
    console.print(f"[green]ingest complete[/]: {len(normalized)} items, {len(queue)} queued")


@app.command("calibrate-coders")
def calibrate_coders_cmd() -> None:
    """Fit coder_calibration.stan against gold-standard set."""
    result = calibrate_coders()
    if result["gate_status"] != "pass":
        console.print(f"[red]coder calibration FAIL_CLOSED[/]: {result['reason']}")
        raise typer.Exit(code=2)
    console.print(f"[green]coder calibration pass[/]: kappa posteriors in {result['summary_path']}")


@app.command()
def fit(
    run_id: str = typer.Option(None, help="Run id; default = autogen"),
) -> None:
    """Fit m01_forward.stan on adjudicated + coder-calibrated data."""
    result = run_fit(run_id=run_id)
    if result["gate_status"] != "pass":
        console.print(f"[red]fit FAIL_CLOSED[/]: {result['reason']}")
        raise typer.Exit(code=2)
    console.print(f"[green]fit pass[/]: artifacts in {result['fit_dir']}")


@app.command()
def sbc(
    forecast: bool = typer.Option(False, "--forecast", help="Run forecast SBC instead of measurement SBC"),
    n_worlds: int = typer.Option(200),
) -> None:
    """Simulation-Based Calibration. Measurement layer by default; forecast layer with --forecast."""
    if forecast:
        result = run_forecast_sbc(n_worlds=n_worlds)
    else:
        result = run_sbc(n_worlds=n_worlds)
    if result["gate_status"] != "pass":
        console.print(f"[red]SBC FAIL_CLOSED[/]: {result['reason']}")
        raise typer.Exit(code=2)
    console.print(f"[green]SBC pass[/]: {result['summary_path']}")


@app.command()
def falsify() -> None:
    """Run F1..F15 falsification battery against the current fit."""
    result = run_falsification()
    if result["gate_status"] != "pass":
        console.print(f"[red]falsification FAIL_CLOSED[/]: {result['failed_tests']}")
        raise typer.Exit(code=2)
    console.print(f"[green]falsification pass[/]: ledger {result['ledger_path']}")


@app.command()
def forecast(
    horizon: int = typer.Option(6, help="months ahead in {3, 6, 9, 12}"),
    cell: str = typer.Option(None, help="optional cell filter c=...,p=...,k=..."),
) -> None:
    """Emit forecasts at horizon h. Pre-route; route applied by `david route`."""
    result = emit_forecasts(horizon_months=horizon, cell_filter=cell)
    console.print(f"[green]forecasts emitted[/]: {result['cells_path']} ({result['n_cells']} cells)")


@app.command()
def route() -> None:
    """Apply FG1..FG6 routing to the latest forecast run."""
    result = apply_forecast_routing()
    console.print(f"[green]routing complete[/]: ledger {result['ledger_path']}")
    for kind, n in result["route_counts"].items():
        console.print(f"  {kind}: {n}")


@app.command()
def serve(
    port: int = typer.Option(8080),
) -> None:
    """Read-only forecast API."""
    from .api import server  # local import to avoid import cost when unused

    server.run(port=port)


@app.command()
def replay(
    run_id: str = typer.Argument(..., help="run id to fully reproduce"),
) -> None:
    """Reproduce a past run end-to-end from recorded versions."""
    result = orchestrator.replay(run_id=run_id)
    console.print(f"[green]replay complete[/]: {result['status']}")


@db_app.command("init")
def db_init() -> None:
    """Create (or re-create) all Postgres tables and indexes.

    Idempotent — safe to run multiple times. Requires Postgres to be running:

        docker compose up -d
        david db init
    """
    from .db.connection import ping
    from .db.schema import create_all
    from .config import DATABASE_URL

    if not ping():
        console.print(
            f"[red]Cannot reach Postgres[/] at [bold]{DATABASE_URL}[/]\n"
            "Start it with:  [bold]docker compose up -d[/]"
        )
        raise typer.Exit(code=1)
    create_all()
    console.print(f"[green]Schema created[/] at [bold]{DATABASE_URL}[/]")


@db_app.command("status")
def db_status() -> None:
    """Show Postgres connectivity and row counts per table."""
    from .db.connection import ping
    from .db.schema import table_row_counts
    from .db.repositories import get_fit_runs
    from .config import DATABASE_URL

    if not ping():
        console.print(
            f"[red]Postgres unreachable[/] — {DATABASE_URL}\n"
            "Run:  [bold]docker compose up -d[/]"
        )
        raise typer.Exit(code=1)

    console.print(f"[green]Connected[/] → [bold]{DATABASE_URL}[/]\n")

    try:
        counts = table_row_counts()
        console.print("[bold]Table row counts:[/]")
        for table, n in counts.items():
            colour = "green" if n > 0 else "yellow"
            console.print(f"  [{colour}]{table:<20}[/] {n:>8,}")
    except Exception as exc:
        console.print(f"[yellow]Schema not yet initialised[/] ({exc})\nRun: [bold]david db init[/]")
        return

    console.print()
    try:
        runs = get_fit_runs(limit=5)
        if runs:
            console.print("[bold]Recent fit runs:[/]")
            for r in runs:
                status_colour = "green" if r["gate_status"] == "pass" else "red"
                console.print(
                    f"  [{status_colour}]{r['gate_status']:<6}[/]  "
                    f"{r['run_id']}  "
                    f"R̂≤{r['rhat_max']:.3f}  "
                    f"started {str(r['started_at'])[:16]}"
                )
        else:
            console.print("[yellow]No fit runs yet.[/]  Run: [bold]david fit[/]")
    except Exception:
        pass


if __name__ == "__main__":
    app()
