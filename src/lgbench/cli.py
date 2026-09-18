"""lgbench command line interface."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from .client import list_models
from .config import BenchConfig, CredentialError, resolve_credentials
from .report import print_table, write_csv, write_json, write_markdown
from .runner import run_benchmark
from .workloads import WORKLOADS

app = typer.Typer(add_completion=False, help="Benchmark the DataRobot LLM Gateway.")
console = Console()


def _creds(endpoint: Optional[str], token: Optional[str]):
    try:
        return resolve_credentials(endpoint=endpoint, token=token)
    except CredentialError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc


@app.command()
def models(
    endpoint: Optional[str] = typer.Option(None, help="DataRobot API endpoint."),
    token: Optional[str] = typer.Option(None, help="DataRobot API token."),
    contains: Optional[str] = typer.Option(None, help="Only show models matching this substring."),
) -> None:
    """List the models the gateway exposes."""
    creds = _creds(endpoint, token)
    try:
        entries = list_models(creds)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Could not list models: {exc}[/red]")
        raise typer.Exit(1) from exc

    if contains:
        entries = [e for e in entries if contains.lower() in (e["model"] or "").lower()]

    table = Table(title=f"Gateway models ({len(entries)})")
    table.add_column("Model")
    table.add_column("Provider")
    for e in sorted(entries, key=lambda e: e["model"] or ""):
        table.add_row(e["model"], e.get("provider") or "-")
    console.print(table)


@app.command()
def workloads() -> None:
    """List the available workloads."""
    table = Table(title="Workloads")
    table.add_column("Name")
    table.add_column("Description")
    table.add_column("~Input tok", justify="right")
    table.add_column("Output budget", justify="right")
    for w in WORKLOADS.values():
        table.add_row(w.name, w.description, str(w.approx_input_tokens), str(w.target_output_tokens))
    console.print(table)


@app.command()
def check(
    model: str = typer.Option("azure/gpt-4o-mini", help="Model to probe."),
    endpoint: Optional[str] = typer.Option(None),
    token: Optional[str] = typer.Option(None),
) -> None:
    """Send one request to confirm credentials and connectivity."""
    from .client import RequestSpec, make_client, run_request

    creds = _creds(endpoint, token)
    console.print(f"Endpoint: [cyan]{creds.gateway_base_url}[/cyan] (credentials from {creds.source})")

    async def probe():
        client = make_client(creds, timeout=60.0, max_retries=0)
        try:
            spec = RequestSpec(
                model=model,
                workload="ping",
                concurrency=1,
                messages=[{"role": "user", "content": "Reply with only the word OK."}],
                max_completion_tokens=5,
                temperature=0.0,
                stream=True,
            )
            return await run_request(client, spec)
        finally:
            await client.close()

    result = asyncio.run(probe())
    if result.ok:
        console.print(
            f"[green]OK[/green] {model}: {result.latency_s:.3f}s total, "
            f"TTFT {result.ttft_s:.3f}s"
            if result.ttft_s is not None
            else f"[green]OK[/green] {model}: {result.latency_s:.3f}s total"
        )
    else:
        console.print(f"[red]FAILED[/red] {model}: {result.error_type}: {result.error_message}")
        raise typer.Exit(1)


@app.command()
def run(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML benchmark config."),
    model: Optional[list[str]] = typer.Option(None, "--model", "-m", help="Model to benchmark; repeatable."),
    workload: Optional[list[str]] = typer.Option(None, "--workload", "-w", help="Workload name; repeatable."),
    concurrency: Optional[list[int]] = typer.Option(None, "--concurrency", help="Concurrency level; repeatable."),
    requests: Optional[int] = typer.Option(None, "-n", "--requests", help="Measured requests per cell."),
    warmup: Optional[int] = typer.Option(None, help="Warmup requests per cell (discarded)."),
    no_stream: bool = typer.Option(False, "--no-stream", help="Disable streaming (TTFT unavailable)."),
    max_tokens: Optional[int] = typer.Option(None, help="Override each workload's output budget."),
    timeout: Optional[float] = typer.Option(None, help="Per-request timeout in seconds."),
    out_dir: Path = typer.Option(Path("results"), "--out-dir", help="Where to write reports."),
    tag: Optional[str] = typer.Option(None, help="Label appended to output filenames."),
    endpoint: Optional[str] = typer.Option(None),
    token: Optional[str] = typer.Option(None),
) -> None:
    """Run the benchmark matrix and write JSON, CSV, and Markdown reports."""
    cfg = BenchConfig.from_yaml(config) if config else BenchConfig()

    if model:
        cfg.models = list(model)
    if workload:
        cfg.workloads = list(workload)
    if concurrency:
        cfg.concurrency = list(concurrency)
    if requests is not None:
        cfg.requests_per_cell = requests
    if warmup is not None:
        cfg.warmup_requests = warmup
    if no_stream:
        cfg.stream = False
    if max_tokens is not None:
        cfg.max_completion_tokens = max_tokens
    if timeout is not None:
        cfg.request_timeout = timeout

    try:
        cfg.validate()
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    creds = _creds(endpoint, token)
    console.print(
        f"Benchmarking [cyan]{creds.gateway_base_url}[/cyan] "
        f"({len(cfg.models)} model(s) x {len(cfg.workloads)} workload(s) x "
        f"{len(cfg.concurrency)} concurrency level(s))"
    )

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    )
    task_id = progress.add_task("starting", total=cfg.total_requests)

    def on_cell_start(m: str, w: str, c: int) -> None:
        progress.update(task_id, description=f"{m} / {w} / c={c}")

    def on_result(_r) -> None:
        progress.advance(task_id)

    with progress:
        run_result = asyncio.run(
            run_benchmark(cfg, creds, on_cell_start=on_cell_start, on_result=on_result)
        )

    console.print()
    print_table(run_result.cells, console)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"bench-{stamp}" + (f"-{tag}" if tag else "")
    json_path = write_json(run_result, out_dir / f"{name}.json")
    csv_path = write_csv(run_result, out_dir / f"{name}.csv")
    md_path = write_markdown(run_result, out_dir / f"{name}.md")
    console.print(f"\nWrote [green]{json_path}[/green], [green]{csv_path}[/green], [green]{md_path}[/green]")

    if any(c.failures for c in run_result.cells):
        raise typer.Exit(2)


@app.command()
def compare(
    baseline: Path = typer.Argument(..., help="Baseline run JSON."),
    candidate: Path = typer.Argument(..., help="Candidate run JSON."),
    metric: str = typer.Option("latency.p50", help="Dotted metric path, e.g. latency.p95 or ttft.p50."),
) -> None:
    """Compare two saved runs cell by cell."""
    base = json.loads(baseline.read_text())
    cand = json.loads(candidate.read_text())

    def key(c: dict) -> tuple:
        return (c["model"], c["workload"], c["concurrency"])

    def value(c: dict) -> float | None:
        node: object = c
        for part in metric.split("."):
            if not isinstance(node, dict):
                return None
            node = node.get(part)
        return node if isinstance(node, (int, float)) else None

    base_cells = {key(c): c for c in base["cells"]}
    table = Table(title=f"{metric}: {baseline.name} -> {candidate.name}")
    table.add_column("Model")
    table.add_column("Workload")
    table.add_column("Conc", justify="right")
    table.add_column("Baseline", justify="right")
    table.add_column("Candidate", justify="right")
    table.add_column("Delta", justify="right")

    for cell in cand["cells"]:
        b = base_cells.get(key(cell))
        bv = value(b) if b else None
        cv = value(cell)
        if bv is None or cv is None:
            delta, style = "-", None
        else:
            pct = (cv - bv) / bv * 100 if bv else 0.0
            delta = f"{pct:+.1f}%"
            style = "red" if pct > 5 else "green" if pct < -5 else None
        table.add_row(
            cell["model"], cell["workload"], str(cell["concurrency"]),
            "-" if bv is None else f"{bv:.3f}",
            "-" if cv is None else f"{cv:.3f}",
            delta, style=style,
        )
    console.print(table)


if __name__ == "__main__":
    app()
