"""Console tables, Markdown reports, and CSV export."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .metrics import CellSummary
from .runner import BenchRun

# Full column set, used for the Markdown and CSV reports.
COLUMNS = [
    ("Model", lambda c: c.model),
    ("Workload", lambda c: c.workload),
    ("Conc", lambda c: str(c.concurrency)),
    ("OK", lambda c: f"{c.successes}/{c.requests}"),
    ("TTFT p50", lambda c: _s(c.ttft.get("p50"))),
    ("TTFT p95", lambda c: _s(c.ttft.get("p95"))),
    ("Lat p50", lambda c: _s(c.latency.get("p50"))),
    ("Lat p95", lambda c: _s(c.latency.get("p95"))),
    ("Out tok/s", lambda c: _n(c.output_tps_mean)),
    ("Req/s", lambda c: _n(c.throughput_rps)),
]

# The console groups by model, so the model column is dropped. TTFT p95 is dropped
# too, to keep the table inside an 80-column terminal; it stays in the reports.
_CONSOLE_DROP = {"Model", "TTFT p95"}
_CONSOLE_RENAME = {"Out tok/s": "tok/s"}
CONSOLE_COLUMNS = [
    (_CONSOLE_RENAME.get(name, name), fn) for name, fn in COLUMNS if name not in _CONSOLE_DROP
]


def _s(v: float | None) -> str:
    return "-" if v is None else f"{v:.3f}s"


def _n(v: float | None) -> str:
    return "-" if v is None else f"{v:.1f}"


def print_table(cells: list[CellSummary], console: Console | None = None) -> None:
    """Print one table per model so long model names never squeeze the metrics."""
    console = console or Console()

    by_model: dict[str, list[CellSummary]] = {}
    for cell in cells:
        by_model.setdefault(cell.model, []).append(cell)

    for model, model_cells in by_model.items():
        table = Table(title=model, title_style="bold cyan", header_style="bold")
        for name, _ in CONSOLE_COLUMNS:
            is_workload = name == "Workload"
            table.add_column(
                name,
                justify="left" if is_workload else "right",
                # Reserve room for the longest workload name before rich shrinks columns.
                min_width=len("long_generation") if is_workload else None,
            )
        for cell in model_cells:
            table.add_row(
                *(fn(cell) for _, fn in CONSOLE_COLUMNS),
                style="red" if cell.failures else None,
            )
        console.print(table)

    failing = [c for c in cells if c.failures]
    if failing:
        console.print("\n[bold red]Errors[/bold red]")
        for cell in failing:
            detail = ", ".join(f"{k} x{v}" for k, v in sorted(cell.errors.items()))
            console.print(f"  {cell.model} / {cell.workload} / c={cell.concurrency}: {detail}")


def write_json(run: BenchRun, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run.to_dict(), indent=2, default=str))
    return path


def write_csv(run: BenchRun, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([name for name, _ in COLUMNS] + ["error_rate", "mean_prompt_tokens", "mean_completion_tokens"])
        for c in run.cells:
            writer.writerow(
                [fn(c) for _, fn in COLUMNS]
                + [f"{c.error_rate:.3f}", _n(c.mean_prompt_tokens), _n(c.mean_completion_tokens)]
            )
    return path


def write_markdown(run: BenchRun, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = run.config
    lines = [
        "# LLM Gateway benchmark",
        "",
        f"- **Endpoint:** `{run.endpoint}`",
        f"- **Started:** {run.started_at}",
        f"- **Finished:** {run.finished_at}",
        f"- **Mode:** {'streaming' if cfg.get('stream') else 'non-streaming'}",
        f"- **Requests per cell:** {cfg.get('requests_per_cell')} "
        f"(plus {cfg.get('warmup_requests')} warmup, discarded)",
        "",
        "## Results",
        "",
        "| " + " | ".join(name for name, _ in COLUMNS) + " |",
        "| " + " | ".join("---" for _ in COLUMNS) + " |",
    ]
    for c in run.cells:
        lines.append("| " + " | ".join(fn(c) for _, fn in COLUMNS) + " |")

    lines += [
        "",
        "## Metric definitions",
        "",
        "- **TTFT** — time to first content token; streaming only. The best single "
        "proxy for perceived responsiveness.",
        "- **Lat** — end-to-end wall time for a complete response.",
        "- **Out tok/s** — mean per-request output tokens divided by that request's latency.",
        "- **Req/s** — completed requests divided by the cell's wall time at the stated "
        "concurrency; this is the aggregate throughput number.",
        "",
        "Token counts come from the gateway's `usage` field when it is returned; where it "
        "is absent on streamed responses, content chunks are counted as a proxy and "
        "token-derived metrics should be treated as approximate.",
    ]

    failing = [c for c in run.cells if c.failures]
    if failing:
        lines += ["", "## Errors", ""]
        for cell in failing:
            detail = ", ".join(f"`{k}` x{v}" for k, v in sorted(cell.errors.items()))
            lines.append(f"- {cell.model} / {cell.workload} / c={cell.concurrency}: {detail}")

    path.write_text("\n".join(lines) + "\n")
    return path
