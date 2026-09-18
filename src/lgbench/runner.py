"""Drives the benchmark matrix: model x workload x concurrency."""

from __future__ import annotations

import asyncio
import platform
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from . import __version__
from .client import RequestSpec, make_client, run_request
from .config import BenchConfig, Credentials
from .metrics import CellSummary, RequestResult, summarize_cell
from .workloads import get_workload


@dataclass
class BenchRun:
    started_at: str
    finished_at: str
    endpoint: str
    config: dict
    cells: list[CellSummary] = field(default_factory=list)
    results: list[RequestResult] = field(default_factory=list)
    environment: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "lgbench_version": __version__,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "endpoint": self.endpoint,
            "environment": self.environment,
            "config": self.config,
            "cells": [c.to_dict() for c in self.cells],
            "results": [r.to_dict() for r in self.results],
        }


async def _run_cell(
    client,
    model: str,
    workload_name: str,
    concurrency: int,
    config: BenchConfig,
    rng: random.Random,
    on_result: Callable[[RequestResult], None] | None = None,
) -> tuple[CellSummary, list[RequestResult]]:
    """Run one cell: `warmup_requests` discarded, then `requests_per_cell` measured.

    Requests are dispatched through a semaphore so exactly `concurrency` are ever
    in flight, which is what makes throughput comparable across cells.
    """
    workload = get_workload(workload_name)
    budget = config.max_completion_tokens or workload.target_output_tokens
    sem = asyncio.Semaphore(concurrency)

    def build_spec() -> RequestSpec:
        return RequestSpec(
            model=model,
            workload=workload_name,
            concurrency=concurrency,
            messages=workload.build_messages(rng),
            max_completion_tokens=budget,
            temperature=config.temperature,
            stream=config.stream,
        )

    async def guarded(spec: RequestSpec, measured: bool) -> RequestResult:
        async with sem:
            result = await run_request(client, spec)
        if measured and on_result:
            on_result(result)
        return result

    if config.warmup_requests:
        warmups = [guarded(build_spec(), measured=False) for _ in range(config.warmup_requests)]
        await asyncio.gather(*warmups)

    specs = [build_spec() for _ in range(config.requests_per_cell)]
    wall_start = time.perf_counter()
    results = await asyncio.gather(*(guarded(s, measured=True) for s in specs))
    wall_time = time.perf_counter() - wall_start

    return summarize_cell(list(results), wall_time), list(results)


async def run_benchmark(
    config: BenchConfig,
    creds: Credentials,
    on_cell_start: Callable[[str, str, int], None] | None = None,
    on_result: Callable[[RequestResult], None] | None = None,
    on_cell_done: Callable[[CellSummary], None] | None = None,
) -> BenchRun:
    config.validate()
    started = datetime.now(timezone.utc).isoformat()
    client = make_client(creds, config.request_timeout, config.max_retries)

    cells: list[CellSummary] = []
    all_results: list[RequestResult] = []

    try:
        for model in config.models:
            for workload_name in config.workloads:
                # Same seed per cell keeps prompts identical across models.
                for concurrency in config.concurrency:
                    rng = random.Random(config.seed)
                    if on_cell_start:
                        on_cell_start(model, workload_name, concurrency)
                    summary, results = await _run_cell(
                        client, model, workload_name, concurrency, config, rng, on_result
                    )
                    cells.append(summary)
                    all_results.extend(results)
                    if on_cell_done:
                        on_cell_done(summary)
    finally:
        await client.close()

    return BenchRun(
        started_at=started,
        finished_at=datetime.now(timezone.utc).isoformat(),
        endpoint=creds.endpoint,
        config=vars(config).copy(),
        cells=cells,
        results=all_results,
        environment={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
    )
