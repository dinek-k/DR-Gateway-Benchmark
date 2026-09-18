"""Per-request records and aggregation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from statistics import mean


@dataclass
class RequestResult:
    """One chat-completion attempt."""

    model: str
    workload: str
    concurrency: int
    stream: bool
    ok: bool
    latency_s: float
    ttft_s: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    chunk_count: int = 0
    error_type: str | None = None
    error_message: str | None = None
    status_code: int | None = None
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def tpot_s(self) -> float | None:
        """Time per output token, excluding the first token."""
        if self.ttft_s is None or not self.completion_tokens or self.completion_tokens < 2:
            return None
        return (self.latency_s - self.ttft_s) / (self.completion_tokens - 1)

    @property
    def output_tps(self) -> float | None:
        """Output tokens per second for this request."""
        if not self.completion_tokens or self.latency_s <= 0:
            return None
        return self.completion_tokens / self.latency_s

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tpot_s"] = self.tpot_s
        d["output_tps"] = self.output_tps
        return d


def percentile(values: list[float], p: float) -> float | None:
    """Linear-interpolated percentile. `p` is in [0, 100]."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (p / 100) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


@dataclass
class CellSummary:
    """Aggregated results for one (model, workload, concurrency) cell."""

    model: str
    workload: str
    concurrency: int
    requests: int
    successes: int
    failures: int
    wall_time_s: float
    latency: dict[str, float | None] = field(default_factory=dict)
    ttft: dict[str, float | None] = field(default_factory=dict)
    tpot_mean_s: float | None = None
    output_tps_mean: float | None = None
    throughput_rps: float | None = None
    total_output_tps: float | None = None
    mean_prompt_tokens: float | None = None
    mean_completion_tokens: float | None = None
    errors: dict[str, int] = field(default_factory=dict)

    @property
    def error_rate(self) -> float:
        return self.failures / self.requests if self.requests else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["error_rate"] = self.error_rate
        return d


def _dist(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p90": None, "p95": None, "p99": None, "min": None, "max": None}
    return {
        "mean": mean(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "min": min(values),
        "max": max(values),
    }


def summarize_cell(results: list[RequestResult], wall_time_s: float) -> CellSummary:
    """Aggregate one cell's requests. `wall_time_s` is the measured window, warmups excluded."""
    if not results:
        raise ValueError("Cannot summarize an empty result set.")

    head = results[0]
    ok = [r for r in results if r.ok]
    bad = [r for r in results if not r.ok]

    errors: dict[str, int] = {}
    for r in bad:
        key = r.error_type or "unknown"
        errors[key] = errors.get(key, 0) + 1

    latencies = [r.latency_s for r in ok]
    ttfts = [r.ttft_s for r in ok if r.ttft_s is not None]
    tpots = [r.tpot_s for r in ok if r.tpot_s is not None]
    tps = [r.output_tps for r in ok if r.output_tps is not None]
    prompt_toks = [r.prompt_tokens for r in ok if r.prompt_tokens is not None]
    completion_toks = [r.completion_tokens for r in ok if r.completion_tokens is not None]

    total_completion = sum(completion_toks) if completion_toks else 0

    return CellSummary(
        model=head.model,
        workload=head.workload,
        concurrency=head.concurrency,
        requests=len(results),
        successes=len(ok),
        failures=len(bad),
        wall_time_s=wall_time_s,
        latency=_dist(latencies),
        ttft=_dist([t for t in ttfts]),
        tpot_mean_s=mean(tpots) if tpots else None,
        output_tps_mean=mean(tps) if tps else None,
        throughput_rps=len(ok) / wall_time_s if wall_time_s > 0 else None,
        total_output_tps=total_completion / wall_time_s if wall_time_s > 0 and total_completion else None,
        mean_prompt_tokens=mean(prompt_toks) if prompt_toks else None,
        mean_completion_tokens=mean(completion_toks) if completion_toks else None,
        errors=errors,
    )
