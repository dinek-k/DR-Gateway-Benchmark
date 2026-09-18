from lgbench.metrics import RequestResult, percentile, summarize_cell


def make(ok=True, latency=1.0, ttft=0.2, tokens=100, **kw):
    return RequestResult(
        model="m", workload="w", concurrency=2, stream=True, ok=ok,
        latency_s=latency, ttft_s=ttft, completion_tokens=tokens, prompt_tokens=10, **kw
    )


def test_percentile_bounds():
    vals = [1.0, 2.0, 3.0, 4.0]
    assert percentile(vals, 0) == 1.0
    assert percentile(vals, 100) == 4.0
    assert percentile(vals, 50) == 2.5
    assert percentile([], 50) is None
    assert percentile([7.0], 99) == 7.0


def test_tpot_and_tps():
    r = make(latency=2.0, ttft=0.5, tokens=101)
    assert r.tpot_s == (2.0 - 0.5) / 100
    assert r.output_tps == 101 / 2.0


def test_tpot_none_without_enough_tokens():
    assert make(tokens=1).tpot_s is None
    assert make(ttft=None).tpot_s is None


def test_summarize_counts_and_errors():
    results = [make(), make(latency=3.0), make(ok=False, error_type="APITimeoutError")]
    cell = summarize_cell(results, wall_time_s=4.0)
    assert cell.requests == 3
    assert cell.successes == 2
    assert cell.failures == 1
    assert cell.errors == {"APITimeoutError": 1}
    assert cell.error_rate == 1 / 3
    # Throughput counts only successful requests.
    assert cell.throughput_rps == 2 / 4.0
    # Failed requests are excluded from latency stats.
    assert cell.latency["max"] == 3.0
    assert cell.total_output_tps == 200 / 4.0


def test_summarize_rejects_empty():
    try:
        summarize_cell([], 1.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
