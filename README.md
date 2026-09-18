# lgbench — DataRobot LLM Gateway benchmarks

Measures latency and throughput of models served through the
[DataRobot LLM Gateway](https://docs.datarobot.com/en/docs/agentic-ai/genai-code/dr-llm-gateway.html),
which exposes an OpenAI-compatible API at `{DATAROBOT_ENDPOINT}/genai/llmgw`.

This benchmarks **serving performance**, not answer quality — it tells you how fast
a model responds through the gateway, not how good the response is.

## Setup

```bash
uv venv && uv pip install -e ".[dev]"
```

Credentials are resolved in this order:

1. `--endpoint` / `--token` flags
2. `DATAROBOT_API_TOKEN` / `DATAROBOT_ENDPOINT` (a local `.env` is loaded automatically)
3. `~/.config/datarobot/drconfig.yaml`

Copy `.env.example` to `.env` if you want to pin a token per-project. `.env` is gitignored.

## Usage

```bash
lgbench models                      # what the gateway exposes (100+ entries)
lgbench models --contains claude    # filter
lgbench workloads                   # the built-in prompt scenarios
lgbench check -m azure/gpt-4o-mini  # one request, confirms auth and connectivity

# ad-hoc run
lgbench run -m azure/gpt-4o-mini -m anthropic/claude-haiku-4-5-20251001 \
            -w short_qa -w long_generation \
            --concurrency 1 --concurrency 4 -n 10

# from a config file
lgbench run -c configs/default.yaml --tag nightly

# regression check between two saved runs
lgbench compare results/bench-A.json results/bench-B.json --metric latency.p95
```

Each run writes JSON (every per-request record), CSV, and a Markdown report to
`results/`. `run` exits 2 if any request failed, so it can gate CI.

## Workloads

| Name | Shape | What it isolates |
| --- | --- | --- |
| `ping` | ~10 in / 5 out | The gateway's fixed overhead floor — auth, routing, connection setup |
| `short_qa` | ~30 in / 120 out | Typical interactive chat turn |
| `long_context` | ~4k in / 200 out | Prefill cost on a large prompt |
| `long_generation` | ~40 in / 800 out | Sustained decode throughput; the clearest read on streaming |

Prompts are seeded per cell, so every model sees identical prompts. Each prompt carries a
random tail so upstream prompt caching does not silently flatter repeat runs.

## Metrics

- **TTFT** — time to the first content token. Streaming only, and the best proxy for
  perceived responsiveness.
- **Lat** — end-to-end wall time for a complete response. `p95` matters more than `p50`
  for capacity planning.
- **tok/s** — mean per-request output tokens ÷ that request's latency (single-stream speed).
- **Req/s** — successful requests ÷ the cell's wall time at the stated concurrency. This is
  the aggregate throughput figure; it should rise with concurrency until the gateway or the
  upstream provider saturates.

Token counts come from the gateway's `usage` field, requested via
`stream_options={"include_usage": True}`. The gateway does return it. Where it is ever
absent, content chunks are counted as a fallback and token-derived metrics become approximate.

## How a run is structured

The matrix is `models x workloads x concurrency`. Each cell runs `warmup_requests`
(discarded, so connection setup and cold routing do not pollute the numbers), then
`requests_per_cell` measured requests dispatched through a semaphore that holds exactly
`concurrency` in flight. The wall-clock window used for throughput covers only the
measured requests.

`max_retries` defaults to 0 so that a recorded latency is always a single attempt —
retries would hide upstream 429s inside an inflated latency figure instead of surfacing
them as errors.

## Reading the results

Concurrency is where gateway behavior shows up. If `Req/s` scales roughly linearly from
c=1 to c=8 while `Lat p50` stays flat, the gateway is not the bottleneck. If latency climbs
in proportion to concurrency while `Req/s` plateaus, requests are queueing somewhere —
either in the gateway or at the upstream provider, and the `ping` workload is what separates
those two: it stays fast if the gateway itself is healthy.

Numbers are only comparable within a run. Provider-side load varies through the day, so use
`lgbench compare` against a baseline captured in the same conditions rather than reading
absolute values across days.

## Caveats

- Measurements include network latency from wherever you run this. Run it from the same
  region as your workload if you care about absolute numbers.
- Reasoning models (`azure/o1`, `gpt-5*`) spend tokens before emitting content, so their
  TTFT is not comparable to non-reasoning models.
- `temperature: 0.0` is set for reproducibility; some models reject or ignore it.

## Tests

```bash
pytest tests/ -q
```
