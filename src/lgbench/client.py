"""Thin wrapper over the OpenAI-compatible DataRobot LLM Gateway."""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx
from openai import AsyncOpenAI

from .config import Credentials
from .metrics import RequestResult


def make_client(creds: Credentials, timeout: float, max_retries: int) -> AsyncOpenAI:
    """Build an AsyncOpenAI client pointed at the gateway.

    Retries default to 0 so that benchmark latencies measure a single attempt.
    """
    return AsyncOpenAI(
        base_url=creds.gateway_base_url,
        api_key=creds.token,
        timeout=timeout,
        max_retries=max_retries,
    )


def list_models(creds: Credentials, timeout: float = 30.0) -> list[dict]:
    """List catalog entries, preferring the DataRobot SDK and falling back to REST."""
    try:
        from datarobot.models.genai.llm_gateway_catalog import LLMGatewayCatalog  # type: ignore
        import datarobot as dr  # type: ignore

        dr.Client(endpoint=creds.endpoint, token=creds.token)
        entries = LLMGatewayCatalog.get_available_models()
        out = []
        for e in entries:
            model = getattr(e, "model", None) or getattr(e, "id", None) or str(e)
            out.append(
                {
                    "model": model,
                    "provider": getattr(e, "provider", None),
                    "source": "datarobot-sdk",
                }
            )
        if out:
            return out
    except Exception:  # noqa: BLE001 - SDK is optional; fall through to REST.
        pass

    resp = httpx.get(
        f"{creds.gateway_base_url}/models",
        headers={"Authorization": f"Bearer {creds.token}"},
        timeout=timeout,
    )
    resp.raise_for_status()
    payload = resp.json()
    items = payload.get("data", payload if isinstance(payload, list) else [])
    return [
        {"model": i.get("id") or i.get("model"), "provider": i.get("owned_by"), "source": "rest"}
        for i in items
    ]


@dataclass
class RequestSpec:
    model: str
    workload: str
    concurrency: int
    messages: list[dict[str, str]]
    max_completion_tokens: int
    temperature: float
    stream: bool


def _classify(exc: BaseException) -> tuple[str, int | None]:
    status = getattr(exc, "status_code", None)
    return type(exc).__name__, status


async def run_request(client: AsyncOpenAI, spec: RequestSpec) -> RequestResult:
    """Issue one chat completion and time it.

    For streaming requests, TTFT is the time to the first chunk carrying content.
    """
    extra_body = {
        "temperature": spec.temperature,
        "max_completion_tokens": spec.max_completion_tokens,
    }
    base = dict(model=spec.model, messages=spec.messages, extra_body=extra_body)

    started = time.perf_counter()
    started_wall = time.time()
    ttft: float | None = None
    chunks = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    try:
        if spec.stream:
            stream = await client.chat.completions.create(
                **base,
                stream=True,
                stream_options={"include_usage": True},
            )
            async for chunk in stream:
                if getattr(chunk, "usage", None):
                    prompt_tokens = chunk.usage.prompt_tokens
                    completion_tokens = chunk.usage.completion_tokens
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    chunks += 1
                    if ttft is None:
                        ttft = time.perf_counter() - started
        else:
            resp = await client.chat.completions.create(**base)
            if resp.usage:
                prompt_tokens = resp.usage.prompt_tokens
                completion_tokens = resp.usage.completion_tokens

        elapsed = time.perf_counter() - started
        if completion_tokens is None and chunks:
            # Gateway did not return usage on the stream; content chunks are the
            # closest available proxy for output token count.
            completion_tokens = chunks

        return RequestResult(
            model=spec.model,
            workload=spec.workload,
            concurrency=spec.concurrency,
            stream=spec.stream,
            ok=True,
            latency_s=elapsed,
            ttft_s=ttft,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            chunk_count=chunks,
            started_at=started_wall,
            finished_at=time.time(),
        )
    except Exception as exc:  # noqa: BLE001 - every failure mode is a datapoint.
        error_type, status = _classify(exc)
        return RequestResult(
            model=spec.model,
            workload=spec.workload,
            concurrency=spec.concurrency,
            stream=spec.stream,
            ok=False,
            latency_s=time.perf_counter() - started,
            ttft_s=ttft,
            chunk_count=chunks,
            error_type=error_type,
            error_message=str(exc)[:500],
            status_code=status,
            started_at=started_wall,
            finished_at=time.time(),
        )
