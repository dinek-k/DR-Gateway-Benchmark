"""Prompt workloads.

Each workload fixes an approximate input size and expected output size so that
latency numbers are comparable across models and across runs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

_FILLER = (
    "DataRobot is an AI platform that supports the end-to-end lifecycle of predictive "
    "and generative models, including data preparation, experimentation, deployment, "
    "monitoring, and governance across hybrid and multi-cloud environments. "
)


def _padding(approx_tokens: int, rng: random.Random) -> str:
    """Build a deterministic filler passage of roughly `approx_tokens` tokens."""
    # ~4 characters per token is a reasonable cross-tokenizer approximation.
    target_chars = approx_tokens * 4
    reps = max(1, target_chars // len(_FILLER) + 1)
    text = (_FILLER * reps)[:target_chars]
    # Vary the tail so providers cannot serve a cached identical prompt.
    return f"{text}\n[context id {rng.randrange(10**9)}]"


@dataclass(frozen=True)
class Workload:
    name: str
    description: str
    approx_input_tokens: int
    target_output_tokens: int

    def build_messages(self, rng: random.Random) -> list[dict[str, str]]:
        raise NotImplementedError


@dataclass(frozen=True)
class ShortQA(Workload):
    def build_messages(self, rng: random.Random) -> list[dict[str, str]]:
        topic = rng.choice(
            [
                "model drift", "feature engineering", "vector databases",
                "retrieval augmented generation", "model governance",
                "batch scoring", "time series forecasting", "guardrails",
            ]
        )
        return [
            {"role": "system", "content": "You are a concise technical assistant."},
            {"role": "user", "content": f"In two sentences, explain {topic}."},
        ]


@dataclass(frozen=True)
class LongContext(Workload):
    def build_messages(self, rng: random.Random) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": "You summarize documents faithfully."},
            {
                "role": "user",
                "content": (
                    f"Summarize the following document in three bullet points.\n\n"
                    f"{_padding(self.approx_input_tokens, rng)}"
                ),
            },
        ]


@dataclass(frozen=True)
class LongGeneration(Workload):
    def build_messages(self, rng: random.Random) -> list[dict[str, str]]:
        n = rng.randrange(10**6)
        return [
            {"role": "system", "content": "You are a thorough technical writer."},
            {
                "role": "user",
                "content": (
                    f"Write a detailed {self.target_output_tokens}-token explanation of how "
                    f"an LLM gateway routes, authenticates, and meters requests. "
                    f"Do not stop early. (variation {n})"
                ),
            },
        ]


@dataclass(frozen=True)
class Ping(Workload):
    """Minimal prompt and output: measures the gateway's fixed overhead floor."""

    def build_messages(self, rng: random.Random) -> list[dict[str, str]]:
        return [{"role": "user", "content": f"Reply with only the word OK. ({rng.randrange(10**6)})"}]


WORKLOADS: dict[str, Workload] = {
    w.name: w
    for w in [
        Ping("ping", "Minimal in/out; gateway overhead floor", 10, 5),
        ShortQA("short_qa", "Small prompt, short answer", 30, 120),
        LongContext("long_context", "~4k token prompt, short answer", 4000, 200),
        LongGeneration("long_generation", "Small prompt, long answer", 40, 800),
    ]
}


def get_workload(name: str) -> Workload:
    try:
        return WORKLOADS[name]
    except KeyError:
        raise ValueError(
            f"Unknown workload {name!r}. Available: {', '.join(sorted(WORKLOADS))}"
        ) from None
