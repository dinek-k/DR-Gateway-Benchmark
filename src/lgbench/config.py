"""Credential resolution and benchmark configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

DEFAULT_ENDPOINT = "https://app.datarobot.com/api/v2"
DRCONFIG = Path.home() / ".config" / "datarobot" / "drconfig.yaml"


class CredentialError(RuntimeError):
    pass


@dataclass(frozen=True)
class Credentials:
    endpoint: str
    token: str
    source: str

    @property
    def gateway_base_url(self) -> str:
        """The OpenAI-compatible base URL the gateway serves.

        The OpenAI client appends /chat/completions, /models, etc. to this.
        """
        return f"{self.endpoint.rstrip('/')}/genai/llmgw"


def resolve_credentials(
    endpoint: str | None = None,
    token: str | None = None,
    env_file: Path | None = None,
) -> Credentials:
    """Resolve credentials from explicit args, then .env / environment, then drconfig.yaml."""
    load_dotenv(dotenv_path=env_file, override=False)

    if token:
        return Credentials(endpoint or DEFAULT_ENDPOINT, token, "argument")

    env_token = os.getenv("DATAROBOT_API_TOKEN")
    if env_token:
        env_endpoint = endpoint or os.getenv("DATAROBOT_ENDPOINT") or DEFAULT_ENDPOINT
        return Credentials(env_endpoint, env_token, "environment")

    if DRCONFIG.exists():
        data = yaml.safe_load(DRCONFIG.read_text()) or {}
        file_token = data.get("token")
        if file_token:
            file_endpoint = endpoint or data.get("endpoint") or DEFAULT_ENDPOINT
            return Credentials(file_endpoint, file_token, str(DRCONFIG))

    raise CredentialError(
        "No DataRobot credentials found. Set DATAROBOT_API_TOKEN (see .env.example) "
        f"or populate {DRCONFIG}."
    )


@dataclass
class BenchConfig:
    """A full benchmark run definition."""

    models: list[str] = field(default_factory=list)
    workloads: list[str] = field(default_factory=lambda: ["short_qa"])
    concurrency: list[int] = field(default_factory=lambda: [1, 4, 8])
    requests_per_cell: int = 10
    warmup_requests: int = 1
    stream: bool = True
    # None = use each workload's own output budget; set a number to override globally.
    max_completion_tokens: int | None = None
    temperature: float = 0.0
    request_timeout: float = 120.0
    max_retries: int = 0
    seed: int = 1234

    @classmethod
    def from_yaml(cls, path: Path) -> "BenchConfig":
        data = yaml.safe_load(Path(path).read_text()) or {}
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
        return cls(**data)

    def validate(self) -> None:
        if not self.models:
            raise ValueError("No models configured. Pass --model or set `models` in the config file.")
        if not self.concurrency or any(c < 1 for c in self.concurrency):
            raise ValueError("`concurrency` must be a list of positive integers.")
        if self.requests_per_cell < 1:
            raise ValueError("`requests_per_cell` must be >= 1.")

    @property
    def total_requests(self) -> int:
        cells = len(self.models) * len(self.workloads) * len(self.concurrency)
        return cells * (self.requests_per_cell + self.warmup_requests)
