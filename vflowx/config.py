"""Which models to use, in one place: `aflow.toml` at the repo root (or the file named by $AFLOW_CONFIG).

    executor  = "<provider>/<model>"     the LLM called inside workflows
    optimizer = "<provider>/<model>"     the LLM that edits workflows ("<model>" or "": the harness's own login)
    harness   = "codex" | "dsh"          the coding-agent program that drives the optimizer (see harness.py)

Command-line flags (--executor / --optimizer / --harness) override the file (VFX_EXECUTOR / VFX_OPTIMIZER /
VFX_HARNESS).
API keys never live in the file: each provider names the environment variable that holds its key.
"""
from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Built-in providers; entries in aflow.toml are merged over these.
BUILTIN = {
    "local": {"base_url": ["http://127.0.0.1:8000/v1"], "concurrency": 96, "responses_api": True},
    "openai": {"base_url": "https://api.openai.com/v1", "api_key_env": "OPENAI_API_KEY", "responses_api": True},
    "deepseek": {"base_url": "https://api.deepseek.com/v1", "api_key_env": "DEEPSEEK_API_KEY",
                 "think_model": "deepseek-reasoner"},
}
# How a provider is talked to (see llm.py); anything not listed is a plain OpenAI-compatible endpoint.
KINDS = {"local": "vllm", "openai": "openai", "deepseek": "deepseek"}
_REASONING_MODEL = re.compile(r"^(gpt-5|o\d)", re.I)


@dataclass
class Model:
    spec: str                 # as written, e.g. "deepseek/deepseek-chat"
    provider: str             # "deepseek"
    model: str | None         # "deepseek-chat" (None: ask the server, local vLLM only)
    kind: str                 # vllm | openai | deepseek | compatible
    base_urls: list[str] = field(default_factory=list)
    api_key_env: str | None = None
    think_model: str | None = None
    think_effort: str = "medium"
    concurrency: int = 16
    max_tokens: int = 6144
    responses_api: bool = False   # also serves the OpenAI Responses API (needed by the codex harness)

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "") if self.api_key_env else ""

    @property
    def reasoning(self) -> bool:
        """OpenAI reasoning model (accepts reasoning_effort)."""
        return self.kind == "openai" and bool(self.model and _REASONING_MODEL.match(self.model))


def config_path() -> str:
    return os.environ.get("AFLOW_CONFIG") or f"{ROOT}/aflow.toml"


def load() -> dict:
    cfg = {"executor": "local", "optimizer": "", "harness": "codex", "providers": {}}
    path = config_path()
    if os.path.exists(path):
        with open(path, "rb") as f:
            cfg.update(tomllib.load(f))
    providers = {k: dict(v) for k, v in BUILTIN.items()}
    for k, v in cfg.get("providers", {}).items():
        providers.setdefault(k, {}).update(v)
    cfg["providers"] = providers
    return cfg


def executor_spec() -> str:
    return os.environ.get("VFX_EXECUTOR") or load()["executor"]


def optimizer_spec() -> str:
    return os.environ.get("VFX_OPTIMIZER", load()["optimizer"])


def harness_name() -> str:
    return os.environ.get("VFX_HARNESS") or load()["harness"]


def resolve(spec: str) -> Model:
    """'<provider>/<model>' (or just '<provider>' for a local server's only model) -> Model."""
    cfg = load()
    provider, _, model = spec.partition("/")
    if provider not in cfg["providers"]:
        raise ValueError(f"unknown provider {provider!r} in {spec!r}; known: {sorted(cfg['providers'])} "
                         f"(add a [providers.{provider}] section to {config_path()})")
    p = cfg["providers"][provider]
    urls = p.get("base_url") or []
    urls = [urls] if isinstance(urls, str) else list(urls)
    if not urls:
        raise ValueError(f"[providers.{provider}] in {config_path()} needs base_url")
    kind = p.get("kind") or KINDS.get(provider, "compatible")
    if not model and kind != "vllm":
        raise ValueError(f"{spec!r}: write it as '{provider}/<model name>'")
    return Model(spec=spec, provider=provider, model=model or None, kind=kind, base_urls=urls,
                 api_key_env=p.get("api_key_env"), think_model=p.get("think_model"),
                 think_effort=p.get("think_effort", "medium"),
                 concurrency=int(p.get("concurrency", 96 if kind == "vllm" else 16)),
                 max_tokens=int(p.get("max_tokens", 6144)), responses_api=bool(p.get("responses_api", False)))
