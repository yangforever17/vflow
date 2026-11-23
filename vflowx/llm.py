"""Thin async client for the executor LLM (local vLLM or a hosted API: OpenAI, DeepSeek, any OpenAI-compatible
endpoint) with per-task usage accounting. Which one is set in aflow.toml (see config.py)."""
from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field

import httpx
import openai
from openai import AsyncOpenAI

from . import config


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_seconds: float = 0.0
    timeouts: int = 0
    # think=True completions cut off by max_tokens before `</think>`: returned as "" (no answer was produced)
    truncated: int = 0
    # codes produced by generate/repair during this problem (for oracle diagnostics on the feedback set)
    candidates: list = field(default_factory=list, repr=False)
    # Common random numbers: when seed_base is set, every call gets a deterministic sampling seed derived
    # from (seed_base, call content, occurrence index of identical calls). Parent and child workflows that
    # issue the same call on the same (problem, repeat) therefore see the same samples.
    seed_base: int | None = None
    _occ: dict = field(default_factory=dict, repr=False)

    def next_seed(self, key: str) -> int | None:
        if self.seed_base is None:
            return None
        k = self._occ.get(key, 0)
        self._occ[key] = k + 1
        h = hashlib.sha1(f"{self.seed_base}|{key}|{k}".encode()).hexdigest()
        return int(h[:8], 16) & 0x7FFFFFFF

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict:
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "total_tokens": self.total_tokens,
                "llm_seconds": round(self.llm_seconds, 2), "timeouts": self.timeouts,
                "truncated": self.truncated}


# Each evaluated problem runs in its own asyncio task with its own Usage object.
CURRENT_USAGE: contextvars.ContextVar[Usage | None] = contextvars.ContextVar("CURRENT_USAGE", default=None)

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


# How `think=True` (reasoning mode) and sampling parameters are sent depends on the provider kind (config.py):
#   vllm        local vLLM serving Qwen3-style models: thinking toggled by the chat template; the reasoning text
#               comes back inline in `content` as "...</think>answer".
#   openai      reasoning models (gpt-5*, o*) get reasoning_effort="none" for think=False (temperature allowed)
#               and reasoning_effort=<think_effort> for think=True (temperature fixed at the API default).
#               Non-reasoning models (gpt-4.1, gpt-4o) ignore `think` unless the provider sets think_model.
#   deepseek    think=True switches to the provider's think_model (deepseek-reasoner); the reasoning comes back
#               separately in `reasoning_content`, `content` is the answer.
#   compatible  any other OpenAI-compatible endpoint: plain parameters (think_model honoured if set).
def _read_cred_file() -> str:
    """Executor key handed to optimizer agents. codex and dsh drop every env var whose NAME contains KEY /
    TOKEN / SECRET from the shells they spawn, so a smoke test run by the agent would lose the key;
    `vflowx.search` writes it to a 0600 file in the run directory and passes the path as VFX_CRED_FILE."""
    path = os.environ.get("VFX_CRED_FILE")
    if path and os.path.isfile(path):
        return open(path).read().strip()
    return ""


class LLM:
    def __init__(self, base_url: str | None = None, model: str | None = None,
                 max_concurrency: int | None = None, default_max_tokens: int | None = None):
        """The executor configured in aflow.toml (`executor = "<provider>/<model>"`, overridable with
        VFX_EXECUTOR / --executor). `base_url` (tests) bypasses the config and means a local vLLM."""
        if base_url:
            self.cfg = config.Model(spec="local", provider="local", model=model, kind="vllm",
                                    base_urls=[base_url])
        else:
            self.cfg = config.resolve(config.executor_spec())
        c = self.cfg
        self.provider, self.kind = c.provider, c.kind
        # Several identical vLLM replicas: a request is routed by the hash of its content, so repeated/CRN-split
        # samples of the same call hit the same replica (prefix cache) and the assignment does not depend on timing.
        self.base_urls = c.base_urls
        self.base_url = self.base_urls[0]
        self.api_key = c.api_key or _read_cred_file() or ("EMPTY" if c.kind == "vllm" or not c.api_key_env else "")
        if not self.api_key:
            raise ValueError(f"executor {c.spec!r} needs an API key: export {c.api_key_env}=...")
        self.clients = [self._client(u) for u in self.base_urls]
        self.client = self.clients[0]
        self.model = model or c.model
        self.think_model = c.think_model
        self.think_effort = c.think_effort
        self.sem = asyncio.Semaphore((max_concurrency or c.concurrency) * len(self.clients))
        self.default_max_tokens = default_max_tokens or c.max_tokens
        self.global_usage = Usage()
        self.call_timeout = float(os.environ.get("VFX_CALL_TIMEOUT", "600"))

    def _client(self, url: str) -> AsyncOpenAI:
        local = re.match(r"https?://(localhost|127\.0\.0\.1|\[::1\])[:/]", url) is not None
        # A local vLLM must never go through the user's (possibly SOCKS) proxy env vars.
        http_client = httpx.AsyncClient(trust_env=False, timeout=1800) if local else None
        return AsyncOpenAI(base_url=url, api_key=self.api_key, timeout=1800,
                           max_retries=2 if local else 6, http_client=http_client)

    async def resolve_model(self) -> str:
        if not self.model:
            models = await self.client.models.list()
            self.model = models.data[0].id
        return self.model

    async def chat(self, messages: list[dict], *, temperature: float = 0.2, n: int = 1,
                   think: bool = False, max_tokens: int | None = None) -> list[str]:
        """Returns n completions (thinking traces stripped).

        With common random numbers active, each of the n samples is its own request with its own seed derived
        from the call content WITHOUT n, so sample j is identical whether it was drawn in a batch of 5 or as
        4+1 (vLLM prefix caching keeps the split cheap). Hosted APIs always get n=1 requests (not every provider
        supports n>1)."""
        u = CURRENT_USAGE.get()
        key = hashlib.sha1(json.dumps([messages, temperature, think, max_tokens], sort_keys=True).encode()).hexdigest()
        if n > 1 and ((u is not None and u.seed_base is not None) or self.kind != "vllm"):
            seeds = [u.next_seed(key) if u else None for _ in range(n)]  # drawn synchronously: deterministic order
            outs = await asyncio.gather(*(self._call(messages, temperature, 1, think, max_tokens, s, key)
                                          for s in seeds))
            return [o[0] for o in outs]
        return await self._call(messages, temperature, n, think, max_tokens, u.next_seed(key) if u else None, key)

    def _charge_failed(self, messages, n, max_tokens, t0, field_name: str):
        # The server already spent the tokens: charge a conservative estimate (prompt chars/3.5 and the full
        # completion budget) so failed calls are not "free" in the cost objective.
        est_p = int(sum(len(m.get("content") or "") for m in messages) / 3.5)
        est_c = n * (max_tokens or self.default_max_tokens)
        for uu in (CURRENT_USAGE.get(), self.global_usage):
            if uu is not None:
                uu.calls += 1
                setattr(uu, field_name, getattr(uu, field_name) + n)
                uu.prompt_tokens += est_p
                uu.completion_tokens += est_c
                uu.llm_seconds += time.time() - t0

    async def _call(self, messages, temperature, n, think, max_tokens, seed, key: str = "") -> list[str]:
        model = await self.resolve_model()
        t0 = time.time()
        async with self.sem:
            try:
                r = await asyncio.wait_for(self._create(model, messages, temperature, n, think, max_tokens, seed,
                                                        key),
                                           self.call_timeout)
            except asyncio.TimeoutError:
                # Degrade instead of killing the whole workflow: an empty reply becomes a failed candidate.
                self._charge_failed(messages, n, max_tokens, t0, "timeouts")
                return [""] * n
            except openai.BadRequestError as e:
                # OpenAI reasoning models reject (instead of truncating) a reply that hits max_completion_tokens.
                if "max_tokens" in str(e) or "output limit" in str(e):
                    self._charge_failed(messages, n, max_tokens, t0, "truncated")
                    return [""] * n
                raise
        dt = time.time() - t0
        return self._account_and_strip(r, dt, think)

    def _params(self, model, temperature, n, think, max_tokens, seed) -> dict:
        """Request parameters for the configured provider kind."""
        mt = max_tokens or self.default_max_tokens
        sampling = {"temperature": temperature, "top_p": 0.95 if temperature > 0 else 1.0}
        if self.kind == "vllm":
            return {"model": model, "n": n, "seed": seed, "max_tokens": mt, **sampling,
                    "extra_body": {"chat_template_kwargs": {"enable_thinking": bool(think)}}}
        if think and self.think_model:
            model = self.think_model
        if self.kind == "openai":
            kw = {"model": model, "n": n, "seed": seed, "max_completion_tokens": mt}
            if self.cfg.reasoning:
                if think:
                    return {**kw, "reasoning_effort": self.think_effort}
                return {**kw, "reasoning_effort": "none", **sampling}
            return {**kw, **sampling}
        return {"model": model, "max_tokens": mt, **sampling}

    async def _create(self, model, messages, temperature, n, think, max_tokens, seed=None, key: str = ""):
        c = self.clients[int(key[:8], 16) % len(self.clients)] if key and len(self.clients) > 1 else self.client
        kw = self._params(model, temperature, n, think, max_tokens, seed)
        return await c.chat.completions.create(messages=messages, **{k: v for k, v in kw.items() if v is not None})

    def _account_and_strip(self, r, dt: float, think: bool = False) -> list[str]:
        outs, cut = [], 0
        inline = self.kind == "vllm"  # only vLLM returns the reasoning inline in `content`
        for c in r.choices:
            txt = c.message.content or ""
            length = getattr(c, "finish_reason", None) == "length"
            if "</think>" in txt:  # template may inject the opening <think> into the prompt
                txt = txt.rsplit("</think>", 1)[1]
            elif inline and (think or length and txt.lstrip().startswith("<think>")):
                # Reasoning never finished (cut off by max_tokens): everything is unfinished thinking, not an
                # answer. Return "" (a failed candidate, like a timeout) instead of handing prose to extract_code.
                txt, cut = "", cut + 1
            elif not inline and think and length:
                cut += 1  # hosted reasoning model hit the limit: whatever partial answer exists is kept
            outs.append(_THINK_RE.sub("", txt).strip())
        for u in (CURRENT_USAGE.get(), self.global_usage):
            if u is None:
                continue
            u.calls += 1
            u.llm_seconds += dt
            u.truncated += cut
            if r.usage:
                u.prompt_tokens += r.usage.prompt_tokens
                u.completion_tokens += r.usage.completion_tokens
        return outs

    async def ask(self, prompt: str, *, system: str | None = None, **kw) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        return (await self.chat(msgs, n=1, **kw))[0]
