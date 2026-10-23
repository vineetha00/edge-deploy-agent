"""Nebius Token Factory client (OpenAI-compatible) with token and cost accounting."""
from __future__ import annotations

import os
from dataclasses import dataclass

import httpx
from prometheus_client import Counter

from .config import BASE_URL, PRICES_PER_M

TOKENS = Counter("eda_llm_tokens_total", "LLM tokens by model and kind", ["model", "kind"])
CALLS = Counter("eda_llm_calls_total", "LLM calls by model and outcome", ["model", "outcome"])


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMResult:
    content: str
    reasoning: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float | None


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    price = PRICES_PER_M.get(model)
    if price is None:
        return None
    return (prompt_tokens * price[0] + completion_tokens * price[1]) / 1_000_000


class NebiusClient:
    def __init__(self, api_key: str | None = None, transport: httpx.BaseTransport | None = None):
        key = api_key or os.environ.get("NEBIUS_API_KEY")
        if not key:
            raise LLMError("NEBIUS_API_KEY is not set")
        self._http = httpx.Client(
            base_url=BASE_URL,
            headers={"Authorization": f"Bearer {key}"},
            timeout=120.0,
            transport=transport,
        )

    def chat(self, model: str, messages: list[dict], max_tokens: int = 1500) -> LLMResult:
        # Reasoning models spend completion tokens on thinking, so keep max_tokens generous.
        resp = self._http.post(
            "/chat/completions",
            json={"model": model, "messages": messages, "max_tokens": max_tokens},
        )
        if resp.status_code != 200:
            CALLS.labels(model=model, outcome=f"http_{resp.status_code}").inc()
            raise LLMError(f"Token Factory returned HTTP {resp.status_code}")
        data = resp.json()
        msg = data["choices"][0]["message"]
        usage = data.get("usage") or {}
        pt = int(usage.get("prompt_tokens", 0))
        ct = int(usage.get("completion_tokens", 0))
        TOKENS.labels(model=model, kind="prompt").inc(pt)
        TOKENS.labels(model=model, kind="completion").inc(ct)
        CALLS.labels(model=model, outcome="ok").inc()
        return LLMResult(
            content=(msg.get("content") or "").strip(),
            reasoning=(msg.get("reasoning") or "").strip(),
            prompt_tokens=pt,
            completion_tokens=ct,
            cost_usd=estimate_cost(model, pt, ct),
        )
