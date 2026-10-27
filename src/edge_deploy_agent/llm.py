"""Nebius Token Factory client (OpenAI-compatible) with token and cost accounting."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx
from prometheus_client import Counter

from .config import BASE_URL, PRICES_PER_M
from .tracing import tracer

TOKENS = Counter("eda_llm_tokens_total", "LLM tokens by model and kind", ["model", "kind"])
CALLS = Counter("eda_llm_calls_total", "LLM calls by model and outcome", ["model", "outcome"])


RETRY_STATUS = {429, 500, 502, 503, 504, 529}


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
    def __init__(self, api_key: str | None = None, transport: httpx.BaseTransport | None = None,
                 max_retries: int = 5, backoff_s: float = 2.0):
        self.max_retries, self.backoff_s, self._sleep = max_retries, backoff_s, time.sleep
        key = api_key or os.environ.get("NEBIUS_API_KEY")
        if not key:
            raise LLMError("NEBIUS_API_KEY is not set")
        self._http = httpx.Client(
            base_url=BASE_URL,
            headers={"Authorization": f"Bearer {key}"},
            timeout=120.0,
            transport=transport,
        )

    def check(self) -> None:
        """Fail fast with a clear message if the key is wrong (free call, no tokens used)."""
        resp = self._http.get("/models")
        if resp.status_code == 401:
            raise LLMError("Token Factory rejected NEBIUS_API_KEY (HTTP 401): check the key")
        if resp.status_code != 200:
            raise LLMError(f"Token Factory /models returned HTTP {resp.status_code}")

    def chat(self, model: str, messages: list[dict], max_tokens: int = 1500) -> LLMResult:
        with tracer.start_as_current_span("llm.chat") as span:
            span.set_attribute("gen_ai.system", "nebius_token_factory")
            span.set_attribute("gen_ai.request.model", model)
            res = self._chat(model, messages, max_tokens)
            span.set_attribute("gen_ai.usage.input_tokens", res.prompt_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", res.completion_tokens)
            if res.cost_usd is not None:
                span.set_attribute("eda.llm.cost_usd", res.cost_usd)
            return res

    def _post_with_retry(self, body: dict, model: str) -> httpx.Response:
        """Retry briefly on overload (429, 529) and server errors; other errors return at once."""
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._http.post("/chat/completions", json=body)
            except httpx.TransportError:
                if attempt == self.max_retries:
                    raise
            else:
                if resp.status_code not in RETRY_STATUS or attempt == self.max_retries:
                    return resp
                CALLS.labels(model=model, outcome=f"retry_http_{resp.status_code}").inc()
            self._sleep(min(2 ** attempt * self.backoff_s, 60.0))
        raise AssertionError("unreachable")

    def _chat(self, model: str, messages: list[dict], max_tokens: int) -> LLMResult:
        # Reasoning models spend completion tokens on thinking, so keep max_tokens generous.
        resp = self._post_with_retry(
            {"model": model, "messages": messages, "max_tokens": max_tokens}, model)
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
