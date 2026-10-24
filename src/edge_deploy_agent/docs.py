"""Tavily web search used by the agent to look up documentation at runtime."""
from __future__ import annotations

import os

import httpx

TAVILY_URL = "https://api.tavily.com"


class DocsError(RuntimeError):
    pass


class TavilyClient:
    def __init__(self, api_key: str | None = None, transport: httpx.BaseTransport | None = None):
        key = api_key or os.environ.get("TAVILY_API_KEY")
        if not key:
            raise DocsError("TAVILY_API_KEY is not set")
        self._http = httpx.Client(
            base_url=TAVILY_URL, headers={"Authorization": f"Bearer {key}"},
            timeout=30.0, transport=transport,
        )

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        r = self._http.post("/search", json={"query": query, "max_results": max_results})
        if r.status_code != 200:
            raise DocsError(f"Tavily returned HTTP {r.status_code}")
        results = r.json().get("results") or []
        return [{"title": x.get("title", ""), "url": x.get("url", ""),
                 "content": x.get("content", "")} for x in results if isinstance(x, dict)]


def make_researcher(client: TavilyClient):
    """Return a callable that fetches docs relevant to a goal."""
    def researcher(goal: dict) -> list[dict]:
        query = (
            "ONNX Runtime INT8 quantization performance Raspberry Pi 4 Cortex-A72 "
            "threads graph optimization " + " ".join(goal.get("candidates", []))
        )
        return client.search(query[:300])
    return researcher
