import httpx
import pytest

from edge_deploy_agent import config
from edge_deploy_agent.llm import LLMError, NebiusClient, estimate_cost

MODEL = config.MODELS["fast"]


def _client(handler):
    return NebiusClient(api_key="test-key", transport=httpx.MockTransport(handler))


def test_chat_parses_content_reasoning_and_usage():
    def handler(request):
        assert request.headers["authorization"] == "Bearer test-key"
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "\nready", "reasoning": "thinking..."}}],
            "usage": {"prompt_tokens": 23, "completion_tokens": 57},
        })

    r = _client(handler).chat(MODEL, [{"role": "user", "content": "hi"}])
    assert r.content == "ready" and r.reasoning == "thinking..."
    assert (r.prompt_tokens, r.completion_tokens) == (23, 57)


def test_http_error_raises():
    with pytest.raises(LLMError):
        _client(lambda req: httpx.Response(401, json={"detail": "no"})).chat(MODEL, [])


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    with pytest.raises(LLMError):
        NebiusClient()


def test_estimate_cost(monkeypatch):
    monkeypatch.setitem(config.PRICES_PER_M, MODEL, (0.30, 0.90))
    from edge_deploy_agent import llm
    monkeypatch.setattr(llm, "PRICES_PER_M", config.PRICES_PER_M)
    assert estimate_cost(MODEL, 1_000_000, 1_000_000) == pytest.approx(1.20)
    assert estimate_cost("unknown/model", 10, 10) is None


def test_known_prices_loaded():
    # one Nano call from the live test: 23 prompt + 57 completion tokens
    assert estimate_cost(config.MODELS["fast"], 23, 57) == pytest.approx(0.00001506, rel=1e-3)
    assert estimate_cost(config.MODELS["planner"], 30_000, 5_000) == pytest.approx(0.045)
