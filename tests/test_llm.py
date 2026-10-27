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


def test_retries_on_overload_then_succeeds():
    import httpx

    from edge_deploy_agent.llm import NebiusClient

    codes = iter([529, 503, 200])
    body = {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    def handler(req):
        c = next(codes)
        return httpx.Response(c, json=body if c == 200 else {})

    c = NebiusClient("k", transport=httpx.MockTransport(handler))
    slept = []
    c._sleep = slept.append
    assert c.chat("m", [{"role": "user", "content": "x"}]).content == "ok"
    assert slept == [2.0, 4.0]


def test_gives_up_after_max_retries_and_does_not_retry_auth_errors():
    import httpx
    import pytest

    from edge_deploy_agent.llm import LLMError, NebiusClient

    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(529 if len(calls) < 100 else 200)

    c = NebiusClient("k", transport=httpx.MockTransport(handler), max_retries=2)
    c._sleep = lambda s: None
    with pytest.raises(LLMError, match="529"):
        c.chat("m", [])
    assert len(calls) == 3
    calls.clear()
    c2 = NebiusClient("k", transport=httpx.MockTransport(lambda r: httpx.Response(401)))
    c2._sleep = lambda s: None
    with pytest.raises(LLMError, match="401"):
        c2.chat("m", [])
