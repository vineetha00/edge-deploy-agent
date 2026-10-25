"""OpenTelemetry tracing for the agent: one span per graph step, one per LLM call.

Spans go to an OTLP collector (e.g. Jaeger, docker-compose `jaeger` service) when
`setup_tracing(otlp_endpoint=...)` is called; otherwise the API's no-op tracer is used.
"""
from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

from opentelemetry import trace

tracer = trace.get_tracer("edge_deploy_agent")


def setup_tracing(otlp_endpoint: str | None = None, service: str = "edge-deploy-agent",
                  exporter: Any = None) -> None:
    """Install an SDK tracer provider. Pass `exporter` (tests) or an OTLP/HTTP endpoint."""
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": service}))
    if exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    elif otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{otlp_endpoint.rstrip('/')}/v1/traces")))
    trace.set_tracer_provider(provider)


def _set(span, key: str, value: Any) -> None:
    if isinstance(value, bool | int | float | str):
        span.set_attribute(key, value)


def annotate(span, update: dict) -> None:
    """Copy the useful parts of a node's state update onto its span."""
    trials = update.get("trials")
    p = update.get("proposal") or (trials[-1].get("proposal") if trials else None) or {}
    for k in ("action", "model", "threads", "iters", "opt"):
        _set(span, f"eda.proposal.{k}", p.get(k))
    v = update.get("verdict") or {}
    _set(span, "eda.verdict.allowed", v.get("allowed"))
    _set(span, "eda.verdict.reason", v.get("reason"))
    if trials:
        last = trials[-1]
        _set(span, "eda.trial.blocked", last.get("blocked"))
        _set(span, "eda.trial.error", last.get("error"))
        r = last.get("result") or {}
        _set(span, "eda.result.p95_ms", (r.get("latency_ms") or {}).get("p95"))
        _set(span, "eda.result.peak_rss_mb", r.get("peak_rss_mb"))
        _set(span, "eda.result.run_valid", r.get("run_valid"))
        _set(span, "eda.result.cooldown_s", r.get("cooldown_s"))
    if "docs" in update:
        docs = update["docs"] or []
        _set(span, "eda.docs.count", len(docs))
        _set(span, "eda.docs.flagged", sum(1 for d in docs if d.get("flags")))
    _set(span, "eda.report", update.get("report"))


def traced_node(name: str, fn: Callable[[dict], dict]) -> Callable[[dict], dict]:
    @wraps(fn)
    def wrapper(state: dict) -> dict:
        with tracer.start_as_current_span(f"agent.{name}") as span:
            _set(span, "eda.step", state.get("steps", 0))
            out = fn(state)
            annotate(span, out or {})
            return out
    return wrapper
