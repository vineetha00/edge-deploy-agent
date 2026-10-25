import json

import httpx
import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from edge_deploy_agent import tracing
from edge_deploy_agent.agent import build_graph, initial_state
from edge_deploy_agent.llm import NebiusClient
from edge_deploy_agent.tracking import MlflowTracker, trial_record
from tests.test_agent import CFG, GOAL, FakePi, ScriptedPlanner, bench

EXPORTER = InMemorySpanExporter()
tracing.setup_tracing(exporter=EXPORTER)  # global provider can only be set once per process


@pytest.fixture
def spans():
    EXPORTER.clear()
    yield EXPORTER
    EXPORTER.clear()


def test_every_graph_step_is_a_span(spans):
    g = build_graph(ScriptedPlanner([bench("int8"), {"action": "finish"}]), FakePi(),
                    auto_approve=True)
    g.invoke(initial_state(GOAL), CFG)
    names = [s.name for s in spans.get_finished_spans()]
    for step in ("research", "propose", "guard", "execute", "evaluate", "report"):
        assert f"agent.{step}" in names
    ex = next(s for s in spans.get_finished_spans() if s.name == "agent.execute")
    assert ex.attributes["eda.proposal.model"] == "int8"
    assert ex.attributes["eda.result.run_valid"] is True


def test_blocked_proposal_span_records_reason(spans):
    bad = {"action": "benchmark", "model": "int8; rm -rf /", "threads": 4, "iters": 5}
    g = build_graph(ScriptedPlanner([bad]), FakePi(), auto_approve=True, max_steps=1)
    g.invoke(initial_state(GOAL), CFG)
    guard = next(s for s in spans.get_finished_spans() if s.name == "agent.guard")
    assert guard.attributes["eda.verdict.allowed"] is False


def test_llm_call_span_has_token_usage(spans):
    body = {"choices": [{"message": {"content": "hi"}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7}}
    c = NebiusClient("k", transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body)))
    c.chat("nvidia/Nemotron-Nano", [{"role": "user", "content": "x"}])
    s = next(s for s in spans.get_finished_spans() if s.name == "llm.chat")
    assert s.attributes["gen_ai.usage.input_tokens"] == 11
    assert s.attributes["gen_ai.usage.output_tokens"] == 7


def test_trial_record_splits_params_metrics_tags():
    goal = {**GOAL, "wer": {"int8": 0.059}, "max_wer": 0.065}
    trial = {"proposal": bench("int8"), "result": {
        "latency_ms": {"p50": 40, "p95": 48, "mean": 41}, "peak_rss_mb": 200,
        "run_valid": True, "model": "int8", "cooldown_s": 10.0}}
    params, metrics, tags = trial_record(trial, goal)
    assert params == {"model": "int8", "threads": 4, "iters": 20}
    assert metrics["latency_p95_ms"] == 48 and metrics["wer"] == 0.059
    assert tags == {"outcome": "ran", "run_valid": "True", "meets_budget": "True"}
    _, m2, t2 = trial_record({"proposal": bench("x"), "blocked": "nope"}, goal)
    assert m2 == {} and t2["outcome"] == "blocked"


def test_mlflow_logs_session_and_nested_trials(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    tracker = MlflowTracker(uri, experiment="test",
                            artifact_location=(tmp_path / "artifacts").as_uri())
    tracker.start(GOAL)
    bad = {"action": "benchmark", "model": "fp32", "threads": 9, "iters": 5}
    g = build_graph(ScriptedPlanner([bench("fp32"), bad, bench("int8")]), FakePi(),
                    auto_approve=True, max_steps=3, tracker=tracker)
    out = g.invoke(initial_state(GOAL), CFG)
    client = mlflow.MlflowClient(uri)
    exp = client.get_experiment_by_name("test")
    runs = client.search_runs([exp.experiment_id])
    parent = client.get_run(tracker.parent_run_id)
    children = [r for r in runs if r.data.tags.get("mlflow.parentRunId") == tracker.parent_run_id]
    assert len(children) == len(out["trials"]) == 3
    assert parent.data.tags["best_model"] == "int8"
    assert parent.data.metrics["n_blocked"] == 1
    path = client.download_artifacts(tracker.parent_run_id, "trials.json")
    with open(path) as f:
        art = json.load(f)
    assert len(art["trials"]) == 3
