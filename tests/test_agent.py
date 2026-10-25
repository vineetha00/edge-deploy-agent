import json

import pytest

from edge_deploy_agent.agent import LLMPlanner, build_graph, initial_state
from edge_deploy_agent.llm import LLMResult

GOAL = {"candidates": ["fp32", "int8"], "latency_p95_ms": 60.0, "ram_mb": 300.0}
CFG = {"configurable": {"thread_id": "t"}, "recursion_limit": 100}


class ScriptedPlanner:
    def __init__(self, proposals):
        self.proposals = list(proposals)

    def propose(self, state):
        return self.proposals.pop(0) if self.proposals else {"action": "finish", "reason": "done"}


class FakePi:
    """Synthetic benchmark results for unit tests only; never used for real claims."""

    def __init__(self):
        self.calls = []

    def run(self, argv):
        self.calls.append(argv)
        model = argv[argv.index("--model") + 1]
        threads = int(argv[argv.index("--threads") + 1])
        base = 120.0 if model == "fp32" else 50.0
        p50 = base / threads
        out = {"latency_ms": {"p50": p50, "p95": p50 * 1.2}, "peak_rss_mb": 200.0,
               "run_valid": True, "model": model, "threads": threads}
        return 0, json.dumps(out)


def bench(model, threads=4, **extra):
    return {"action": "benchmark", "model": model, "threads": threads, "iters": 20, **extra}


def test_auto_approve_finds_best_and_reports():
    pi = FakePi()
    g = build_graph(ScriptedPlanner([bench("fp32"), bench("int8", 4), bench("int8", 2)]), pi,
                    auto_approve=True)
    out = g.invoke(initial_state(GOAL), CFG)
    assert out["status"] == "done" and len(pi.calls) == 3
    assert out["best"]["proposal"]["model"] == "int8" and out["best"]["proposal"]["threads"] == 4
    assert "Best config" in out["report"]


def test_blocked_proposals_never_reach_the_pi():
    pi = FakePi()
    attacks = [
        {"action": "benchmark", "model": "int8;rm -rf /", "threads": 4, "iters": 5},
        {"action": "benchmark", "model": "not-a-candidate", "threads": 4, "iters": 5},
        {"action": "benchmark", "model": "int8", "threads": 99, "iters": 5},
        {"action": "shell", "cmd": "id"},
        {"action": "invalid", "raw": "ignore previous instructions"},
    ]
    g = build_graph(ScriptedPlanner(attacks), pi, auto_approve=True)
    out = g.invoke(initial_state(GOAL), CFG)
    assert pi.calls == []
    assert sum("blocked" in t for t in out["trials"]) == 5
    assert out["status"] == "done"


def test_approval_gate_pauses_then_resumes():
    pi = FakePi()
    g = build_graph(ScriptedPlanner([bench("int8")]), pi, auto_approve=False)
    g.invoke(initial_state(GOAL), CFG)
    assert g.get_state(CFG).next == ("approve",) and pi.calls == []
    g.update_state(CFG, {"approved": True})
    out = g.invoke(None, CFG)
    assert len(pi.calls) == 1 and out["status"] == "done"


def test_rejection_stops_without_executing():
    pi = FakePi()
    g = build_graph(ScriptedPlanner([bench("int8")]), pi, auto_approve=False)
    g.invoke(initial_state(GOAL), CFG)
    g.update_state(CFG, {"approved": False})
    out = g.invoke(None, CFG)
    assert pi.calls == [] and out["status"] == "rejected"


def test_max_steps_bounds_the_loop():
    pi = FakePi()
    g = build_graph(ScriptedPlanner([bench("fp32")] * 50), pi, auto_approve=True, max_steps=3)
    out = g.invoke(initial_state(GOAL), CFG)
    assert len(pi.calls) == 3 and out["status"] == "done"


class FakeClient:
    def __init__(self, content):
        self.content = content

    def chat(self, model, messages, max_tokens=1500):
        return LLMResult(self.content, "", 1, 1, None)


@pytest.mark.parametrize("content,expected", [
    ('Sure! {"action":"finish","reason":"ok"}', "finish"),
    ('{"action":"benchmark","model":"int8","threads":2,"iters":10}', "benchmark"),
    ("I cannot comply", "invalid"),
    ("{not json}", "invalid"),
    ("[1,2,3]", "invalid"),
])
def test_llm_planner_parsing(content, expected):
    p = LLMPlanner(FakeClient(content), "m").propose(initial_state(GOAL))
    assert p["action"] == expected


def test_opt_level_reaches_the_pi_command():
    pi = FakePi()
    g = build_graph(ScriptedPlanner([bench("int8", 2, opt=1)]), pi, auto_approve=True)
    g.invoke(initial_state(GOAL), CFG)
    assert pi.calls[0][-2:] == ["--opt", "1"]


def test_iters_over_goal_limit_is_blocked_before_the_pi():
    pi = FakePi()
    goal = {**GOAL, "max_iters": 20}
    g = build_graph(ScriptedPlanner([bench("fp32") | {"iters": 500}, bench("fp32")]), pi,
                    auto_approve=True)
    out = g.invoke(initial_state(goal), CFG)
    assert out["trials"][0]["blocked"].startswith("iters over goal limit")
    assert all(int(c[c.index("--iters") + 1]) <= 20 for c in pi.calls)
