#!/usr/bin/env python3
"""Run the agent against the control plane (and, through it, the Pi worker).

  EDA_API_TOKEN=... NEBIUS_API_KEY=... python scripts/run_agent.py \
      --api http://localhost:8000 --candidates smoke_fp32,smoke_int8 \
      --p95-ms 5 --ram-mb 300 [--auto-approve] [--planner-model fast|editor|planner]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import httpx

from edge_deploy_agent.agent import LLMPlanner, build_graph, initial_state
from edge_deploy_agent.config import MODELS
from edge_deploy_agent.docs import DocsError, TavilyClient, make_researcher
from edge_deploy_agent.llm import LLMError, NebiusClient
from edge_deploy_agent.tracing import setup_tracing
from edge_deploy_agent.tracking import MlflowTracker, SafeTracker
from edge_deploy_agent.transport import HttpTransport


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--p95-ms", type=float, required=True)
    ap.add_argument("--ram-mb", type=float, required=True)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--planner-model", choices=list(MODELS), default="editor")  # Nemotron Super
    ap.add_argument("--auto-approve", action="store_true")
    ap.add_argument("--wer-file", type=Path, help="JSON {variant: WER} from scripts/eval_wer.py")
    ap.add_argument("--mlflow-uri", help="MLflow tracking URI, e.g. http://127.0.0.1:5001 "
                    "(avoid 5000 on macOS: AirPlay Receiver answers it with HTTP 403)")
    ap.add_argument("--otlp-endpoint", help="OTLP/HTTP collector, e.g. http://localhost:4318")
    ap.add_argument("--max-iters", type=int, help="cap on benchmark iterations per run")
    ap.add_argument("--save", type=Path, help="write goal, trials and best config as JSON")
    ap.add_argument("--guardrails", action="store_true",
                    help="NeMo Guardrails input rail on docs + output rail on proposals")
    ap.add_argument("--no-docs", action="store_true", help="skip the Tavily doc lookup")
    ap.add_argument("--max-wer", type=float, help="accuracy gate, e.g. 0.05 for 5%% WER")
    args = ap.parse_args()

    goal = {"candidates": args.candidates.split(","), "latency_p95_ms": args.p95_ms,
            "ram_mb": args.ram_mb}
    if args.max_iters is not None:
        goal["max_iters"] = args.max_iters
    if args.max_wer is not None:
        if not args.wer_file:
            raise SystemExit("--max-wer needs --wer-file")
        goal["max_wer"] = args.max_wer
        goal["wer"] = json.loads(args.wer_file.read_text())
    client = httpx.Client(base_url=args.api, timeout=30.0)
    llm, researcher = preflight(client, require_docs=not args.no_docs)
    if args.otlp_endpoint:
        setup_tracing(args.otlp_endpoint)
    if args.mlflow_uri:
        check_mlflow(args.mlflow_uri)
    tracker = SafeTracker(MlflowTracker(args.mlflow_uri)) if args.mlflow_uri else None
    if tracker:
        tracker.start(goal)
    try:
        run(args, goal, client, llm, researcher, tracker)
    except BaseException as exc:  # includes Ctrl+C: mark the MLflow run failed, then re-raise
        if tracker:
            tracker.fail(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if args.otlp_endpoint:
            from opentelemetry import trace

            trace.get_tracer_provider().force_flush()


def check_mlflow(uri: str) -> None:
    try:
        r = httpx.get(f"{uri.rstrip('/')}/health", timeout=5.0)
    except httpx.HTTPError as exc:
        raise SystemExit(f"MLflow not reachable at {uri} ({exc}); is `mlflow server` running?")
    if r.status_code != 200 or "OK" not in r.text:
        hint = (" Port 5000 on macOS is usually AirPlay Receiver: run MLflow with --port 5001."
                if r.status_code == 403 else "")
        raise SystemExit(f"{uri} answered HTTP {r.status_code}, not MLflow.{hint}")
    print("MLflow tracking: server OK")


def preflight(client: httpx.Client, require_docs: bool):
    """Check every dependency before starting, so a bad key fails in seconds with a clear message."""
    try:
        h = client.get("/health")
        h.raise_for_status()
    except httpx.HTTPError as exc:
        raise SystemExit(f"API not reachable at {client.base_url} ({exc}); is uvicorn running?")
    if not os.environ.get("EDA_API_TOKEN"):
        raise SystemExit("EDA_API_TOKEN is not set")
    try:
        llm = NebiusClient()
        llm.check()
    except LLMError as exc:
        raise SystemExit(str(exc))
    print("Nebius Token Factory: key OK")
    if not require_docs:
        print("Tavily doc lookup: off (--no-docs)")
        return llm, None
    try:
        tavily = TavilyClient()
        tavily.search("ONNX Runtime quantization", max_results=1)
    except DocsError as exc:
        raise SystemExit(f"Tavily check failed: {exc}. Fix TAVILY_API_KEY or pass --no-docs.")
    print("Tavily doc lookup: key OK")
    return llm, make_researcher(tavily)


def run(args, goal, client, llm, researcher, tracker) -> None:
    rails = None
    if args.guardrails:
        from edge_deploy_agent.guardrails import NemoRails

        rails = NemoRails(MODELS["fast"])
        print(f"NeMo Guardrails: on (judge {MODELS['fast']})")
    graph = build_graph(
        LLMPlanner(llm, MODELS[args.planner_model]),
        HttpTransport(client, os.environ["EDA_API_TOKEN"]),
        auto_approve=args.auto_approve, max_steps=args.max_steps,
        researcher=researcher, tracker=tracker, rails=rails,
    )
    cfg = {"configurable": {"thread_id": "cli"}, "recursion_limit": 200}
    state = graph.invoke(initial_state(goal), cfg)
    while graph.get_state(cfg).next:
        proposal = graph.get_state(cfg).values["proposal"]
        answer = input(f"Approve {proposal}? [y/N] ")
        graph.update_state(cfg, {"approved": answer.strip().lower() == "y"})
        state = graph.invoke(None, cfg)
    for i, t in enumerate(state["trials"], 1):
        print(i, t.get("proposal"), "->", t.get("blocked") or t.get("error") or (
            f"p50 {t['result']['latency_ms']['p50']:.3f} ms, "
            f"p95 {t['result']['latency_ms']['p95']:.3f} ms, valid={t['result']['run_valid']}"))
    docs = state.get("docs") or []
    flagged = sum(1 for d in docs if d.get("flags"))
    print(f"Docs used: {len(docs)} ({flagged} with injection-like lines removed)")
    print(state["report"])
    if tracker and not tracker.disabled:
        print(f"MLflow run: {args.mlflow_uri}")
    if args.save:
        save_run(args.save, goal, state)
        print(f"Saved run to {args.save}")


def save_run(path: Path, goal: dict, state: dict) -> None:
    """Record of the run: the evidence behind any number quoted from it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "goal": goal,
        "trials": state.get("trials", []),
        "best": state.get("best"),
        "report": state.get("report"),
        "docs": [{"title": d.get("title"), "url": d.get("url"), "flags": d.get("flags")}
                 for d in state.get("docs") or []],
    }, indent=2))


if __name__ == "__main__":
    main()
