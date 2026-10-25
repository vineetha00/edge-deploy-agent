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
from edge_deploy_agent.docs import TavilyClient, make_researcher
from edge_deploy_agent.llm import NebiusClient
from edge_deploy_agent.tracing import setup_tracing
from edge_deploy_agent.tracking import MlflowTracker
from edge_deploy_agent.transport import HttpTransport


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--p95-ms", type=float, required=True)
    ap.add_argument("--ram-mb", type=float, required=True)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--planner-model", choices=list(MODELS), default="fast")
    ap.add_argument("--auto-approve", action="store_true")
    ap.add_argument("--wer-file", type=Path, help="JSON {variant: WER} from scripts/eval_wer.py")
    ap.add_argument("--mlflow-uri", help="MLflow tracking URI, e.g. http://localhost:5000")
    ap.add_argument("--otlp-endpoint", help="OTLP/HTTP collector, e.g. http://localhost:4318")
    ap.add_argument("--max-iters", type=int, help="cap on benchmark iterations per run")
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
    if args.otlp_endpoint:
        setup_tracing(args.otlp_endpoint)
    tracker = MlflowTracker(args.mlflow_uri) if args.mlflow_uri else None
    if tracker:
        tracker.start(goal)
    researcher = make_researcher(TavilyClient()) if os.environ.get("TAVILY_API_KEY") else None
    print("Tavily doc lookup:", "on" if researcher else "off (TAVILY_API_KEY not set)")
    graph = build_graph(
        LLMPlanner(NebiusClient(), MODELS[args.planner_model]),
        HttpTransport(client, os.environ["EDA_API_TOKEN"]),
        auto_approve=args.auto_approve, max_steps=args.max_steps,
        researcher=researcher, tracker=tracker,
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
    if tracker:
        print(f"MLflow run: {args.mlflow_uri} (run id {tracker.parent_run_id})")
    if args.otlp_endpoint:
        from opentelemetry import trace

        trace.get_tracer_provider().force_flush()


if __name__ == "__main__":
    main()
