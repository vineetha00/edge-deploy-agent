# Edge Deploy Agent

A Nemotron-driven agent that optimizes and benchmarks models on a Raspberry Pi 4
within a latency and RAM budget, with guardrails, observability and evals.

Status: early skeleton for the Nebius x NVIDIA Global AI Hackathon (Coding & Agentic
Engineering track). Pre-existing work: none; this repo was created after Aug 26, 2026.

## Planned scope
- LangGraph agent with checkpoints and a human-approval gate
- OpenTelemetry traces and Prometheus metrics
- Allowlisted SSH runner, NeMo Guardrails, red-team suite with block-rate report
- FastAPI control plane, Docker Compose locally
- pgvector run memory, MLflow experiment tracking
- Terraform + Nebius managed Kubernetes deployment
- Triton Inference Server serving the reference model for parity checks

## Local development
    pip install -e ".[dev]"
    ruff check . && pytest tests -v
