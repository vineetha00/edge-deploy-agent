from __future__ import annotations

from fastapi import FastAPI
from prometheus_client import Counter, make_asgi_app
from pydantic import BaseModel

from . import __version__
from .runner import validate

COMMANDS = Counter("eda_runner_commands_total", "Runner command checks by verdict", ["verdict"])

app = FastAPI(title="Edge Deploy Agent", version=__version__)
app.mount("/metrics", make_asgi_app())


class CommandRequest(BaseModel):
    argv: list[str]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/runner/validate")
def runner_validate(req: CommandRequest) -> dict:
    verdict = validate(req.argv)
    COMMANDS.labels(verdict="allowed" if verdict.allowed else "blocked").inc()
    return {"allowed": verdict.allowed, "reason": verdict.reason}
