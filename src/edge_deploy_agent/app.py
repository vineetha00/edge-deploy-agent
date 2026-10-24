from __future__ import annotations

import hmac
import os

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from prometheus_client import Counter, make_asgi_app
from pydantic import BaseModel

from . import __version__
from .jobs import JobStore
from .runner import validate

COMMANDS = Counter("eda_runner_commands_total", "Runner command checks by verdict", ["verdict"])

app = FastAPI(title="Edge Deploy Agent", version=__version__)
app.mount("/metrics", make_asgi_app())
store = JobStore()


def _require(env_name: str):
    """Bearer-token check; fails closed when the token env var is not configured."""
    def dep(authorization: str = Header(default="")) -> None:
        expected = os.environ.get(env_name)
        ok = bool(expected) and hmac.compare_digest(
            authorization.encode(), f"Bearer {expected}".encode()
        )
        if not ok:
            raise HTTPException(status_code=401, detail="unauthorized")
    return dep


api_auth = Depends(_require("EDA_API_TOKEN"))
worker_auth = Depends(_require("EDA_WORKER_TOKEN"))


class CommandRequest(BaseModel):
    argv: list[str]


class ResultRequest(BaseModel):
    id: str
    rc: int
    stdout: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/runner/validate")
def runner_validate(req: CommandRequest) -> dict:
    verdict = validate(req.argv)
    COMMANDS.labels(verdict="allowed" if verdict.allowed else "blocked").inc()
    return {"allowed": verdict.allowed, "reason": verdict.reason}


@app.post("/jobs", dependencies=[api_auth])
def create_job(req: CommandRequest) -> dict:
    verdict = validate(req.argv)
    COMMANDS.labels(verdict="allowed" if verdict.allowed else "blocked").inc()
    if not verdict.allowed:
        raise HTTPException(status_code=403, detail=verdict.reason)
    return {"id": store.enqueue(req.argv)}


@app.get("/jobs/{job_id}", dependencies=[api_auth])
def get_job(job_id: str) -> dict:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="no such job")
    return job


@app.post("/worker/next", dependencies=[worker_auth], response_model=None)
def worker_next() -> Response | dict:
    job = store.next_job()
    return job if job else Response(status_code=204)


@app.post("/worker/result", dependencies=[worker_auth])
def worker_result(req: ResultRequest) -> dict:
    if not store.complete(req.id, req.rc, req.stdout):
        raise HTTPException(status_code=409, detail="job not running")
    return {"ok": True}
