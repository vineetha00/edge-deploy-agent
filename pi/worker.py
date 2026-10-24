#!/usr/bin/env python3
"""Pi worker: polls the control plane over outbound HTTPS and runs allowlisted commands.

Nothing listens on the Pi. Every job is validated against the allowlist locally before it
runs, even though the control plane already checked it (defense in depth).
Deploy next to a copy of runner.py:  /home/agent/worker/worker.py + runner.py
Env: EDA_API_URL, EDA_WORKER_TOKEN
"""
from __future__ import annotations

import os
import subprocess
import time

import httpx

try:
    from edge_deploy_agent.runner import validate
except ImportError:  # on the Pi, runner.py sits next to this file
    from runner import validate


def execute(argv: list[str]) -> tuple[int, str]:
    verdict = validate(argv)
    if not verdict.allowed:
        return 126, f"blocked by worker allowlist: {verdict.reason}"
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=600, check=False)
    except subprocess.TimeoutExpired:
        return 124, "command timed out"
    return p.returncode, p.stdout if p.returncode == 0 else (p.stdout + p.stderr)[-4000:]


def process_one(client: httpx.Client, token: str, run=execute) -> bool:
    headers = {"Authorization": f"Bearer {token}"}
    r = client.post("/worker/next", headers=headers)
    if r.status_code == 204:
        return False
    r.raise_for_status()
    job = r.json()
    rc, out = run(job["argv"])
    client.post("/worker/result", json={"id": job["id"], "rc": rc, "stdout": out},
                headers=headers).raise_for_status()
    return True


def main() -> None:
    token = os.environ["EDA_WORKER_TOKEN"]
    with httpx.Client(base_url=os.environ["EDA_API_URL"], timeout=30.0) as client:
        while True:
            try:
                busy = process_one(client, token)
            except httpx.HTTPError as exc:
                print(f"worker: {exc}", flush=True)
                busy = False
            if not busy:
                time.sleep(2)


if __name__ == "__main__":
    main()
