"""Transport that sends validated commands to the Pi through the control-plane job queue."""
from __future__ import annotations

import time

import httpx


class HttpTransport:
    def __init__(self, client: httpx.Client, token: str, poll_s: float = 1.0,
                 timeout_s: float = 900.0):
        self.client = client
        self.headers = {"Authorization": f"Bearer {token}"}
        self.poll_s, self.timeout_s = poll_s, timeout_s

    def run(self, argv: list[str]) -> tuple[int, str]:
        r = self.client.post("/jobs", json={"argv": argv}, headers=self.headers)
        if r.status_code != 200:
            return 126, f"job rejected: HTTP {r.status_code}"
        job_id = r.json()["id"]
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            job = self.client.get(f"/jobs/{job_id}", headers=self.headers).json()
            if job["status"] == "done":
                return job["rc"], job["stdout"]
            time.sleep(self.poll_s)
        return 124, "timed out waiting for the Pi worker"
