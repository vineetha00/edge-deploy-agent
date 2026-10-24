"""In-memory job queue between the control plane and the Pi worker (Postgres later)."""
from __future__ import annotations

import threading
import uuid
from collections import deque


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}
        self._queue: deque[str] = deque()

    def enqueue(self, argv: list[str]) -> str:
        job_id = uuid.uuid4().hex
        with self._lock:
            self._jobs[job_id] = {"id": job_id, "argv": argv, "status": "queued",
                                  "rc": None, "stdout": None}
            self._queue.append(job_id)
        return job_id

    def next_job(self) -> dict | None:
        with self._lock:
            if not self._queue:
                return None
            job = self._jobs[self._queue.popleft()]
            job["status"] = "running"
            return {"id": job["id"], "argv": job["argv"]}

    def complete(self, job_id: str, rc: int, stdout: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job["status"] != "running":
                return False
            job.update(status="done", rc=rc, stdout=stdout)
            return True

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None
