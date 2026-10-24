import importlib.util
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from edge_deploy_agent.app import app
from edge_deploy_agent.runner import BENCH_SCRIPT, PI_PYTHON
from edge_deploy_agent.transport import HttpTransport

spec = importlib.util.spec_from_file_location(
    "pi_worker", Path(__file__).parent.parent / "pi" / "worker.py")
worker = importlib.util.module_from_spec(spec)
sys.modules["pi_worker"] = worker
spec.loader.exec_module(worker)

BENCH = [PI_PYTHON, BENCH_SCRIPT, "--model", "m", "--threads", "2", "--iters", "5"]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("EDA_API_TOKEN", "api-secret")
    monkeypatch.setenv("EDA_WORKER_TOKEN", "worker-secret")
    return TestClient(app)


def test_auth_required_and_fails_closed(client, monkeypatch):
    assert client.post("/jobs", json={"argv": ["nproc"]}).status_code == 401
    bad = {"Authorization": "Bearer nope"}
    assert client.post("/worker/next", headers=bad).status_code == 401
    # API token must not work as a worker token and vice versa
    assert client.post("/worker/next", headers={"Authorization": "Bearer api-secret"}
                       ).status_code == 401
    monkeypatch.delenv("EDA_API_TOKEN")
    assert client.post("/jobs", json={"argv": ["nproc"]},
                       headers={"Authorization": "Bearer "}).status_code == 401


def test_disallowed_command_rejected_at_the_api(client):
    r = client.post("/jobs", json={"argv": ["rm", "-rf", "/"]},
                    headers={"Authorization": "Bearer api-secret"})
    assert r.status_code == 403


def test_worker_blocks_even_if_queue_is_poisoned(client):
    from edge_deploy_agent.app import store
    while store.next_job():  # drain jobs left by other tests
        pass
    job_id = store.enqueue(["rm", "-rf", "/"])  # bypasses the API check on purpose
    assert worker.process_one(client, "worker-secret") is True  # default executor validates
    job = store.get(job_id)
    assert job["status"] == "done" and job["rc"] == 126 and "blocked" in job["stdout"]


def test_end_to_end_job_roundtrip(client):
    from edge_deploy_agent.app import store
    while store.next_job():  # drain jobs left by other tests
        pass
    t = HttpTransport(client, "api-secret", poll_s=0.05, timeout_s=10)

    def worker_thread():
        for _ in range(100):
            if worker.process_one(client, "worker-secret", run=lambda a: (0, '{"ok": true}')):
                return
            time.sleep(0.05)

    th = threading.Thread(target=worker_thread)
    th.start()
    rc, out = t.run(BENCH)
    th.join()
    assert rc == 0 and out == '{"ok": true}'
