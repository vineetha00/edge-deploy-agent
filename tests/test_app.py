from fastapi.testclient import TestClient

from edge_deploy_agent.app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_validate_endpoint_and_metrics():
    assert client.post("/runner/validate", json={"argv": ["nproc"]}).json()["allowed"] is True
    assert client.post("/runner/validate", json={"argv": ["rm", "-rf", "/"]}).json()["allowed"] is False
    body = client.get("/metrics").text
    assert 'eda_runner_commands_total{verdict="blocked"}' in body
