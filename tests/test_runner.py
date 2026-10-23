import pytest

from edge_deploy_agent.runner import validate

ALLOWED = [
    ["uname", "-a"],
    ["python3", "--version"],
    ["nproc"],
    ["free", "-h"],
    ["vcgencmd", "measure_temp"],
    ["vcgencmd", "get_throttled"],
    ["python3", "/home/agent/bench/run.py", "--model", "mobilenet_v2.int8"],
]


@pytest.mark.parametrize("argv", ALLOWED)
def test_allowed(argv):
    assert validate(argv).allowed
