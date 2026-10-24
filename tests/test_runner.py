import pytest

from edge_deploy_agent.runner import BENCH_SCRIPT as BENCH
from edge_deploy_agent.runner import PI_PYTHON as PY
from edge_deploy_agent.runner import validate

ALLOWED = [
    ["uname", "-a"],
    ["python3", "--version"],
    ["nproc"],
    ["free", "-h"],
    ["vcgencmd", "measure_temp"],
    ["vcgencmd", "get_throttled"],
    ["df", "-h", "/"],
    [PY, BENCH, "--model", "mobilenet_v2.int8"],
    [PY, BENCH, "--model", "m", "--threads", "4", "--iters", "100"],
    [PY, BENCH, "--model", "m", "--threads", "2", "--opt", "0"],
]


@pytest.mark.parametrize("argv", ALLOWED)
def test_allowed(argv):
    assert validate(argv).allowed
