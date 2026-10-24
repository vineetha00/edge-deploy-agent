import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("onnxruntime")
onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper

SCRIPT = Path(__file__).parent.parent / "pi" / "bench" / "run.py"
spec = importlib.util.spec_from_file_location("pi_bench_run", SCRIPT)
bench = importlib.util.module_from_spec(spec)
sys.modules["pi_bench_run"] = bench
spec.loader.exec_module(bench)


def _tiny_model(path: Path):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 8])
    graph = helper.make_graph([helper.make_node("Relu", ["x"], ["y"])], "g", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))


def test_decode_throttled():
    assert bench.decode_throttled(0x80000) == ["soft_temp_limit_occurred"]
    assert bench.decode_throttled(0) == []
    assert "under_voltage_now" in bench.decode_throttled(0x1)


def test_run_benchmark_on_tiny_model(tmp_path):
    path = tmp_path / "tiny.onnx"
    _tiny_model(path)
    result = bench.run_benchmark(path, threads=1, iters=10, warmup=2)
    json.dumps(result)  # must be JSON-serializable
    assert result["model"] == "tiny" and result["iters"] == 10
    assert result["latency_ms"]["p95"] >= result["latency_ms"]["p50"] > 0
    # no vcgencmd off-Pi -> telemetry unavailable -> validity unknown, never "valid"
    assert result["telemetry_ok"] is False and result["run_valid"] is None


def test_run_valid_reflects_throttling(tmp_path, monkeypatch):
    path = tmp_path / "tiny.onnx"
    _tiny_model(path)
    temps = iter([50.0, 60.0])
    thr = iter([0x0, 0x40000])  # throttling occurred during the run
    monkeypatch.setattr(bench, "read_temp_c", lambda: next(temps))
    monkeypatch.setattr(bench, "read_throttled", lambda: next(thr))
    r = bench.run_benchmark(path, threads=1, iters=5, warmup=1)
    assert r["telemetry_ok"] is True
    assert r["new_throttle_flags"] == ["throttled_occurred"]
    assert r["run_valid"] is False


def test_graph_opt_level_is_recorded(tmp_path):
    path = tmp_path / "tiny.onnx"
    _tiny_model(path)
    r = bench.run_benchmark(path, threads=1, iters=5, warmup=1, opt=0)
    assert r["graph_opt"] == 0
