import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("onnxruntime")
ROOT = Path(__file__).parent.parent
spec = importlib.util.spec_from_file_location("make_variants", ROOT / "scripts" / "make_variants.py")
mv = importlib.util.module_from_spec(spec)
sys.modules["make_variants"] = mv
spec.loader.exec_module(mv)


def test_int8_variant_is_smaller_and_runs(tmp_path):
    import numpy as np
    import onnxruntime as ort

    fp32, int8 = mv.make_variants(ROOT / "pi" / "bench" / "smoke.onnx", tmp_path, "smoke")
    assert int8.stat().st_size < fp32.stat().st_size
    x = np.random.default_rng(0).random((1, 256), dtype=np.float32)
    a = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
    b = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
    assert a.shape == b.shape
