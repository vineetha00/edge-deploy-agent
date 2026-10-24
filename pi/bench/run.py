#!/usr/bin/env python3
"""Pi-side ONNX benchmark. Prints exactly one JSON object to stdout.

Runs inside the `agent` venv on the Raspberry Pi:
  /home/agent/venv/bin/python /home/agent/bench/run.py --model NAME [--threads N] [--iters N] [--opt 0-3]
The model is read from $EDA_MODEL_DIR/NAME.onnx (default /home/agent/models).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import resource
import subprocess
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

THROTTLE_FLAGS = {
    0: "under_voltage_now",
    1: "freq_capped_now",
    2: "throttled_now",
    3: "soft_temp_limit_now",
    16: "under_voltage_occurred",
    17: "freq_capped_occurred",
    18: "throttled_occurred",
    19: "soft_temp_limit_occurred",
}
_OPT_LEVELS = {
    0: ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    1: ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    2: ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    3: ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}
_DTYPES = {"tensor(float)": np.float32, "tensor(int64)": np.int64, "tensor(int32)": np.int32}


def decode_throttled(value: int) -> list[str]:
    return [name for bit, name in THROTTLE_FLAGS.items() if (value >> bit) & 1]


def _vcgencmd(*args: str) -> str | None:
    try:
        out = subprocess.run(["vcgencmd", *args], capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def read_temp_c() -> float | None:
    m = re.search(r"temp=([\d.]+)", _vcgencmd("measure_temp") or "")
    return float(m.group(1)) if m else None


def read_throttled() -> int | None:
    m = re.search(r"throttled=(0x[0-9a-fA-F]+)", _vcgencmd("get_throttled") or "")
    return int(m.group(1), 16) if m else None


# Fixed benchmark clip for raw-audio models (e.g. wav2vec2): 5 s at 16 kHz.
AUDIO_SAMPLES = 80000


def _shape(inp) -> list[int]:
    dims = list(inp.shape)
    fixed = [d if isinstance(d, int) and d > 0 else 1 for d in dims]
    # Raw-audio input: float [batch, samples] with a dynamic samples axis.
    if (inp.type == "tensor(float)" and len(dims) == 2
            and not (isinstance(dims[1], int) and dims[1] > 0)):
        fixed[1] = AUDIO_SAMPLES
    return fixed


def make_inputs(session: ort.InferenceSession) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    feeds = {}
    for inp in session.get_inputs():
        shape = _shape(inp)
        dtype = _DTYPES.get(inp.type)
        if dtype is None:
            raise SystemExit(f"unsupported input type {inp.type}")
        feeds[inp.name] = (
            rng.random(shape, dtype=np.float32)
            if dtype == np.float32
            else rng.integers(0, 10, size=shape).astype(dtype)
        )
    return feeds


def run_benchmark(
    model_path: Path, threads: int, iters: int, warmup: int = 5, opt: int = 3
) -> dict:
    temp_before, thr_before = read_temp_c(), read_throttled()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.graph_optimization_level = _OPT_LEVELS[opt]
    session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
    feeds = make_inputs(session)
    for _ in range(warmup):
        session.run(None, feeds)
    times_ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        session.run(None, feeds)
        times_ms.append((time.perf_counter() - t0) * 1000)
    temp_after, thr_after = read_temp_c(), read_throttled()
    telemetry_ok = None not in (temp_before, temp_after, thr_before, thr_after)
    new_flags = sorted(
        set(decode_throttled(thr_after or 0)) - set(decode_throttled(thr_before or 0))
    )
    arr = np.array(times_ms)
    return {
        "model": model_path.stem,
        "provider": "CPUExecutionProvider",
        "threads": threads,
        "graph_opt": opt,
        "iters": iters,
        "input_shapes": {k: list(v.shape) for k, v in feeds.items()},
        "latency_ms": {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "mean": float(arr.mean()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        },
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "temp_c_before": temp_before,
        "temp_c_after": temp_after,
        "throttled_before": hex(thr_before) if thr_before is not None else None,
        "throttled_after": hex(thr_after) if thr_after is not None else None,
        "new_throttle_flags": new_flags,
        "telemetry_ok": telemetry_ok,
        # None = unknown (sensors unreadable); never report a run as valid without heat data.
        "run_valid": (not new_flags) if telemetry_ok else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--opt", type=int, default=3, choices=[0, 1, 2, 3])
    args = ap.parse_args()
    model_dir = Path(os.environ.get("EDA_MODEL_DIR", "/home/agent/models"))
    path = model_dir / f"{args.model}.onnx"
    if not path.is_file():
        raise SystemExit(f"model not found: {path}")
    print(json.dumps(run_benchmark(path, args.threads, args.iters, opt=args.opt)))


if __name__ == "__main__":
    main()
