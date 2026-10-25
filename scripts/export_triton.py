#!/usr/bin/env python3
"""Turn the agent's winning configuration into a Triton model repository.

  python scripts/export_triton.py --from-run results/latest.json --models-dir models \
      --out triton_repo
  python scripts/export_triton.py --model models/w2v_int8dyn.onnx --threads 4 --opt 2 \
      --out triton_repo

writes triton_repo/<name>/1/model.onnx and triton_repo/<name>/config.pbtxt, carrying the
agent's thread count and ONNX Runtime graph-optimization level into Triton's ORT backend.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import onnx

# Our benchmark's --opt (0..3) -> Triton ORT backend `optimization { graph { level } }`.
# Triton levels: -1 basic, 1 basic+extended, 2 disabled; omitting the block = all enabled.
TRITON_GRAPH_LEVEL = {0: 2, 1: -1, 2: 1, 3: None}

_TYPES = {
    onnx.TensorProto.FLOAT: "TYPE_FP32", onnx.TensorProto.FLOAT16: "TYPE_FP16",
    onnx.TensorProto.INT64: "TYPE_INT64", onnx.TensorProto.INT32: "TYPE_INT32",
    onnx.TensorProto.INT8: "TYPE_INT8", onnx.TensorProto.UINT8: "TYPE_UINT8",
    onnx.TensorProto.BOOL: "TYPE_BOOL",
}


def _io(values) -> list[dict]:
    out = []
    for v in values:
        t = v.type.tensor_type
        dims = [d.dim_value if d.HasField("dim_value") and d.dim_value > 0 else -1
                for d in t.shape.dim]
        dtype = _TYPES.get(t.elem_type)
        if dtype is None:
            raise SystemExit(f"unsupported ONNX type {t.elem_type} for {v.name}")
        out.append({"name": v.name, "data_type": dtype, "dims": dims})
    return out


def model_io(model_path: Path) -> tuple[list[dict], list[dict]]:
    m = onnx.load(str(model_path), load_external_data=False)
    inits = {i.name for i in m.graph.initializer}
    return _io([i for i in m.graph.input if i.name not in inits]), _io(m.graph.output)


def render_config(name: str, inputs: list[dict], outputs: list[dict], threads: int,
                  opt: int) -> str:
    if opt not in TRITON_GRAPH_LEVEL:
        raise SystemExit(f"opt must be 0-3, got {opt}")
    if not 1 <= threads <= 64:
        raise SystemExit(f"threads must be 1-64, got {threads}")

    def block(kind: str, items: list[dict]) -> str:
        rows = ",\n".join(
            f'  {{ name: "{i["name"]}", data_type: {i["data_type"]}, dims: {i["dims"]} }}'
            for i in items)
        return f"{kind} [\n{rows}\n]"

    lines = [
        f'name: "{name}"',
        'backend: "onnxruntime"',
        "max_batch_size: 0  # dims below include the batch axis",
        block("input", inputs),
        block("output", outputs),
        "instance_group [ { kind: KIND_CPU, count: 1 } ]",
        f'parameters {{ key: "intra_op_thread_count" value: {{ string_value: "{threads}" }} }}',
        'parameters { key: "inter_op_thread_count" value: { string_value: "1" } }',
    ]
    level = TRITON_GRAPH_LEVEL[opt]
    if level is not None:
        lines.append(f"optimization {{ graph {{ level: {level} }} }}"
                     f"  # agent opt={opt}")
    return "\n".join(lines) + "\n"


def export(model_path: Path, out: Path, name: str, threads: int, opt: int) -> Path:
    inputs, outputs = model_io(model_path)
    d = out / name
    (d / "1").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(model_path, d / "1" / "model.onnx")
    (d / "config.pbtxt").write_text(render_config(name, inputs, outputs, threads, opt))
    return d


def best_from_run(run_file: Path) -> dict:
    run = json.loads(run_file.read_text())
    best = run.get("best")
    if not best:
        raise SystemExit(f"{run_file} has no configuration that met the budget")
    return best["proposal"]


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-run", type=Path, help="JSON saved by run_agent.py --save")
    src.add_argument("--model", type=Path, help="ONNX file to export directly")
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--opt", type=int, default=3)
    ap.add_argument("--name", default="w2v")
    ap.add_argument("--out", type=Path, default=Path("triton_repo"))
    a = ap.parse_args()
    if a.from_run:
        p = best_from_run(a.from_run)
        model, threads, opt = a.models_dir / f"{p['model']}.onnx", p["threads"], p.get("opt", 3)
    else:
        model, threads, opt = a.model, a.threads, a.opt
    d = export(model, a.out, a.name, threads, opt)
    print(f"wrote {d} from {model.name} (threads={threads}, opt={opt})")
    print((d / "config.pbtxt").read_text())


if __name__ == "__main__":
    main()
