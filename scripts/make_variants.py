#!/usr/bin/env python3
"""Make FP32 and dynamic-INT8 ONNX variants of a model for the Pi.

  python scripts/make_variants.py path/to/model.onnx out_dir NAME
writes out_dir/NAME_fp32.onnx and out_dir/NAME_int8.onnx
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def make_variants(src: Path, out_dir: Path, name: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32, int8 = out_dir / f"{name}_fp32.onnx", out_dir / f"{name}_int8.onnx"
    shutil.copyfile(src, fp32)
    quantize_dynamic(str(src), str(int8), weight_type=QuantType.QInt8)
    return [fp32, int8]


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    for p in make_variants(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]):
        print(p, f"{p.stat().st_size / 1e6:.2f} MB")
