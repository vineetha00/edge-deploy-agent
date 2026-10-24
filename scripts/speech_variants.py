#!/usr/bin/env python3
"""Build the speech-model variants the agent searches over.

  python scripts/speech_variants.py data/model.onnx data/eval_set models/
writes models/w2v_fp32.onnx, models/w2v_int8dyn.onnx (dynamic INT8 on MatMul/Gemm),
models/w2v_int8qdq.onnx (static QDQ INT8 on MatMul/Gemm/Conv, calibrated on real speech).
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_wer import load_eval_set, normalize

CLIP = 80000  # 5 s at 16 kHz, same as the Pi benchmark input


class SpeechCalib(CalibrationDataReader):
    def __init__(self, input_name: str, audio: list[np.ndarray]):
        clips = []
        for a in audio:
            a = normalize(a)[:CLIP]
            clips.append(np.pad(a, (0, CLIP - len(a)))[None, :])
        self._it = iter([{input_name: c} for c in clips])

    def get_next(self):
        return next(self._it, None)


def build(src: Path, eval_dir: Path, out: Path, n_calib: int = 20) -> list[Path]:
    import onnxruntime as ort

    out.mkdir(parents=True, exist_ok=True)
    fp32, dyn, qdq = (out / f"w2v_{s}.onnx" for s in ("fp32", "int8dyn", "int8qdq"))
    shutil.copyfile(src, fp32)
    # MatMul/Gemm only: ConvInteger with int8 weights has no CPU kernel in ONNX Runtime.
    quantize_dynamic(str(src), str(dyn), weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul", "Gemm"])
    input_name = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"]) \
        .get_inputs()[0].name
    audio, _ = load_eval_set(eval_dir, n_calib)
    with tempfile.TemporaryDirectory() as tmp:
        pre = Path(tmp) / "pre.onnx"
        quant_pre_process(str(src), str(pre), skip_symbolic_shape=False)
        quantize_static(str(pre), str(qdq), SpeechCalib(input_name, audio),
                        quant_format=QuantFormat.QDQ, per_channel=True,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                        op_types_to_quantize=["MatMul", "Gemm", "Conv"])
    return [fp32, dyn, qdq]


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    for p in build(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])):
        print(p, f"{p.stat().st_size / 1e6:.1f} MB")
