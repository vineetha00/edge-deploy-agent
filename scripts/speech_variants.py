#!/usr/bin/env python3
"""Build the speech-model variants the agent searches over.

  python scripts/speech_variants.py data/model.onnx data/eval_set models/
writes models/w2v_fp32.onnx, models/w2v_int8dyn.onnx (dynamic INT8 on MatMul/Gemm),
models/w2v_int8qdq.onnx (static QDQ INT8 on weight MatMuls, calibrated on real speech).

Static settings were chosen by measurement on wav2vec2-base-960h (10 held-out clips, ARM CPU):
quantizing Conv too gave 100% WER; all MatMuls 27%; weight MatMuls only ~7% (FP32 ~6%).
Calibration uses the LAST n_calib clips of the eval set; report WER on the first
(len - n_calib) clips (eval_wer.py --n) so calibration audio is never scored.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import version_converter
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

MIN_QDQ_OPSET = 13
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
    audio, _ = load_eval_set(eval_dir)
    audio = audio[-n_calib:]  # held out from WER scoring (see module docstring)
    with tempfile.TemporaryDirectory() as tmp:
        # Per-channel QDQ needs DequantizeLinear's `axis` attribute, added in opset 13.
        # Exports like Xenova's are opset 11/12, so upgrade the graph first.
        up = Path(tmp) / "up.onnx"
        model = onnx.load(str(src))
        if _opset(model) < MIN_QDQ_OPSET:
            model = version_converter.convert_version(model, MIN_QDQ_OPSET)
        onnx.save(model, str(up))
        pre = Path(tmp) / "pre.onnx"
        quant_pre_process(str(up), str(pre), skip_symbolic_shape=False)
        quantize_static(str(pre), str(qdq), SpeechCalib(input_name, audio),
                        quant_format=QuantFormat.QDQ, per_channel=True,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                        op_types_to_quantize=["MatMul", "Gemm"],
                        nodes_to_exclude=activation_matmuls(pre))
    for p in (fp32, dyn, qdq):  # fail here on the Mac, not later on the Pi
        ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    return [fp32, dyn, qdq]


def activation_matmuls(path: Path) -> list[str]:
    """MatMuls with no weight input (attention scores/context): too outlier-heavy for INT8."""
    m = onnx.load(str(path), load_external_data=False)
    inits = {i.name for i in m.graph.initializer}
    return [n.name for n in m.graph.node
            if n.op_type in ("MatMul", "Gemm") and not any(x in inits for x in n.input)]


def _opset(model) -> int:
    return next((o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), 0)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    for p in build(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])):
        print(p, f"{p.stat().st_size / 1e6:.1f} MB")
