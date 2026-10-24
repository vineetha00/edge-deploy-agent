import json

import eval_wer
import numpy as np
import onnx
import onnxruntime as ort
import pytest
import speech_variants
from onnx import TensorProto, helper, numpy_helper

from edge_deploy_agent.agent import accuracy_ok, meets_budget

VOCAB = {"<pad>": 0, "|": 1, "A": 2, "B": 3, "C": 4}


def tiny_ctc_model(path, opset=17):
    """wav2vec2-shaped stand-in: raw audio [1, T] -> Conv -> MatMul -> logits [1, F, V]."""
    rng = np.random.default_rng(0)
    w1 = numpy_helper.from_array(rng.standard_normal((8, 1, 400)).astype(np.float32) * 0.05, "w1")
    w2 = numpy_helper.from_array(rng.standard_normal((8, 5)).astype(np.float32), "w2")
    ax = numpy_helper.from_array(np.array([1], dtype=np.int64), "ax")
    nodes = [
        helper.make_node("Unsqueeze", ["input_values", "ax"], ["u"]) if opset >= 13
        else helper.make_node("Unsqueeze", ["input_values"], ["u"], axes=[1]),
        helper.make_node("Conv", ["u", "w1"], ["c"], strides=[320]),
        helper.make_node("Transpose", ["c"], ["t"], perm=[0, 2, 1]),
        helper.make_node("MatMul", ["t", "w2"], ["logits"]),
    ]
    g = helper.make_graph(
        nodes, "tiny",
        [helper.make_tensor_value_info("input_values", TensorProto.FLOAT, [1, "T"])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, "F", 5])],
        [w1, w2, ax] if opset >= 13 else [w1, w2])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", opset)])
    m.ir_version = 9
    onnx.save(m, path)


def eval_set(d, n=8):
    d.mkdir()
    rng = np.random.default_rng(1)
    np.savez(d / "audio.npz", **{f"a{i}": rng.standard_normal(16000 + 800 * i).astype(np.float32)
                                for i in range(n)})
    (d / "texts.json").write_text(json.dumps(["ab c"] * n))


def test_ctc_greedy_collapses_and_drops_blank():
    id2tok = {v: k for k, v in VOCAB.items()}
    ids = [2, 2, 0, 2, 3, 1, 1, 4, 0]
    logits = np.eye(5)[ids]
    assert eval_wer.ctc_greedy(logits, id2tok, 0) == "AAB C"


def test_corpus_wer():
    assert eval_wer.corpus_wer(["A B C", "D"], ["A B C", "D"]) == 0
    assert eval_wer.corpus_wer(["A B C D"], ["A X C"]) == pytest.approx(2 / 4)


@pytest.mark.parametrize("opset", [11, 17])  # 11 = like the Xenova export that broke QDQ
def test_variants_build_load_and_score(tmp_path, opset):
    src = tmp_path / "model.onnx"
    tiny_ctc_model(src, opset)
    eval_set(tmp_path / "eval")
    paths = speech_variants.build(src, tmp_path / "eval", tmp_path / "models", n_calib=4)
    assert [p.name for p in paths] == ["w2v_fp32.onnx", "w2v_int8dyn.onnx", "w2v_int8qdq.onnx"]
    audio, refs = eval_wer.load_eval_set(tmp_path / "eval")
    ops = {n.op_type for n in onnx.load(paths[2]).graph.node}
    assert "QuantizeLinear" in ops and "DequantizeLinear" in ops
    for p in paths:
        ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])  # loads on CPU EP
        hyps = eval_wer.transcribe(p, audio, VOCAB, threads=1)
        assert len(hyps) == len(refs)
        assert 0 <= eval_wer.corpus_wer(refs, hyps)


def test_bench_feeds_5s_audio(tmp_path):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("bench", Path("pi/bench/run.py"))
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    tiny_ctc_model(tmp_path / "m.onnx")
    s = ort.InferenceSession(str(tmp_path / "m.onnx"), providers=["CPUExecutionProvider"])
    assert bench.make_inputs(s)["input_values"].shape == (1, 80000)


GOAL = {"latency_p95_ms": 100, "ram_mb": 500, "max_wer": 0.05,
        "wer": {"w2v_fp32": 0.03, "w2v_int8qdq": 0.12}}
RES = {"run_valid": True, "latency_ms": {"p95": 50}, "peak_rss_mb": 300}


def test_wer_gate():
    assert meets_budget({**RES, "model": "w2v_fp32"}, GOAL)
    assert not meets_budget({**RES, "model": "w2v_int8qdq"}, GOAL)      # over the gate
    assert not meets_budget({**RES, "model": "w2v_int8dyn"}, GOAL)      # never measured
    assert accuracy_ok("anything", {"latency_p95_ms": 1, "ram_mb": 1})  # no gate set


def test_activation_matmuls_excluded(tmp_path):
    """Attention-style MatMul (both inputs are activations) must stay in float."""
    w = numpy_helper.from_array(np.ones((4, 4), dtype=np.float32), "w")
    nodes = [helper.make_node("MatMul", ["x", "w"], ["h"], name="weight_mm"),
             helper.make_node("Transpose", ["h"], ["ht"], perm=[1, 0]),
             helper.make_node("MatMul", ["h", "ht"], ["y"], name="attn_mm")]
    g = helper.make_graph(nodes, "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [3, 4])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 3])], [w])
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]), tmp_path / "m.onnx")
    assert speech_variants.activation_matmuls(tmp_path / "m.onnx") == ["attn_mm"]
