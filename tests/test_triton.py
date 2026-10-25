import json

import export_triton
import httpx
import numpy as np
import pytest
import triton_bench

from tests.test_speech import VOCAB, eval_set, tiny_ctc_model


def test_config_carries_agent_threads_and_opt(tmp_path):
    src = tmp_path / "m.onnx"
    tiny_ctc_model(src)
    d = export_triton.export(src, tmp_path / "repo", "w2v", threads=4, opt=2)
    cfg = (d / "config.pbtxt").read_text()
    assert (d / "1" / "model.onnx").read_bytes() == src.read_bytes()
    assert 'backend: "onnxruntime"' in cfg
    assert 'key: "intra_op_thread_count" value: { string_value: "4" }' in cfg
    assert "optimization { graph { level: 1 } }" in cfg          # agent opt 2 = extended
    assert '{ name: "input_values", data_type: TYPE_FP32, dims: [1, -1] }' in cfg
    assert '{ name: "logits", data_type: TYPE_FP32, dims: [1, -1, 5] }' in cfg


@pytest.mark.parametrize("opt,expect", [(0, "level: 2"), (1, "level: -1"), (3, None)])
def test_opt_level_mapping(opt, expect):
    cfg = export_triton.render_config("m", [], [], threads=1, opt=opt)
    if expect:
        assert expect in cfg
    else:
        assert "optimization" not in cfg  # all optimizations = Triton default


def test_bad_settings_rejected():
    with pytest.raises(SystemExit):
        export_triton.render_config("m", [], [], threads=1, opt=7)
    with pytest.raises(SystemExit):
        export_triton.render_config("m", [], [], threads=0, opt=3)


def test_from_run_uses_best_proposal(tmp_path):
    run = tmp_path / "run.json"
    run.write_text(json.dumps({"best": {"proposal": {"model": "w2v_int8dyn", "threads": 4,
                                                     "opt": 2}}}))
    assert export_triton.best_from_run(run)["model"] == "w2v_int8dyn"
    run.write_text(json.dumps({"best": None}))
    with pytest.raises(SystemExit):
        export_triton.best_from_run(run)


def fake_triton(logits_fn):
    """Minimal KServe v2 server: metadata, infer, stats."""
    state = {"count": 0, "ns": 0}

    def handler(req: httpx.Request):
        path = req.url.path
        if path.endswith("/ready"):
            return httpx.Response(200)
        if path.endswith("/stats"):
            return httpx.Response(200, json={"model_stats": [{"inference_stats": {
                "compute_infer": {"count": state["count"], "ns": state["ns"]}}}]})
        if path.endswith("/infer"):
            inp = json.loads(req.content)["inputs"][0]
            x = np.asarray(inp["data"], dtype=np.float32).reshape(inp["shape"])
            y = logits_fn(x)
            state["count"] += 1
            state["ns"] += 2_000_000
            return httpx.Response(200, json={"outputs": [
                {"name": "logits", "shape": list(y.shape), "data": y.ravel().tolist()}]})
        return httpx.Response(200, json={"inputs": [{"name": "input_values"}]})
    return httpx.MockTransport(handler)


def test_latency_reports_e2e_and_server_compute():
    t = triton_bench.Triton("http://t", "w2v", transport=fake_triton(
        lambda x: np.zeros((1, 3, 5), dtype=np.float32)))
    r = triton_bench.latency(t, t.input_name(), iters=4, warmup=1)
    assert r["iters"] == 4 and r["e2e_p95_ms"] >= r["e2e_p50_ms"] > 0
    assert r["server_compute_mean_ms"] == pytest.approx(2.0)


def test_parity_matches_direct_onnxruntime(tmp_path):
    """Served transcriptions must equal running the same ONNX file in-process."""
    import eval_wer
    import onnxruntime as ort

    src = tmp_path / "m.onnx"
    tiny_ctc_model(src)
    eval_set(tmp_path / "eval")
    (tmp_path / "vocab.json").write_text(json.dumps(VOCAB))
    sess = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
    t = triton_bench.Triton("http://t", "w2v", transport=fake_triton(
        lambda x: sess.run(None, {"input_values": x})[0]))
    served = triton_bench.parity(t, "input_values", tmp_path / "eval", tmp_path / "vocab.json", 5)
    audio, refs = eval_wer.load_eval_set(tmp_path / "eval", 5)
    direct = eval_wer.corpus_wer([r.upper() for r in refs],
                                 eval_wer.transcribe(src, audio, VOCAB, threads=1))
    assert served["n"] == 5 and served["wer"] == pytest.approx(direct)


def test_save_run_records_evidence(tmp_path):
    import run_agent

    out = tmp_path / "results" / "r.json"
    run_agent.save_run(out, {"candidates": ["a"]}, {"trials": [{"x": 1}], "best": None,
                                                     "report": "r", "docs": [
                                                         {"title": "t", "url": "u",
                                                          "content": "big", "flags": []}]})
    data = json.loads(out.read_text())
    assert data["trials"] == [{"x": 1}] and "content" not in data["docs"][0]


@pytest.mark.parametrize("opt", [0, 1, 2, 3])
def test_config_parses_with_tritons_own_schema(tmp_path, opt):
    pytest.importorskip("tritonclient.grpc")
    from google.protobuf import text_format
    from tritonclient.grpc import model_config_pb2 as mc

    src = tmp_path / "m.onnx"
    tiny_ctc_model(src)
    d = export_triton.export(src, tmp_path / "repo", "w2v", threads=4, opt=opt)
    cfg = text_format.Parse((d / "config.pbtxt").read_text(), mc.ModelConfig())
    assert cfg.backend == "onnxruntime"
    assert cfg.instance_group[0].kind == mc.ModelInstanceGroup.KIND_CPU
    assert cfg.parameters["intra_op_thread_count"].string_value == "4"
    assert cfg.optimization.graph.level == {0: 2, 1: -1, 2: 1, 3: 0}[opt]
