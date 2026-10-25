#!/usr/bin/env python3
"""Benchmark a model served by Triton and check its transcriptions match ONNX Runtime.

  python scripts/triton_bench.py --url http://127.0.0.1:8080 --iters 20 \
      --eval-dir data/eval_set --vocab data/vocab.json --n 10

Latency is reported twice: end to end over HTTP/JSON (what a client sees) and Triton's own
server-side compute time from /v2/models/<name>/stats (what the model costs).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_wer import corpus_wer, ctc_greedy, load_eval_set, normalize

CLIP = 80000  # 5 s at 16 kHz, same as the Pi benchmark


class Triton:
    def __init__(self, url: str, model: str, transport: httpx.BaseTransport | None = None):
        self.model = model
        self.http = httpx.Client(base_url=url, timeout=120.0, transport=transport)

    def ready(self) -> bool:
        try:
            return self.http.get(f"/v2/models/{self.model}/ready").status_code == 200
        except httpx.HTTPError:
            return False

    def input_name(self) -> str:
        meta = self.http.get(f"/v2/models/{self.model}").json()
        return meta["inputs"][0]["name"]

    def infer(self, name: str, x: np.ndarray) -> np.ndarray:
        body = {"inputs": [{"name": name, "shape": list(x.shape), "datatype": "FP32",
                            "data": x.ravel().tolist()}]}
        r = self.http.post(f"/v2/models/{self.model}/infer", json=body)
        r.raise_for_status()
        out = r.json()["outputs"][0]
        return np.asarray(out["data"], dtype=np.float32).reshape(out["shape"])

    def compute_ns(self) -> tuple[int, int]:
        """(count, total ns) of server-side model compute so far."""
        s = self.http.get(f"/v2/models/{self.model}/stats").json()["model_stats"][0]
        c = s["inference_stats"]["compute_infer"]
        return int(c["count"]), int(c["ns"])


def latency(t: Triton, name: str, iters: int, warmup: int = 3) -> dict:
    x = normalize(np.random.default_rng(0).standard_normal(CLIP).astype(np.float32))[None, :]
    for _ in range(warmup):
        t.infer(name, x)
    c0, ns0 = t.compute_ns()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        t.infer(name, x)
        times.append((time.perf_counter() - t0) * 1000)
    c1, ns1 = t.compute_ns()
    arr = np.array(times)
    return {"iters": iters, "e2e_p50_ms": float(np.percentile(arr, 50)),
            "e2e_p95_ms": float(np.percentile(arr, 95)),
            "server_compute_mean_ms": (ns1 - ns0) / max(c1 - c0, 1) / 1e6}


def parity(t: Triton, name: str, eval_dir: Path, vocab_path: Path, n: int) -> dict:
    audio, refs = load_eval_set(eval_dir, n)
    vocab = json.loads(vocab_path.read_text())
    id2tok, blank = {v: k for k, v in vocab.items()}, vocab.get("<pad>", 0)
    hyps = [ctc_greedy(t.infer(name, normalize(a)[None, :])[0], id2tok, blank) for a in audio]
    return {"n": len(refs), "wer": corpus_wer([r.upper() for r in refs], hyps)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="w2v")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--eval-dir", type=Path)
    ap.add_argument("--vocab", type=Path)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--wait-s", type=int, default=600, help="wait this long for model ready")
    a = ap.parse_args()
    t = Triton(a.url, a.model)
    deadline = time.time() + a.wait_s
    while not t.ready():
        if time.time() > deadline:
            raise SystemExit(f"model {a.model} not ready at {a.url}")
        time.sleep(5)
    name = t.input_name()
    result = {"latency": latency(t, name, a.iters)}
    if a.eval_dir and a.vocab:
        result["parity"] = parity(t, name, a.eval_dir, a.vocab, a.n)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
