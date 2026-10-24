#!/usr/bin/env python3
"""Word error rate of wav2vec2-style CTC ONNX models on a saved eval set.

Self-contained (numpy + onnxruntime only) so the same file runs on the Mac and on the Pi.

  python scripts/eval_wer.py --eval-dir data/eval_set --vocab data/vocab.json \
      --out wer.json models/w2v_fp32.onnx models/w2v_int8dyn.onnx ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def load_eval_set(eval_dir: Path, n: int | None = None) -> tuple[list[np.ndarray], list[str]]:
    data = np.load(eval_dir / "audio.npz")
    texts = json.loads((eval_dir / "texts.json").read_text())
    audio = [data[f"a{i}"].astype(np.float32) for i in range(len(texts))]
    if n is not None:
        audio, texts = audio[:n], texts[:n]
    return audio, texts


def normalize(x: np.ndarray) -> np.ndarray:
    # Same as the wav2vec2 feature extractor (do_normalize=True): zero mean, unit variance.
    return ((x - x.mean()) / np.sqrt(x.var() + 1e-7)).astype(np.float32)


def ctc_greedy(logits: np.ndarray, id2tok: dict[int, str], blank: int) -> str:
    ids = logits.argmax(-1)
    out, prev = [], None
    for i in ids.tolist():
        if i != prev and i != blank:
            out.append(id2tok.get(i, ""))
        prev = i
    text = "".join(t for t in out if not (t.startswith("<") and t.endswith(">")))
    return " ".join(text.replace("|", " ").split())


def word_edits(ref: list[str], hyp: list[str]) -> int:
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
            prev, d[j] = d[j], cur
    return d[len(hyp)]


def corpus_wer(refs: list[str], hyps: list[str]) -> float:
    edits = sum(word_edits(r.split(), h.split()) for r, h in zip(refs, hyps, strict=True))
    words = sum(len(r.split()) for r in refs)
    return edits / max(words, 1)


def transcribe(model: Path, audio: list[np.ndarray], vocab: dict[str, int],
               threads: int = 4) -> list[str]:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(model), opts, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    id2tok = {v: k for k, v in vocab.items()}
    blank = vocab.get("<pad>", 0)
    return [ctc_greedy(sess.run(None, {name: normalize(a)[None, :]})[0][0], id2tok, blank)
            for a in audio]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", type=Path)
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--vocab", type=Path, required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("wer.json"))
    args = ap.parse_args()
    audio, refs = load_eval_set(args.eval_dir, args.n)
    vocab = json.loads(args.vocab.read_text())
    results = json.loads(args.out.read_text()) if args.out.exists() else {}
    for m in args.models:
        wer = corpus_wer([r.upper() for r in refs], transcribe(m, audio, vocab, args.threads))
        results[m.stem] = round(wer, 4)
        print(f"{m.stem}: WER {wer:.2%} on {len(refs)} utterances")
    args.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
