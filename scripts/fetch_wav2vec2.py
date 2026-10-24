#!/usr/bin/env python3
"""Download wav2vec2-base-960h ONNX + vocab and a small LibriSpeech eval set (run on the Mac).

  pip install huggingface_hub "datasets<4" soundfile
  python scripts/fetch_wav2vec2.py --out data
writes data/model.onnx, data/vocab.json, data/eval_set/{audio.npz,texts.json}
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
from pathlib import Path

import numpy as np

REPO = "Xenova/wav2vec2-base-960h"
EVAL_DATASET = "hf-internal-testing/librispeech_asr_dummy"  # 73 clean LibriSpeech utterances


def fetch_model(out: Path) -> None:
    from huggingface_hub import hf_hub_download, list_repo_files

    files = list_repo_files(REPO)
    for want, dest in [("onnx/model.onnx", "model.onnx"), ("vocab.json", "vocab.json")]:
        if want not in files:
            raise SystemExit(f"{want} not in {REPO}; files are: {sorted(files)}")
        shutil.copyfile(hf_hub_download(REPO, want), out / dest)
        print("saved", out / dest)


def fetch_eval_set(out: Path, n: int) -> None:
    import soundfile as sf
    from datasets import Audio, load_dataset

    ds = load_dataset(EVAL_DATASET, "clean", split="validation")
    ds = ds.cast_column("audio", Audio(decode=False))
    audio, texts = {}, []
    for i, row in enumerate(ds.select(range(min(n, len(ds))))):
        wav, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        if sr != 16000:
            raise SystemExit(f"expected 16 kHz audio, got {sr}")
        audio[f"a{i}"] = wav
        texts.append(row["text"])
    d = out / "eval_set"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "audio.npz", **audio)
    (d / "texts.json").write_text(json.dumps(texts, indent=1))
    print(f"saved {len(texts)} utterances to {d}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--n", type=int, default=73)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    fetch_model(a.out)
    fetch_eval_set(a.out, a.n)
