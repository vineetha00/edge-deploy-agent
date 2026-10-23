"""Model roles and endpoint settings. Model IDs come from the Token Factory catalog."""
from __future__ import annotations

import os

BASE_URL = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1")

MODELS = {
    "planner": "nvidia/Nemotron-3-Ultra-550b-a55b",
    "editor": "nvidia/nemotron-3-super-120b-a12b",
    "fast": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
}

# USD per 1M tokens: {model_id: (input, output)}, from `/v1/models?verbose=true` on 2026-10-23
# (the API reports per-token prices; multiplied by 1e6). Unknown models leave cost as None.
PRICES_PER_M: dict[str, tuple[float, float]] = {
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B": (0.06, 0.24),
    "nvidia/Nemotron-3_5-Lightning": (0.06, 0.24),
    "nvidia/nemotron-3-super-120b-a12b": (0.30, 0.90),
    "nvidia/Nemotron-3.5-Super-VL": (0.30, 0.90),
    "nvidia/Nemotron-3-Ultra-550b-a55b": (1.00, 3.00),
}
