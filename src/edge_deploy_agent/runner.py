"""Allowlisted command validation for the remote (Raspberry Pi) runner.

Every command the agent wants to run on the Pi is an argv list that must pass
`validate` first. Anything not explicitly allowed is denied (fail closed). The Pi-side
worker runs the same check again before executing (defense in depth).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

PI_PYTHON = "/home/agent/venv/bin/python"
BENCH_SCRIPT = "/home/agent/bench/run.py"

_EXACT = {
    ("uname", "-a"),
    ("python3", "--version"),
    ("nproc",),
    ("free", "-h"),
    ("df", "-h", "/"),
    ("vcgencmd", "measure_temp"),
    ("vcgencmd", "get_throttled"),
}
_MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_FORBIDDEN_CHARS = set(";|&$`<>\n\r\\\"'()*?~!{}\x00")
_INT_FLAGS = {"--threads": (1, 4), "--iters": (1, 1000), "--opt": (0, 3)}


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    reason: str


def _valid_bench_options(rest: list[str]) -> bool:
    if len(rest) % 2:
        return False
    seen = set()
    for flag, value in zip(rest[::2], rest[1::2]):
        if flag not in _INT_FLAGS or flag in seen or not re.fullmatch(r"[0-9]{1,4}", value):
            return False
        lo, hi = _INT_FLAGS[flag]
        if not lo <= int(value) <= hi:
            return False
        seen.add(flag)
    return True


def validate(argv: list[str]) -> Verdict:
    if not argv or not all(isinstance(a, str) and a for a in argv):
        return Verdict(False, "empty or non-string argv")
    if any(ch in arg for arg in argv for ch in _FORBIDDEN_CHARS):
        return Verdict(False, "forbidden character in argument")
    if tuple(argv) in _EXACT:
        return Verdict(True, "exact allowlist match")
    if (
        len(argv) >= 4
        and argv[0] == PI_PYTHON
        and argv[1] == BENCH_SCRIPT
        and argv[2] == "--model"
        and _MODEL_NAME.match(argv[3])
        and _valid_bench_options(argv[4:])
    ):
        return Verdict(True, "benchmark script with valid model name and options")
    return Verdict(False, "not on allowlist")
