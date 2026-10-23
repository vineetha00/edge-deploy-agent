"""Allowlisted command validation for the remote (Raspberry Pi) runner.

Every command the agent wants to run over SSH is an argv list that must pass
`validate` first. Anything not explicitly allowed is denied (fail closed).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_EXACT = {
    ("uname", "-a"),
    ("python3", "--version"),
    ("nproc",),
    ("free", "-h"),
    ("vcgencmd", "measure_temp"),
    ("vcgencmd", "get_throttled"),
}
_BENCH_SCRIPT = re.compile(r"^/home/agent/bench/run\.py$")
_MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_FORBIDDEN_CHARS = set(";|&$`<>\n\r\\\"'()*?~!{}\x00")


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    reason: str


def validate(argv: list[str]) -> Verdict:
    if not argv or not all(isinstance(a, str) and a for a in argv):
        return Verdict(False, "empty or non-string argv")
    if any(ch in arg for arg in argv for ch in _FORBIDDEN_CHARS):
        return Verdict(False, "forbidden character in argument")
    if tuple(argv) in _EXACT:
        return Verdict(True, "exact allowlist match")
    if (
        len(argv) == 4
        and argv[0] == "python3"
        and _BENCH_SCRIPT.match(argv[1])
        and argv[2] == "--model"
        and _MODEL_NAME.match(argv[3])
    ):
        return Verdict(True, "benchmark script with a valid model name")
    return Verdict(False, "not on allowlist")
