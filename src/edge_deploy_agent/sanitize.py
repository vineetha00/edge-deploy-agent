"""First rail for untrusted text (web docs) before it reaches the planner prompt.

Deterministic and cheap: drops lines that look like instructions or shell commands aimed at
the agent, truncates, and records what was flagged. NeMo Guardrails is a second rail on top;
the allowlist guard is the last line and never trusts the planner at all.
"""
from __future__ import annotations

import re

from prometheus_client import Counter

DOC_FLAGS = Counter("eda_doc_injection_flags_total", "Untrusted doc lines flagged", ["rule"])

MAX_DOC_CHARS = 1500
MAX_DOCS = 5

_RULES = {
    "override_instructions": re.compile(
        r"(ignore|disregard|forget)\b.{0,40}\b(previous|prior|above|earlier|all)\b"
        r".{0,40}\b(instructions?|prompts?|rules?)", re.IGNORECASE),
    "role_hijack": re.compile(r"\b(you are now|new instructions?|system prompt|developer message)\b",
                              re.IGNORECASE),
    "run_command": re.compile(
        r"\b(run|execute|exec|invoke)\b.{0,60}\b(curl|wget|bash|sh|sudo|rm|nc|ssh|python3?)\b", re.IGNORECASE),
    "pipe_to_shell": re.compile(r"\b(curl|wget)\b[^\n]*\|\s*(ba|z)?sh\b", re.IGNORECASE),
    "destructive": re.compile(r"\brm\s+-[a-z]*r[a-z]*f?\b|\bsudo\b|/etc/(passwd|shadow)|authorized_keys",
                              re.IGNORECASE),
    "exfiltrate": re.compile(r"\b(send|post|upload|exfiltrate)\b.{0,50}\b(key|token|secret|password)s?\b",
                             re.IGNORECASE),
}


def screen_text(text: str) -> tuple[str, list[str]]:
    kept, flags = [], []
    for line in text.splitlines():
        hit = [name for name, rx in _RULES.items() if rx.search(line)]
        if hit:
            flags.extend(hit)
            for name in hit:
                DOC_FLAGS.labels(rule=name).inc()
            continue
        kept.append(line)
    return "\n".join(kept)[:MAX_DOC_CHARS], flags


def screen_doc(doc: dict) -> dict:
    content, flags = screen_text(str(doc.get("content", "")))
    title, tflags = screen_text(str(doc.get("title", "")))
    return {"title": title[:200], "url": str(doc.get("url", ""))[:300],
            "content": content, "flags": flags + tflags}
