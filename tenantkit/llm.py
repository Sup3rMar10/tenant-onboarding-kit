"""The optional language-model arbiter for columns the rules leave ambiguous.

The offline default is no arbiter at all: the mapper's own rules decide, and
anything still unclear is flagged for a human. An arbiter is a plain function that
takes a prompt string and returns text, so any provider (or a fake in tests) can be
plugged in. `AnthropicArbiter` is a thin adapter for the Anthropic API. It is off by
default, needs the optional `anthropic` package and an API key, and has NOT been run
against the live API in this project; only the prompt building and answer parsing are
tested, with a fake model.

Privacy: by default the prompt carries the column's name, shape statistics and its
candidate fields, never cell values. Pass `send_samples=True` to include up to three
masked samples (letters become x, digits become 9).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

LLMFunction = Callable[[str], str]


@dataclass
class Question:
    filename: str
    entity: str
    source: str
    summary: dict
    candidates: list[str]
    samples: list[str]


def mask(value: str) -> str:
    return re.sub(r"[A-Za-z]", "x", re.sub(r"\d", "9", value))


def build_prompt(q: Question) -> str:
    lines = [
        "You map one column of a customer's data export to a canonical field. Answer with exactly one of the",
        "candidate names, or NONE if the column should not be mapped. No explanation.",
        f"Table: {q.filename} (looks like: {q.entity})",
        f"Column name: {q.source}",
        "Column statistics: " + ", ".join(f"{k}={v}" for k, v in q.summary.items()),
        "Candidates: " + ", ".join(q.candidates),
    ]
    if q.samples:
        lines.append("Masked samples: " + " | ".join(q.samples))
    return "\n".join(lines)


def parse_answer(text: str, candidates: list[str]) -> str | None:
    """Accept only an answer that names a candidate. Anything else is ignored."""
    t = text.strip().strip(".`\"' ")
    if t.upper() == "NONE":
        return None
    for c in candidates:
        if t == c or t == c.split(".", 1)[-1]:
            return c
    return None


class Arbiter:
    """Wraps an LLM function. Asked only about ambiguous columns, and only ever allowed to pick a candidate."""

    def __init__(self, llm: LLMFunction, send_samples: bool = False, max_questions: int = 20):
        self.llm = llm
        self.send_samples = send_samples
        self.max_questions = max_questions
        self.asked = 0

    def resolve(self, q: Question) -> str | None:
        if self.asked >= self.max_questions:
            return None
        self.asked += 1
        return parse_answer(self.llm(build_prompt(q)), q.candidates)


class AnthropicArbiter(Arbiter):
    """Optional adapter. Not exercised against the live API here."""

    def __init__(self, model: str = "claude-haiku-5-5", **kw):
        import anthropic  # optional dependency

        client = anthropic.Anthropic()

        def call(prompt: str) -> str:
            msg = client.messages.create(model=model, max_tokens=20, messages=[{"role": "user", "content": prompt}])
            return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")

        super().__init__(call, **kw)


def consult_arbiter(sig, mapping, params, arbiter: Arbiter) -> None:
    """Ask about columns that are unmapped-but-plausible or mapped-but-doubtful, and apply a valid answer."""
    low = params.threshold * 0.6
    for fs in sig.files:
        fm = mapping.files[fs.name]
        by_src = {c.source: (i, c) for i, c in enumerate(fs.columns)}
        taken = {c.field for c in fm.columns if c.field}
        for cm in fm.columns:
            i, col = by_src[cm.source]
            best = cm.alternatives[0][1] if cm.alternatives else 0.0
            ambiguous = (cm.field is None and best >= low) or (cm.field is not None and cm.review and cm.margin < params.review_margin)
            if not ambiguous:
                continue
            cands = [k for k, _ in cm.alternatives[:3]]
            p = col.profile
            summary = {"kind": p.kind, "unique_ratio": round(p.uniq_ratio, 2), "null_rate": round(p.null_rate, 2),
                       "mean_length": round(p.len_mean, 1), "shape": p.shape_top}
            samples = []
            if arbiter.send_samples:
                samples = [mask(v) for v in sorted(p.values)[:3]]
            ans = arbiter.resolve(Question(fs.name, fm.entity, cm.source, summary, cands, samples))
            if ans is None:
                continue
            fname = ans.split(".", 1)[1]
            if fname in taken and fname != cm.field:
                continue                      # one column per field: the arbiter may not steal
            if cm.field:
                taken.discard(cm.field)
            cm.field, cm.decided_by, cm.review = fname, "arbiter", True
            taken.add(fname)
