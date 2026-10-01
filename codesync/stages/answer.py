"""Write the answer from the passages, then check every citation
(work package 8).

The model may only cite tags that exist and may only quote words that are
in the passages; the check runs on the finished answer and reports what
failed, so a reader knows which parts to distrust.
"""
from __future__ import annotations

import re
from collections.abc import Iterator

from .. import prompts
from ..models.chat import engine
from .chunk import Chunk
from .retrieve import _passage_text

_TAG = re.compile(r"\[(P\d+)\]")
# a quotation: opens with a quote mark, starts with a letter, digit or
# bracket, and stays inside one paragraph (straight quotes cannot tell an
# opening from a closing mark, so the text BETWEEN two quotations must not
# be taken for one)
_QUOTE = re.compile(r"[\"“]([A-Za-z0-9(][^\"”\n]{11,400})[\"”]")


def format_passages(passages: list[dict], question: str | None = None) -> str:
    out = []
    for p in passages:
        c: Chunk = p["chunk"]
        where = f"{c.code or 'code'} {c.edition}".strip()
        page = f"page {c.page_label}" if c.page_label else f"PDF page {c.page}"
        head = f"[{p['tag']}] {where}, {c.ref} ({page})"
        # a clause with its nomenclature block can run to 4-5k characters;
        # eight such passages are still well inside the model's context
        out.append(head + "\n" + _passage_text(c, 6_000, question))
    return "\n\n".join(out)


def messages_for(question: str, passages: list[dict]) -> list[dict]:
    return [{"role": "system", "content": prompts.ASK_SYSTEM},
            {"role": "user", "content": prompts.ASK_USER.format(
                question=question, passages=format_passages(passages, question))}]


def stream_answer(question: str, passages: list[dict], max_tokens: int = 1_500
                  ) -> Iterator[dict]:
    """Yields {"delta": str} pieces, then {"done": True, ...}. Falls back to
    one non-streamed call if the engine's stream method fails."""
    messages = messages_for(question, passages)
    eng = engine()
    try:
        for piece in eng.stream.remote_gen(messages, max_tokens, 0.0):
            yield piece
        return
    except Exception as e:                               # noqa: BLE001
        fallback = f"{type(e).__name__}: {e}"
    res = eng.chat.remote([{"messages": messages, "max_tokens": max_tokens}], model="llm")
    r = res["results"][0]
    yield {"delta": r["text"]}
    yield {"done": True, "finish_reason": r.get("finish_reason", ""),
           "prompt_tokens": r.get("prompt_tokens", 0),
           "completion_tokens": r.get("completion_tokens", 0),
           "stream_fallback": fallback}


def _norm(s: str) -> str:
    s = s.lower().replace("’", "'").replace("‘", "'")
    s = re.sub(r"[–—]", "-", s)
    # "tem-peratures" (a line break in the scan) and "temperatures" are the
    # same words; so are "side-step" and "sidestep" for this purpose
    s = re.sub(r"(?<=[a-z])-\s*(?=[a-z])", "", s)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def check_citations(answer: str, passages: list[dict]) -> dict:
    """Which tags and quotes in the answer are backed by the passages."""
    tags_ok = {p["tag"] for p in passages}
    cited = sorted(set(_TAG.findall(answer)), key=lambda t: int(t[1:]))
    bad_tags = [t for t in cited if t not in tags_ok]
    hay = _norm(" ".join(_passage_text(p["chunk"], 100_000) for p in passages))
    quotes = [q.strip() for q in _QUOTE.findall(answer)]
    bad_quotes = [q for q in quotes if _norm(q) not in hay]
    not_found = bool(re.search(r"not found in the passages", answer, re.I))
    return {"cited": cited, "bad_tags": bad_tags, "quotes": len(quotes),
            "bad_quotes": bad_quotes, "not_found": not_found,
            "ok": not bad_tags and not bad_quotes and (bool(cited) or not_found)}
