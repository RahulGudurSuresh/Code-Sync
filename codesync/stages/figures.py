"""Figure reading: each drawing sheet -> the requirements drawn on it.

The reader is any chat client that accepts images; by default the
Qwen3.5-9B in the Modal engine. It is deliberately a different model from
the 27B that later checks every statement against the sheet (see
`extract.check_drawing_statements`): a statement counts only when two
readings agree.
"""
from __future__ import annotations

import re
from collections.abc import Callable

from .. import config, library, prompts
from ..models.chat import chat_json, image_part


def _flag_ambiguous(reqs: list[dict]) -> list[dict]:
    """Drop exact repeats and flag statements that cannot all be right.

    A vision model that cannot tell what a bare dimension line measures
    tends to hang it on the nearest label, producing a run of statements
    giving different values for the same property of the same component
    ("the unsupported length is 4 ft", "... 7 ft", "... 1 ft"). One sheet
    can give such a property a minimum and a maximum, not three or more
    values, so a group of three or more is flagged and kept out of the
    comparison rather than guessed at."""
    seen: set[str] = set()
    unique = []
    for r in reqs:
        key = r["statement"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(r)
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in unique:
        groups.setdefault((r["subject"].lower(), r["parameter"].lower()),
                          []).append(r)
    for (subject, parameter), members in groups.items():
        flagged = bool(subject or parameter) and len(members) >= 3
        for r in members:
            r["ambiguous"] = flagged
    # A second symptom of the same failure is a run of statements that are
    # the same sentence with the numbers changed ("the distance from the
    # sixth rung to the seventh is 3", "... seventh to the eighth is 3").
    shapes: dict[str, list[dict]] = {}
    for r in unique:
        shapes.setdefault(_shape(r["statement"]), []).append(r)
    for members in shapes.values():
        if len(members) >= 3:
            for r in members:
                r["ambiguous"] = True
    return unique


_ORDINALS = ("first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|"
             "tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|"
             "sixteenth|seventeenth|eighteenth|nineteenth|twentieth|"
             "thirtieth|twenty|thirty")


def _shape(statement: str) -> str:
    """The sentence with every number and ordinal word blanked out."""
    s = re.sub(rf"\b(?:{_ORDINALS})(?:-\w+)?\b", "#", statement.lower())
    return re.sub(r"[\d'\"/.,-]+", "#", s)


def read_figures(meta, parsed, client,
                 img_tokens: int = config.LLM_IMG_TOKENS,
                 progress: Callable[[str], None] | None = None) -> list[dict]:
    """[{"element_id", "sheet_title", "views", "requirements", "reader"}],
    one per figure, cached per reader and prompt version. All sheets of a
    document go to the reader in one batch."""
    say = progress or (lambda _m: None)
    key = f"{client.name}|{prompts.FIGURE_VERSION}|{img_tokens}"
    cached = library.load_json(meta.doc_id, "figure_readings.json")
    done = {r["element_id"]: r for r in (cached or {}).get("readings", [])
            } if cached and cached.get("key") == key else {}

    figures = [e for e in parsed.elements if e.kind == "figure"]
    fig_dir = library.doc_dir(meta.doc_id) / "figures"
    todo = [f for f in figures if f.id not in done]
    if todo:
        say(f"reading {len(todo)} drawing sheets ({client.name})")
        requests = [{"messages": [
            {"role": "system", "content": prompts.FIGURE_SYSTEM},
            {"role": "user", "content": [
                image_part(fig_dir / fig.image, img_tokens),
                {"type": "text", "text": prompts.FIGURE_USER.format(
                    doc_name=meta.name, ref=fig.ref,
                    labels=" | ".join(fig.extra.get("labels", []))
                    or "(none)")}]}],
            "max_tokens": 3500} for fig in todo]
        for fig, (answer, raw) in zip(todo, chat_json(client, requests)):
            answer = answer if isinstance(answer, dict) else {}
            reqs = [{k: str(r.get(k, "")).strip() for k in (
                        "detail", "statement", "subject", "parameter", "value")}
                    for r in answer.get("requirements") or []
                    if isinstance(r, dict)
                    and str(r.get("statement", "")).strip()]
            views = [{k: str(v.get(k, "")).strip()
                      for k in ("title", "component", "arrangement")}
                     for v in answer.get("views") or [] if isinstance(v, dict)]
            reqs = _flag_ambiguous(reqs)
            done[fig.id] = {
                "element_id": fig.id, "reader": client.name,
                "sheet_title": str(answer.get("sheet_title", "")),
                "views": views, "requirements": reqs,
                "parse_failed": not answer,
                "ambiguous": sum(r["ambiguous"] for r in reqs),
                "truncated": raw.get("finish_reason") == "length"}
        library.save_json(meta.doc_id, "figure_readings.json",
                          {"key": key, "readings": list(done.values())})
    return [done[f.id] for f in figures]
