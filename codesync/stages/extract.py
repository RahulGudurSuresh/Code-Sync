"""Clauses, tables and figure readings -> atomic requirements.

Clauses are sent to the LLM in blocks — everything under one heading — so
a list item arrives with the lead-in sentence it depends on. Each clause in
a block is tagged with its reference and the model cites that tag, which is
how a requirement stays traceable to the page and clause it came from.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from .. import config, library, prompts
from ..models.chat import chat_json, image_part
from ..schema import DocumentMeta, Element, ParsedDocument, Requirement

MODALITIES = {"shall", "shall not", "should", "should not", "may", "info"}


@dataclass
class Block:
    section: str
    lines: list[str] = field(default_factory=list)
    elements: list[Element] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def build_blocks(parsed: ParsedDocument) -> list[Block]:
    by_id = {e.id: e for e in parsed.elements}

    def heading_of(e: Element) -> Element | None:
        p = by_id.get(e.parent) if e.parent else None
        while p is not None and not p.is_heading:
            p = by_id.get(p.parent) if p.parent else None
        return p

    blocks: list[Block] = []
    cur: Block | None = None
    cur_key: str | None = "?"
    for e in parsed.elements:
        if e.kind != "clause" or e.is_heading or not e.text.strip() \
                or e.extra.get("front_matter"):
            continue
        head = heading_of(e)
        key = head.id if head else None
        base = head.depth + 1 if head else 0
        top_level = e.parent == key or e.parent is None
        if cur is None or key != cur_key or (
                top_level and len(cur.text) > config.BLOCK_MAX_CHARS):
            cur = Block(section=" > ".join(e.heading_trail) or "(untitled)")
            blocks.append(cur)
            cur_key = key
            if not top_level:
                # block starts mid-subtree: repeat the lead-in clauses
                chain, p = [], by_id.get(e.parent)
                while p is not None and not p.is_heading:
                    chain.append(p)
                    p = by_id.get(p.parent) if p.parent else None
                for a in reversed(chain):
                    cur.lines.append("  " * (a.depth - base)
                                     + f"[{a.ref}] {a.text}  (context)")
        cur.lines.append("  " * max(e.depth - base, 0) + f"[{e.ref}] {e.text}")
        cur.elements.append(e)
    return blocks


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _ref_key(s: str) -> str:
    return re.sub(r"[^a-z0-9¶•]", "", s.lower())


def _requirement(raw: dict, element: Element, doc_id: str, n: int,
                 haystack: str, from_figure: bool = False,
                 ref: str | None = None) -> Requirement | None:
    statement = str(raw.get("statement", "")).strip()
    if not statement:
        return None
    modality = str(raw.get("modality", "")).strip().lower()
    modality = {"must": "shall", "must not": "shall not",
                "will": "shall"}.get(modality, modality)
    if from_figure:
        modality = "detail"
    elif modality not in MODALITIES:
        modality = "info"
    quote = str(raw.get("quote", "")).strip()
    refs = raw.get("references") or []
    return Requirement(
        id=f"{doc_id}:r{n}", doc_id=doc_id, element_id=element.id,
        ref=ref or element.ref, page=element.page, statement=statement,
        subject=str(raw.get("subject", "")).strip(),
        parameter=str(raw.get("parameter", "")).strip(),
        value=str(raw.get("value", "")).strip(), modality=modality,
        quote=quote,
        quote_verified=_quoted(quote, haystack),
        references=[str(r) for r in refs] if isinstance(refs, list) else [],
        from_figure=from_figure)


def _quoted(quote: str, haystack: str) -> bool:
    """True if `quote` is verbatim source text. The model joins a lead-in
    and a list item with an ellipsis ("shall be provided: ... for access
    to platforms"); each side of the ellipsis must then be found."""
    pieces = [_norm(p) for p in re.split(r"\.{3,}|…", quote)]
    pieces = [p for p in pieces if p]
    return bool(pieces) and all(p in haystack for p in pieces)


def _view_statements(reading: dict) -> list[dict]:
    """One statement per titled view: what the standard offers.

    What a standard draws is what it offers. A clause in another document
    that limits a component to one type ("ladders should be side-step")
    can only be matched against a drawing if the drawing's views are
    stated as the types they show."""
    out = []
    for v in reading.get("views") or []:
        title, component = v.get("title", ""), v.get("component", "")
        arrangement = v.get("arrangement", "")
        if not title or not (component or arrangement):
            continue
        what = ": ".join(p for p in (component, arrangement) if p)
        out.append({
            "detail": title, "subject": component,
            "parameter": "type of arrangement", "value": arrangement,
            "statement": f"The standard offers this arrangement as a "
                         f'standard detail: {what} (view "{title}").'})
    return out


def extraction_key(client_name: str) -> str:
    return f"{client_name}|{prompts.PROMPT_VERSION}|{config.BLOCK_MAX_CHARS}"


def extract_requirements(meta: DocumentMeta, parsed: ParsedDocument,
                         figure_readings: list[dict], llm,
                         progress: Callable[[str], None] | None = None
                         ) -> tuple[list[Requirement], dict]:
    """(requirements, stats). One LLM request per block and per table; the
    figure readings are already requirement-shaped and pass straight in."""
    say = progress or (lambda _m: None)
    blocks = build_blocks(parsed)
    tables = [e for e in parsed.elements if e.kind == "table"
              and e.text.strip()]
    fig_dir = library.doc_dir(meta.doc_id) / "figures"

    requests, owners = [], []
    for b in blocks:
        requests.append({"messages": [
            {"role": "system", "content": prompts.EXTRACT_SYSTEM},
            {"role": "user", "content": prompts.EXTRACT_USER.format(
                doc_name=meta.name, section=b.section, block=b.text)}],
            "max_tokens": 6000})
        owners.append(b)
    for t in tables:
        cap = t.extra.get("caption", "")
        text = prompts.TABLE_USER.format(
            doc_name=meta.name, section=" > ".join(t.heading_trail) or "-",
            caption=f' "{cap}"' if cap else "", ref=t.ref,
            table=t.text[:12_000])
        content: list | str = text
        if t.extra.get("method") != "pdf-vector" and t.image:
            # OCR-derived table text can be wrong; let the LLM see the table
            content = [image_part(fig_dir / t.image, config.LLM_IMG_TOKENS),
                       {"type": "text", "text": text}]
        requests.append({"messages": [
            {"role": "system", "content": prompts.EXTRACT_SYSTEM},
            {"role": "user", "content": content}], "max_tokens": 6000})
        owners.append(t)

    say(f"extracting from {len(blocks)} clause blocks and "
        f"{len(tables)} tables")
    answers = chat_json(llm, requests) if requests else []

    reqs: list[Requirement] = []
    stats = {"blocks": len(blocks), "tables": len(tables), "parse_failed": 0,
             "truncated": 0, "unverified_quotes": 0, "unknown_clause": 0,
             "ambiguous_drawing_statements": 0}
    for owner, (answer, raw) in zip(owners, answers):
        if raw.get("finish_reason") == "length":
            stats["truncated"] += 1
        items = answer.get("requirements") if isinstance(answer, dict) else None
        if not isinstance(items, list):
            stats["parse_failed"] += 1
            continue
        if isinstance(owner, Block):
            by_ref = {_ref_key(e.ref): e for e in owner.elements}
            block_text = _norm(owner.text)
        for item in items:
            if not isinstance(item, dict):
                continue
            if isinstance(owner, Block):
                element = by_ref.get(_ref_key(str(item.get("clause", ""))))
                if element is None:
                    stats["unknown_clause"] += 1
                    element = owner.elements[0]
                    haystack = block_text
                else:
                    haystack = _norm(element.text)
            else:
                element, haystack = owner, _norm(owner.text)
            r = _requirement(item, element, meta.doc_id, len(reqs) + 1,
                             haystack)
            if r:
                # a quote may legitimately span parent and child clauses
                if not r.quote_verified and isinstance(owner, Block):
                    r.quote_verified = _quoted(r.quote, block_text)
                stats["unverified_quotes"] += not r.quote_verified
                reqs.append(r)

    by_id = {e.id: e for e in parsed.elements}
    for reading in figure_readings:
        fig = by_id[reading["element_id"]]
        for item in _view_statements(reading) + reading["requirements"]:
            if item.get("ambiguous"):
                stats["ambiguous_drawing_statements"] += 1
                continue
            detail = item.get("detail", "").strip()
            r = _requirement(item, fig, meta.doc_id, len(reqs) + 1, "",
                             from_figure=True,
                             ref=f"{fig.ref} · {detail}" if detail else fig.ref)
            if r:
                reqs.append(r)
    stats["from_text"] = sum(not r.from_figure for r in reqs)
    stats["from_figures"] = sum(r.from_figure for r in reqs)
    stats["drawing_check"] = check_drawing_statements(meta, parsed, reqs,
                                                      llm, say)
    return reqs, stats


def check_drawing_statements(meta: DocumentMeta, parsed: ParsedDocument,
                             reqs: list[Requirement], llm,
                             say: Callable[[str], None]) -> dict:
    """Second reading of every drawing statement, by the LLM's own vision.

    The two models fail differently: the local one misattributes bare
    dimensions, and the LLM's re-reading is not dependable enough to
    replace a statement it disputes. So neither is trusted alone — a
    statement takes part in the comparison only when both agree, and the
    rest are kept, marked, for a person to read from the sheet."""
    targets = [r for r in reqs if r.from_figure]
    stats = {"checked": len(targets), "confirmed": 0, "incorrect": 0,
             "not_found": 0, "unread": 0}
    if not targets:
        return stats
    by_id = {e.id: e for e in parsed.elements}
    fig_dir = library.doc_dir(meta.doc_id) / "figures"
    images: dict[str, dict] = {}

    def image(element_id: str) -> dict:
        if element_id not in images:
            images[element_id] = image_part(
                fig_dir / by_id[element_id].image, config.LLM_IMG_TOKENS)
        return images[element_id]

    requests = [{"messages": [
        {"role": "system", "content": prompts.VERIFY_SYSTEM},
        {"role": "user", "content": [
            image(r.element_id),
            {"type": "text", "text": prompts.VERIFY_USER.format(
                doc_name=meta.name, ref=by_id[r.element_id].ref,
                statement=r.statement)}]}],
        "max_tokens": 400} for r in targets]
    say(f"checking {len(targets)} drawing statements against their sheets")
    for r, (answer, _raw) in zip(targets, chat_json(llm, requests)):
        verdict = "unread"
        if isinstance(answer, dict):
            v = str(answer.get("verdict", "")).strip().lower()
            if v in ("confirmed", "incorrect", "not_found"):
                verdict = v
        r.drawing_check = verdict
        stats[verdict] += 1
        if verdict == "confirmed":
            continue
        r.excluded = True
        if verdict == "incorrect":
            reread = str(answer.get("correct_statement", "")).strip()
            evidence = str(answer.get("evidence", "")).strip()
            r.check_note = f'second reading: "{reread}" (from: {evidence})'
        elif verdict == "not_found":
            r.check_note = "second reading could not find this on the sheet"
        else:
            r.check_note = "second reading returned no answer"
    return stats
