"""Parsed elements -> searchable chunks (work package 3 of the build plan).

One chunk per clause, carrying the heading path above it, the lead-in text
of its parents, its notes, its equations, and the cross-references it makes
("see para. 102.3.2") as links. Tables and figures are chunks of their own.

Rules that matter:
  * the original wording is kept word for word; nothing here rewrites,
    summarises or drops source text — a description can be ADDED later,
    beside the text, never instead of it;
  * every non-heading element lands in exactly one chunk (checked);
  * short clauses are kept with their sub-items ("(a) ... (b) ...") so a
    list item never travels without the sentence that gives it meaning;
  * a "where ..." nomenclature block stays with its equation.
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field

from ..schema import DocumentMeta, Element, ParsedDocument

MAX_CHUNK_CHARS = 1_500          # a clause and its sub-items up to this size
NOTE_START = re.compile(r"^\s*(?:GENERAL\s+)?NOTES?\b", re.I)
WHERE_START = re.compile(r"^\s*where\b", re.I)
EQUATION = re.compile(r"\[EQUATION(?:\s+\(?([\w]+)\)?)?:\s*\$\$(.*?)\$\$\]", re.S)
DEFINITION_HEADING = re.compile(r"\bdefinitions?\b", re.I)
# "arc welding: a group of ..." — a defined term, as the parser splits them
TERM = re.compile(r"^[a-z][\w ,\-/()']{0,60}:\s+\S")


class Ref(BaseModel):
    kind: str                     # para | table | fig | appendix | eq | chapter
    target: str                   # "102.3.2", "104.1.2(A)", "A"
    text: str                     # as written
    chunk_id: str | None = None   # resolved within the document, if found


class Chunk(BaseModel):
    id: str
    doc_id: str
    code: str = ""
    edition: str = ""
    kind: str                     # clause | table | figure | definition
    ref: str                      # citation: "104.1.2 A", "Table 104.1.2(A)"
    page: int                     # page in the stored PDF
    page_label: str = ""          # printed page number
    heading_path: list[str] = Field(default_factory=list)
    context: str = ""             # lead-in text of non-heading parents, verbatim
    text: str                     # verbatim
    notes: list[str] = Field(default_factory=list)
    latex: list[str] = Field(default_factory=list)
    table: dict | None = None     # {"caption", "columns", "rows", "html"}
    image: str | None = None
    description: str = ""         # added by a model; never replaces text
    refs: list[Ref] = Field(default_factory=list)
    element_ids: list[str] = Field(default_factory=list)

    @property
    def embed_text(self) -> str:
        """What the embedder sees: where the chunk sits, then what it says."""
        head = " > ".join(self.heading_path)
        parts = [f"{head}: " if head else "", self.ref + ". " if self.ref else ""]
        if self.context:
            parts.append(self.context + " ")
        parts.append(self.text)
        if self.notes:
            parts.append(" " + " ".join(self.notes))
        if self.description:
            parts.append(" " + self.description)
        return "".join(parts)[:6_000]


# ---- cross-references ------------------------------------------------------
_PARA_NUM = r"\d{3}(?:\.\d+)*(?:\([A-Z](?:\.\d+)*\))?"
_REF_PATTERNS = [
    ("para", re.compile(rf"\bparas?\.?\s+({_PARA_NUM}(?:\s*(?:,|and|or|through|to)\s*{_PARA_NUM})*)", re.I)),
    ("table", re.compile(r"\bTables?\s+([A-Z]?-?\d+(?:\.\d+)*(?:\([A-Z0-9.]+\))*(?:\s*(?:,|and|or)\s*[A-Z]?-?\d+(?:\.\d+)*(?:\([A-Z0-9.]+\))*)*)")),
    ("fig", re.compile(r"\bFig(?:s|ures?)?\.?\s+([A-Z]?-?\d+(?:\.\d+)*(?:\([A-Z0-9.]+\))*(?:\s*(?:,|and|or)\s*[A-Z]?-?\d+(?:\.\d+)*(?:\([A-Z0-9.]+\))*)*)")),
    ("appendix", re.compile(r"\b(?:Mandatory\s+|Nonmandatory\s+)?Appendi(?:x|ces)\s+([A-Z])\b")),
    ("eq", re.compile(r"\beqs?\.?\s*\((\d+[A-Z]?)\)(?:\s*(?:,|and|or)\s*\((\d+[A-Z]?)\))*", re.I)),
    ("chapter", re.compile(r"\bChapter\s+([IVX]+)\b")),
]
_SPLIT = re.compile(r"\s*(?:,|\band\b|\bor\b|\bthrough\b|\bto\b)\s*")


def find_refs(text: str) -> list[Ref]:
    out: list[Ref] = []
    seen: set[tuple[str, str]] = set()
    for kind, rx in _REF_PATTERNS:
        for m in rx.finditer(text):
            groups = [g for g in m.groups() if g]
            targets = []
            for g in groups:
                targets += [t.strip() for t in _SPLIT.split(g) if t.strip()]
            for t in targets:
                t = re.sub(r"\s+", "", t)
                if (kind, t) not in seen:
                    seen.add((kind, t))
                    out.append(Ref(kind=kind, target=t, text=m.group(0)))
    return out


def _norm_ref(s: str) -> str:
    return re.sub(r"[\s.]+$", "", re.sub(r"\s+", "", s)).lower()


def resolve_refs(chunks: list[Chunk]) -> dict:
    """Point each reference at a chunk of the same document when one
    matches. Returns counts {resolved, unresolved}."""
    by_para: dict[str, str] = {}
    for c in chunks:
        if c.kind in ("clause", "definition"):
            key = _norm_ref(c.ref.split(" ")[0]) if c.ref else ""
            # "104.1.2 A" -> also reachable as "104.1.2(a)"
            full = _norm_ref(c.ref.replace(" ", "(") + ")") if " " in c.ref else key
            for k in {key, full, _norm_ref(c.ref)}:
                by_para.setdefault(k, c.id)
    tables = [(c, _norm_ref((c.table or {}).get("caption", "") or c.text[:80]))
              for c in chunks if c.kind == "table"]
    figures = [(c, _norm_ref(c.text[:80] + " " + c.ref)) for c in chunks if c.kind == "figure"]
    eq_owner: dict[str, str] = {}
    for c in chunks:
        for m in EQUATION.finditer(c.text):
            if m.group(1):
                eq_owner.setdefault(m.group(1).strip("()").lower(), c.id)
    stats = {"resolved": 0, "unresolved": 0}
    for c in chunks:
        for r in c.refs:
            t = _norm_ref(r.target)
            hit = None
            if r.kind == "para":
                hit = by_para.get(t) or by_para.get(t.replace("(", "").replace(")", ""))
            elif r.kind == "table":
                hit = next((tc.id for tc, cap in tables if "table" + t in cap), None)
            elif r.kind == "fig":
                hit = next((fc.id for fc, cap in figures if "fig" in cap and t in cap), None)
            elif r.kind == "eq":
                hit = eq_owner.get(t)
            if hit and hit != c.id:
                r.chunk_id = hit
                stats["resolved"] += 1
            else:
                stats["unresolved"] += 1
    return stats


# ---- tables ----------------------------------------------------------------
_TAG = re.compile(r"<[^>]+>")


def table_to_grid(text: str) -> tuple[list[str], list[list[str]]]:
    """Rows and columns from the parser's table text (HTML from the table
    model, or Markdown from a vector table). Cell text is kept verbatim."""
    rows: list[list[str]] = []
    if "<table" in text.lower():
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", text, re.S | re.I):
            cells = []
            for m in re.finditer(r"<t[dh]([^>]*)>(.*?)</t[dh]>", tr, re.S | re.I):
                attrs, body = m.group(1), m.group(2)
                span = re.search(r'colspan="?(\d+)', attrs)
                value = re.sub(r"\s+", " ", _TAG.sub(" ", body)).strip()
                cells += [value] * int(span.group(1) if span else 1)
            if cells:
                rows.append(cells)
    else:
        for line in text.splitlines():
            if "|" in line and not re.match(r"^\s*\|?\s*:?-{2,}", line):
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                rows.append(cells)
    if not rows:
        return [], []
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    header, body = rows[0], rows[1:]
    if not any(header):
        header, body = [f"col{i + 1}" for i in range(width)], rows
    return header, body


# ---- building ----------------------------------------------------------------
def _is_note(e: Element) -> bool:
    return bool(e.extra.get("note")) or bool(NOTE_START.match(e.text))


def _is_term(e: Element) -> bool:
    """"t_m = minimum required wall thickness ..." — a nomenclature line."""
    return "=" in e.text[:60]


def _subtree_text(e: Element, children: dict[str, list[Element]],
                  only_terms: bool = False) -> str:
    """The element's text and its sub-items', in order; notes are left out
    (they travel beside the chunk). With `only_terms`, only the leading
    run of nomenclature children is kept."""
    parts = [e.text]
    for c in children.get(e.id, []):
        if c.kind != "clause" or c.is_heading or _is_note(c):
            continue
        if only_terms and not _is_term(c):
            break
        label = c.number or ""
        parts.append((f"{label} " if label else "") + _subtree_text(c, children))
    return " ".join(p for p in parts if p)


def _subtree_ids(e: Element, children: dict[str, list[Element]],
                 only_terms: bool = False) -> list[str]:
    ids = [e.id]
    for c in children.get(e.id, []):
        if c.kind != "clause" or c.is_heading or _is_note(c):
            continue
        if only_terms and not _is_term(c):
            break
        ids += _subtree_ids(c, children)
    return ids


def _subtree_notes(e: Element, children: dict[str, list[Element]],
                   ids: list[str]) -> tuple[list[str], list[str]]:
    """Note children of any element in `ids`: (texts, their ids)."""
    texts, nids = [], []
    for eid in ids:
        for c in children.get(eid, []):
            if c.kind == "clause" and _is_note(c):
                texts.append(_subtree_text(c, children))
                nids += _subtree_ids(c, children)
    return texts, nids


def build_chunks(meta: DocumentMeta, parsed: ParsedDocument) -> tuple[list[Chunk], dict]:
    by_id = {e.id: e for e in parsed.elements}
    children: dict[str, list[Element]] = {}
    for e in parsed.elements:
        if e.parent:
            children.setdefault(e.parent, []).append(e)
    labels = {p.page: p.label for p in parsed.pages}

    def heading_path(e: Element) -> list[str]:
        path, p = [], by_id.get(e.parent) if e.parent else None
        while p is not None:
            if p.is_heading:
                path.append(f"{p.ref} {p.text}".strip())
            p = by_id.get(p.parent) if p.parent else None
        return path[::-1]

    def lead_in(e: Element) -> str:
        texts, p = [], by_id.get(e.parent) if e.parent else None
        while p is not None and not p.is_heading:
            if p.kind == "clause":
                texts.append(p.text)
            p = by_id.get(p.parent) if p.parent else None
        return " ".join(texts[::-1])

    def in_definitions(e: Element) -> bool:
        return any(DEFINITION_HEADING.search(h) for h in e.heading_trail)

    chunks: list[Chunk] = []
    taken: set[str] = set()
    order = [e for e in parsed.elements if not e.extra.get("front_matter")]

    def new_chunk(kind: str, e: Element, text: str, ids: list[str], **kw) -> Chunk:
        c = Chunk(id=f"{meta.doc_id}:c{len(chunks) + 1}", doc_id=meta.doc_id,
                  code=meta.code or meta.name, edition=meta.edition, kind=kind, ref=e.ref,
                  page=e.page, page_label=labels.get(e.page, ""),
                  heading_path=heading_path(e), text=text, element_ids=ids, **kw)
        chunks.append(c)
        taken.update(ids)
        return c

    for i, e in enumerate(order):
        if e.id in taken or e.is_heading:
            continue
        if e.kind == "table":
            columns, rows = table_to_grid(e.text)
            caption = e.extra.get("caption", "") or ""
            new_chunk("table", e, caption or e.ref, [e.id], image=e.image,
                      table={"caption": caption, "columns": columns, "rows": rows,
                             "html": e.text if "<table" in e.text.lower() else "",
                             "method": e.extra.get("method", "")})
            continue
        if e.kind == "figure":
            new_chunk("figure", e, e.text or e.ref, [e.id], image=e.image)
            continue
        if e.kind != "clause":
            continue

        # notes and footnotes attach to the chunk before them
        if (NOTE_START.match(e.text) or e.extra.get("note")) and chunks \
                and chunks[-1].page >= e.page - 1:
            prev = chunks[-1]
            prev.notes.append(_subtree_text(e, children))
            prev.element_ids += _subtree_ids(e, children)
            taken.update(prev.element_ids)
            continue

        # a definitions list: one chunk per defined term
        if in_definitions(e) and TERM.match(e.text):
            term = e.text.split(":", 1)[0].strip()
            c = new_chunk("definition", e, _subtree_text(e, children),
                          _subtree_ids(e, children), context=lead_in(e))
            c.ref = f"Definition: {term}"
            continue

        text = _subtree_text(e, children)
        if len(text) <= MAX_CHUNK_CHARS:
            ids = _subtree_ids(e, children)
        elif EQUATION.search(e.text):
            # too big as a whole, but an equation keeps the nomenclature
            # items written under it, whatever their size
            text = _subtree_text(e, children, only_terms=True)
            ids = _subtree_ids(e, children, only_terms=True)
        else:
            text, ids = e.text, [e.id]        # children become their own chunks
        notes, note_ids = _subtree_notes(e, children, ids)
        c = new_chunk("clause", e, text, ids + note_ids, context=lead_in(e))
        c.notes = notes

        # an equation keeps its "where ..." nomenclature with it
        if EQUATION.search(text):
            c.latex = [m.group(2).strip() for m in EQUATION.finditer(text)]
            j = i + 1
            folded = 0
            last_page = e.page
            while j < len(order) and folded < 30:
                nxt = order[j]
                j += 1
                if nxt.id in taken or nxt.is_heading or nxt.kind != "clause":
                    if nxt.is_heading:
                        break
                    continue
                if nxt.page > last_page + 1:          # a different place
                    break
                is_where = bool(WHERE_START.match(nxt.text))
                is_term = _is_term(nxt) or (nxt.number and len(nxt.text) < 400 and folded)
                if not (is_where or (folded and is_term)):
                    break
                c.text += "\n" + _subtree_text(nxt, children)
                c.element_ids += _subtree_ids(nxt, children)
                taken.update(c.element_ids)
                last_page = nxt.page
                folded += 1

    for c in chunks:
        c.refs = find_refs(c.text + " " + " ".join(c.notes))
    ref_stats = resolve_refs(chunks)

    # nothing lost: every non-heading element is in exactly one chunk
    counted: dict[str, int] = {}
    for c in chunks:
        for eid in c.element_ids:
            counted[eid] = counted.get(eid, 0) + 1
    expected = [e.id for e in order if not e.is_heading and e.kind in ("clause", "table", "figure")]
    missing = [eid for eid in expected if eid not in counted]
    duplicated = [eid for eid, n in counted.items() if n > 1]
    stats = {"chunks": len(chunks),
             "by_kind": {k: sum(c.kind == k for c in chunks) for k in ("clause", "definition", "table", "figure")},
             "elements_expected": len(expected), "elements_missing": len(missing),
             "elements_duplicated": len(duplicated), "refs": ref_stats,
             "with_latex": sum(bool(c.latex) for c in chunks),
             "with_notes": sum(bool(c.notes) for c in chunks)}
    if missing or duplicated:
        stats["missing_ids"] = missing[:10]
        stats["duplicated_ids"] = duplicated[:10]
    return chunks, stats
