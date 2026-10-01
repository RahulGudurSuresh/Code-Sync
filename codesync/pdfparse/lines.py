"""Page text -> visual rows in reading order.

A PDF text layer hands back fragments, not lines: the enumerator "a)" and
the sentence after it are separate objects that merely share a baseline.
This module rebuilds the rows a reader sees, drops running headers and
footers, and orders two-column pages column by column. OCR output is fed
through the same path so scanned pages parse identically.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import fitz


@dataclass
class Frag:
    """One text fragment as the PDF (or OCR) emitted it."""
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    size: float = 10.0
    bold: bool = False

    @property
    def yc(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass
class Row:
    """Fragments sharing a baseline, left to right."""
    page: int
    parts: list[Frag] = field(default_factory=list)
    source: str = "text-layer"

    @property
    def x0(self) -> float:
        return self.parts[0].x0

    @property
    def y0(self) -> float:
        return min(p.y0 for p in self.parts)

    @property
    def x1(self) -> float:
        return max(p.x1 for p in self.parts)

    @property
    def y1(self) -> float:
        return max(p.y1 for p in self.parts)

    @property
    def size(self) -> float:
        return max(p.size for p in self.parts)

    @property
    def bold(self) -> bool:
        return all(p.bold for p in self.parts if p.text.strip())

    @property
    def text(self) -> str:
        return _clean(" ".join(p.text for p in self.parts))

    @property
    def bbox(self) -> list[float]:
        return [self.x0, self.y0, self.x1, self.y1]


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


_REVISION_MARK = re.compile(r"^\(0\d\)$")   # "(07)": ASME edition change marker


def text_fragments(page: fitz.Page) -> list[Frag]:
    """Horizontal text-layer fragments of a page."""
    out: list[Frag] = []
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            dx, dy = line["dir"]
            if abs(dy) > 0.1 or dx <= 0:      # rotated text: drawing labels
                continue
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            text = "".join(s["text"] for s in line["spans"])
            if _REVISION_MARK.match(text.strip()):
                continue
            x0, y0, x1, y1 = line["bbox"]
            out.append(Frag(
                x0, y0, x1, y1, text,
                size=max(s["size"] for s in spans),
                bold=all(bool(s["flags"] & 16) or "Bold" in s["font"]
                         for s in spans)))
    return out


# ---- running headers / footers ---------------------------------------------
_PAGE_LABEL = re.compile(r"\bpage\s+(\d+)\s+of\s+\d+\b", re.I)


def _norm_running(text: str) -> str:
    return re.sub(r"\d+", "#", _clean(text).lower())


def find_running_text(pages: list[tuple[float, list[Frag]]]) -> set[str]:
    """Normalised texts that repeat in the top/bottom margin across pages.

    `pages` is [(page_height, fragments)]. Digits are masked so "Page 3 of
    9" and "Page 4 of 9" count as the same running line."""
    seen: Counter[str] = Counter()
    for height, frags in pages:
        keys = {_norm_running(f.text) for f in frags
                if f.y1 < 0.10 * height or f.y0 > 0.92 * height}
        seen.update(k for k in keys if k)
    need = max(2, round(0.3 * len(pages)))
    return {k for k, n in seen.items() if n >= need}


def split_running(frags: list[Frag], height: float, running: set[str],
                  width: float | None = None) -> tuple[list[Frag], list[Frag]]:
    """(body, running) fragments of one page. A landscape page bound into
    a portrait book carries its header and footer down the side margins,
    so those count as margins too when the text is a known running line."""
    body, run = [], []
    long_keys = [k for k in running if len(k) >= 20]
    for f in frags:
        in_margin = f.y1 < 0.10 * height or f.y0 > 0.92 * height
        at_side = width is not None and (f.x1 < 0.08 * width or f.x0 > 0.9 * width)
        key = _norm_running(f.text)
        known = key in running or any(key.startswith(k) for k in long_keys)
        if (in_margin and (known or _PAGE_LABEL.search(f.text))) \
                or (at_side and known):
            run.append(f)
        else:
            body.append(f)
    return body, run


def page_label(running: list[Frag]) -> str:
    for f in running:
        m = _PAGE_LABEL.search(f.text)
        if m:
            return m.group(1)
    return ""


# ---- columns and rows ------------------------------------------------------
def _column_split(frags: list[Frag], width: float) -> float | None:
    """x of the gutter if the page is two-column, else None.

    A gutter is a vertical strip near the middle that almost no fragment
    crosses while both sides carry real text."""
    if len(frags) < 16:
        return None
    step = 2.0
    lo, hi = int(0.35 * width / step), int(0.65 * width / step)
    cover = [0] * (hi - lo + 1)
    for f in frags:
        a = max(lo, int(f.x0 / step))
        b = min(hi, int(f.x1 / step))
        for i in range(a, b + 1):
            cover[i - lo] += 1
    best = min(range(len(cover)), key=cover.__getitem__)
    gutter = (best + lo) * step
    if cover[best] > 0.05 * len(frags):
        return None
    left = sum(1 for f in frags if f.x1 <= gutter)
    right = sum(1 for f in frags if f.x0 >= gutter)
    if min(left, right) < max(6, 0.2 * len(frags)):
        return None
    return gutter


def build_rows(frags: list[Frag], page: int, width: float,
               source: str = "text-layer", columns: bool = True) -> list[Row]:
    """Rows in reading order: left column top-to-bottom, then right.
    Pass `columns=False` for fragments already known to be one column (a
    single layout region)."""
    gutter = _column_split(frags, width) if columns else None
    columns = ([frags] if gutter is None else
               [[f for f in frags if (f.x0 + f.x1) / 2 < gutter],
                [f for f in frags if (f.x0 + f.x1) / 2 >= gutter]])
    rows: list[Row] = []
    for col in columns:
        col_rows: list[Row] = []
        for f in sorted(col, key=lambda f: (f.yc, f.x0)):
            for r in reversed(col_rows[-3:]):
                h = min(f.y1 - f.y0, r.y1 - r.y0)
                ryc = (r.y0 + r.y1) / 2
                if abs(f.yc - ryc) < 0.5 * max(h, 4.0):
                    r.parts.append(f)
                    break
            else:
                col_rows.append(Row(page=page, parts=[f], source=source))
        for r in col_rows:
            r.parts.sort(key=lambda p: p.x0)
        col_rows.sort(key=lambda r: r.y0)
        rows.extend(col_rows)
    return rows


_TOC_LEADER = re.compile(r"\.{5,}\s*\d+")


def is_toc_row(text: str) -> bool:
    return bool(_TOC_LEADER.search(text))


def is_toc_page(frags: list[Frag]) -> bool:
    """A contents page: several dot-leader lines ("1.1 Purpose ...... 2")."""
    return sum(1 for f in frags if _TOC_LEADER.search(f.text)) >= 3
