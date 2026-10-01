"""PDF -> ParsedDocument (clauses, tables, figures, equations).

Pages are routed by what they contain:

  text      a usable text layer; clauses come straight from it
  drawing   vector-heavy with no text (CAD sheets drawn as strokes); the
            whole sheet becomes a figure for the vision model
  scanned   an image with no text layer; OCR supplies the rows

Layout detection runs only on pages that may hold tables, figures or
display equations — it costs seconds per page on CPU, and a plain text page
has nothing for it to find.
"""
from __future__ import annotations

import json
import math
import re
import statistics
from collections.abc import Callable
from pathlib import Path

import fitz

from .. import config
from ..schema import Element, PageInfo, ParsedDocument
from . import vision
from .lines import (_REVISION_MARK, Frag, Row, build_rows, find_running_text,
                    is_toc_page, page_label, split_running, text_fragments)
from .numbering import ClauseTree, Node, node_ref

_SHEET = re.compile(r"PAGE\s*(\d+)\s*OF\s*(\d+)", re.I)
FIGURE_LABELS = {"image", "chart"}
TEXT_LABELS = {"text", "paragraph_title", "content", "abstract", "reference",
               "reference_content", "doc_title", "footnote", "aside_text"}
RUNNING_LABELS = {"header", "footer", "number", "header_image",
                  "footer_image", "seal"}


def _inside(f: Frag, bbox: list[float], pad: float = 2.0) -> bool:
    cx, cy = (f.x0 + f.x1) / 2, (f.y0 + f.y1) / 2
    return (bbox[0] - pad <= cx <= bbox[2] + pad
            and bbox[1] - pad <= cy <= bbox[3] + pad)


def _image_cover(page: fitz.Page) -> float:
    """Largest share of the page covered by a single raster image."""
    area = page.rect.width * page.rect.height
    best = 0.0
    for info in page.get_image_info():
        r = fitz.Rect(info["bbox"]) & page.rect
        best = max(best, r.width * r.height / area)
    return best


def _tiles(rect: fitz.Rect) -> list[fitz.Rect]:
    """Split a region so each tile fits the vision token budget at
    FIGURE_DPI. A Letter sheet is one tile; an A1 drawing becomes a grid."""
    budget = config.VLM_IMG_TOKENS * config.PX_PER_TOKEN
    k = config.FIGURE_DPI / 72.0
    px = rect.width * k * rect.height * k
    if px <= budget:
        return [rect]
    n = math.ceil(px / budget)
    cols = math.ceil(math.sqrt(n * rect.width / rect.height))
    rows = math.ceil(n / cols)
    ow, oh = 0.08 * rect.width / cols, 0.08 * rect.height / rows
    w, h = rect.width / cols, rect.height / rows
    return [fitz.Rect(rect.x0 + c * w - ow, rect.y0 + r * h - oh,
                      rect.x0 + (c + 1) * w + ow, rect.y0 + (r + 1) * h + oh)
            & rect for r in range(rows) for c in range(cols)]


def _save_crop(page: fitz.Page, rect: fitz.Rect, path: Path) -> None:
    page.get_pixmap(dpi=config.FIGURE_DPI, clip=rect, alpha=False).save(path)


class _Builder:
    def __init__(self, doc: fitz.Document, doc_id: str, fig_dir: Path):
        self.doc, self.doc_id, self.fig_dir = doc, doc_id, fig_dir
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.n_fig = 0
        # model outputs that only depend on the page image: OCR lines,
        # table HTML, formula LaTeX. A re-parse after a parser change then
        # costs seconds, not the minutes the models take.
        self.cache_dir = fig_dir.parent / "ocr"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def cached(self, kind: str, page_no: int, bbox: list[float] | None,
               compute: Callable[[], object]) -> object:
        key = f"{kind}_p{page_no}" + (
            "_" + "_".join(str(round(v)) for v in bbox) if bbox else "")
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        value = compute()
        path.write_text(json.dumps(value), encoding="utf-8")
        return value

    # ---- figures -----------------------------------------------------------
    def figure_payloads(self, page: fitz.Page, rect: fitz.Rect, caption: str,
                        labels: list[str], whole_sheet: bool) -> list[dict]:
        tiles = _tiles(rect)
        out = []
        for i, tile in enumerate(tiles):
            self.n_fig += 1
            name = f"fig{self.n_fig:03d}_p{page.number + 1}.png"
            _save_crop(page, tile, self.fig_dir / name)
            kind = "Drawing" if whole_sheet else "Figure"
            ref = _captioned_ref(caption) or f"{kind} p.{page.number + 1}"
            # OCR may split the title block ("PAGE 5" / "OF9"): search the
            # labels joined together
            m = _SHEET.search(" ".join(labels))
            sheet = f"{m.group(1)} of {m.group(2)}" if m else ""
            if whole_sheet and sheet:
                ref += f" (sheet {sheet})"
            if len(tiles) > 1:
                ref += f" tile {i + 1}/{len(tiles)}"
            out.append({"kind": "figure", "page": page.number + 1,
                        "bbox": list(tile), "image": name, "ref": ref,
                        "text": caption,
                        "extra": {"labels": labels, "sheet": sheet,
                                  "whole_sheet": whole_sheet}})
        return out

    def drawing_sheet(self, page: fitz.Page, frags: list[Frag]) -> list[dict]:
        """A full-page drawing. Its lettering is strokes, not text, so OCR
        recovers the labels: they make the sheet searchable and give the
        vision model a list of strings to anchor its reading on."""
        if sum(len(f.text) for f in frags) >= config.MIN_TEXT_CHARS:
            labels = [f.text.strip() for f in frags]
        else:
            labels = [t for _, t, s in vision.ocr_page(page, config.OCR_DPI)
                      if s >= 0.6]
        return self.figure_payloads(page, page.rect, "", labels, True)

    # ---- tables ------------------------------------------------------------
    def table_payload(self, page: fitz.Page, bbox: list[float], caption: str,
                      frags: list[Frag]) -> dict:
        rect = fitz.Rect(bbox)
        text, how = "", ""
        try:
            found = page.find_tables(clip=rect + (-3, -3, 3, 3))
            if found.tables and any(c for row in found.tables[0].extract()
                                    for c in row):
                text, how = found.tables[0].to_markdown(), "pdf-vector"
        except Exception:                                # noqa: BLE001
            pass
        if not text:
            try:
                text = self.cached("table", page.number + 1, bbox,
                                   lambda: vision.table_html(page, bbox))
                how = "table-ocr"
            except Exception as e:                       # noqa: BLE001
                how = f"failed: {type(e).__name__}"
        if not text:                                     # last resort
            rows = build_rows([f for f in frags if _inside(f, bbox)],
                              page.number + 1, page.rect.width)
            text, how = "\n".join(r.text for r in rows), how or "plain-rows"
        self.n_fig += 1
        name = f"tab{self.n_fig:03d}_p{page.number + 1}.png"
        _save_crop(page, rect + (-4, -4, 4, 4) & page.rect,
                   self.fig_dir / name)
        return {"kind": "table", "page": page.number + 1, "bbox": bbox,
                "image": name,
                "ref": _captioned_ref(caption) or f"Table p.{page.number + 1}",
                "text": text, "extra": {"caption": caption, "method": how}}


# "Table 104.1.2(A) Values of y" -> "Table 104.1.2(A)"; "Fig. 104.8.4 ..."
_CAPTION_REF = re.compile(r"^\s*((?:Table|Fig(?:ure|s)?\.?)\s+[A-Z]?-?\d[\w.()-]*)",
                          re.I)


def _captioned_ref(caption: str) -> str:
    m = _CAPTION_REF.match(caption or "")
    return m.group(1).strip().rstrip(".") if m else ""


def _caption_for(region, titles, frags) -> str:
    """Text of the nearest figure/table title region that overlaps in x."""
    best, best_d = None, 40.0
    for t in titles:
        if min(t.bbox[2], region.bbox[2]) - max(t.bbox[0], region.bbox[0]) <= 0:
            continue
        d = min(abs(t.bbox[1] - region.bbox[3]), abs(region.bbox[1] - t.bbox[3]))
        if d < best_d:
            best, best_d = t, d
    if best is None:
        return ""
    return " ".join(f.text.strip() for f in frags if _inside(f, best.bbox))


def _page_number(regions: list, frags: list[Frag]) -> str:
    """The printed page number, from the layout's "number" region."""
    for g in regions:
        if g.label == "number":
            text = "".join(f.text for f in frags if _inside(f, g.bbox)).strip()
            if text.isdigit():
                return text
    return ""


_NOISY = re.compile(r"[\\~|{}^]|[A-Za-z][0-9][A-Za-z]|(?<![0-9.])[0-9](?=[a-z]{2})"
                    r"|VV|\bl\b|rn(?=[a-z]*ent\b)")


def text_noise(text: str) -> float:
    """Share of tokens that look like OCR damage ("VVeld", "me8sured",
    "\\vall"). Used to decide whether a page's embedded text layer is worth
    keeping or should be re-read with the OCR model."""
    tokens = text.split()
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if _NOISY.search(t)) / len(tokens)


def _ordered_by_layout(page: fitz.Page, regions: list, frags: list[Frag],
                       b: _Builder, size: float, source: str) -> list[object]:
    """Rows, tables, figures and equations of one page in reading order.

    The layout model returns its regions in reading order (full-width
    blocks, then the left column, then the right), which is more reliable
    on scans than inferring columns from fragment positions. Each text
    region becomes rows; fragments the model left outside every region are
    slotted in by position afterwards."""
    page_no = page.number + 1
    area = page.rect.width * page.rect.height
    frags = [f for f in frags if not any(
        _inside(f, g.bbox) for g in regions if g.label in RUNNING_LABELS)]
    titles = [g for g in regions if g.label == "figure_title"]
    numbers = [g for g in regions if g.label == "formula_number"]
    used: set[int] = set()

    def take(bbox: list[float]) -> list[Frag]:
        members = [f for f in frags if id(f) not in used and _inside(f, bbox)]
        used.update(id(f) for f in members)
        return members

    seq: list[tuple[list[float], object]] = []
    for g in regions:
        if g.label in RUNNING_LABELS or g.label == "formula_number":
            continue
        if g.label == "figure_title":
            take(g.bbox)               # travels with its figure or table
        elif g.label in FIGURE_LABELS:
            if g.area <= 0.02 * area:
                continue
            labels = [f.text.strip() for f in take(g.bbox)]
            for p in b.figure_payloads(page, fitz.Rect(g.bbox) & page.rect,
                                       _caption_for(g, titles, frags),
                                       labels, False):
                seq.append((g.bbox, p))
        elif g.label == "table" and g.score >= 0.5:
            payload = b.table_payload(page, g.bbox,
                                      _caption_for(g, titles, frags), frags)
            take(g.bbox)
            seq.append((g.bbox, payload))
        elif g.label == "display_formula" and g.score >= 0.5:
            try:
                latex = b.cached("formula", page_no, g.bbox,
                                 lambda: vision.formula_latex(page, g.bbox))
            except Exception:                            # noqa: BLE001
                latex = ""
            if not latex:
                continue               # keep whatever text the layer has
            mid = (g.bbox[1] + g.bbox[3]) / 2
            # its number: the nearest one to the right at the same height
            near = sorted((n for n in numbers
                           if abs((n.bbox[1] + n.bbox[3]) / 2 - mid) < 12
                           and n.bbox[0] >= g.bbox[0]),
                          key=lambda n: n.bbox[0] - g.bbox[2])
            num = _eq_label(" ".join(f.text for f in take(near[0].bbox))) \
                if near else ""
            take(g.bbox)
            head = f"EQUATION {num}".strip()
            eq = Frag(g.bbox[0], g.bbox[1], g.bbox[2], g.bbox[3],
                      f"[{head}: $${latex}$$]", size=size)
            seq.append((g.bbox, Row(page=page_no, parts=[eq],
                                    source="formula")))
        else:
            src = "footnote" if g.label == "footnote" else source
            for row in build_rows(take(g.bbox), page_no, page.rect.width,
                                  src, columns=False):
                seq.append((row.bbox, row))

    width = page.rect.width

    def column(bx: list[float]) -> int:
        """-1 left, 1 right, 0 spans the page (or sits on its centre)."""
        c = (bx[0] + bx[2]) / 2
        if bx[2] - bx[0] > 0.55 * width:
            return 0
        if c < 0.4 * width:
            return -1
        if c > 0.6 * width:
            return 1
        return 0

    orphans = [f for f in frags if id(f) not in used]
    for row in build_rows(orphans, page_no, width, source):
        at = 0                         # after the last item above it ...
        for k, (bx, _obj) in enumerate(seq):
            same = column(bx) == 0 or column(row.bbox) == 0 \
                or column(bx) == column(row.bbox)        # ... in its column
            if same and bx[1] <= row.y0 + 2:
                at = k + 1
        seq.insert(at, (row.bbox, row))

    # Two columns: read each band between full-width items left column
    # first, top to bottom, then the right column. The model's own order
    # is usually that, but not always (a short last paragraph of the left
    # column can come after the right column's first).
    cols = [column(bx) for bx, _obj in seq]
    if cols.count(-1) >= 3 and cols.count(1) >= 3:
        ordered: list[tuple[list[float], object]] = []
        band: list[tuple[list[float], object]] = []

        def flush() -> None:
            band.sort(key=lambda it: (column(it[0]), it[0][1]))
            ordered.extend(band)
            band.clear()
        for item in seq:
            if column(item[0]) == 0:
                flush()
                ordered.append(item)
            else:
                band.append(item)
        flush()
        seq = ordered
    return [obj for _bx, obj in seq]


_EQ_NUM = re.compile(r"\(\s*(\d{1,3}\s*[A-Za-z]?)\s*\)")


def _eq_label(text: str) -> str:
    """"(3)" from the printed number, which OCR may read as "(3)3" (a
    footnote mark) or "(4)^3}$"."""
    m = _EQ_NUM.search(text)
    if m:
        return "(" + re.sub(r"\s+", "", m.group(1)).upper() + ")"
    return re.sub(r"[^\w().-]+", "", text)


def _left_edge(frags: list[Frag], width: float, right: bool = False
               ) -> float | None:
    """Left edge of the text block: where the full lines in the left half
    of the page start. With `right`, the edge of a right-hand column: where
    at least five full lines start in the right half."""
    xs = sorted(f.x0 for f in frags if len(f.text.strip()) > 20
                and ((f.x0 >= 0.5 * width) if right else (f.x0 < 0.5 * width)))
    if right:
        return xs[0] if len(xs) >= 5 else None
    return xs[0] if xs else None


def _mirrored_offsets(raw: dict[int, dict], todo: list[int]) -> None:
    """A bound book shifts the text block left and right on alternate
    pages (the inner margin is wider). The running header, which is what
    the offset normally follows, sits at the page centre and does not move
    with the block, so clause indents would jump between pages. When the
    left edge alternates with period two, the edge itself is the offset."""
    pages = [i for i in todo if raw[i].get("left") is not None]
    if len(pages) < 4:
        return
    lefts = [raw[i]["left"] for i in pages]
    step1 = statistics.median(abs(a - b) for a, b in zip(lefts, lefts[1:]))
    step2 = statistics.median(abs(a - b) for a, b in zip(lefts, lefts[2:]))
    if step1 > 8 and step2 < 4:
        for i in pages:
            raw[i]["offset"] = raw[i]["left"]


def _ocr_cached(doc: fitz.Document, i: int, cache_dir: Path) -> list:
    """OCR lines of page `i`, cached on disk (OCR is 10-25 s per page)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"page{i + 1}_{config.OCR_DPI}dpi.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    lines = vision.ocr_page(doc[i], config.OCR_DPI)
    path.write_text(json.dumps(lines), encoding="utf-8")
    return lines


def _layout_cached(b: "_Builder", page: fitz.Page) -> list:
    """Layout regions, cached like the other model outputs."""
    raw = b.cached("layout", page.number + 1, None, lambda: [
        {"label": g.label, "bbox": list(g.bbox), "score": g.score}
        for g in vision.detect_layout(page, config.LAYOUT_DPI)])
    return [vision.Region(**g) for g in raw]


def parse_pdf(pdf_path: Path, doc_id: str, fig_dir: Path,
              pages: range | None = None,
              progress: Callable[[str], None] | None = None) -> ParsedDocument:
    say = progress or (lambda _m: None)
    doc = fitz.open(pdf_path)
    todo = list(pages) if pages is not None else list(range(len(doc)))
    b = _Builder(doc, doc_id, fig_dir)

    # ---- pass 1: text layer, page kinds, running headers -------------------
    raw: dict[int, dict] = {}
    for i in todo:
        page = doc[i]
        frags = text_fragments(page)
        chars = sum(len(f.text.strip()) for f in frags)
        n_draw = len(page.get_drawings())
        cover = _image_cover(page)
        if is_toc_page(frags):
            kind = "toc"
        elif chars >= config.MIN_TEXT_CHARS:
            kind = "text"
        elif n_draw >= config.VECTOR_HEAVY:
            kind = "drawing"
        elif cover > 0.5:
            kind = "scanned"
        else:
            kind = "text" if chars else "empty"
        raw[i] = {"frags": frags, "kind": kind, "n_draw": n_draw,
                  "cover": cover}

    # OCR scanned pages now so they take part in header/footer detection.
    # A scan that carries a noisy embedded text layer (an archive.org OCR
    # pass: "VVeld", "me8sured") is re-read too; the fresh OCR is cleaner.
    for i in todo:
        r = raw[i]
        noisy = (r["kind"] == "text" and r["cover"] > 0.5 and config.REOCR
                 and text_noise(" ".join(f.text for f in r["frags"]))
                 > config.REOCR_NOISE)
        if r["kind"] == "scanned" or noisy:
            say(f"ocr page {i + 1}" + (" (noisy text layer)" if noisy else ""))
            raw[i]["frags"] = [
                Frag(bx[0], bx[1], bx[2], bx[3], text,
                     size=0.75 * (bx[3] - bx[1]))
                for bx, text, score in _ocr_cached(doc, i, fig_dir.parent / "ocr")
                if score >= 0.5 and not _REVISION_MARK.match(text.strip())]
            raw[i]["source"] = "ocr"

    running = find_running_text(
        [(doc[i].rect.height, raw[i]["frags"]) for i in todo
         if raw[i]["kind"] in ("text", "scanned")])

    sizes, pitches = [], []
    body: dict[int, list[Frag]] = {}
    infos: list[PageInfo] = []
    header_text: list[str] = []
    for i in todo:
        page, r = doc[i], raw[i]
        bd, run = split_running(r["frags"], page.rect.height, running,
                                page.rect.width)
        body[i], r["offset"] = bd, min(
            (f.x0 for f in (run or bd)), default=0.0)
        r["run"] = run
        r["left"] = _left_edge(bd, page.rect.width)
        r["right"] = _left_edge(bd, page.rect.width, right=True)
    _mirrored_offsets(raw, todo)
    for i in todo:
        page, r = doc[i], raw[i]
        bd, run = body[i], r["run"]
        # two columns: a row in the right column measures its indent from
        # that column's edge, on the same scale as the left column
        if r["right"] is not None and r["left"] is not None:
            r["col2"] = (0.5 * page.rect.width,
                         r["right"] - (r["left"] - r["offset"]))
        infos.append(PageInfo(page=i + 1, kind=r["kind"],
                              label=page_label(run),
                              source=r.get("source", "text-layer"),
                              width=page.rect.width, height=page.rect.height))
        for f in run:
            t = f.text.strip()
            if f.y1 < 0.10 * page.rect.height and t not in header_text \
                    and any(c.isalpha() for c in t):
                header_text.append(t)
        if r["kind"] in ("text", "scanned"):
            rows = build_rows(bd, i + 1, page.rect.width)
            sizes += [row.size for row in rows]
            for a, c in zip(rows, rows[1:]):
                d = c.y0 - a.y0
                if 0.8 * a.size < d < 1.7 * a.size:
                    pitches.append(d)
    size = statistics.median(sizes) if sizes else 10.0
    pitch = statistics.median(pitches) if pitches else 1.18 * size
    tree = ClauseTree(pitch=pitch, size=size)

    by_page = {info.page: info for info in infos}

    def sheet(page: fitz.Page, frags: list[Frag]) -> None:
        for p in b.drawing_sheet(page, frags):
            tree.add_payload(p, top_level=True)
            info = by_page[page.number + 1]
            info.kind = "drawing"
            info.label = info.label or p["extra"]["sheet"].split(" of ")[0]

    # ---- pass 2: regions and the clause tree -------------------------------
    for i in todo:
        page, r = doc[i], raw[i]
        if r["kind"] in ("empty", "toc"):
            continue
        if r["kind"] == "drawing":
            say(f"drawing sheet page {i + 1}")
            sheet(page, r["frags"])
            continue

        frags = body[i]
        source = r.get("source", "text-layer")
        needs_layout = (r["kind"] == "scanned" or r["n_draw"] >= 30
                        or r["cover"] > 0.05)
        if not needs_layout:
            ordered: list[object] = list(
                build_rows(frags, i + 1, page.rect.width, source))
        else:
            say(f"layout page {i + 1}")
            regions = _layout_cached(b, page)
            info = by_page[i + 1]
            info.label = info.label or _page_number(regions, r["frags"])
            area = page.rect.width * page.rect.height
            text_area = sum(g.area for g in regions if g.label in TEXT_LABELS)
            fig_area = sum(g.area for g in regions if g.label in FIGURE_LABELS
                           and g.area > 0.02 * area)
            if fig_area > 0.35 * area and fig_area > 2 * text_area:
                # a drawing sheet that happens to carry a text layer or
                # came in as a scan
                sheet(page, r["frags"])
                continue
            ordered = _ordered_by_layout(page, regions, frags, b, size,
                                         source)
        for obj in ordered:
            if isinstance(obj, Row):
                tree.add_row(obj, r["offset"], glue=obj.source == "formula",
                             col2=r.get("col2"))
            else:
                tree.add_payload(obj)

    root = tree.finish()
    elements = _emit(root, doc_id)
    return ParsedDocument(doc_id=doc_id, pages=infos, elements=elements,
                          header_text=header_text)


def _emit(root: Node, doc_id: str) -> list[Element]:
    out: list[Element] = []
    seen: dict[str, int] = {}

    def unique(ref: str) -> str:
        seen[ref] = seen.get(ref, 0) + 1
        return ref if seen[ref] == 1 else f"{ref} #{seen[ref]}"

    numbered_seen = False

    def walk(node: Node, chain: list[Node], parent_id: str | None,
             trail: list[str], front: bool = False) -> None:
        nonlocal numbered_seen
        for child in node.children:
            eid = f"{doc_id}:e{len(out) + 1}"
            if node is root:
                # cover page, copyright notice, revision history: anything
                # before the first numbered clause
                numbered_seen = numbered_seen or bool(child.enum)
                front = not numbered_seen and not child.payload
            if child.payload:
                p = child.payload
                out.append(Element(
                    id=eid, kind=p["kind"], page=p["page"], bbox=p["bbox"],
                    ref=unique(p["ref"]), parent=parent_id, depth=len(chain),
                    text=p["text"], heading_trail=trail, image=p["image"],
                    source="text-layer", extra=p["extra"]))
                continue
            first = child.rows[0]
            same_page = [r for r in child.rows if r.page == first.page]
            bbox = [min(r.x0 for r in same_page), min(r.y0 for r in same_page),
                    max(r.x1 for r in same_page), max(r.y1 for r in same_page)]
            nchain = chain + [child]
            src = "ocr" if first.source == "ocr" else "text-layer"
            extra: dict = {"front_matter": True} if front else {}
            if child.note:
                extra["note"] = True
            out.append(Element(
                id=eid, kind="clause", page=first.page, bbox=bbox,
                number=child.enum.raw if child.enum else "",
                ref=unique(node_ref(nchain, first.page)), parent=parent_id,
                depth=len(chain), text=child.text,
                is_heading=child.is_heading, heading_trail=trail, source=src,
                extra=extra))
            walk(child, nchain, eid,
                 trail + [child.text] if child.is_heading else trail, front)

    walk(root, [], None, [])
    return out
