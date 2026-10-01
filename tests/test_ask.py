"""Ask the Code: chunking, the store, fusion and the citation check — all
without a PDF, a GPU or the network."""
import numpy as np
import pytest

from codesync import config, store
from codesync.schema import DocumentMeta, Element, PageInfo, ParsedDocument
from codesync.stages.answer import check_citations
from codesync.stages.chunk import Chunk, build_chunks, find_refs, table_to_grid
from codesync.stages.retrieve import _rrf

META = DocumentMeta(doc_id="d1", name="ASME B31.1 extract", label="B31.1",
                    filename="x.pdf", sha256="0", code="ASME B31.1", edition="2007")


def _el(n, kind, text, *, page=1, number="", ref="", parent=None, depth=0,
        heading=False, trail=(), extra=None):
    return Element(id=f"d1:e{n}", kind=kind, page=page, bbox=[0, 0, 1, 1],
                   number=number, ref=ref, parent=parent, depth=depth, text=text,
                   is_heading=heading, heading_trail=list(trail), extra=extra or {})


def _doc(elements):
    return ParsedDocument(doc_id="d1", elements=elements,
                          pages=[PageInfo(page=1, kind="scanned", label="7",
                                          width=612, height=792)])


def test_chunks_keep_text_verbatim_with_equation_notes_and_refs():
    els = [
        _el(1, "clause", "PRESSURE DESIGN OF COMPONENTS", ref="104", heading=True),
        _el(2, "clause", "Straight Pipe Under Internal Pressure", ref="104.1.2",
            parent="d1:e1", depth=1, heading=True, trail=["104"]),
        _el(3, "clause", "Minimum Wall Thickness. The minimum thickness shall "
            "not be less than that determined by eq. (3) as follows: "
            "[EQUATION (3): $$t_m = \\frac{PD_o}{2(SE+Py)} + A$$]",
            number="(A)", ref="104.1.2 A", parent="d1:e2", depth=2,
            trail=["104", "104.1.2"]),
        _el(4, "clause", "where", parent="d1:e2", depth=2, trail=["104", "104.1.2"]),
        _el(5, "clause", "t_m = minimum required wall thickness, see para. 102.3.2 "
            "and Table 104.1.2(A)", parent="d1:e2", depth=2, trail=["104", "104.1.2"]),
        _el(6, "table", "<table><tr><th>Material</th><th>y</th></tr>"
            "<tr><td>Ferritic steels</td><td>0.4</td></tr></table>",
            ref="Table 104.1.2(A)", parent="d1:e2", depth=2,
            extra={"caption": "Table 104.1.2(A) Values of y"}),
        _el(7, "clause", "GENERAL NOTE: Ratios of t to D are limited.",
            parent="d1:e2", depth=2, trail=["104", "104.1.2"]),
        _el(8, "clause", "Limits for Sustained Stresses", ref="102.3.2",
            heading=True),
        _el(9, "clause", "The stress shall not exceed the allowable.",
            ref="102.3.2 ¶1", parent="d1:e8", depth=1, trail=["102.3.2"]),
    ]
    chunks, stats = build_chunks(META, _doc(els))

    assert stats["elements_missing"] == 0 and stats["elements_duplicated"] == 0
    by_ref = {c.ref: c for c in chunks}
    a = by_ref["104.1.2 A"]
    # verbatim text, equation kept, "where" block folded in
    assert a.text.startswith("Minimum Wall Thickness. The minimum thickness")
    assert a.latex == ["t_m = \\frac{PD_o}{2(SE+Py)} + A"]
    assert "where\nt_m = minimum required wall thickness" in a.text
    assert a.heading_path == ["104 PRESSURE DESIGN OF COMPONENTS",
                              "104.1.2 Straight Pipe Under Internal Pressure"]
    assert a.code == "ASME B31.1" and a.edition == "2007" and a.page_label == "7"
    # cross-references are links, resolved inside the document
    kinds = {(r.kind, r.target): r.chunk_id for r in a.refs}
    assert kinds[("para", "102.3.2")] == by_ref["102.3.2 ¶1"].id
    assert kinds[("table", "104.1.2(A)")] == by_ref["Table 104.1.2(A)"].id
    # the table is data, and its note stays with it
    t = by_ref["Table 104.1.2(A)"]
    assert t.text == "Table 104.1.2(A) Values of y"
    assert t.table["columns"] == ["Material", "y"]
    assert t.table["rows"] == [["Ferritic steels", "0.4"]]
    assert t.notes == ["GENERAL NOTE: Ratios of t to D are limited."]
    # the heading path and reference are what the embedder sees first
    assert a.embed_text.startswith("104 PRESSURE DESIGN OF COMPONENTS > ")


def test_definitions_become_one_chunk_per_term():
    els = [
        _el(1, "clause", "Definitions", ref="100.2", heading=True),
        _el(2, "clause", "arc welding: a group of welding processes.",
            ref="100.2 ¶1", parent="d1:e1", depth=1, trail=["100.2 Definitions"]),
        _el(3, "clause", "base metal: the metal to be welded.",
            ref="100.2 ¶2", parent="d1:e1", depth=1, trail=["100.2 Definitions"]),
    ]
    chunks, _ = build_chunks(META, _doc(els))
    assert [c.kind for c in chunks] == ["definition", "definition"]
    assert [c.ref for c in chunks] == ["Definition: arc welding", "Definition: base metal"]
    assert chunks[0].text == "arc welding: a group of welding processes."


def test_find_refs_and_table_grid():
    refs = find_refs("see paras. 102.3.2 and 104.8.1, Table A-1, Fig. 104.3.1(D), "
                     "eqs. (3) and (3A), Mandatory Appendix A, Chapter II")
    assert {(r.kind, r.target) for r in refs} >= {
        ("para", "102.3.2"), ("para", "104.8.1"), ("table", "A-1"),
        ("fig", "104.3.1(D)"), ("eq", "3"), ("eq", "3A"), ("appendix", "A"),
        ("chapter", "II")}
    cols, rows = table_to_grid("| NPS | Wall |\n|---|---|\n| 2 | 0.154 |\n| 3 | 0.216 |")
    assert cols == ["NPS", "Wall"] and rows == [["2", "0.154"], ["3", "0.216"]]


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_DIR", tmp_path / "lancedb")
    return store


def _chunk(i, doc, text, code="ASME B31.1", edition="2007", kind="clause"):
    return Chunk(id=f"{doc}:c{i}", doc_id=doc, code=code, edition=edition, kind=kind,
                 ref=f"104.1.{i}" if kind == "clause" else f"Definition: term{i}",
                 page=1, text=text, heading_path=["104 PRESSURE DESIGN"])


def test_store_vector_keyword_and_filters(tmp_store):
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(3, config.EMBED_DIM)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    chunks = [_chunk(1, "d1", "The minimum wall thickness shall be determined by eq. (3)."),
              _chunk(2, "d1", "Radiography shall be performed on welds."),
              _chunk(3, "d1", "corrosion allowance: thickness added for corrosion.", kind="definition")]
    assert tmp_store.replace_document(chunks, vecs) == 3
    other = [_chunk(1, "d2", "Wall thickness of steel platforms.", code="PIP", edition="2016")]
    tmp_store.replace_document(other, vecs[:1])
    assert tmp_store.indexed_documents() == {"d1": 3, "d2": 1}

    hits = tmp_store.vector_search(vecs[1], k=2)
    assert hits[0][0].id == "d1:c2" and hits[0][1] == pytest.approx(1.0, abs=1e-4)
    # the same vector, scoped to the other document, cannot return d1 rows
    assert [c.doc_id for c, _ in tmp_store.vector_search(vecs[1], k=5, doc_ids=["d2"])] == ["d2"]
    assert [c.doc_id for c, _ in tmp_store.vector_search(vecs[1], k=5, edition="2016")] == ["d2"]

    kw = tmp_store.keyword_search("radiography welds", k=5, doc_ids=["d1"])
    assert kw and kw[0][0].id == "d1:c2"
    assert [c.ref for c in tmp_store.definitions(["d1"])] == ["Definition: term3"]
    assert set(tmp_store.get(["d1:c1", "d2:c1"])) == {"d1:c1", "d2:c1"}

    # re-indexing a document replaces its rows instead of adding to them
    tmp_store.replace_document(chunks[:2], vecs[:2])
    assert tmp_store.count("d1") == 2


def test_table_rows_follow_the_question():
    from codesync.stages.retrieve import table_rows_for
    rows = [[f"Material {i}", str(i)] for i in range(40)]
    rows[25] = ["A106 Grade B", "17.1"]
    rows[33] = ["A53 Grade B", "17.1"]
    table = {"columns": ["Spec", "Stress"], "rows": rows}
    shown = table_rows_for(table, "allowable stress of A106 Grade B at 500 F", limit=5)
    assert rows[25] in shown and rows[33] in shown           # both mention "grade b"
    assert shown.index(rows[25]) < shown.index(rows[33])     # table order kept
    assert len(shown) == 5
    assert table_rows_for(table, None, limit=3) == rows[:3]


def test_rrf_prefers_chunks_ranked_by_both_lists():
    a, b, c = (_chunk(i, "d1", f"text {i}") for i in (1, 2, 3))
    fused = _rrf([(a, 0.9), (b, 0.8)], [(b, 5.0), (c, 4.0)])
    assert [x.id for x in fused][0] == b.id


def test_check_citations_flags_unknown_tags_and_invented_quotes():
    passages = [{"tag": "P1", "chunk": _chunk(1, "d1", "The minimum thickness shall not be less than t_m.")}]
    good = 'Per [P1], "the minimum thickness shall not be less than t_m".'
    assert check_citations(good, passages)["ok"]
    bad = 'Per [P2], "walls may be thinner".'
    r = check_citations(bad, passages)
    assert r["bad_tags"] == ["P2"] and r["bad_quotes"] == ["walls may be thinner"] and not r["ok"]
    assert check_citations("Not found in the passages supplied.", passages)["ok"]
    # the text between two quotations is not a quotation
    two = 'It says "the minimum thickness shall not be less than t_m" [P1].\n\nAlso "the minimum thickness shall not" [P1].'
    r = check_citations(two, passages)
    assert r["quotes"] == 2 and r["bad_quotes"] == []
