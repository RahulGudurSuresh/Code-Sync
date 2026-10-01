"""The parser against the two real test codes (text pages only, so no
OCR or vision model is needed and the tests stay fast)."""
from pathlib import Path

import pytest

from codesync.pdfparse import parse_pdf
from codesync.stages.extract import build_blocks

CODES = Path(__file__).resolve().parent.parent / "codes" / "test_codes"
BP = CODES / "WBU-GP-04-20-1_platform_xtract.pdf"
PIP = CODES / "PIP STF05501 Fixed Ladders and Cages.pdf"

pytestmark = pytest.mark.skipif(not BP.exists() or not PIP.exists(),
                                reason="test code PDFs not present")


@pytest.fixture(scope="module")
def bp(tmp_path_factory):
    return parse_pdf(BP, "bp", tmp_path_factory.mktemp("bp"))


def test_bp_clause_tree(bp):
    by_ref = {e.ref: e for e in bp.elements}
    assert len(bp.elements) == 103
    assert [p.label for p in bp.pages] == ["55", "56", "57", "58", "59"]

    leaf = by_ref["11.6 a.3.c.ii"]
    assert leaf.text == "40 in. for passageway"
    assert leaf.heading_trail == [
        "Platforms, Stairways, Steel Flooring, and Ladders", "Platforms",
        "Dimensions"]
    assert by_ref[leaf.ref].page == 2

    # a clause that continues onto the next page stays one element
    assert by_ref["11.6 a.7.c"].page == 3
    assert by_ref["11.6 c.12"].text == "Ladders should be side-step design."
    assert by_ref["11.6 a.6"].is_heading and not by_ref["11.6 c.6"].is_heading


def test_bp_running_headers_are_not_clauses(bp):
    text = " ".join(e.text for e in bp.elements)
    assert "February 2020" not in text and "Page 55 of 59" not in text


def test_bp_blocks_carry_lead_in_context(bp):
    blocks = build_blocks(bp)
    block = next(b for b in blocks if "[11.6 a.3.c.ii]" in b.text)
    assert block.section.endswith("Platforms > Dimensions")
    # the list item travels with the sentence that gives it meaning
    assert "[11.6 a.3.c] Minimum horizontal clearance" in block.text


def test_pip_text_pages(tmp_path):
    pip = parse_pdf(PIP, "pip", tmp_path, pages=range(0, 5))
    assert [p.kind for p in pip.pages] == ["text", "text", "toc", "text",
                                           "text"]
    body = {e.ref: e for e in pip.elements if not e.extra.get("front_matter")}
    assert body["3"].is_heading and body["3"].text == "Design"
    assert body["3.8.1"].parent == body["3.8"].id
    assert body["3.2"].text == ("All structural material shall be ASTM A36 "
                                "steel unless otherwise noted.")
    # cover page and notices are kept out of the requirement flow
    assert any(e.extra.get("front_matter") for e in pip.elements)
    assert not any("Table of Contents" in e.text for e in pip.elements)
