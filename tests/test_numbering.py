"""Clause-tree construction from rows, without touching a PDF."""
from codesync.pdfparse.lines import Frag, Row
from codesync.pdfparse.numbering import (ClauseTree, detect_enum, node_ref)


def _row(y: float, *parts: tuple[float, str], page: int = 1,
         bold: bool = False) -> Row:
    frags = [Frag(x, y, x + 6 * len(t), y + 10, t, size=10, bold=bold)
             for x, t in parts]
    return Row(page=page, parts=frags)


def _refs(tree: ClauseTree) -> dict[str, str]:
    """{citation: text} for every node, in document order."""
    out: dict[str, str] = {}

    def walk(node, chain):
        for child in node.children:
            if child.payload:              # tables/figures have no clause ref
                continue
            out[node_ref(chain + [child], 1)] = child.text
            walk(child, chain + [child])
    walk(tree.finish(), [])
    return out


def test_detect_enum_forms():
    assert detect_enum("3.8.1").dec == (3, 8, 1)
    assert detect_enum("a)").interps == [("alpha-lower", 1)]
    assert ("roman-lower", 1) in detect_enum("i)").interps
    assert ("alpha-lower", 9) in detect_enum("i)").interps
    assert detect_enum("(A.1)").path == ("A", 1)
    assert detect_enum("(Al.l)").path == ("A", 1, 1)   # OCR: 1 read as l
    assert detect_enum("in.") is None                  # a unit, not a list
    assert detect_enum("No.") is None


def test_nested_list_like_bp():
    """11.6 > a. > 1. > a) > i), the BP layout: enumerators are their own
    fragments and each level is indented."""
    t = ClauseTree(pitch=11.6, size=10)
    rows = [
        _row(72, (54, "11.6"), (108, "Platforms and Ladders")),
        _row(95, (108, "a."), (135, "Platforms")),
        _row(118, (135, "1."), (162, "Requirements")),
        _row(141, (162, "a)"), (189, "A platform shall be provided.")),
        _row(153, (162, "b)"), (189, "Platforms should be provided for:")),
        _row(165, (189, "i)"), (216, "Elevated manways")),
        _row(177, (189, "ii)"), (216, "Instrument locations")),
        _row(200, (135, "2."), (162, "Dimensions")),
        _row(223, (162, "a)"), (189, "Walkways shall be clear.")),
        _row(246, (108, "b."), (135, "Steel Flooring")),
    ]
    for r in rows:
        t.add_row(r, offset=54)
    refs = _refs(t)
    assert refs["11.6 a.1.b.ii"] == "Instrument locations"
    assert refs["11.6 a.2.a"] == "Walkways shall be clear."
    assert "11.6 b" in refs


def test_letter_i_continues_alpha_sequence():
    """"i)" after "h)" is the ninth letter, not roman one."""
    t = ClauseTree(pitch=12, size=10)
    y = 0
    for letter in "abcdefghij":
        y += 12
        t.add_row(_row(y, (60, f"{letter})"), (90, f"item {letter}")), 0)
    refs = _refs(t)
    assert set(refs) == set("abcdefghij")


def test_wrapped_line_continues_clause():
    t = ClauseTree(pitch=12, size=10)
    t.add_row(_row(10, (60, "a)"), (90, "Toe plates shall be along")), 0)
    t.add_row(_row(22, (90, "all edges.")), 0)
    assert _refs(t) == {"a": "Toe plates shall be along all edges."}


def test_decimal_children_attach_by_number():
    """PIP layout: "3." heading, then 3.1 ... 3.8.1 found by their prefix."""
    t = ClauseTree(pitch=13, size=11)
    t.add_row(_row(10, (72, "3."), (108, "Design"), bold=True), 72)
    t.add_row(_row(30, (108, "3.8"), (144, "Assemblies shall be coated.")), 72)
    t.add_row(_row(50, (157, "3.8.1"), (190, "Galvanizing shall follow.")), 72)
    t.add_row(_row(70, (108, "3.9"), (144, "Erection shall comply.")), 72)
    refs = _refs(t)
    assert list(refs) == ["3", "3.8", "3.8.1", "3.9"]


def test_inline_decimal_needs_a_sequence():
    """A wrapped line that happens to start with a number is text."""
    t = ClauseTree(pitch=13, size=11)
    t.add_row(_row(10, (72, "3."), (108, "Design"), bold=True), 0)
    t.add_row(_row(30, (108, "3.6"), (144, "Connections are designed for")), 0)
    t.add_row(_row(43, (144, "4.4 kN on one side rail.")), 0)
    refs = _refs(t)
    assert refs["3.6"] == "Connections are designed for 4.4 kN on one side rail."


def test_asme_sections_and_subparagraphs():
    """ASME prints every level flush left, with the number in the same run
    of text as the paragraph."""
    t = ClauseTree(pitch=11, size=9)
    lines = [
        "104 PRESSURE DESIGN OF COMPONENTS",
        "104.1 Straight Pipe",
        "104.1.2 Straight Pipe Under Internal Pressure",
        "(A) Minimum Wall Thickness. The minimum thickness",
        "of pipe wall shall not be less than required.",
        "(A.1) If pipe is ordered by nominal wall thickness,",
        "the tolerance shall be taken into account.",
        "104.1.3 Straight Pipe Under External Pressure. See UG-28.",
    ]
    y = 0
    for text in lines:
        y += 11
        indent = 60 if not text[0].islower() else 50   # first-line indent
        t.add_row(_row(y, (indent, text)), 0)
    refs = _refs(t)
    assert "104.1.2 A" in refs and refs["104.1.2 A"].endswith("required.")
    assert "104.1.2 A.1" in refs
    assert "104.1.3" in refs


def test_titled_paragraph_is_a_heading_without_its_section():
    """An extract may start at "100.2 Definitions" with no "100 GENERAL"
    before it; the Title Case title makes it a heading anyway."""
    t = ClauseTree(pitch=11, size=9)
    t.add_row(_row(11, (60, "100.2 Definitions")), 0)
    t.add_row(_row(22, (60, "arc welding: a group of welding processes.")), 0)
    t.add_row(_row(33, (60, "wherein coalescence is produced by heat.")), 0)
    t.add_row(_row(44, (60, "base metal: the metal to be welded.")), 0)
    t.add_row(_row(55, (60, "The ratio is 2:1 in all cases.")), 0)
    root = t.finish()
    (defs,) = root.children
    assert defs.is_heading and defs.text == "Definitions"
    texts = [c.text for c in defs.children]
    # each term starts its own paragraph; wrapped lines and ordinary
    # sentences containing a colon do not
    assert texts == ["arc welding: a group of welding processes. wherein "
                     "coalescence is produced by heat.",
                     "base metal: the metal to be welded. The ratio is 2:1 "
                     "in all cases."]


def test_equation_joins_the_clause_that_introduced_it():
    """A table printed between a clause ending "as follows:" and its
    equation must not capture the equation (nor must the table's notes)."""
    t = ClauseTree(pitch=11, size=9)
    t.add_row(_row(11, (60, "(A) The thickness shall be determined as follows:")), 0)
    t.add_payload({"kind": "table", "text": "<table></table>"})
    t.add_row(_row(300, (60, "[1] Radiography shall be performed.")), 0)
    t.add_row(_row(330, (60, "[EQUATION (3): $$t_m = x$$]")), 0, glue=True)
    t.add_row(_row(345, (60, "where")), 0)
    t.add_row(_row(356, (60, "t_m = minimum required thickness")), 0)
    texts = list(_refs(t).values())
    clause = next(x for x in texts if "as follows:" in x)
    assert "[EQUATION (3)" in clause
    assert "where t_m = minimum required thickness" in clause
    note = next(x for x in texts if x.startswith("[1]"))
    assert "EQUATION" not in note


def test_heading_needs_to_be_a_title():
    t = ClauseTree(pitch=12, size=10)
    t.add_row(_row(10, (60, "6."), (90, "Gates")), 0)
    t.add_row(_row(22, (90, "a)"), (120, "Gates shall be self-closing.")), 0)
    t.add_row(_row(40, (60, "7."), (90, "Plan dimension shall not exceed 1 in. for")), 0)
    t.add_row(_row(52, (90, "a)"), (120, "Joints")), 0)
    root = t.finish()
    gates, plan = root.children
    assert gates.is_heading
    assert not plan.is_heading        # it obliges, so it is a requirement
