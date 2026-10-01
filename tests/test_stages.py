"""Stage logic that needs no model: JSON salvage, the ambiguity filter,
candidate matching and the assembly of findings."""
import numpy as np

from codesync.models.chat import parse_json
from codesync.schema import DocumentMeta, Requirement
from codesync.stages.figures import _flag_ambiguous
from codesync.stages.judge import build_findings
from codesync.stages.match import candidate_pairs


def test_parse_json_plain_fenced_and_truncated():
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}
    assert parse_json("no json here") is None
    cut = '{"requirements": [{"s": "one"}, {"s": "two"}, {"s": "thr'
    assert parse_json(cut) == {"requirements": [{"s": "one"}, {"s": "two"}]}


def _stmt(subject: str, parameter: str, value: str) -> dict:
    return {"detail": "D", "subject": subject, "parameter": parameter,
            "value": value, "statement": f"{subject} {parameter} is {value}."}


def test_ambiguous_runs_are_flagged_not_min_max_pairs():
    reqs = _flag_ambiguous([
        _stmt("ladder", "unsupported length", "4 ft"),
        _stmt("ladder", "unsupported length", "7 ft"),
        _stmt("ladder", "unsupported length", "1 ft"),
        _stmt("platform", "height", "15 in min"),
        _stmt("platform", "height", "21 in max"),
        _stmt("rung", "diameter", "3/4 in"),
        _stmt("rung", "diameter", "3/4 in"),          # exact repeat
    ])
    flagged = {(r["subject"], r["value"]) for r in reqs if r["ambiguous"]}
    assert flagged == {("ladder", "4 ft"), ("ladder", "7 ft"),
                       ("ladder", "1 ft")}
    assert sum(r["subject"] == "rung" for r in reqs) == 1


def _unit(*vals: float) -> np.ndarray:
    v = np.array(vals, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_candidate_pairs_keeps_both_directions_above_floor():
    a = np.stack([_unit(1, 0, 0), _unit(0, 1, 0)])
    b = np.stack([_unit(1, 0.1, 0), _unit(0, 0, 1), _unit(0.9, 0.4, 0)])
    pairs, sim = candidate_pairs((a, a), (b, b), top_k=1, min_sim=0.5)
    assert sim.shape == (2, 3)
    got = {(i, j) for i, j, _s in pairs}
    assert (0, 0) in got               # A0's best match
    assert (0, 2) in got               # B2's best match is A0
    assert all(j != 1 for _i, j in got)   # B1 matches nothing


def _req(doc: str, n: int, statement: str, **kw) -> Requirement:
    return Requirement(id=f"{doc}:r{n}", doc_id=doc, element_id=f"{doc}:e{n}",
                       ref=f"{n}", page=1, statement=statement, **kw)


def test_findings_categories_merge_and_review_flag():
    meta_a = DocumentMeta(doc_id="a", name="Client", label="A",
                          filename="a.pdf", sha256="")
    meta_b = DocumentMeta(doc_id="b", name="Industry", label="B",
                          filename="b.pdf", sha256="")
    reqs_a = [_req("a", 1, "Gates shall be self-closing."),
              _req("a", 2, "Platform shall be 20 in wide."),
              _req("a", 3, "Flooring shall be galvanized.")]
    reqs_b = [_req("b", 1, "Provide a self-closing double bar safety gate."),
              _req("b", 2, "Provide a self-closing double bar safety gate "
                           "conforming to OSHA.", from_figure=True),
              _req("b", 3, "Landing platform shall be 30 in wide minimum."),
              _req("b", 4, "Rungs shall be round bar.")]
    judged = [
        {"a": "a:r1", "b": "b:r1", "similarity": 0.7, "relation": "equivalent",
         "reason": "same"},
        {"a": "a:r1", "b": "b:r2", "similarity": 0.6, "relation": "equivalent",
         "reason": "same"},
        {"a": "a:r2", "b": "b:r3", "similarity": 0.7,
         "relation": "b_stricter", "reason": "30 > 20", "topic": "width",
         "severity": "medium", "action": "Use 20 in."},
        {"a": "a:r3", "b": "b:r4", "similarity": 0.5, "relation": "unrelated",
         "reason": ""},
    ]
    findings = build_findings(meta_a, meta_b, reqs_a, reqs_b, judged, meta_a)
    by_type = {}
    for f in findings:
        by_type.setdefault(f.type, []).append(f)

    # the gate note repeated in B collapses into one finding
    (similar,) = by_type["equivalent"]
    assert similar.b.id == "b:r1" and len(similar.also_b) == 1

    # the non-governing document being stricter is a question for a person
    (stricter,) = by_type["more_stringent"]
    assert stricter.category == "conflict" and stricter.stricter == "B"
    assert stricter.needs_review and stricter.governing == "A"

    assert [f.a.id for f in by_type["only_in_a"]] == ["a:r3"]
    assert [f.b.id for f in by_type["only_in_b"]] == ["b:r4"]
    assert [f.id for f in findings] == [f"F{n:04d}"
                                        for n in range(1, len(findings) + 1)]


def test_quote_verification_handles_ellipsis_joins():
    from codesync.stages.extract import _norm, _quoted
    source = _norm("Fixed ladders shall be provided:\n  For access to "
                   "platforms not served by a stairway")
    assert _quoted("Fixed ladders shall be provided: ... For access to "
                   "platforms not served by a stairway", source)
    assert _quoted("for access to platforms", source)
    assert not _quoted("Fixed ladders shall be provided: ... on every "
                       "vessel", source)
    assert not _quoted("", source)
