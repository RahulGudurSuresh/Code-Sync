"""API surface with the pipeline stubbed out: routing, validation and the
job runner, without loading a model."""
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from codesync import config, library, pipeline

PDF = (Path(__file__).resolve().parent.parent / "codes" / "test_codes"
       / "WBU-GP-04-20-1_platform_xtract.pdf")

pytestmark = pytest.mark.skipif(not PDF.exists(),
                                reason="test code PDFs not present")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LIBRARY_DIR", tmp_path / "library")
    monkeypatch.setattr(config, "COMPARISON_DIR", tmp_path / "comparisons")
    config.LIBRARY_DIR.mkdir()
    config.COMPARISON_DIR.mkdir()
    calls = []

    def fake_ingest(doc_ids, *a, progress=None, **kw):
        calls.append(("ingest", tuple(doc_ids)))
        for d in doc_ids:
            meta = library.load_meta(d)
            meta.status = "ready"
            library.save_meta(meta)

    def fake_compare(cmp_id, *a, progress=None, **kw):
        calls.append(("compare", cmp_id))
        cmp = library.load_comparison(cmp_id)
        cmp.status = "done"
        library.save_comparison(cmp)
        library.save_findings(cmp_id, [])

    monkeypatch.setattr(pipeline, "ingest", fake_ingest)
    monkeypatch.setattr(pipeline, "compare", fake_compare)
    from codesync.api import app
    c = TestClient(app)
    c.calls = calls
    return c


def _wait(client, job_id: str) -> dict:
    for _ in range(100):
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def _upload(client, label: str, data: bytes) -> dict:
    r = client.post("/documents", data={"label": label, "name": label},
                    files={"file": (f"{label}.pdf", data, "application/pdf")})
    assert r.status_code == 200, r.text
    return r.json()


def test_upload_ingest_and_compare(client):
    assert client.get("/health").json()["status"] == "ok"
    pdf = PDF.read_bytes()
    a = _upload(client, "BP", pdf)
    assert _wait(client, a["job"]["id"])["status"] == "done"
    doc_a = a["document"]["doc_id"]
    assert client.get(f"/documents/{doc_a}").json()["status"] == "ready"

    # the same bytes are the same document: no second entry, no second job
    again = _upload(client, "BP again", pdf)
    assert again["document"]["doc_id"] == doc_a and again["job"] is None

    b = _upload(client, "PIP", pdf + b"\n%different")
    doc_b = b["document"]["doc_id"]
    _wait(client, b["job"]["id"])
    assert len(client.get("/documents").json()) == 2

    r = client.post("/comparisons", json={"doc_a": doc_a, "doc_b": doc_b})
    assert r.status_code == 200
    body = r.json()
    assert body["comparison"]["governing"] == doc_a   # default: doc_a governs
    _wait(client, body["job"]["id"])
    cmp_id = body["comparison"]["cmp_id"]
    assert client.get(f"/comparisons/{cmp_id}").json()["status"] == "done"
    assert client.get(f"/comparisons/{cmp_id}/findings").json() == []
    assert ("compare", cmp_id) in client.calls

    assert client.get(f"/documents/{doc_a}/pdf").content == pdf


def test_validation_errors(client):
    assert client.get("/documents/nope").status_code == 404
    assert client.get("/comparisons/nope").status_code == 404
    r = client.post("/documents", data={"label": "X"},
                    files={"file": ("notes.txt", b"hi", "text/plain")})
    assert r.status_code == 400
    doc = _upload(client, "BP", PDF.read_bytes())["document"]["doc_id"]
    same = client.post("/comparisons", json={"doc_a": doc, "doc_b": doc})
    assert same.status_code == 400
    assert client.get(f"/documents/{doc}/figures/..%5Cmeta.json"
                      ).status_code in (400, 404)


def test_review_decisions_are_saved_per_finding(client, monkeypatch):
    from codesync import library
    from codesync.schema import Finding, Requirement

    doc = _upload(client, "BP", PDF.read_bytes())["document"]["doc_id"]
    other = _upload(client, "PIP", PDF.read_bytes() + b"\n%x")["document"]["doc_id"]
    started = client.post("/comparisons", json={"doc_a": doc, "doc_b": other}).json()
    cmp_id = started["comparison"]["cmp_id"]
    _wait(client, started["job"]["id"])      # the stubbed run writes findings.json
    req = Requirement(id="a:r1", doc_id=doc, element_id="a:e1", ref="1", page=1, statement="x")
    library.save_findings(cmp_id, [Finding(id="F0001", category="unique", type="only_in_a", a=req)])

    assert client.get(f"/comparisons/{cmp_id}/reviews").json() == {}
    r = client.put(f"/comparisons/{cmp_id}/reviews/F0001", json={"decision": "accepted", "note": "RFI-7"})
    assert r.status_code == 200 and r.json()["decision"] == "accepted"
    saved = client.get(f"/comparisons/{cmp_id}/reviews").json()
    assert saved["F0001"]["note"] == "RFI-7" and saved["F0001"]["at"]
    assert client.put(f"/comparisons/{cmp_id}/reviews/F9999", json={"decision": "accepted"}).status_code == 404
    assert client.put(f"/comparisons/{cmp_id}/reviews/F0001", json={"decision": "maybe"}).status_code == 422
