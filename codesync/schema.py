"""Data model shared by every stage and returned by the API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ElementKind = Literal["clause", "table", "figure", "equation"]
PageKind = Literal["text", "drawing", "scanned", "toc", "empty"]


class Element(BaseModel):
    """One addressable piece of a code: a clause, table, figure or equation."""
    id: str                               # "<doc_id>:e<n>"
    kind: ElementKind
    page: int                             # 1-based PDF page
    bbox: list[float]                     # [x0, y0, x1, y1] in PDF points
    number: str = ""                      # own enumerator as printed: "a)", "3.7"
    ref: str = ""                         # citation: "11.6 a.3.b.ii"
    parent: str | None = None             # element id
    depth: int = 0
    text: str = ""
    is_heading: bool = False
    heading_trail: list[str] = Field(default_factory=list)
    source: Literal["text-layer", "ocr", "vlm"] = "text-layer"
    image: str | None = None              # crop file name under figures/
    extra: dict = Field(default_factory=dict)


class PageInfo(BaseModel):
    page: int
    kind: PageKind
    label: str = ""                       # printed page number, if found
    source: str = "text-layer"            # "ocr" when the page was re-read
    width: float
    height: float


class ParsedDocument(BaseModel):
    doc_id: str
    pages: list[PageInfo]
    elements: list[Element]
    header_text: list[str] = Field(default_factory=list)


Modality = Literal["shall", "shall not", "should", "should not", "may",
                   "detail", "info"]


class Requirement(BaseModel):
    """One atomic, self-contained requirement extracted from an element."""
    id: str                               # "<doc_id>:r<n>"
    doc_id: str
    element_id: str
    ref: str                              # clause citation or figure title
    page: int
    statement: str                        # normalised, self-contained
    subject: str = ""
    parameter: str = ""
    value: str = ""
    modality: Modality = "shall"
    quote: str = ""                       # verbatim source text
    quote_verified: bool = False          # quote found in the source element
    references: list[str] = Field(default_factory=list)
    from_figure: bool = False
    # Drawing statements are read by the local vision model and then
    # checked against the sheet by the LLM: "confirmed", "incorrect",
    # "not_found" or "unread". Only confirmed ones take part in matching.
    drawing_check: str = ""
    check_note: str = ""
    excluded: bool = False


class DocumentMeta(BaseModel):
    doc_id: str
    name: str                             # display name, e.g. "BP WBU-GP 04-20-1"
    label: str                            # short tag used in reports: "BP"
    filename: str
    sha256: str
    code: str = ""                        # the standard, e.g. "ASME B31.1"
    edition: str = ""                     # its edition, e.g. "2007"
    # when the stored PDF is a cut of a larger one: stored page -> original
    page_map: list[int] = Field(default_factory=list)
    pages: int = 0
    status: Literal["new", "parsed", "figures", "extracted", "ready",
                    "failed"] = "new"
    stage: str = ""                       # what is running right now
    error: str = ""
    counts: dict = Field(default_factory=dict)
    timings: dict = Field(default_factory=dict)
    created_at: str = ""


Relation = Literal["conflict", "a_stricter", "b_stricter", "equivalent",
                   "related", "unrelated"]
Category = Literal["conflict", "similar", "unique"]


class Finding(BaseModel):
    id: str
    category: Category
    # conflict: "contradiction" | "more_stringent"; similar: "equivalent";
    # unique: "only_in_a" | "only_in_b"
    type: str
    topic: str = ""
    a: Requirement | None = None
    b: Requirement | None = None
    similarity: float | None = None
    # other places in document B that state the same requirement as `b`
    also_b: list[str] = Field(default_factory=list)
    rationale: str = ""
    governing: str = ""                   # document label whose text applies
    stricter: str = ""                    # label of the stricter document
    action: str = ""
    severity: Literal["high", "medium", "low", ""] = ""
    # "confirmed" | "incorrect" | "not_found" when a drawing-derived
    # statement was re-checked against the sheet by the LLM
    drawing_check: str = ""
    needs_review: bool = False
    review_reason: str = ""


class ComparisonMeta(BaseModel):
    cmp_id: str
    doc_a: str
    doc_b: str
    governing: str                        # doc_id that takes precedence
    status: Literal["queued", "running", "done", "failed"] = "queued"
    stage: str = ""
    error: str = ""
    summary: dict = Field(default_factory=dict)
    timings: dict = Field(default_factory=dict)
    created_at: str = ""
