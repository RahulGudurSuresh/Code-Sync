"""Small local vision models used while parsing: layout detection, OCR,
table structure and formula recognition (PaddleOCR 3.x, CPU).

Each model is loaded on first use and kept for the process lifetime; none
of them touch the GPU, so they never compete with the VLM or the embedder
for the 12 GB card.
"""
from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from functools import lru_cache

import fitz
import numpy as np

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
warnings.filterwarnings("ignore", module="paddle")
logging.getLogger("paddlex").setLevel(logging.ERROR)


def render(page: fitz.Page, dpi: int, clip: fitz.Rect | None = None
           ) -> np.ndarray:
    """Page (or a clip of it) as a BGR uint8 array."""
    pix = page.get_pixmap(dpi=dpi, clip=clip, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.h, pix.w, pix.n)
    return img[:, :, :3][:, :, ::-1].copy()


@dataclass
class Region:
    label: str
    score: float
    bbox: list[float]              # PDF points

    @property
    def area(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0]) * \
            max(0.0, self.bbox[3] - self.bbox[1])


@lru_cache(maxsize=1)
def _layout_model():
    from paddleocr import LayoutDetection
    return LayoutDetection(model_name="PP-DocLayoutV3")


def detect_layout(page: fitz.Page, dpi: int) -> list[Region]:
    img = render(page, dpi)
    out = _layout_model().predict(img, batch_size=1, layout_nms=True)
    k = 72.0 / dpi
    return [Region(b["label"], float(b["score"]),
                   [float(v) * k for v in b["coordinate"]])
            for b in out[0].json["res"]["boxes"]]


@lru_cache(maxsize=1)
def _ocr_model():
    from paddleocr import PaddleOCR
    return PaddleOCR(text_detection_model_name="PP-OCRv5_server_det",
                     text_recognition_model_name="PP-OCRv5_server_rec",
                     use_doc_orientation_classify=False,
                     use_doc_unwarping=False,
                     use_textline_orientation=False)


def ocr_page(page: fitz.Page, dpi: int, clip: fitz.Rect | None = None
             ) -> list[tuple[list[float], str, float]]:
    """[(bbox in PDF points, text, score)] for every text line found."""
    img = render(page, dpi, clip)
    res = _ocr_model().predict(img)[0]
    k = 72.0 / dpi
    ox, oy = (clip.x0, clip.y0) if clip else (0.0, 0.0)
    out = []
    for box, text, score in zip(res["rec_boxes"], res["rec_texts"],
                                res["rec_scores"]):
        if not str(text).strip():
            continue
        x0, y0, x1, y1 = (float(v) * k for v in box)
        out.append(([x0 + ox, y0 + oy, x1 + ox, y1 + oy], str(text),
                    float(score)))
    return out


@lru_cache(maxsize=1)
def _formula_model():
    from paddleocr import FormulaRecognition
    return FormulaRecognition(model_name="PP-FormulaNet_plus-L")


def formula_latex(page: fitz.Page, bbox: list[float], dpi: int = 200) -> str:
    img = render(page, dpi, fitz.Rect(bbox) + (-3, -3, 3, 3))
    out = _formula_model().predict(img, batch_size=1)
    return str(out[0]["rec_formula"]).strip()


@lru_cache(maxsize=1)
def _table_pipeline():
    from paddleocr import TableRecognitionPipelineV2
    return TableRecognitionPipelineV2(use_doc_orientation_classify=False,
                                      use_doc_unwarping=False,
                                      use_layout_detection=False)


def table_html(page: fitz.Page, bbox: list[float], dpi: int = 200) -> str:
    """Structure + OCR of a table image region -> HTML (no text layer)."""
    img = render(page, dpi, fitz.Rect(bbox) + (-4, -4, 4, 4))
    out = _table_pipeline().predict(img)
    tables = out[0]["table_res_list"]
    return str(tables[0]["pred_html"]) if tables else ""
