"""Paths, model choices and tunables for the whole backend.

Everything a deployment might change lives here; stages import from this
module rather than hard-coding values. Environment variables override the
few settings that differ per machine.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("CODESYNC_DATA", ROOT / "data"))
LIBRARY_DIR = DATA_DIR / "library"          # one folder per ingested code
COMPARISON_DIR = DATA_DIR / "comparisons"   # one folder per A-vs-B run

# ---- the Modal engine (LLM, vision model, embedder on one GPU) -------------
# Model choices themselves live in modal_app/codesync_modal.py; the names
# here only label caches and reports.
MODAL_APP = "codesync-llm"
MODAL_CLASS = "Engine"
LLM_BATCH = 48                  # chat requests per Engine.chat call
LLM_IMG_TOKENS = 4_096          # must not exceed IMG_TOKENS_CAP in the app
PX_PER_TOKEN = 1024             # Qwen3.x vision: 16 px patches merged 2x2

# Who reads drawings first: "modal" (Qwen3.5-9B in the engine, default),
# "llm" (the 27B reads as well as checks — untested as a safeguard, see
# BOTTLENECKS.md item 7), or "local" (llama.cpp on this machine, developer
# fallback, needs `codesync setup`).
FIGURE_READER = os.environ.get("CODESYNC_FIGURE_READER", "modal")

# Embedding model in the engine. Changing it re-embeds the library: vectors
# from different models are not comparable (the cache key carries `name`).
EMBED_MODEL = "Qwen/Qwen3-Embedding-8B"
EMBED_DIM = 4096
EMBED_BATCH = 256               # texts per Engine.embed call
EMBED_TASK = ("Given a requirement from an engineering standard, retrieve "
              "requirements from another standard that address the same "
              "subject and parameter")

# ---- optional local drawing reader (llama.cpp) -----------------------------
LLAMA_SERVER = Path(os.environ.get(
    "CODESYNC_LLAMA_SERVER", ROOT / "tools" / "llama" / "llama-server.exe"))
VLM_REPO = "unsloth/Qwen3.5-9B-GGUF"
VLM_FILE = "Qwen3.5-9B-Q6_K.gguf"
VLM_MMPROJ = "mmproj-F16.gguf"
VLM_PORT = 8765
VLM_CTX = 16_384
VLM_IMG_TOKENS = 4_096

# ---- parsing ---------------------------------------------------------------
FIGURE_DPI = 190                # a Letter sheet -> ~3.6k image tokens
LAYOUT_DPI = 110
OCR_DPI = 200
MIN_TEXT_CHARS = 200            # below this a page has no usable text layer
VECTOR_HEAVY = 400              # drawing ops that mark a page as a drawing
# A scanned page whose embedded text layer has more than this share of
# damaged tokens is re-read with PP-OCRv5 (10-25 s per page on CPU).
REOCR = True
REOCR_NOISE = 0.003

# ---- requirement extraction ------------------------------------------------
BLOCK_MAX_CHARS = 5_000         # clause text per extraction request

# ---- matching --------------------------------------------------------------
# Calibrated on BP WBU-GP 04-20-1 vs PIP STF05501 with the local Q8 file:
# scores between the two codes averaged 0.37, true counterparts sat at
# 0.56-0.76, and the best candidate for a requirement with no counterpart
# was around 0.49. Re-checked against the full-precision model on Modal
# before the thresholds were kept.
TOP_K = 5                       # candidates kept per requirement, each way
MIN_SIMILARITY = 0.50           # cosine floor for a candidate pair

for _d in (LIBRARY_DIR, COMPARISON_DIR):
    _d.mkdir(parents=True, exist_ok=True)
