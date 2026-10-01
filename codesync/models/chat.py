"""Chat clients. All speak the same request shape, so a stage can be
pointed at any of them.

A request is {"messages": [...OpenAI style...], "max_tokens": int,
"temperature": float}; a result is {"text": str, "finish_reason": str}.

  ModalLLM   Qwen3.8-27B on Modal (extraction, judging, drawing checks)
  ModalVLM   Qwen3.5-9B on Modal, same container (first reading of drawings)
  LocalVLM   Qwen3.5-9B through llama.cpp on this machine — a developer
             fallback only; the pipeline does not use it by default
"""
from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path

import httpx
from PIL import Image

from .. import config


def image_part(src: Path | bytes, max_tokens: int) -> dict:
    """An `image_url` content part, downscaled to fit `max_tokens`."""
    im = Image.open(io.BytesIO(src) if isinstance(src, bytes) else src)
    im = im.convert("RGB")
    budget = max_tokens * config.PX_PER_TOKEN
    if im.width * im.height > budget:
        k = (budget / (im.width * im.height)) ** 0.5
        im = im.resize((int(im.width * k), int(im.height * k)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return {"type": "image_url", "image_url": {
        "url": "data:image/png;base64," + base64.b64encode(
            buf.getvalue()).decode()}}


def engine():
    """Handle on the deployed Modal engine (`modal_app/codesync_modal.py`)."""
    import modal
    return modal.Cls.from_name(config.MODAL_APP, config.MODAL_CLASS)()


class ModalChat:
    """One of the two chat models in the Modal engine. Requests go in
    batches; vLLM runs each batch concurrently on the GPU."""

    def __init__(self, model: str):
        self.model = model
        self.name = f"modal-{model}"
        self._engine = None
        self.stats = {"calls": 0, "requests": 0, "elapsed_s": 0.0,
                      "prompt_tokens": 0, "completion_tokens": 0}

    def _get(self):
        if self._engine is None:
            self._engine = engine()
        return self._engine

    def warm(self) -> None:
        """Start booting the container without waiting for it, so a cold
        start overlaps local work instead of following it. Best-effort."""
        try:
            self._get().ping.spawn()
        except Exception:                                # noqa: BLE001
            pass

    def chat(self, requests: list[dict]) -> list[dict]:
        out: list[dict] = []
        for i in range(0, len(requests), config.LLM_BATCH):
            batch = requests[i:i + config.LLM_BATCH]
            res = self._get().chat.remote(batch, model=self.model)
            out += res["results"]
            self.stats["calls"] += 1
            self.stats["requests"] += len(batch)
            self.stats["elapsed_s"] += res["elapsed_s"]
            for r in res["results"]:
                self.stats["prompt_tokens"] += r.get("prompt_tokens", 0)
                self.stats["completion_tokens"] += r.get(
                    "completion_tokens", 0)
        return out


class ModalLLM(ModalChat):
    def __init__(self):
        super().__init__("llm")


class ModalVLM(ModalChat):
    def __init__(self):
        super().__init__("vlm")


class LocalVLM:
    """Qwen3.5-9B through llama-server on this machine. One request at a
    time: the card holds a single slot's context. Needs `codesync setup`."""
    name = "local-vlm"

    def chat(self, requests: list[dict]) -> list[dict]:
        from .llama_server import local_models
        server = local_models.use("vlm")
        out = []
        with httpx.Client(base_url=server.base_url, timeout=600.0) as http:
            for r in requests:
                resp = http.post("/v1/chat/completions", json={
                    "messages": r["messages"],
                    "max_tokens": r.get("max_tokens", 1024),
                    "temperature": r.get("temperature", 0.0),
                    "seed": r.get("seed", 42)})
                resp.raise_for_status()
                choice = resp.json()["choices"][0]
                out.append({"text": choice["message"].get("content") or "",
                            "finish_reason": choice.get("finish_reason", "")})
        return out


# ---- JSON out of model text ------------------------------------------------
_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


def parse_json(text: str):
    """Best-effort JSON from a model reply: strips code fences, takes the
    outermost object/array, and salvages a truncated array by cutting it
    back to its last complete item. Returns None when nothing parses."""
    t = _FENCE.sub("", text.strip()).strip()
    starts = [i for i in (t.find("{"), t.find("[")) if i >= 0]
    if not starts:
        return None
    t = t[min(starts):]
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    end = max(t.rfind("}"), t.rfind("]"))
    while end > 0:
        chunk = t[:end + 1]
        for closer in ("", "]", "]}", "}", "}]", "}]}"):
            try:
                return json.loads(chunk + closer)
            except json.JSONDecodeError:
                continue
        end = t.rfind("}", 0, end)
    return None


def chat_json(client, requests: list[dict], retries: int = 1
              ) -> list[tuple[object | None, dict]]:
    """Run `requests`, parse each reply as JSON, and re-ask the ones that
    did not parse (warmer, different seed). Returns [(parsed, raw_result)].
    A reply is retried only when it could not be READ — never because its
    content was unwelcome."""
    results = client.chat(requests)
    parsed = [parse_json(r["text"]) for r in results]
    for attempt in range(retries):
        bad = [i for i, p in enumerate(parsed) if p is None]
        if not bad:
            break
        again = client.chat([{**requests[i], "temperature": 0.3,
                              "seed": 1000 + attempt} for i in bad])
        for i, r in zip(bad, again):
            results[i], parsed[i] = r, parse_json(r["text"])
    return list(zip(parsed, results))
