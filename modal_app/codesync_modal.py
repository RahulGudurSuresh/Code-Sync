"""CodeSync engine on Modal: three models on one RTX PRO 6000.

  llm    Qwen3.8-27B (NVFP4) via vLLM — extraction, judging, review, and
         the second reading of drawing statements (it is a vision model)
  vlm    Qwen3.5-9B via vLLM — first reading of drawing sheets. A second,
         different model, so that a drawing statement is used only when two
         readings agree.
  embed  Qwen3-Embedding-8B via transformers — requirement vectors
  rerank Qwen3-Reranker-0.6B via transformers — rescores search candidates
         against a question (Ask the Code)

The serving recipe is DTSPL's `fast_pipeline/fast_modal.py`, in its own app
with its own volumes (nothing is shared with `bom-extractor-fast` except the
Modal workspace): vLLM 0.27.1 in-process, weights on a Volume,
torch.compile artifacts on a second Volume, GPU memory snapshot taken after
a warm-up of the real request shapes, sleep mode never invoked (it corrupts
the NVFP4 checkpoint — DTSPL finding, 2026-08).

Memory budget on the 96 GB card (see BOTTLENECKS.md, item 1), measured
at the first boot:
  llm   40%  38 GiB: 22.1 weights, 8.3 peak activations (image encoder
             profiling), the rest working space
  vlm   30%  28.5 GiB: 17.7 weights, several GiB of image-encoder
             activations, the rest working space. 22% was not enough:
             vLLM found no room for cache blocks and refused to start.
  embed      ~17 GiB: 15 weights + activations, allocated as it goes
  rerank     ~1.5 GiB: 0.6B weights + activations
  free       ~8 GiB for CUDA contexts and spikes
vLLM engines are started first, largest first, because each one checks
that its share is free when it initialises.

The engine is deliberately dumb: OpenAI-style messages in, text out; texts
in, vectors out. Every prompt lives client-side in `codesync/prompts.py`,
so a prompt change never redeploys this app — and a redeploy is what
invalidates the snapshot.

Deploy:   modal deploy modal_app/codesync_modal.py
Audit:    modal run modal_app/codesync_modal.py::imgcheck
Smoke:    modal run modal_app/codesync_modal.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

APP_NAME = "codesync-llm"
LLM_ID = "unsloth/Qwen3.8-27B-NVFP4"
VLM_ID = "Qwen/Qwen3.5-9B"
EMBED_ID = "Qwen/Qwen3-Embedding-8B"
RERANK_ID = "Qwen/Qwen3-Reranker-0.6B"
GPU_KIND = "RTX-PRO-6000"
# The trailing `a` is REQUIRED: the CUTLASS FP4 kernels are arch-conditional
# and plain "12.0" empties flashinfer's FP4 arch list.
CUDA_ARCH = "12.0a"

LLM_SHARE = 0.40
VLM_SHARE = 0.30

PX_PER_TOKEN = 1024          # Qwen3.x vision: patch 16, merge 2
# Largest image the client sends (a full drawing sheet at ~190 DPI). The
# warm-up exercises exactly this shape so its JIT paths are in the snapshot.
IMG_TOKENS_CAP = 4_096
MAX_IMAGES_PER_PROMPT = 2    # one drawing from each code in a verdict
LLM_MAX_MODEL_LEN = 40_960
VLM_MAX_MODEL_LEN = 16_384
VLM_MAX_SEQS = 8             # sheets read side by side
EMBED_MAX_TOKENS = 8_192
# Texts are batched by a token budget, not a count: a batch of 32 clauses
# of 2,000 tokens each would need far more activation memory than the
# ~10 GB the embedder has beside the two vLLM models (OOM seen on the
# first Ask-the-Code index). Short requirement statements still go 100+
# to a batch.
EMBED_BATCH_TOKENS = 8_192
EMBED_BATCH_MAX = 128
RERANK_MAX_TOKENS = 4_096
RERANK_BATCH = 16
# Qwen3-Reranker answers "yes"/"no" to a judged pair; the score is P(yes).
RERANK_PREFIX = ("<|im_start|>system\nJudge whether the Document meets the "
                 "requirements based on the Query and the Instruct provided. "
                 "Note that the answer can only be \"yes\" or \"no\"."
                 "<|im_end|>\n<|im_start|>user\n")
RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

# Snapshot restore is placement-sensitive: quick reap->reboot cycles land on
# the same worker and restore in ~30-80 s, long gaps move workers and miss.
SCALEDOWN_S = 2 * 60

# First three layers are byte-identical to DTSPL's image on purpose: it reuses
# the built layers and, more importantly, the pip resolution that was
# validated (vllm 0.27.1 / torch 2.13). Do NOT add packages to these layers —
# DTSPL saw an unrelated addition silently resolve vllm 0.22.1. The embedder
# uses transformers, which vLLM already brings in, so nothing was added.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.3-devel-ubuntu24.04", add_python="3.12"
    )
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        # pinned to the validated stack — do NOT float these
        "vllm==0.27.1", "transformers>=5.14.0", "Pillow",
        "pymupdf", "huggingface_hub[hf_transfer]",
    )
    .pip_install("flashinfer-python==0.6.16.post3",
                 "flashinfer-jit-cache==0.6.16.post3",
                 extra_options="--extra-index-url "
                               "https://flashinfer.ai/whl/cu130/")
    # flashinfer refuses to import if `flashinfer-cubin`'s version differs
    # from `flashinfer-python`'s; it arrives as a transitive dependency.
    .run_commands("pip uninstall -y flashinfer-cubin || true")
    .env({
        "HF_HOME": "/cache",
        # The Xet downloader keeps log files open under /cache/xet; a file
        # still open during snapshot creation makes the snapshot restore fail
        # ("failed to walk .../xet/logs"). Plain HTTP downloads are fine here.
        "HF_HUB_DISABLE_XET": "1",
        # sm_120 traps — do not "simplify" these:
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "TORCH_CUDA_ARCH_LIST": CUDA_ARCH,
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        # Required for GPU memory snapshots: multi-threaded Torch compile can
        # make snapshot creation fail.
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
    })
)

app = modal.App(APP_NAME)
weights = modal.Volume.from_name("codesync-qwen38-nvfp4-weights",
                                 create_if_missing=True)
# vLLM writes torch_compile_cache/ and flashinfer_autotune_cache/ here; on
# ephemeral storage every cold start would recompile (~3.5 min per engine).
VLLM_CACHE_DIR = "/root/.cache/vllm"
compile_cache = modal.Volume.from_name("codesync-vllm-compile",
                                       create_if_missing=True)


def _blank_image_url(tokens: int) -> str:
    import base64
    import io

    from PIL import Image
    side = int((tokens * PX_PER_TOKEN) ** 0.5)
    buf = io.BytesIO()
    Image.new("RGB", (side, side), "white").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.cls(gpu=GPU_KIND, image=image,
         volumes={"/cache": weights, VLLM_CACHE_DIR: compile_cache},
         # Snapshots are ONLY created for DEPLOYED apps; `modal run` never
         # produces one.
         enable_memory_snapshot=True,
         experimental_options={"enable_gpu_snapshot": True},
         # RAM for loading the embedder through CPU memory (see load())
         memory=24 * 1024,
         timeout=60 * 60, scaledown_window=SCALEDOWN_S, max_containers=1)
class Engine:
    """Pure compute: batches of chat requests in, completions out; texts
    in, vectors out."""

    @modal.enter(snap=True)
    def load(self):
        """Runs BEFORE the snapshot, so the CUDA contexts, compiled kernels,
        CUDA graphs and all three sets of weights are captured in it."""
        import torch
        from vllm import LLM

        cdir = Path(VLLM_CACHE_DIR) / "torch_compile_cache"
        warm = cdir.exists() and any(cdir.iterdir())
        print(f"[cache] {VLLM_CACHE_DIR} warm={warm}", flush=True)
        t0 = time.time()

        self.llm = LLM(
            model=LLM_ID, trust_remote_code=True,
            max_model_len=LLM_MAX_MODEL_LEN,
            kv_cache_dtype="fp8",
            limit_mm_per_prompt={"image": MAX_IMAGES_PER_PROMPT},
            enable_prefix_caching=True,   # shared system prompt per stage
            gpu_memory_utilization=LLM_SHARE,
            # Kept from the validated DTSPL config. sleep() itself is never
            # called — see the module docstring.
            enable_sleep_mode=True,
            # MTP speculative decoding via the checkpoint's own draft heads.
            speculative_config={"method": "mtp",
                                "num_speculative_tokens": 2},
        )
        t1 = time.time()
        print(f"[load] llm ready in {t1 - t0:.1f}s", flush=True)

        self.vlm = LLM(
            model=VLM_ID, trust_remote_code=True,
            max_model_len=VLM_MAX_MODEL_LEN,
            max_num_seqs=VLM_MAX_SEQS,
            limit_mm_per_prompt={"image": 1},
            enable_prefix_caching=True,
            gpu_memory_utilization=VLM_SHARE,
        )
        t2 = time.time()
        print(f"[load] vlm ready in {t2 - t1:.1f}s", flush=True)

        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(EMBED_ID, padding_side="left")
        # No `device_map`: that needs the `accelerate` package, which is not
        # in the image (see the dependency note above). Load on the CPU and
        # move; the 15 GB CPU copy is what the `memory=` reservation is for.
        self.emb = AutoModel.from_pretrained(
            EMBED_ID, dtype=torch.bfloat16).to("cuda").eval()
        # Qwen3-Embedding pools the last token, which must be the end-of-text
        # token. Some tokenizer configs append it, some do not: check once.
        ids = self.tok("check")["input_ids"]
        self.add_eos = ids[-1] != self.tok.eos_token_id
        t3 = time.time()
        print(f"[load] embedder ready in {t3 - t2:.1f}s "
              f"(add_eos={self.add_eos})", flush=True)

        from transformers import AutoModelForCausalLM
        self.rr_tok = AutoTokenizer.from_pretrained(RERANK_ID, padding_side="left")
        self.rr = AutoModelForCausalLM.from_pretrained(
            RERANK_ID, dtype=torch.bfloat16).to("cuda").eval()
        self.rr_yes = self.rr_tok.convert_tokens_to_ids("yes")
        self.rr_no = self.rr_tok.convert_tokens_to_ids("no")
        self.rr_prefix = self.rr_tok.encode(RERANK_PREFIX, add_special_tokens=False)
        self.rr_suffix = self.rr_tok.encode(RERANK_SUFFIX, add_special_tokens=False)
        t4 = time.time()
        print(f"[load] reranker ready in {t4 - t3:.1f}s", flush=True)
        self.load_s = t4 - t0
        free, total = torch.cuda.mem_get_info()
        print(f"[load] all models in {self.load_s:.1f}s | "
              f"{torch.cuda.get_device_name(0)} | GPU used "
              f"{(total - free) / 2**30:.1f} of {total / 2**30:.1f} GiB",
              flush=True)

        # Container writes to a Volume are not durable without a commit.
        # Best-effort: a failed commit only costs the next cold start.
        if not warm:
            for name, vol in (("compile", compile_cache), ("weights", weights)):
                try:
                    vol.commit()
                    print(f"[cache] {name} volume committed", flush=True)
                except Exception as e:                   # noqa: BLE001
                    print(f"[cache] {name} commit FAILED "
                          f"({type(e).__name__}: {e})", flush=True)

        # Warm up every real shape so the snapshot carries the JIT artifacts
        # instead of re-earning them on every restore.
        t4 = time.time()
        try:
            url = _blank_image_url(IMG_TOKENS_CAP)
            image_req = {"messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": "Is this image blank? "
                                         "Answer yes or no."}]}],
                "max_tokens": 16}
            self._run("llm", [
                {"messages": [
                    {"role": "system", "content": "Reply with JSON only."},
                    {"role": "user", "content": 'Return {"ok": true}'}],
                 "max_tokens": 16},
                image_req])
            self._run("vlm", [image_req] * 2)
            self._embed(["warm-up text"] * 4)
            self._rerank("warm-up question", ["warm-up passage"] * 2, "")
            for _ in self._stream([{"role": "user", "content": "Say ok."}], 8, 0.0):
                pass
            print(f"[warmup] all shapes exercised in {time.time()-t4:.1f}s",
                  flush=True)
        except Exception as e:                           # noqa: BLE001
            # A failed warm-up only means a slower first real request.
            print(f"[warmup] skipped ({type(e).__name__}: {e})", flush=True)

    @modal.enter(snap=False)
    def restore(self):
        """Runs after a snapshot restore (and on the first boot, right after
        `load`). Nothing to wake: no engine was ever put to sleep."""
        self.restored_at = time.time()
        print("[restore] engine live", flush=True)

    # -- helpers ------------------------------------------------------------
    def _run(self, model: str, requests: list[dict]) -> list[dict]:
        """One `llm.chat` call per distinct sampling config, results
        returned in request order."""
        from vllm import SamplingParams

        engine = self.llm if model == "llm" else self.vlm
        groups: dict[tuple, list[int]] = {}
        for i, r in enumerate(requests):
            key = (int(r.get("max_tokens", 1024)),
                   float(r.get("temperature", 0.0)),
                   int(r.get("seed", 42)))
            groups.setdefault(key, []).append(i)

        results: list[dict | None] = [None] * len(requests)
        for (max_tokens, temperature, seed), idxs in groups.items():
            sp = SamplingParams(temperature=temperature,
                                max_tokens=max_tokens, seed=seed)
            # Reasoning is ON by default on both models; callers that want
            # a rationale ask for it as a JSON field instead.
            outs = engine.chat(
                [requests[i]["messages"] for i in idxs], sp, use_tqdm=False,
                chat_template_kwargs={"enable_thinking": False})
            for i, o in zip(idxs, outs):
                c = o.outputs[0]
                results[i] = {
                    "text": c.text,
                    "finish_reason": c.finish_reason,
                    "prompt_tokens": len(o.prompt_token_ids or []),
                    "completion_tokens": len(c.token_ids or []),
                }
        return results                                   # type: ignore[return-value]

    def _embed(self, texts: list[str]):
        """L2-normalised float32 vectors, shape (len(texts), 4096)."""
        import torch
        import torch.nn.functional as F

        if self.add_eos:
            texts = [t + self.tok.eos_token for t in texts]
        # tokenise once, then batch by length so padding is small and the
        # (count x longest) product stays under the budget
        encoded = self.tok(texts, truncation=True, max_length=EMBED_MAX_TOKENS,
                           padding=False)["input_ids"]
        order = sorted(range(len(texts)), key=lambda i: len(encoded[i]))
        batches: list[list[int]] = []
        cur: list[int] = []
        for i in order:
            longest = len(encoded[i])          # ascending: the newest is longest
            if cur and (len(cur) >= EMBED_BATCH_MAX
                        or (len(cur) + 1) * longest > EMBED_BATCH_TOKENS):
                batches.append(cur)
                cur = []
            cur.append(i)
        if cur:
            batches.append(cur)
        out = torch.empty(len(texts), self.emb.config.hidden_size, dtype=torch.float32)
        with torch.inference_mode():
            for idx in batches:
                batch = self.tok.pad({"input_ids": [encoded[i] for i in idx]},
                                     padding=True, return_tensors="pt").to("cuda")
                hidden = self.emb(**batch).last_hidden_state
                # left padding: the last position is the last real token
                vec = F.normalize(hidden[:, -1].float(), p=2, dim=1)
                out[idx] = vec.cpu()
                del hidden, batch
        # give the reserved pool back: the vision encoder of the 9B takes
        # its activations from the same ~10 GB that is free beside the
        # models, and found 95 MB after a big embed
        torch.cuda.empty_cache()
        return out.numpy()

    def _rerank(self, query: str, docs: list[str], instruction: str
                ) -> list[float]:
        """P(yes) for each (query, doc) pair, per the Qwen3-Reranker card."""
        import torch

        instruction = instruction or ("Given a question about an engineering "
                                      "code, retrieve passages of the code "
                                      "that answer it")
        scores: list[float] = []
        budget = RERANK_MAX_TOKENS - len(self.rr_prefix) - len(self.rr_suffix)
        with torch.inference_mode():
            for i in range(0, len(docs), RERANK_BATCH):
                pairs = [f"<Instruct>: {instruction}\n<Query>: {query}\n"
                         f"<Document>: {d}" for d in docs[i:i + RERANK_BATCH]]
                enc = self.rr_tok(pairs, padding=False, truncation="longest_first",
                                  max_length=budget, add_special_tokens=False)
                ids = [self.rr_prefix + x + self.rr_suffix for x in enc["input_ids"]]
                width = max(len(x) for x in ids)
                pad = self.rr_tok.pad_token_id
                input_ids = torch.tensor([[pad] * (width - len(x)) + x for x in ids],
                                         device="cuda")
                attention = (input_ids != pad).long()
                for k, x in enumerate(ids):            # a real pad token inside
                    attention[k, width - len(x):] = 1   # the text stays visible
                logits = self.rr(input_ids=input_ids,
                                 attention_mask=attention).logits[:, -1, :]
                pair = torch.stack([logits[:, self.rr_no], logits[:, self.rr_yes]], dim=1)
                scores += torch.log_softmax(pair.float(), dim=1)[:, 1].exp().tolist()
        torch.cuda.empty_cache()
        return scores

    def _stream(self, messages: list[dict], max_tokens: int, temperature: float):
        """Token-by-token generation from the 27B, through the engine's own
        step loop (the batch `chat` call only returns when everything is
        done, which is wrong for a person watching an answer appear)."""
        from vllm import SamplingParams

        tok = self.llm.get_tokenizer()
        prompt = tok.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True,
                                         enable_thinking=False)
        engine = self.llm.llm_engine
        rid = f"ask-{time.time_ns()}"
        engine.add_request(rid, prompt, SamplingParams(
            temperature=temperature, max_tokens=max_tokens, seed=42))
        sent = ""
        while engine.has_unfinished_requests():
            for out in engine.step():
                if out.request_id != rid:
                    continue
                text = out.outputs[0].text
                if len(text) > len(sent):
                    delta, sent = text[len(sent):], text
                    yield {"delta": delta}
                if out.finished:
                    yield {"done": True,
                           "finish_reason": out.outputs[0].finish_reason,
                           "prompt_tokens": len(out.prompt_token_ids or []),
                           "completion_tokens": len(out.outputs[0].token_ids or [])}

    @staticmethod
    def _die_if_engine_dead(e: BaseException) -> None:
        """A vLLM engine that has died (an OOM inside it, a CUDA fault)
        stays dead for the life of the process. Ending the container makes
        Modal start a fresh one, the way it does after a preemption."""
        if type(e).__name__ == "EngineDeadError" or "EngineDeadError" in str(e):
            print(f"engine dead ({e!r}); ending the container", flush=True)
            os._exit(3)

    # -- API ----------------------------------------------------------------
    @modal.method()
    def rerank(self, query: str, docs: list[str], instruction: str = "") -> dict:
        """Relevance of each passage to the question, 0 to 1."""
        t0 = time.time()
        return {"scores": self._rerank(query, docs, instruction),
                "elapsed_s": round(time.time() - t0, 2)}

    @modal.method()
    def stream(self, messages: list[dict], max_tokens: int = 1500,
               temperature: float = 0.0):
        """Streams {"delta": text} chunks, then {"done": True, ...}."""
        try:
            yield from self._stream(messages, max_tokens, temperature)
        except Exception as e:                           # noqa: BLE001
            self._die_if_engine_dead(e)
            raise

    @modal.method()
    def chat(self, requests: list[dict], model: str = "llm") -> dict:
        """`requests`: [{"messages": [...], "max_tokens": int,
        "temperature": float, "seed": int}] with OpenAI-style messages
        (images as `image_url` data URLs). `model` is "llm" (27B) or "vlm"
        (9B). Returns {"results": [{"text", "finish_reason",
        "prompt_tokens", "completion_tokens"}], "elapsed_s", "load_s"}."""
        if model not in ("llm", "vlm"):
            raise ValueError(f"unknown model {model!r}")
        t0 = time.time()
        try:
            results = self._run(model, requests)
        except Exception as e:                           # noqa: BLE001
            self._die_if_engine_dead(e)
            raise
        return {"results": results, "elapsed_s": round(time.time() - t0, 2),
                "load_s": round(self.load_s, 1)}

    @modal.method()
    def embed(self, texts: list[str]) -> dict:
        """Vectors for `texts` (query instructions are the caller's job).
        Returned as raw float32 bytes: 40,000 vectors as JSON would be over
        a gigabyte."""
        t0 = time.time()
        vecs = self._embed(texts)
        return {"shape": list(vecs.shape), "dtype": "float32",
                "data": vecs.astype("float32").tobytes(),
                "elapsed_s": round(time.time() - t0, 2)}

    @modal.method()
    def ping(self) -> dict:
        """Boots the container (seeding the snapshot) without doing work."""
        import torch
        free, total = torch.cuda.mem_get_info()
        return {"llm": LLM_ID, "vlm": VLM_ID, "embed": EMBED_ID,
                "rerank": RERANK_ID,
                "gpu": GPU_KIND, "load_s": round(self.load_s, 1),
                "gpu_used_gib": round((total - free) / 2**30, 1),
                "gpu_total_gib": round(total / 2**30, 1)}


@app.function(image=image, timeout=300)
def imgcheck() -> dict:
    """CPU-only image audit — cheaper than discovering a bad dependency set
    six minutes into a GPU cold start. Expect vllm 0.27.1, torch 2.13.x,
    flashinfer-cubin ABSENT."""
    import importlib.metadata as md
    out = {}
    for p in ("flashinfer-python", "flashinfer-cubin", "flashinfer-jit-cache",
              "vllm", "torch", "transformers"):
        try:
            out[p] = md.version(p)
        except Exception:                                # noqa: BLE001
            out[p] = "ABSENT"
    print(json.dumps(out, indent=2), flush=True)
    return out


@app.local_entrypoint()
def smoke():
    """One request per model through an ephemeral engine (no snapshot is
    created by `modal run`; use the deployed app for real work)."""
    import numpy as np
    engine = Engine()
    print(json.dumps(engine.ping.remote(), indent=2))
    for model in ("llm", "vlm"):
        res = engine.chat.remote([{
            "messages": [{"role": "user", "content":
                          'Reply with exactly this JSON: {"ok": true}'}],
            "max_tokens": 32}], model=model)
        print(model, json.dumps(res))
    res = engine.embed.remote(["ladder rung diameter", "stair tread width"])
    vecs = np.frombuffer(res["data"], dtype=np.float32).reshape(res["shape"])
    print("embed", res["shape"], "cos", float(vecs[0] @ vecs[1]))
    print("rerank", engine.rerank.remote("minimum wall thickness of pipe", [
        "The minimum thickness of pipe wall shall not be less than tm = PDo/2(SE+Py) + A.",
        "Platform exits to vertical ladders shall have self-closing gates."]))
    for piece in engine.stream.remote_gen(
            [{"role": "user", "content": "Count from one to five, words only."}], 32):
        print(piece, flush=True)
