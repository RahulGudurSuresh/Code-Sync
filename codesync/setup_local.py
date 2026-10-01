"""Optional: the local developer-fallback drawing reader (llama.cpp and the
Qwen3.5-9B GGUF). Production reads drawings on Modal and needs none of it.

    codesync setup

The llama.cpp Windows release ships the CUDA *backend* and the CUDA
*runtime* as two separate archives. With only the first, llama-server
starts, finds no GPU and silently runs on the CPU (about ten times
slower), so both are fetched and the GPU is checked afterwards.
"""
from __future__ import annotations

import io
import subprocess
import urllib.request
import zipfile

from . import config
from .models.llama_server import gguf_path

LLAMA_BUILD = "b11284"
_BASE = ("https://github.com/ggml-org/llama.cpp/releases/download/"
         f"{LLAMA_BUILD}/")
ARCHIVES = (f"llama-{LLAMA_BUILD}-bin-win-cuda-12.4-x64.zip",
            "cudart-llama-bin-win-cuda-12.4-x64.zip")


def _fetch(name: str) -> None:
    print(f"downloading {name} ...", flush=True)
    with urllib.request.urlopen(_BASE + name) as resp:
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(config.LLAMA_SERVER.parent)


def gpu_devices() -> str:
    out = subprocess.run([str(config.LLAMA_SERVER), "--list-devices"],
                         capture_output=True, text=True, timeout=60)
    return (out.stdout + out.stderr).split("Available devices:")[-1].strip()


def main() -> int:
    target = config.LLAMA_SERVER.parent
    target.mkdir(parents=True, exist_ok=True)
    if not config.LLAMA_SERVER.exists():
        _fetch(ARCHIVES[0])
    if not list(target.glob("cudart64_*.dll")):
        _fetch(ARCHIVES[1])
    devices = gpu_devices()
    print(f"llama.cpp devices: {devices or '(none)'}")
    if "CUDA" not in devices:
        print("WARNING: llama.cpp sees no CUDA device; local models would "
              "run on the CPU. Check the NVIDIA driver.")

    for repo, name in ((config.VLM_REPO, config.VLM_MMPROJ),
                       (config.VLM_REPO, config.VLM_FILE)):
        print(f"model {repo}/{name} -> {gguf_path(repo, name)}", flush=True)
    return 0 if "CUDA" in devices else 1
