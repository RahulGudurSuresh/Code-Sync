"""Optional local llama.cpp server for the developer-fallback drawing
reader (`FIGURE_READER=local`). Production reads drawings on Modal.

Kept because it exists and is tested; not promised to anyone. Needs
`codesync setup` to fetch llama.cpp and the GGUF files.
"""
from __future__ import annotations

import atexit
import subprocess
import threading
import time
from pathlib import Path

import httpx
from huggingface_hub import hf_hub_download

from .. import config


def gguf_path(repo: str, filename: str) -> str:
    """Path of a GGUF in the Hugging Face cache, downloading if absent."""
    try:
        return hf_hub_download(repo, filename, local_files_only=True)
    except Exception:                                    # noqa: BLE001
        return hf_hub_download(repo, filename)


class LlamaServer:
    def __init__(self, name: str, args: list[str], port: int):
        self.name, self.args, self.port = name, args, port
        self.proc: subprocess.Popen | None = None
        self.log_path = config.DATA_DIR / "logs" / f"llama-{name}.log"
        self.load_s = 0.0

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, timeout: float = 300.0) -> None:
        if self.running():
            return
        if not config.LLAMA_SERVER.exists():
            raise FileNotFoundError(
                f"llama-server not found at {config.LLAMA_SERVER}; run "
                f"`codesync setup` or use FIGURE_READER=modal")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(self.log_path, "w", encoding="utf-8", errors="replace")
        cmd = [str(config.LLAMA_SERVER), *self.args,
               "--host", "127.0.0.1", "--port", str(self.port)]
        t0 = time.time()
        self.proc = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server ({self.name}) exited during load:\n"
                    + self._log_tail())
            try:
                if httpx.get(self.base_url + "/health",
                             timeout=2.0).status_code == 200:
                    self.load_s = time.time() - t0
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        self.stop()
        raise TimeoutError(f"llama-server ({self.name}) not ready after "
                           f"{timeout:.0f}s:\n" + self._log_tail())

    def stop(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        self.proc = None

    def _log_tail(self, n: int = 25) -> str:
        try:
            lines = Path(self.log_path).read_text(
                encoding="utf-8", errors="replace").splitlines()
            return "\n".join(lines[-n:])
        except OSError:
            return "(no log)"


def _vlm_server() -> LlamaServer:
    return LlamaServer("vlm", [
        "-m", gguf_path(config.VLM_REPO, config.VLM_FILE),
        "--mmproj", gguf_path(config.VLM_REPO, config.VLM_MMPROJ),
        # no -ngl: llama.cpp fits layers to the VRAM that is free right
        # now, spilling the rest to CPU instead of into shared memory
        "-c", str(config.VLM_CTX), "-np", "1",
        # Qwen vision needs >= 1024 image tokens to read fine detail
        "--image-min-tokens", "1024",
        "--image-max-tokens", str(config.VLM_IMG_TOKENS),
        # a thinking pass per drawing would dwarf the answer
        "--reasoning", "off",
    ], config.VLM_PORT)


class LocalModels:
    """At most one local GPU model resident at a time."""

    def __init__(self):
        self._lock = threading.RLock()
        self._servers: dict[str, LlamaServer] = {}

    def use(self, which: str) -> LlamaServer:
        with self._lock:
            for name, s in self._servers.items():
                if name != which:
                    s.stop()
            if which not in self._servers:
                if which != "vlm":
                    raise ValueError(f"no local server for {which!r}")
                self._servers[which] = _vlm_server()
            server = self._servers[which]
            server.start()
            return server

    def release(self) -> None:
        """Unload whatever is resident, freeing the card."""
        with self._lock:
            for s in self._servers.values():
                s.stop()


local_models = LocalModels()
atexit.register(local_models.release)
