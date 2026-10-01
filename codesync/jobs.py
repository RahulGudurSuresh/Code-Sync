"""Background job runner for the API.

One worker thread, one job at a time: every job needs the single local GPU
and the single Modal container, so running two at once would only make
both slower. Job progress is kept in memory for polling; the durable state
lives in each document's and comparison's meta.json.
"""
from __future__ import annotations

import itertools
import queue
import threading
import traceback
from collections import deque
from collections.abc import Callable

from .library import now

Progress = Callable[[str], None]


class JobRunner:
    def __init__(self):
        self._queue: queue.Queue = queue.Queue()
        self._jobs: dict[str, dict] = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def _ensure_worker(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._work, daemon=True,
                                            name="codesync-jobs")
            self._thread.start()

    def submit(self, kind: str, target: str,
               fn: Callable[[Progress], None]) -> dict:
        with self._lock:
            job = {"id": f"job-{next(self._ids)}", "kind": kind,
                   "target": target, "status": "queued", "message": "",
                   "log": deque(maxlen=200), "error": "",
                   "created_at": now(), "finished_at": ""}
            self._jobs[job["id"]] = job
            self._queue.put((job, fn))
            self._ensure_worker()
        return self.view(job)

    def _work(self) -> None:
        while True:
            job, fn = self._queue.get()
            job["status"] = "running"

            def progress(msg: str, job=job) -> None:
                job["message"] = msg
                job["log"].append(f"{now()} {msg}")

            try:
                fn(progress)
                job["status"] = "done"
            except Exception as e:                       # noqa: BLE001
                job["status"] = "failed"
                job["error"] = f"{type(e).__name__}: {e}"
                job["log"].append(traceback.format_exc(limit=6))
            finally:
                job["finished_at"] = now()
                self._queue.task_done()

    @staticmethod
    def view(job: dict) -> dict:
        return {**job, "log": list(job["log"])}

    def get(self, job_id: str) -> dict | None:
        job = self._jobs.get(job_id)
        return self.view(job) if job else None

    def list(self) -> list[dict]:
        return [self.view(j) for j in self._jobs.values()]


runner = JobRunner()
