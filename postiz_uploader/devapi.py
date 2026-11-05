"""A RunPod Serverless stand-in for local development.

POST /run         queue a job, answer at once with its id
GET  /status/<id> report on a job, never run it
POST /runsync     queue a job and wait for it

The paths take an optional /v2/<endpoint> prefix, so a caller pointed at
https://api.runpod.ai/v2/<endpoint> only swaps the host. Jobs run through
pipeline.process, the function handler.py hands every RunPod job to, on
WORKER_CONCURRENCY threads.

The SDK's own `handler.py --rp_serve_api` runs a job inside the first status call and
forgets it afterwards, which a caller polling with a request timeout cannot use. This is
dev tooling and is never reachable in production.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROUTE = re.compile(r"^(?:/v2/[^/]+)?/(run|runsync|status)(?:/([^/?]+))?/?(?:\?.*)?$")
# RunPod keeps /run results for 30 minutes and waits 90 seconds on /runsync by default
RETENTION_SECONDS = 30 * 60
RUNSYNC_WAIT_SECONDS = 90
PUBLIC_FIELDS = ("id", "status", "delayTime", "executionTime", "output", "error")


class Jobs:
    def __init__(self, workers: int, run: Callable[[object], dict]):
        self._pool = ThreadPoolExecutor(max_workers=workers)
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._run = run

    def submit(self, job_input: object) -> dict:
        job = {
            "id": f"local-{uuid.uuid4()}",
            "status": "IN_QUEUE",
            "submitted": time.monotonic(),
            "done": threading.Event(),
        }
        with self._lock:
            self._expire()
            self._jobs[job["id"]] = job
        self._pool.submit(self._execute, job, job_input)
        return job

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            self._expire()
            return self._jobs.get(job_id)

    def _execute(self, job: dict, job_input: object) -> None:
        started = time.monotonic()
        job["delayTime"] = int((started - job["submitted"]) * 1000)
        job["status"] = "IN_PROGRESS"
        try:
            job["output"] = self._run(job_input)
            status = "COMPLETED"
        except Exception as exc:  # a handler that raises is a FAILED job on RunPod
            job["error"] = str(exc)
            status = "FAILED"
        job["executionTime"] = int((time.monotonic() - started) * 1000)
        job["finished"] = time.monotonic()
        # last, so a poll that sees a terminal status also sees the timing fields
        job["status"] = status
        job["done"].set()

    def _expire(self) -> None:
        now = time.monotonic()
        for job_id, job in list(self._jobs.items()):
            if "finished" in job and now - job["finished"] > RETENTION_SECONDS:
                del self._jobs[job_id]

    @staticmethod
    def public(job: dict) -> dict:
        return {k: job[k] for k in PUBLIC_FIELDS if k in job}


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, jobs: Jobs, **kwargs):
        self.jobs = jobs
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:  # noqa: N802 - http.server naming
        self._route("GET")

    def do_POST(self) -> None:  # noqa: N802 - http.server naming
        self._route("POST")

    def _route(self, method: str) -> None:
        match = ROUTE.match(self.path)
        if not match:
            return self._send(404, {"error": "not found"})
        operation, job_id = match.groups()

        if operation == "status":
            if method != "GET" or not job_id:
                return self._send(405, {"error": "status is GET /status/<id>"})
            job = self.jobs.get(job_id)
            if not job:
                return self._send(404, {"error": "job not found"})
            return self._send(200, Jobs.public(job))

        if method != "POST" or job_id:
            return self._send(405, {"error": f"{operation} is POST /{operation}"})
        body = self._read_json()
        if not isinstance(body, dict):
            return self._send(400, {"error": "body must be a JSON object"})
        job = self.jobs.submit(body.get("input"))
        if operation == "run":
            return self._send(200, {"id": job["id"], "status": "IN_QUEUE"})
        job["done"].wait(RUNSYNC_WAIT_SECONDS)
        return self._send(200, Jobs.public(job))

    def _read_json(self) -> object:
        length = int(self.headers.get("Content-Length", "0"))
        try:
            return json.loads(self.rfile.read(length) or b"null")
        except ValueError:
            return None

    def _send(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        if os.environ.get("DEVSERVER_LOG"):
            super().log_message(format, *args)


def serve(
    host: str = "127.0.0.1", port: int = 0, *, workers: int, run: Callable[[object], dict]
) -> tuple[ThreadingHTTPServer, threading.Thread]:
    server = ThreadingHTTPServer((host, port), partial(Handler, jobs=Jobs(workers, run)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


if __name__ == "__main__":
    from postiz_uploader import pot
    from postiz_uploader.config import get_settings
    from postiz_uploader.log import setup_logging
    from postiz_uploader.pipeline import process

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    setup_logging()
    settings = get_settings()
    # same warm-up as handler.py, ingest jobs wait for it if needed
    pot.ensure(settings, wait=False)
    server, thread = serve(args.host, args.port, workers=settings.worker_concurrency, run=process)
    print(f"RunPod stand-in on http://{args.host}:{args.port}, {settings.worker_concurrency} worker(s)", flush=True)
    try:
        thread.join()
    except KeyboardInterrupt:
        server.shutdown()
