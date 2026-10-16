"""The RunPod stand-in answers the way the real /run and /status do."""

from __future__ import annotations

import os
import shutil
import threading
import time

import pytest
import requests

from postiz_uploader.devapi import serve
from postiz_uploader.pipeline import process
from postiz_uploader.schema import validate_result
from tests.conftest import make_job, needs_ffmpeg


@pytest.fixture
def api():
    gate = threading.Event()

    def run(job_input):
        if job_input == "raise":
            raise RuntimeError("handler blew up")
        if job_input == "wait":
            gate.wait(10)
        return {"echo": job_input}

    server, _ = serve(workers=1, run=run)
    host, port = server.server_address[:2]
    yield {"base": f"http://{host}:{port}/v2/endpoint", "gate": gate}
    gate.set()
    server.shutdown()


def _status(api, job_id):
    return requests.get(f"{api['base']}/status/{job_id}", timeout=5)


def _until(api, job_id, status):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        body = _status(api, job_id).json()
        if body["status"] == status:
            return body
        time.sleep(0.05)
    raise AssertionError(f"{job_id} never reached {status}")


def test_run_answers_before_the_job_finishes(api):
    started = time.monotonic()
    body = requests.post(f"{api['base']}/run", json={"input": "wait"}, timeout=5).json()
    assert time.monotonic() - started < 2
    assert body == {"id": body["id"], "status": "IN_QUEUE"}

    _until(api, body["id"], "IN_PROGRESS")
    api["gate"].set()
    done = _until(api, body["id"], "COMPLETED")
    assert done["output"] == {"echo": "wait"}
    assert "executionTime" in done and "delayTime" in done
    # status only reports: asking again returns the same result
    assert _status(api, body["id"]).json() == done


def test_jobs_queue_behind_the_worker_limit(api):
    first = requests.post(f"{api['base']}/run", json={"input": "wait"}, timeout=5).json()
    second = requests.post(f"{api['base']}/run", json={"input": "next"}, timeout=5).json()
    _until(api, first["id"], "IN_PROGRESS")
    assert _status(api, second["id"]).json()["status"] == "IN_QUEUE"
    api["gate"].set()
    assert _until(api, second["id"], "COMPLETED")["output"] == {"echo": "next"}


def test_a_raising_handler_fails_the_job(api):
    body = requests.post(f"{api['base']}/run", json={"input": "raise"}, timeout=5).json()
    failed = _until(api, body["id"], "FAILED")
    assert failed["error"] == "handler blew up"
    assert "output" not in failed


def test_runsync_waits_for_the_result(api):
    body = requests.post(f"{api['base']}/runsync", json={"input": "now"}, timeout=5).json()
    assert body["status"] == "COMPLETED" and body["output"] == {"echo": "now"}


def test_paths_without_the_endpoint_prefix(api):
    bare = api["base"].removesuffix("/v2/endpoint")
    body = requests.post(f"{bare}/runsync", json={"input": "bare"}, timeout=5).json()
    assert body["output"] == {"echo": "bare"}


def test_unknown_jobs_and_wrong_methods(api):
    assert _status(api, "local-missing").status_code == 404
    assert requests.post(f"{api['base']}/status/local-missing", timeout=5).status_code == 405
    assert requests.get(f"{api['base']}/run", timeout=5).status_code == 405
    assert requests.post(f"{api['base']}/run", data="not json", timeout=5).status_code == 400
    assert requests.post(f"{api['base']}/cancel/x", timeout=5).status_code == 404


@needs_ffmpeg
def test_a_real_job_through_the_stand_in(bucket, fixtures_dir, work_dir):
    shutil.copy(os.path.join(fixtures_dir, "compliant-1080p.mp4"), os.path.join(bucket["root"], "devapi.mp4"))
    job = make_job(
        type_="video",
        source_url=f"{bucket['base']}/devapi.mp4",
        output_url=f"{bucket['base']}/out/devapi.mp4",
        output_type="video/mp4",
    )
    server, _ = serve(workers=1, run=process)
    host, port = server.server_address[:2]
    base = f"http://{host}:{port}/v2/endpoint"
    try:
        job_id = requests.post(f"{base}/run", json={"input": job}, timeout=5).json()["id"]
        deadline = time.monotonic() + 120
        body = requests.get(f"{base}/status/{job_id}", timeout=5).json()
        while body["status"] in ("IN_QUEUE", "IN_PROGRESS"):
            assert time.monotonic() < deadline
            time.sleep(0.2)
            body = requests.get(f"{base}/status/{job_id}", timeout=5).json()
    finally:
        server.shutdown()
    assert body["status"] == "COMPLETED"
    validate_result(body["output"])
    assert body["output"]["status"] == "unchanged"
