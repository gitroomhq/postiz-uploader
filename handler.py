"""RunPod Serverless entrypoint.

python handler.py                  # production worker loop
python handler.py --rp_serve_api   # local HTTP API on :8000 (RunPod SDK flag)
"""

from __future__ import annotations

import asyncio

import runpod

from postiz_uploader.config import get_settings
from postiz_uploader.log import setup_logging
from postiz_uploader.pipeline import process


async def handler(event: dict) -> dict:
    # The SDK runs a sync handler on its event loop, which would serialize every job
    # behind one ffmpeg. A thread per job is what lets WORKER_CONCURRENCY mean anything.
    return await asyncio.to_thread(process, event.get("input") if isinstance(event, dict) else None)


def concurrency_modifier(current_concurrency: int) -> int:
    return get_settings().worker_concurrency


def _init_sentry() -> None:
    dsn = get_settings().sentry_dsn
    if not dsn:
        return
    import sentry_sdk

    sentry_sdk.init(dsn=dsn, traces_sample_rate=0.0, send_default_pii=False)


if __name__ == "__main__":
    setup_logging()
    _init_sentry()
    runpod.serverless.start({"handler": handler, "concurrency_modifier": concurrency_modifier})
