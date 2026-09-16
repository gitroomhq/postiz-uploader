from __future__ import annotations

import os
import time

import requests

from postiz_uploader import errors
from postiz_uploader.errors import JobError

ATTEMPTS = 3
BACKOFF_SECONDS = (1, 3)
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 120


def upload(path: str, url: str, content_type: str) -> int:
    """PUT a file from disk to a presigned URL. Retries inside the job; PUT is idempotent."""
    size = os.path.getsize(path)
    headers = {"Content-Type": content_type, "Content-Length": str(size)}
    last: str = ""
    for attempt in range(ATTEMPTS):
        try:
            with open(path, "rb") as fh:
                response = requests.put(url, data=fh, headers=headers, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
            if response.status_code < 300:
                return size
            last = f"HTTP {response.status_code}"
            # a 4xx other than 408/429 will not change on retry (expired signature, wrong key)
            if 400 <= response.status_code < 500 and response.status_code not in (408, 429):
                break
        except requests.RequestException as err:
            last = err.__class__.__name__
        if attempt < ATTEMPTS - 1:
            time.sleep(BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)])
    raise JobError(errors.UPLOAD_FAILED, f"upload failed after {ATTEMPTS} attempts: {last}", retryable=True)
