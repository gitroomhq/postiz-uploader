from __future__ import annotations

import os
from urllib.parse import urlsplit

import requests

from postiz_uploader import errors
from postiz_uploader.errors import JobError

CHUNK = 1024 * 1024
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 60


def host_allowed(url: str, allowed: tuple[str, ...]) -> bool:
    """Exact hostname match, or `*.suffix` wildcard match (the suffix itself excluded)."""
    host = (urlsplit(url).hostname or "").lower()
    if not host:
        return False
    for pattern in allowed:
        if pattern.startswith("*."):
            if host.endswith(pattern[1:]) and host != pattern[2:]:
                return True
        elif host == pattern:
            return True
    return False


def download(url: str, dest: str, *, max_bytes: int, deadline: float | None = None) -> int:
    """Stream `url` to `dest`. Abort as soon as the byte count passes `max_bytes`.

    `deadline` is an absolute time.monotonic() value; the read loop checks it so a
    slow origin cannot stretch a job past the job timeout.
    """
    import time

    try:
        # redirects are not followed: the allowlist was checked against this exact host
        response = requests.get(url, stream=True, allow_redirects=False, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
    except requests.RequestException as err:
        raise JobError(errors.DOWNLOAD_FAILED, f"fetch failed: {err.__class__.__name__}", retryable=True) from err

    with response:
        if 300 <= response.status_code < 400:
            raise JobError(
                errors.DOWNLOAD_FAILED, f"source redirected (HTTP {response.status_code}); redirects are not followed"
            )
        if response.status_code >= 400:
            raise JobError(
                errors.DOWNLOAD_FAILED,
                f"source returned HTTP {response.status_code}",
                retryable=response.status_code >= 500 or response.status_code == 429,
            )

        declared = response.headers.get("Content-Length")
        if declared and declared.isdigit() and int(declared) > max_bytes:
            raise JobError(errors.INPUT_TOO_LARGE, f"declared size {declared} exceeds {max_bytes}")

        received = 0
        try:
            with open(dest, "wb") as fh:
                for chunk in response.iter_content(chunk_size=CHUNK):
                    if not chunk:
                        continue
                    received += len(chunk)
                    if received > max_bytes:
                        raise JobError(errors.INPUT_TOO_LARGE, f"source exceeds {max_bytes} bytes")
                    if deadline is not None and time.monotonic() > deadline:
                        raise JobError(errors.TIMEOUT, "job timeout during download", retryable=True)
                    fh.write(chunk)
        except requests.RequestException as err:
            raise JobError(errors.DOWNLOAD_FAILED, f"read failed: {err.__class__.__name__}", retryable=True) from err

    if received == 0:
        raise JobError(errors.DOWNLOAD_FAILED, "source is empty")

    return os.path.getsize(dest)
