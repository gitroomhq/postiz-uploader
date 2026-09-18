"""yt-dlp as a subprocess: one metadata pass, one download pass that reuses it.

A subprocess (not the Python API) for the same reason ffmpeg is one: a hard SIGKILL
timeout, and nothing yt-dlp does can take the worker down with it.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from dataclasses import dataclass

from postiz_uploader import errors, ffmpeg
from postiz_uploader.errors import JobError

DESCRIPTION_MAX = 5000
METADATA_TIMEOUT = 120

# matched against yt-dlp's stderr, lower-cased; first hit wins
_BLOCKED = ("sign in to confirm", "not a bot", "http error 429", "too many requests", "http error 403", "captcha")
_UNAVAILABLE = (
    "private video",
    "video unavailable",
    "has been removed",
    "members-only",
    "members only",
    "confirm your age",
    "age-restricted",
    "not available in your country",
    "not made this video available",
    "premieres in",
    "live event will begin",
    "account associated with this video has been terminated",
    "copyright",
)
_UNSUPPORTED = ("unsupported url", "no video formats found", "no media found")


@dataclass(frozen=True)
class Metadata:
    info_path: str
    extractor: str | None
    id: str | None
    title: str | None
    description: str | None
    uploader: str | None
    webpage_url: str | None
    thumbnail_url: str | None
    language: str | None
    duration_seconds: float | None
    is_live: bool
    bytes_estimate: int | None


def _scrub(text: str, proxy: str | None) -> str:
    return text.replace(proxy, "<proxy>") if proxy else text


def classify(stderr: str) -> tuple[str, bool]:
    """(error code, retryable) for a failed yt-dlp run."""
    lowered = stderr.lower()
    if any(marker in lowered for marker in _BLOCKED):
        return errors.SOURCE_BLOCKED, True
    if any(marker in lowered for marker in _UNAVAILABLE):
        return errors.SOURCE_UNAVAILABLE, False
    if any(marker in lowered for marker in _UNSUPPORTED):
        return errors.UNSUPPORTED_INPUT, False
    return errors.DOWNLOAD_FAILED, True


def _base(proxy: str | None, max_height: int, ffmpeg_bin: str) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--ignore-config",
        "--no-cache-dir",
        "--no-playlist",
        "--no-progress",
        "--socket-timeout",
        "30",
        "--retries",
        "3",
        # closest rendition at or under max_height first, then prefer H.264/AAC so the
        # stored source plays in a browser and the merge below is a plain remux. Stereo,
        # because some clients also list a 5.1 AAC track that yt-dlp would rank higher.
        "-S",
        f"res:{max_height},vcodec:h264,acodec:aac,channels:2",
        "--merge-output-format",
        "mp4",
    ]
    if os.path.sep in ffmpeg_bin:
        cmd += ["--ffmpeg-location", ffmpeg_bin]
    if proxy:
        cmd += ["--proxy", proxy]
    return cmd


def _fail(outcome: ffmpeg.RunOutcome, proxy: str | None) -> JobError:
    tail = _scrub(outcome.stderr_tail, proxy)
    code, retryable = classify(tail)
    last = next((line for line in reversed(tail.strip().splitlines()) if line.strip()), "yt-dlp failed")
    return JobError(code, last.strip()[:500], retryable=retryable, stderr_tail=tail)


def _size(fmt: dict) -> int | None:
    value = fmt.get("filesize") or fmt.get("filesize_approx")
    return int(value) if isinstance(value, (int, float)) and value > 0 else None


# yt-dlp's logged-out default (visionos) takes no PO token. mweb does, and with one it
# lists the same https formats, so a player client YouTube breaks is not an outage.
POT_PLAYER_CLIENTS = "default,mweb"


def fetch_metadata(
    url: str,
    workdir: str,
    *,
    proxy: str | None,
    max_height: int,
    ffmpeg_bin: str,
    deadline: float,
    player_clients: str | None = None,
) -> Metadata:
    """Resolve the page without downloading media. Writes <workdir>/meta.info.json."""
    cmd = _base(proxy, max_height, ffmpeg_bin)
    if player_clients:
        cmd += ["--extractor-args", f"youtube:player_client={player_clients}"]
    cmd += [
        "--skip-download",
        "--write-info-json",
        "--no-write-comments",
        "-o",
        os.path.join(workdir, "meta"),
        url,
    ]
    outcome = ffmpeg.run(cmd, timeout=min(METADATA_TIMEOUT, deadline - time.monotonic()))
    info_path = os.path.join(workdir, "meta.info.json")
    if outcome.returncode != 0 or not os.path.exists(info_path):
        raise _fail(outcome, proxy)

    with open(info_path, encoding="utf-8") as fh:
        info = json.load(fh)

    selected = info.get("requested_formats") or [info]
    sizes = [_size(fmt) for fmt in selected]
    duration = info.get("duration")
    description = info.get("description")
    return Metadata(
        info_path=info_path,
        extractor=info.get("extractor_key") or info.get("extractor"),
        id=str(info["id"]) if info.get("id") is not None else None,
        title=info.get("title"),
        description=description[:DESCRIPTION_MAX] if isinstance(description, str) else None,
        uploader=info.get("uploader") or info.get("channel"),
        webpage_url=info.get("webpage_url"),
        thumbnail_url=info.get("thumbnail"),
        language=info.get("language"),
        duration_seconds=float(duration) if isinstance(duration, (int, float)) else None,
        is_live=bool(info.get("is_live")) or info.get("live_status") in ("is_live", "is_upcoming"),
        bytes_estimate=sum(s for s in sizes if s) if all(sizes) else None,
    )


def download_media(
    meta: Metadata,
    workdir: str,
    *,
    proxy: str | None,
    max_height: int,
    max_bytes: int,
    ffmpeg_bin: str,
    deadline: float,
) -> str:
    """Download what fetch_metadata resolved. Returns the path of the merged file."""
    cmd = _base(proxy, max_height, ffmpeg_bin) + [
        "--load-info-json",
        meta.info_path,
        # per stream, so it cannot bound the merged file; the size check after does
        "--max-filesize",
        str(max_bytes),
        "-o",
        os.path.join(workdir, "media.%(ext)s"),
    ]
    outcome = ffmpeg.run(cmd, timeout=deadline - time.monotonic())
    if outcome.returncode != 0:
        raise _fail(outcome, proxy)

    files = [p for p in glob.glob(os.path.join(workdir, "media.*")) if not p.endswith((".part", ".ytdl"))]
    if len(files) != 1:
        # --max-filesize skips the stream and still exits 0
        raise JobError(errors.INPUT_TOO_LARGE, f"source exceeds {max_bytes} bytes or produced no single file")
    if os.path.getsize(files[0]) > max_bytes:
        raise JobError(errors.INPUT_TOO_LARGE, f"source exceeds {max_bytes} bytes")
    return files[0]
