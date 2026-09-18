"""Oxylabs as the YouTube route: they talk to YouTube, this worker never does.

Three calls, all keyed by video id:
  youtube_metadata   realtime; title and duration, so limits are checked before paying
  youtube_subtitles  realtime; YouTube's own captions, which can replace transcription
  youtube_download   async; Oxylabs writes the file into an S3-compatible bucket
                     (OXYLABS_STORAGE_URL) and this module fetches and deletes it

Downloads are billed per GB, which is why the caller asks for captions first, audio
only when there are none, and video in trimmed segments.
"""

from __future__ import annotations

import datetime
import hashlib
import hmac
import json
import math
import re
import time
from dataclasses import dataclass
from urllib.parse import parse_qs, quote, unquote, urlsplit

import requests

from postiz_uploader import errors
from postiz_uploader.config import Settings
from postiz_uploader.errors import JobError
from postiz_uploader.log import get_logger, log

logger = get_logger(__name__)

REALTIME_URL = "https://realtime.oxylabs.io/v1/queries"
QUERIES_URL = "https://data.oxylabs.io/v1/queries"
CONNECT_TIMEOUT = 10
REALTIME_TIMEOUT = 150
POLL_SECONDS = 4
# "done" is reported a moment before the object is readable on some stores
DELIVERY_WAIT_SECONDS = 30
QUALITIES = (144, 360, 480, 720, 1080, 1440, 2160, 4320)
ORIGINS = ("auto_generated", "uploader_provided")
# a spoken word held on screen through a pause reads as a frozen caption
MAX_WORD_SECONDS = 2.0

# result status codes of the YouTube sources that mean "this video, from anywhere"
_UNAVAILABLE = {
    11201: "video is deleted",
    11203: "video is private",
    11204: "video is geo-restricted",
    11205: "video is live",
    11206: "video is age-restricted",
    11207: "video is for channel members only",
    11208: "video requires YouTube Premium",
}
_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


def configured(settings: Settings) -> bool:
    return bool(settings.oxylabs_username and settings.oxylabs_password)


def video_id(url: str) -> str:
    """The 11 character id out of a watch, short, shorts, live or embed URL."""
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    segments = [s for s in parts.path.split("/") if s]
    candidate = None
    if host == "youtu.be":
        candidate = segments[0] if segments else None
    elif segments and segments[0] in ("shorts", "live", "embed", "v") and len(segments) > 1:
        candidate = segments[1]
    else:
        candidate = (parse_qs(parts.query).get("v") or [None])[0]
    if not candidate or not _VIDEO_ID.match(candidate):
        raise JobError(errors.UNSUPPORTED_INPUT, "source.url is not a YouTube video URL")
    return candidate


# ---------------------------------------------------------------- API


def _request(settings: Settings, method: str, url: str, payload: dict | None, timeout: float) -> dict:
    try:
        response = requests.request(
            method,
            url,
            json=payload,
            auth=(settings.oxylabs_username, settings.oxylabs_password),
            timeout=(CONNECT_TIMEOUT, max(timeout, 1)),
        )
    except requests.RequestException as err:
        raise JobError(
            errors.DOWNLOAD_FAILED, f"oxylabs request failed: {err.__class__.__name__}", retryable=True
        ) from err
    if response.status_code in (401, 403):
        # the worker's own login: no other worker and no retry will do better
        raise JobError(
            errors.DOWNLOAD_FAILED, f"oxylabs rejected the worker's credentials (HTTP {response.status_code})"
        )
    if response.status_code == 429 or response.status_code >= 500:
        raise JobError(errors.DOWNLOAD_FAILED, f"oxylabs returned HTTP {response.status_code}", retryable=True)
    try:
        body = response.json()
    except ValueError:
        body = {}
    if response.status_code >= 400:
        message = str(body.get("message") or f"HTTP {response.status_code}")[:300]
        raise JobError(errors.DOWNLOAD_FAILED, f"oxylabs refused the request: {message}")
    return body


def _realtime(settings: Settings, payload: dict, deadline: float) -> tuple[int, object]:
    """(result status code, content) of a synchronous query."""
    body = _request(settings, "POST", REALTIME_URL, payload, min(REALTIME_TIMEOUT, deadline - time.monotonic()))
    result = (body.get("results") or [{}])[0]
    return int(result.get("status_code") or 0), result.get("content")


def _raise_unavailable(status: int) -> None:
    if status in _UNAVAILABLE:
        raise JobError(errors.SOURCE_UNAVAILABLE, _UNAVAILABLE[status])


def _number(value: object) -> float | None:
    """Oxylabs sends numbers as strings ("19")."""
    try:
        return float(value) if value is not None and not isinstance(value, bool) else None
    except (TypeError, ValueError):
        return None


def _flag(value: object) -> bool | None:
    """...and booleans as "True"/"False", where bool("False") would be True."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    return None


def _thumbnail(data: dict) -> str | None:
    thumbs = [t for t in data.get("thumbnails") or [] if isinstance(t, dict) and isinstance(t.get("url"), str)]
    if thumbs:
        return max(thumbs, key=lambda t: _number(t.get("width")) or 0)["url"]
    return data.get("thumbnail") if isinstance(data.get("thumbnail"), str) else None


def metadata(settings: Settings, vid: str, deadline: float) -> dict:
    status, content = _realtime(settings, {"source": "youtube_metadata", "query": vid, "parse": True}, deadline)
    _raise_unavailable(status)
    if status == 404:
        raise JobError(errors.SOURCE_UNAVAILABLE, "video not found")
    if status != 200 or not isinstance(content, dict):
        raise JobError(errors.DOWNLOAD_FAILED, f"oxylabs metadata returned status {status}", retryable=True)
    data = content.get("results") if isinstance(content.get("results"), dict) else content
    uploaded = data.get("user_subtitle_languages")
    return {
        "title": data.get("title"),
        "description": data.get("description"),
        "uploader": data.get("uploader") or data.get("channel"),
        "thumbnail_url": _thumbnail(data),
        "duration_seconds": _number(data.get("duration")),
        "is_live": _flag(data.get("is_live")) is True,
        # None when the field is missing: then captions are simply asked for
        "captions_available": _flag(data.get("is_transcript_available")),
        "uploaded_languages": tuple(x for x in uploaded if isinstance(x, str)) if isinstance(uploaded, list) else None,
    }


# ---------------------------------------------------------------- captions


@dataclass(frozen=True)
class Transcript:
    language: str
    origin: str
    word_level: bool
    segments: list[dict]
    words: list[dict]

    def to_json(self) -> dict:
        return {
            "version": 1,
            "language": self.language,
            "origin": self.origin,
            "word_level": self.word_level,
            "segments": self.segments,
            "words": self.words,
        }


def _events(content: object, depth: int = 0) -> list[dict] | None:
    """YouTube's json3 caption events. Oxylabs nests them as {origin: {language: json3}},
    so the search is by shape, not by key."""
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except ValueError:
            return None
    if depth > 6:
        return None
    if isinstance(content, dict):
        events = content.get("events")
        if isinstance(events, list) and events:
            return [e for e in events if isinstance(e, dict)]
        children = list(content.values())
    elif isinstance(content, list):
        children = content
    else:
        return None
    for child in children:
        if isinstance(child, (dict, list, str)):
            found = _events(child, depth + 1)
            if found:
                return found
    return None


def parse_captions(content: object) -> tuple[list[dict], list[dict], bool] | None:
    """(segments, words, word_level) from json3 events, or None when there is no text.

    Auto-generated tracks time every word (`tOffsetMs` inside the event); uploaded
    tracks time lines only. Word timings are never invented: without offsets `words`
    stays empty and the caller decides how to caption.
    """
    events = _events(content)
    if not events:
        return None
    timed: list[tuple[float, float, list[tuple[float, str]]]] = []
    word_level = False
    for event in events:
        segs = event.get("segs")
        start_ms = event.get("tStartMs")
        if not isinstance(segs, list) or not isinstance(start_ms, (int, float)):
            continue
        pieces = []
        for seg in segs:
            text = seg.get("utf8") if isinstance(seg, dict) else None
            if not isinstance(text, str) or not text.strip():
                continue
            if "tOffsetMs" in seg:
                word_level = True
            pieces.append(((start_ms + (seg.get("tOffsetMs") or 0)) / 1000, " ".join(text.split())))
        if pieces:
            end = (start_ms + (event.get("dDurationMs") or 0)) / 1000
            timed.append((start_ms / 1000, end, pieces))
    if not timed:
        return None

    words: list[dict] = []
    if word_level:
        flat = [(start, text, end) for _, end, pieces in timed for start, text in pieces]
        for i, (start, text, event_end) in enumerate(flat):
            # rolling captions overlap the next event, so the next word is the real end
            following = flat[i + 1][0] if i + 1 < len(flat) else event_end
            end = min(max(following, start), start + MAX_WORD_SECONDS)
            for token in text.split():
                words.append({"text": token, "start": round(start, 3), "end": round(max(end, start + 0.01), 3)})

    segments = []
    for i, (start, end, pieces) in enumerate(timed):
        if word_level and i + 1 < len(timed):
            end = min(end, timed[i + 1][0])
        segments.append(
            {"start": round(start, 3), "end": round(max(end, start), 3), "text": " ".join(t for _, t in pieces)}
        )
    return segments, words, word_level


def transcript(
    settings: Settings,
    vid: str,
    languages: tuple[str, ...],
    deadline: float,
    *,
    available: bool | None = None,
    uploaded_languages: tuple[str, ...] | None = None,
) -> Transcript | None:
    """YouTube's captions in the first language that has any, or None.

    Auto-generated first: it is the track with word timings. Each miss is one cheap
    request, so the metadata's hints are used to skip the ones that cannot hit.
    """
    if available is False:
        return None
    for language in languages:
        for origin in ORIGINS:
            if origin == "uploader_provided" and uploaded_languages is not None:
                if language not in uploaded_languages:
                    continue
            status, content = _realtime(
                settings,
                {
                    "source": "youtube_subtitles",
                    "query": vid,
                    "context": [
                        {"key": "language_code", "value": language},
                        {"key": "subtitle_origin", "value": origin},
                    ],
                },
                deadline,
            )
            _raise_unavailable(status)
            if status != 200:
                continue
            parsed = parse_captions(content)
            if parsed is None:
                log(logger, "oxylabs captions had no readable events", language=language, origin=origin)
                continue
            segments, words, word_level = parsed
            return Transcript(language, origin, word_level, segments, words)
    return None


# ---------------------------------------------------------------- download


def quality(max_height: int, cap: int) -> str:
    """The tallest rendition Oxylabs offers at or under both limits."""
    limit = min(max_height, cap)
    return str(max((q for q in QUALITIES if q <= limit), default=QUALITIES[0]))


def clock(seconds: float) -> str:
    whole = int(seconds)
    return f"{whole // 3600:02d}:{whole % 3600 // 60:02d}:{whole % 60:02d}"


def trim_window(start: float | None, end: float | None) -> tuple[int, int] | None:
    """Whole seconds that contain [start, end]: the API takes hh:mm:ss only."""
    if start is None and end is None:
        return None
    if end is None:
        raise JobError(errors.INVALID_JOB, "source.end_seconds is required with start_seconds")
    first = int(math.floor(start or 0))
    last = int(math.ceil(end))
    if last <= first:
        raise JobError(errors.INVALID_JOB, "source.end_seconds must be after start_seconds")
    return first, last


def submit_download(
    settings: Settings, vid: str, *, audio_only: bool, max_height: int, window: tuple[int, int] | None
) -> str:
    context = [{"key": "download_type", "value": "audio" if audio_only else "audio_video"}]
    if not audio_only:
        context.append({"key": "video_quality", "value": quality(max_height, settings.oxylabs_max_height)})
    if window:
        context += [{"key": "start_at", "value": clock(window[0])}, {"key": "end_at", "value": clock(window[1])}]
    body = _request(
        settings,
        "POST",
        QUERIES_URL,
        {
            "source": "youtube_download",
            "query": vid,
            "context": context,
            "storage_type": "s3_compatible",
            "storage_url": settings.oxylabs_storage_url,
        },
        60,
    )
    job_id = body.get("id")
    if not job_id:
        raise JobError(errors.DOWNLOAD_FAILED, "oxylabs accepted the download without a job id", retryable=True)
    return str(job_id)


def wait_download(settings: Settings, job_id: str, deadline: float) -> None:
    while True:
        if time.monotonic() + POLL_SECONDS > deadline:
            raise JobError(errors.TIMEOUT, "job timeout while oxylabs was downloading", retryable=True)
        time.sleep(POLL_SECONDS)
        body = _request(settings, "GET", f"{QUERIES_URL}/{job_id}", None, 30)
        status = body.get("status")
        if status == "done":
            return
        if status == "faulted":
            for entry in body.get("statuses") or []:
                code = entry.get("status_code") if isinstance(entry, dict) else None
                if isinstance(code, int):
                    _raise_unavailable(code)
            raise JobError(errors.DOWNLOAD_FAILED, "oxylabs could not download the video", retryable=True)


# ---------------------------------------------------------------- storage


@dataclass(frozen=True)
class Storage:
    endpoint: str  # scheme://host[:port]
    access_key: str
    secret_key: str
    bucket: str
    prefix: str
    region: str


def parse_storage(url: str, region: str) -> Storage:
    parts = urlsplit(url)
    path = [s for s in parts.path.split("/") if s]
    if not (parts.scheme and parts.hostname and parts.username and parts.password and len(path) >= 2):
        # Oxylabs cannot write to a bucket root, so a folder is part of the contract
        raise RuntimeError("OXYLABS_STORAGE_URL must be https://KEY:SECRET@host/bucket/folder")
    host = parts.hostname + (f":{parts.port}" if parts.port else "")
    return Storage(
        endpoint=f"{parts.scheme}://{host}",
        access_key=unquote(parts.username),
        secret_key=unquote(parts.password),
        bucket=path[0],
        prefix="/".join(path[1:]),
        region=region,
    )


def _sign(key: bytes, message: str) -> bytes:
    return hmac.new(key, message.encode(), hashlib.sha256).digest()


def presign(
    storage: Storage, method: str, key: str, *, expires: int = 900, now: datetime.datetime | None = None
) -> str:
    """SigV4 query-string URL, path style. Small enough not to be worth an SDK."""
    now = now or datetime.datetime.now(datetime.timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    scope = f"{stamp[:8]}/{storage.region}/s3/aws4_request"
    host = urlsplit(storage.endpoint).netloc
    path = "/" + quote(f"{storage.bucket}/{key}", safe="/-_.~")
    query = "&".join(
        f"{quote(k, safe='')}={quote(v, safe='')}"
        for k, v in sorted(
            {
                "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
                "X-Amz-Credential": f"{storage.access_key}/{scope}",
                "X-Amz-Date": stamp,
                "X-Amz-Expires": str(expires),
                "X-Amz-SignedHeaders": "host",
            }.items()
        )
    )
    canonical = "\n".join([method, path, query, f"host:{host}\n", "host", "UNSIGNED-PAYLOAD"])
    to_sign = "\n".join(["AWS4-HMAC-SHA256", stamp, scope, hashlib.sha256(canonical.encode()).hexdigest()])
    signing_key = _sign(
        _sign(_sign(_sign(f"AWS4{storage.secret_key}".encode(), stamp[:8]), storage.region), "s3"), "aws4_request"
    )
    signature = hmac.new(signing_key, to_sign.encode(), hashlib.sha256).hexdigest()
    return f"{storage.endpoint}{path}?{query}&X-Amz-Signature={signature}"


def delivered_url(storage: Storage, vid: str, job_id: str, *, audio_only: bool, deadline: float) -> tuple[str, str]:
    """(presigned GET, object key) of what the job wrote.

    Named `{video id}_{job id}` plus an extension that the docs and the service
    disagree on for audio, so every known one is probed.
    """
    extensions = ("aac", "m4a", "mp4") if audio_only else ("mp4",)
    give_up = min(time.monotonic() + DELIVERY_WAIT_SECONDS, deadline)
    while True:
        for extension in extensions:
            key = f"{storage.prefix}/{vid}_{job_id}.{extension}"
            try:
                head = requests.head(presign(storage, "HEAD", key), timeout=(CONNECT_TIMEOUT, 30))
            except requests.RequestException:
                continue
            if head.status_code == 200:
                return presign(storage, "GET", key), key
            if head.status_code in (401, 403):
                raise JobError(errors.DOWNLOAD_FAILED, "the oxylabs bucket rejected the worker's storage key")
        if time.monotonic() > give_up:
            raise JobError(
                errors.DOWNLOAD_FAILED, "oxylabs reported done but the file is not in the bucket", retryable=True
            )
        time.sleep(2)


def discard(storage: Storage, key: str) -> None:
    """Best effort: a bucket lifecycle rule is the real cleanup."""
    try:
        requests.delete(presign(storage, "DELETE", key), timeout=(CONNECT_TIMEOUT, 30))
    except requests.RequestException:
        pass
