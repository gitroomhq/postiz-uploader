"""Ingest stage: fetch a long source once -> faststart MP4, speech-grade audio, captions."""

from __future__ import annotations

import json
import os
import random
import time
from urllib.parse import urlsplit

from postiz_uploader import errors, ffmpeg, oxylabs, pot, ytdlp
from postiz_uploader.config import Settings
from postiz_uploader.download import download, host_allowed
from postiz_uploader.errors import JobError
from postiz_uploader.log import get_logger, log, redact_url
from postiz_uploader.probe import SourceInfo, probe
from postiz_uploader.schema import IngestJob, VideoRules
from postiz_uploader.sniff import sniff
from postiz_uploader.upload import upload

logger = get_logger(__name__)
SNIFF_BYTES = 512


def _remaining(deadline: float) -> float:
    return deadline - time.monotonic()


def _check_duration(duration: float | None, job: IngestJob) -> None:
    if duration is not None and duration > job.limits.max_duration_seconds:
        raise JobError(
            errors.DURATION_TOO_LONG, f"duration {duration:.1f}s exceeds {job.limits.max_duration_seconds}s"
        )


def _routes(job: IngestJob, settings: Settings) -> list[str | None]:
    """The order yt-dlp tries: this worker's own address, then proxies. None is direct.

    The pool is sampled, not walked in order: the worker keeps no state between jobs,
    so random choice is what spreads load across the addresses and keeps one flagged
    proxy from failing every job. A job's own proxy replaces the pool.
    """
    if job.source.proxy:
        proxies = [job.source.proxy]
    else:
        pool = list(settings.ingest_proxies)
        proxies = random.sample(pool, min(settings.ingest_proxy_attempts, len(pool)))
    # bandwidth through a proxy is the expensive part of an ingest, so it is only
    # used once the platform has refused this worker's own address
    direct: list[str | None] = [None] if settings.ingest_direct_first or not proxies else []
    return direct + proxies


def _route_name(route: str | None) -> str:
    """Host of a route for logs; never the credentials."""
    return (urlsplit(route).hostname or "proxy") if route else "direct"


def _fetch_ytdlp(job: IngestJob, workdir: str, settings: Settings, deadline: float, max_bytes: int, partial: dict):
    if not host_allowed(job.source.url, settings.allowed_ingest_hosts):
        raise JobError(
            errors.SOURCE_HOST_NOT_ALLOWED, f"host of source.url is not allowed: {redact_url(job.source.url)}"
        )
    routes = _routes(job, settings)
    # without the token server mweb formats are unusable and only cost requests
    player_clients = ytdlp.POT_PLAYER_CLIENTS if pot.ensure(settings) else None

    for i, route in enumerate(routes):
        try:
            meta = ytdlp.fetch_metadata(
                job.source.url,
                workdir,
                proxy=route,
                max_height=job.source.max_height,
                ffmpeg_bin=settings.ffmpeg_bin,
                deadline=deadline,
                player_clients=player_clients,
            )
            partial["source"] = {
                "via": "ytdlp",
                "extractor": meta.extractor,
                "id": meta.id,
                "title": meta.title,
                "description": meta.description,
                "uploader": meta.uploader,
                "webpage_url": meta.webpage_url,
                "thumbnail_url": meta.thumbnail_url,
                "language": meta.language,
                "duration_seconds": meta.duration_seconds,
                "proxied": route is not None,
            }
            if meta.is_live:
                raise JobError(errors.UNSUPPORTED_INPUT, "live and upcoming streams cannot be ingested")
            _check_duration(meta.duration_seconds, job)
            if meta.bytes_estimate and meta.bytes_estimate > max_bytes:
                raise JobError(errors.INPUT_TOO_LARGE, f"estimated size {meta.bytes_estimate} exceeds {max_bytes}")
            return ytdlp.download_media(
                meta,
                workdir,
                proxy=route,
                max_height=job.source.max_height,
                max_bytes=max_bytes,
                ffmpeg_bin=settings.ffmpeg_bin,
                deadline=deadline,
            )
        except JobError as err:
            if err.code != errors.SOURCE_BLOCKED or i + 1 == len(routes):
                raise
            log(
                logger,
                "blocked, trying the next route",
                reference=job.reference,
                blocked=_route_name(route),
                next=_route_name(routes[i + 1]),
            )
    raise AssertionError("unreachable")


def _fetch_direct(job: IngestJob, workdir: str, settings: Settings, deadline: float, max_bytes: int, partial: dict):
    if not host_allowed(job.source.url, settings.allowed_source_hosts):
        raise JobError(
            errors.SOURCE_HOST_NOT_ALLOWED, f"host of source.url is not allowed: {redact_url(job.source.url)}"
        )
    path = os.path.join(workdir, "input")
    download(job.source.url, path, max_bytes=max_bytes, deadline=deadline)
    with open(path, "rb") as fh:
        sniffed = sniff(fh.read(SNIFF_BYTES))
    if sniffed is None or sniffed.kind != "video":
        raise JobError(errors.UNSUPPORTED_INPUT, "source is not a video file")
    partial["source"] = {"via": "direct", "proxied": False}
    return path


def _prepare_oxylabs(job: IngestJob, settings: Settings, deadline: float, partial: dict) -> dict:
    """Everything that is known before a byte is paid for: id, metadata, the window."""
    if not host_allowed(job.source.url, settings.allowed_ingest_hosts):
        raise JobError(
            errors.SOURCE_HOST_NOT_ALLOWED, f"host of source.url is not allowed: {redact_url(job.source.url)}"
        )
    if not oxylabs.configured(settings):
        raise JobError(errors.INVALID_JOB, "source.via oxylabs is not configured on this worker")
    vid = oxylabs.video_id(job.source.url)
    window = oxylabs.trim_window(job.source.start_seconds, job.source.end_seconds)
    meta = oxylabs.metadata(settings, vid, deadline)
    partial["source"] = {
        "via": "oxylabs",
        "extractor": "youtube",
        "id": vid,
        "title": meta["title"],
        "description": meta["description"][: ytdlp.DESCRIPTION_MAX] if isinstance(meta["description"], str) else None,
        "uploader": meta["uploader"],
        "webpage_url": f"https://www.youtube.com/watch?v={vid}",
        "thumbnail_url": meta["thumbnail_url"],
        "language": None,
        "duration_seconds": meta["duration_seconds"],
        "proxied": False,
        "trim": {"start_seconds": window[0], "end_seconds": window[1]} if window else None,
    }
    if meta["is_live"]:
        raise JobError(errors.UNSUPPORTED_INPUT, "live and upcoming streams cannot be ingested")
    # a window is what gets fetched and processed, so it is what the limit is about
    _check_duration(float(window[1] - window[0]) if window else meta["duration_seconds"], job)
    return {
        "id": vid,
        "window": window,
        "captions_available": meta["captions_available"],
        "uploaded_languages": meta["uploaded_languages"],
    }


def _fetch_oxylabs(
    job: IngestJob, workdir: str, settings: Settings, deadline: float, max_bytes: int, ctx: dict, *, audio_only: bool
) -> str:
    if not settings.oxylabs_storage_url:
        raise JobError(errors.INVALID_JOB, "OXYLABS_STORAGE_URL is not configured on this worker")
    try:
        storage = oxylabs.parse_storage(settings.oxylabs_storage_url, settings.oxylabs_storage_region)
    except RuntimeError as err:
        # the worker's configuration: retrying the job cannot fix it
        raise JobError(errors.INVALID_JOB, str(err)) from err
    job_id = oxylabs.submit_download(
        settings, ctx["id"], audio_only=audio_only, max_height=job.source.max_height, window=ctx["window"]
    )
    log(logger, "oxylabs download submitted", reference=job.reference, job_id=job_id, audio_only=audio_only)
    oxylabs.wait_download(settings, job_id, deadline)
    url, key = oxylabs.delivered_url(storage, ctx["id"], job_id, audio_only=audio_only, deadline=deadline)
    path = os.path.join(workdir, "input")
    try:
        download(url, path, max_bytes=max_bytes, deadline=deadline)
    finally:
        oxylabs.discard(storage, key)
    return path


def _windowed(transcript: oxylabs.Transcript, window: tuple[int, int] | None) -> dict:
    """The transcript file. With a window, only what falls inside it, on its clock."""
    data = transcript.to_json()
    if window:
        first, last = window
        for field in ("segments", "words"):
            data[field] = [
                {**item, "start": round(item["start"] - first, 3), "end": round(item["end"] - first, 3)}
                for item in data[field]
                if item["end"] > first and item["start"] < last
            ]
    return data


def _video_block(info: SourceInfo, content_type: str) -> dict:
    return {
        "width": info.width,
        "height": info.height,
        "duration_seconds": info.duration_seconds,
        "bytes": info.bytes,
        "content_type": content_type,
        "video_codec": info.video_codec,
        "audio_codec": info.audio_codec,
        "fps": info.fps,
    }


def process_ingest(
    job: IngestJob, workdir: str, settings: Settings, deadline: float, timing: dict, partial: dict
) -> dict:
    max_bytes = min(job.limits.max_input_bytes, settings.max_input_bytes_cap)

    ctx = None
    transcript_path = None
    transcript = None
    if job.source.via == "oxylabs":
        t = time.monotonic()
        ctx = _prepare_oxylabs(job, settings, deadline, partial)
        timing["metadata"] = int((time.monotonic() - t) * 1000)
        if job.transcript:
            t = time.monotonic()
            found = oxylabs.transcript(
                settings,
                ctx["id"],
                job.transcript.languages,
                deadline,
                available=ctx["captions_available"],
                uploaded_languages=ctx["uploaded_languages"],
            )
            timing["transcript"] = int((time.monotonic() - t) * 1000)
            if found:
                data = _windowed(found, ctx["window"])
                transcript_path = os.path.join(workdir, "transcript.json")
                with open(transcript_path, "w", encoding="utf-8") as fh:
                    json.dump(data, fh, ensure_ascii=False, separators=(",", ":"))
                transcript = {
                    "language": found.language,
                    "origin": found.origin,
                    "word_level": found.word_level,
                    "segments": len(data["segments"]),
                    "words": len(data["words"]),
                    "bytes": os.path.getsize(transcript_path),
                    "content_type": job.transcript.content_type,
                }
                partial["source"]["language"] = found.language
            log(logger, "captions", reference=job.reference, found=bool(found), origin=found.origin if found else None)

    # captions make the audio pointless, and with it the only reason to download at all
    want_audio = bool(job.audio) and not (job.audio.unless_transcript and transcript is not None)
    video_path = None
    video = None
    audio_path = None
    audio = None

    if job.video or want_audio:
        t = time.monotonic()
        if ctx is not None:
            media_path = _fetch_oxylabs(job, workdir, settings, deadline, max_bytes, ctx, audio_only=not job.video)
        else:
            fetch = _fetch_ytdlp if job.source.via == "ytdlp" else _fetch_direct
            media_path = fetch(job, workdir, settings, deadline, max_bytes, partial)
        timing["download"] = int((time.monotonic() - t) * 1000)

        t = time.monotonic()
        info = probe(settings.ffprobe_bin, media_path, timeout=min(60, max(_remaining(deadline), 1)))
        timing["probe"] = int((time.monotonic() - t) * 1000)
        known = partial["source"].get("duration_seconds")
        partial["source"].update(
            {
                # a window is shorter than the video the metadata described
                "duration_seconds": known if ctx is not None and known else info.duration_seconds or known,
                "width": info.width,
                "height": info.height,
                "fps": info.fps,
                "video_codec": info.video_codec,
                "audio_codec": info.audio_codec,
                "bytes": info.bytes,
            }
        )
        log(logger, "fetched", reference=job.reference, via=job.source.via, bytes=info.bytes)
        # pages that do not declare a duration are only caught here
        _check_duration(info.duration_seconds, job)
        if want_audio and not info.has_audio:
            raise JobError(errors.UNSUPPORTED_INPUT, "source has no audio track to extract")

        t = time.monotonic()
        if job.video:
            rules = VideoRules()
            if info.container == "mp4" and info.faststart is True:
                video_path, out_info = media_path, info
            else:
                video_path = os.path.join(workdir, "source.mp4")
                outcome = ffmpeg.run(
                    ffmpeg.build_remux(settings.ffmpeg_bin, media_path, video_path, info, rules),
                    timeout=_remaining(deadline),
                )
                if outcome.returncode != 0:
                    raise JobError(
                        errors.ENCODE_FAILED,
                        f"remux to mp4 failed with status {outcome.returncode}",
                        stderr_tail=outcome.stderr_tail,
                    )
                out_info = probe(settings.ffprobe_bin, video_path, timeout=min(60, max(_remaining(deadline), 1)))
            video = _video_block(out_info, job.video.content_type)

        if want_audio:
            audio_path = os.path.join(workdir, "audio.ogg")
            outcome = ffmpeg.run(
                ffmpeg.build_audio_extract(
                    settings.ffmpeg_bin,
                    media_path,
                    audio_path,
                    bitrate_kbps=job.audio.bitrate_kbps,
                    sample_rate=job.audio.sample_rate,
                ),
                timeout=_remaining(deadline),
            )
            if outcome.returncode != 0 or not os.path.exists(audio_path):
                raise JobError(
                    errors.ENCODE_FAILED,
                    f"audio extract failed with status {outcome.returncode}",
                    stderr_tail=outcome.stderr_tail,
                )
            audio = {
                "duration_seconds": info.duration_seconds,
                "bytes": os.path.getsize(audio_path),
                "content_type": job.audio.content_type,
                "codec": "opus",
                "bitrate_kbps": job.audio.bitrate_kbps,
                "sample_rate": job.audio.sample_rate,
            }
        timing["process"] = int((time.monotonic() - t) * 1000)

    if _remaining(deadline) <= 0:
        raise JobError(errors.TIMEOUT, "job timeout reached before upload", retryable=True)
    t = time.monotonic()
    if video_path and job.video:
        upload(video_path, job.video.url, job.video.content_type)
    if audio_path and job.audio:
        upload(audio_path, job.audio.url, job.audio.content_type)
    if transcript_path and job.transcript:
        upload(transcript_path, job.transcript.url, job.transcript.content_type)
    timing["upload"] = int((time.monotonic() - t) * 1000)

    return {
        "status": "completed",
        "source": partial["source"],
        "video": video,
        "audio": audio,
        "transcript": transcript,
    }
