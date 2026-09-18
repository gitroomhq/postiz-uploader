"""Ingest stage: fetch a long source once -> faststart MP4 and/or speech-grade audio."""

from __future__ import annotations

import os
import time

from postiz_uploader import errors, ffmpeg, ytdlp
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


def _fetch_ytdlp(job: IngestJob, workdir: str, settings: Settings, deadline: float, max_bytes: int, partial: dict):
    if not host_allowed(job.source.url, settings.allowed_ingest_hosts):
        raise JobError(
            errors.SOURCE_HOST_NOT_ALLOWED, f"host of source.url is not allowed: {redact_url(job.source.url)}"
        )
    proxy = job.source.proxy or settings.ingest_proxy or None
    # bandwidth through a proxy is the expensive part of an ingest, so it is only
    # used once the platform has refused this worker's own address
    routes: list[str | None] = [None, proxy] if proxy and settings.ingest_direct_first else [proxy]

    for i, route in enumerate(routes):
        try:
            meta = ytdlp.fetch_metadata(
                job.source.url,
                workdir,
                proxy=route,
                max_height=job.source.max_height,
                ffmpeg_bin=settings.ffmpeg_bin,
                deadline=deadline,
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
            log(logger, "blocked on the direct route, retrying through the proxy", reference=job.reference)
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

    t = time.monotonic()
    fetch = _fetch_ytdlp if job.source.via == "ytdlp" else _fetch_direct
    media_path = fetch(job, workdir, settings, deadline, max_bytes, partial)
    timing["download"] = int((time.monotonic() - t) * 1000)

    t = time.monotonic()
    info = probe(settings.ffprobe_bin, media_path, timeout=min(60, max(_remaining(deadline), 1)))
    timing["probe"] = int((time.monotonic() - t) * 1000)
    partial["source"].update(
        {
            "duration_seconds": info.duration_seconds or partial["source"].get("duration_seconds"),
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
    if job.audio and not info.has_audio:
        raise JobError(errors.UNSUPPORTED_INPUT, "source has no audio track to extract")

    t = time.monotonic()
    video_path = None
    video = None
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

    audio_path = None
    audio = None
    if job.audio:
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
    timing["upload"] = int((time.monotonic() - t) * 1000)

    return {"status": "completed", "source": partial["source"], "video": video, "audio": audio}
