"""Slideshow stage: N (image, audio) pairs -> one crossfaded, subtitled video.

Every slide is rendered to its own part first (same canvas, frame rate and codecs), so
the assemble pass only has to crossfade streams that already match. One output, so
unlike clips there is no partial result: a slide that fails fails the job.
"""

from __future__ import annotations

import os
import time

from postiz_uploader import errors, ffmpeg
from postiz_uploader.captions import build_cue_ass
from postiz_uploader.config import Settings
from postiz_uploader.download import download, host_allowed
from postiz_uploader.errors import JobError
from postiz_uploader.ffmpeg import Encoders
from postiz_uploader.log import get_logger, log, redact_url
from postiz_uploader.probe import probe
from postiz_uploader.schema import SlideshowJob
from postiz_uploader.sniff import sniff
from postiz_uploader.upload import upload

logger = get_logger(__name__)
SNIFF_BYTES = 512
# AAC pads its last packet and every crossfade lands on a frame boundary
DURATION_TOLERANCE = 1.0


def _remaining(deadline: float) -> float:
    return deadline - time.monotonic()


def timeline(durations: list[float], transition: float) -> tuple[list[float], float]:
    """Where each slide starts on the output timeline, and the total length.

    A crossfade overlaps the end of one slide with the start of the next, so slide k
    starts at `sum(durations[:k]) - transition * k` and the video is
    `sum(durations) - transition * (len - 1)` long. The caller places its subtitle
    cues with the same arithmetic.
    """
    starts = []
    elapsed = 0.0
    for k, duration in enumerate(durations):
        starts.append(elapsed - transition * k)
        elapsed += duration
    return starts, elapsed - transition * max(len(durations) - 1, 0)


def _fetch(
    job: SlideshowJob, workdir: str, settings: Settings, deadline: float, max_bytes: int
) -> tuple[list[tuple[str, str]], int]:
    """Download every image and audio into the workdir; returns (image, audio) paths and the bytes fetched."""
    paths = []
    received = 0
    for index, slide in enumerate(job.slides):
        for field, url in (("image", slide.image_url), ("audio", slide.audio_url)):
            if not host_allowed(url, settings.allowed_source_hosts):
                raise JobError(
                    errors.SOURCE_HOST_NOT_ALLOWED,
                    f"host of slides.{index}.{field}.url is not allowed: {redact_url(url)}",
                )
        image = os.path.join(workdir, f"slide-{index}.image")
        audio = os.path.join(workdir, f"slide-{index}.audio")
        received += download(slide.image_url, image, max_bytes=max_bytes, deadline=deadline)
        with open(image, "rb") as fh:
            sniffed = sniff(fh.read(SNIFF_BYTES))
        if sniffed is None or sniffed.kind != "image":
            raise JobError(errors.UNSUPPORTED_INPUT, f"slides.{index}.image is not an image file")
        received += download(slide.audio_url, audio, max_bytes=max_bytes, deadline=deadline)
        paths.append((image, audio))
    return paths, received


def _encode(
    job: SlideshowJob, build, dst: str, settings: Settings, deadline: float, encoders: Encoders, what: str
) -> None:
    """Run `build(encoder)` through the encoder ladder until one succeeds."""
    last: JobError | None = None
    for attempt in list(encoders.attempts):
        encoder, _ = attempt
        outcome = ffmpeg.run(build(encoder), timeout=_remaining(deadline))
        if outcome.returncode == 0:
            encoders.used = attempt
            return
        last = JobError(
            errors.ENCODE_FAILED, f"ffmpeg exited with status {outcome.returncode}", stderr_tail=outcome.stderr_tail
        )
        log(
            logger,
            "slideshow encode failed",
            reference=job.reference,
            step=what,
            encoder=encoder,
            stderr=outcome.stderr_tail[-500:],
        )
        encoders.drop(attempt)
        if os.path.exists(dst):
            os.remove(dst)
    else:
        assert last is not None
        raise last


def process_slideshow(job: SlideshowJob, workdir: str, settings: Settings, deadline: float, timing: dict) -> dict:
    max_bytes = min(job.limits.max_input_bytes, settings.max_input_bytes_cap)
    t = time.monotonic()
    paths, received = _fetch(job, workdir, settings, deadline, max_bytes)
    timing["download"] = int((time.monotonic() - t) * 1000)
    log(logger, "downloaded", reference=job.reference, slides=len(paths), bytes=received)

    t = time.monotonic()
    durations = []
    for index, (slide, (_, audio)) in enumerate(zip(job.slides, paths, strict=True)):
        info = probe(settings.ffprobe_bin, audio, timeout=min(60, max(_remaining(deadline), 1)), audio_only=True)
        if not info.has_audio or not info.duration_seconds:
            raise JobError(errors.UNSUPPORTED_INPUT, f"slides.{index}.audio has no audio stream")
        durations.append(slide.duration_seconds or info.duration_seconds + job.tail_seconds)
    timing["probe"] = int((time.monotonic() - t) * 1000)

    transition = job.transition_seconds if len(durations) > 1 else 0.0
    if transition and transition >= min(durations):
        raise JobError(
            errors.INVALID_JOB,
            f"transition_seconds {transition:g} must be below the shortest slide ({min(durations):.2f}s)",
        )
    offsets, total = timeline(durations, transition)

    ass_path = None
    cue_count = 0
    if job.subtitles:
        document, cue_count = build_cue_ass(
            job.subtitles.cues, job.subtitles.style, width=job.frame.width, height=job.frame.height, total=total
        )
        if cue_count:
            if not ffmpeg.has_filter(settings.ffmpeg_bin, "ass"):
                raise JobError(errors.INTERNAL, "this ffmpeg build has no libass, subtitles cannot be burned in")
            ass_path = os.path.join(workdir, "subtitles.ass")
            with open(ass_path, "w", encoding="utf-8") as fh:
                fh.write(document)

    rules = job.video
    encoders = Encoders(settings, cuda_decode=False)
    vf = ffmpeg.slide_filter(job.frame, fit=job.fit, fps=rules.fps, pixel_format=rules.pixel_format)
    t = time.monotonic()
    parts = []
    for index, ((image, audio), duration) in enumerate(zip(paths, durations, strict=True)):
        part = os.path.join(workdir, f"part-{index}.mp4")
        _encode(
            job,
            lambda encoder, image=image, audio=audio, part=part, duration=duration: ffmpeg.build_slide(
                settings.ffmpeg_bin,
                image,
                audio,
                part,
                duration=duration,
                audio_delay=job.audio_delay_seconds,
                vf=vf,
                rules=rules,
                encoder=encoder,
            ),
            part,
            settings,
            deadline,
            encoders,
            f"slide {index}",
        )
        parts.append(part)
        for path in (image, audio):
            os.remove(path)

    dst = os.path.join(workdir, "slideshow.mp4")
    _encode(
        job,
        lambda encoder: ffmpeg.build_slideshow(
            settings.ffmpeg_bin,
            parts,
            dst,
            offsets=offsets,
            transition=transition,
            rules=rules,
            encoder=encoder,
            ass_path=ass_path,
            fonts_dir=settings.fonts_dir or None,
        ),
        dst,
        settings,
        deadline,
        encoders,
        "assemble",
    )
    timing["process"] = int((time.monotonic() - t) * 1000)

    out = probe(settings.ffprobe_bin, dst, timeout=min(60, max(_remaining(deadline), 1)))
    problems = []
    if (out.width, out.height) != (job.frame.width, job.frame.height):
        problems.append(f"dimensions={out.width}x{out.height}")
    if out.video_codec != "h264":
        problems.append(f"video_codec={out.video_codec}")
    if out.pixel_format != rules.pixel_format:
        problems.append(f"pixel_format={out.pixel_format}")
    if out.duration_seconds is None or abs(out.duration_seconds - total) > DURATION_TOLERANCE:
        problems.append(f"duration={out.duration_seconds} expected {total:.2f}")
    if out.faststart is not True:
        problems.append("faststart=missing")
    if problems:
        raise JobError(errors.ENCODE_FAILED, "slideshow failed post-check: " + ", ".join(problems))

    if _remaining(deadline) <= 0:
        raise JobError(errors.TIMEOUT, "job timeout reached before upload", retryable=True)
    t = time.monotonic()
    upload(dst, job.output.url, job.output.content_type)
    timing["upload"] = int((time.monotonic() - t) * 1000)

    used = encoders.used
    return {
        "status": "completed",
        "output": {
            "width": out.width,
            "height": out.height,
            "duration_seconds": out.duration_seconds,
            "bytes": out.bytes or 0,
            "content_type": job.output.content_type,
            "video_codec": out.video_codec,
            "audio_codec": out.audio_codec,
            "fps": out.fps,
            "slides": len(parts),
            "subtitle_cues": cue_count,
        },
        "_encoder": used[0] if used else None,
        "_decode": "software" if used else None,
    }
