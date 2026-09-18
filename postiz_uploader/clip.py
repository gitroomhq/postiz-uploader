"""Clip stage: one downloaded source -> N cut, reframed, captioned clips.

Each clip is uploaded as soon as it is rendered, so a job that fails or times out
half way keeps what it finished and the caller resubmits only the rest.
"""

from __future__ import annotations

import os
import time

from postiz_uploader import errors, ffmpeg
from postiz_uploader.captions import build_ass
from postiz_uploader.config import Settings
from postiz_uploader.errors import JobError
from postiz_uploader.log import get_logger, log
from postiz_uploader.probe import SourceInfo, probe
from postiz_uploader.schema import Clip, ClipJob
from postiz_uploader.upload import upload

logger = get_logger(__name__)
# a cut lands on a frame boundary and AAC pads its last packet
DURATION_TOLERANCE = 1.0


def _remaining(deadline: float) -> float:
    return deadline - time.monotonic()


class _Encoders:
    """The (encoder, cuda decode) combinations to try, best first.

    A combination that failed for one clip will fail for the next one too (it is a
    property of the host and the source, not of the cut), so it is dropped for the
    rest of the job.
    """

    def __init__(self, settings: Settings, tonemap: bool):
        if settings.gpu:
            self.attempts = [(settings.encoder, False), ("libx264", False)]
            if not tonemap:
                self.attempts.insert(0, (settings.encoder, True))
        else:
            self.attempts = [(settings.encoder, False)]
        self.used: tuple[str, bool] | None = None

    def drop(self, attempt: tuple[str, bool]) -> None:
        if len(self.attempts) > 1:
            self.attempts.remove(attempt)


def _render(
    job: ClipJob,
    clip: Clip,
    index: int,
    window: tuple[float, float],
    info: SourceInfo,
    src: str,
    workdir: str,
    settings: Settings,
    deadline: float,
    encoders: _Encoders,
) -> dict:
    start, end = window
    duration = end - start
    ass_path = None
    caption_lines = 0
    if job.captions:
        document, caption_lines = build_ass(
            job.captions.words, job.captions.style, start=start, end=end, width=job.frame.width, height=job.frame.height
        )
        if caption_lines:
            if not ffmpeg.has_filter(settings.ffmpeg_bin, "ass"):
                raise JobError(errors.INTERNAL, "this ffmpeg build has no libass, captions cannot be burned in")
            ass_path = os.path.join(workdir, f"clip-{index}.ass")
            with open(ass_path, "w", encoding="utf-8") as fh:
                fh.write(document)

    rules = job.video
    fps_cap = rules.fps_max if (rules.fps_max and info.fps and info.fps > rules.fps_max + 0.01) else None
    vf = ffmpeg.clip_filter(
        job.frame,
        focus_x=clip.focus_x if clip.focus_x is not None else job.frame.focus_x,
        focus_y=clip.focus_y if clip.focus_y is not None else job.frame.focus_y,
        fps_cap=fps_cap,
        tonemap=info.is_hdr,
        pixel_format=rules.pixel_format,
        ass_path=ass_path,
        fonts_dir=settings.fonts_dir or None,
    )

    dst = os.path.join(workdir, f"clip-{index}.{rules.container}")
    last: JobError | None = None
    for attempt in list(encoders.attempts):
        encoder, cuda = attempt
        cmd = ffmpeg.build_clip(
            settings.ffmpeg_bin,
            src,
            dst,
            start=start,
            duration=duration,
            vf=vf,
            has_audio=info.has_audio,
            rules=rules,
            encoder=encoder,
            cuda_decode=cuda,
        )
        outcome = ffmpeg.run(cmd, timeout=_remaining(deadline))
        if outcome.returncode == 0:
            encoders.used = attempt
            break
        last = JobError(
            errors.ENCODE_FAILED, f"ffmpeg exited with status {outcome.returncode}", stderr_tail=outcome.stderr_tail
        )
        log(
            logger,
            "clip encode failed",
            reference=job.reference,
            clip=clip.reference,
            encoder=encoder,
            decode="cuda" if cuda else "software",
            stderr=outcome.stderr_tail[-500:],
        )
        encoders.drop(attempt)
        if os.path.exists(dst):
            os.remove(dst)
    else:
        assert last is not None
        raise last

    out = probe(settings.ffprobe_bin, dst, timeout=min(60, max(_remaining(deadline), 1)))
    problems = []
    if (out.width, out.height) != (job.frame.width, job.frame.height):
        problems.append(f"dimensions={out.width}x{out.height}")
    if out.video_codec != rules.video_codec:
        problems.append(f"video_codec={out.video_codec}")
    if out.duration_seconds is None or abs(out.duration_seconds - duration) > DURATION_TOLERANCE:
        problems.append(f"duration={out.duration_seconds} expected {duration:.2f}")
    if out.faststart is not True:
        problems.append("faststart=missing")
    if problems:
        raise JobError(errors.ENCODE_FAILED, "clip failed post-check: " + ", ".join(problems))

    thumbnail = None
    thumbnail_path = None
    if clip.thumbnail:
        thumbnail_path = os.path.join(workdir, f"clip-{index}.jpg")
        ts = min(clip.thumbnail.timestamp_seconds, max(duration - 0.1, 0))
        cmd = ffmpeg.build_thumbnail(
            settings.ffmpeg_bin, dst, thumbnail_path, timestamp=ts, width=out.width, height=out.height
        )
        try:
            shot = ffmpeg.run(cmd, timeout=max(min(_remaining(deadline), 120), 1))
        except JobError:
            shot = None
        if shot is not None and shot.returncode == 0 and os.path.exists(thumbnail_path):
            thumbnail = {"width": out.width, "height": out.height, "bytes": os.path.getsize(thumbnail_path)}
        else:
            # same rule as the video stage: a thumbnail never fails the clip
            log(logger, "clip thumbnail failed", reference=job.reference, clip=clip.reference)
            thumbnail_path = None

    if _remaining(deadline) <= 0:
        raise JobError(errors.TIMEOUT, "job timeout reached before upload", retryable=True)
    upload(dst, clip.output.url, clip.output.content_type)
    if thumbnail_path and clip.thumbnail:
        upload(thumbnail_path, clip.thumbnail.url, clip.thumbnail.content_type)

    for path in (dst, ass_path, thumbnail_path):
        if path and os.path.exists(path):
            os.remove(path)

    return {
        "output": {
            "width": out.width,
            "height": out.height,
            "duration_seconds": out.duration_seconds,
            "bytes": out.bytes or 0,
            "content_type": clip.output.content_type,
            "video_codec": out.video_codec,
            "audio_codec": out.audio_codec,
            "fps": out.fps,
        },
        "thumbnail": thumbnail,
        "caption_lines": caption_lines,
    }


def process_clips(
    job: ClipJob, input_path: str, workdir: str, settings: Settings, deadline: float, timing: dict, partial: dict
) -> dict:
    t = time.monotonic()
    info = probe(settings.ffprobe_bin, input_path, timeout=min(60, max(_remaining(deadline), 1)))
    timing["probe"] = int((time.monotonic() - t) * 1000)
    partial["source"] = info.to_dict()

    encoders = _Encoders(settings, info.is_hdr)
    results = []
    t = time.monotonic()
    for index, clip in enumerate(job.clips):
        entry = {
            "reference": clip.reference,
            "status": "failed",
            "start_seconds": clip.start_seconds,
            "end_seconds": clip.end_seconds,
            "output": None,
            "thumbnail": None,
            "caption_lines": 0,
            "failure": None,
        }
        results.append(entry)
        try:
            if _remaining(deadline) <= 0:
                raise JobError(errors.TIMEOUT, "job timeout reached before this clip started", retryable=True)
            end = clip.end_seconds
            if info.duration_seconds is not None:
                if clip.start_seconds >= info.duration_seconds:
                    raise JobError(
                        errors.INVALID_JOB,
                        f"start_seconds {clip.start_seconds} is past the end of the source "
                        f"({info.duration_seconds:.1f}s)",
                    )
                end = min(end, info.duration_seconds)
            entry["end_seconds"] = end
            started = time.monotonic()
            window = (clip.start_seconds, end)
            entry.update(_render(job, clip, index, window, info, input_path, workdir, settings, deadline, encoders))
            entry["status"] = "completed"
            log(
                logger,
                "clip done",
                reference=job.reference,
                clip=clip.reference,
                seconds=round(end - clip.start_seconds, 2),
                render_ms=int((time.monotonic() - started) * 1000),
            )
        except JobError as err:
            entry["failure"] = err.to_dict()
            log(logger, "clip failed", reference=job.reference, clip=clip.reference, code=err.code, message=err.message)
    timing["process"] = int((time.monotonic() - t) * 1000)

    done = sum(1 for r in results if r["status"] == "completed")
    status = "completed" if done == len(results) else ("partial" if done else "failed")
    failure = next((r["failure"] for r in results if r["failure"]), None) if status == "failed" else None
    used = encoders.used
    return {
        "status": status,
        "source": partial["source"],
        "clips": results,
        "failure": failure,
        "_encoder": used[0] if used else None,
        "_decode": ("cuda" if used[1] else "software") if used else None,
    }
