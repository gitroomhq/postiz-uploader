"""Video stage: probe -> plan -> ffmpeg -> post-check -> optional thumbnail."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from postiz_uploader import errors, ffmpeg
from postiz_uploader.config import Settings
from postiz_uploader.errors import JobError
from postiz_uploader.log import get_logger, log
from postiz_uploader.planner import ENCODE, REMUX, UNCHANGED, Plan, plan_video
from postiz_uploader.probe import SourceInfo, probe
from postiz_uploader.schema import Job

logger = get_logger(__name__)


@dataclass
class VideoOutcome:
    status: str  # completed | unchanged
    actions: tuple[str, ...]
    source: SourceInfo
    plan: Plan
    output_path: str | None
    output: dict | None
    thumbnail_path: str | None
    thumbnail: dict | None
    decode: str | None  # cuda | software | None (no encode ran)
    encoder: str | None  # h264_nvenc | libx264 | None (no encode ran)


def _remaining(deadline: float) -> float:
    return deadline - time.monotonic()


def _run_encode(
    job: Job, plan: Plan, info: SourceInfo, src: str, dst: str, settings: Settings, deadline: float
) -> tuple[str, str]:
    """Run the ENCODE plan. On the GPU flavour try CUDA decode first and fall back to
    software decode when ffmpeg rejects it (NVDEC cannot decode every source, 4:2:2
    HEVC from iPhones being the usual case), then to libx264 when the encoder itself
    cannot open (a host whose container lacks libnvidia-encode). Returns the decode
    mode and the encoder used."""
    rules = job.rules.video
    # (encoder, cuda decode) in the order they are tried
    attempts: list[tuple[str, bool]] = []
    if settings.gpu and plan.video_encode:
        if not plan.tonemap:
            attempts.append((settings.encoder, True))
        attempts.append((settings.encoder, False))
        attempts.append(("libx264", False))
    else:
        attempts.append((settings.encoder, False))

    last: JobError | None = None
    for i, (encoder, cuda) in enumerate(attempts):
        cmd = ffmpeg.build_encode(settings.ffmpeg_bin, src, dst, plan, info, rules, encoder=encoder, cuda_decode=cuda)
        outcome = ffmpeg.run(cmd, timeout=_remaining(deadline))
        if outcome.returncode == 0:
            return ("cuda" if cuda else "software"), encoder
        last = JobError(
            errors.ENCODE_FAILED,
            f"ffmpeg exited with status {outcome.returncode}",
            stderr_tail=outcome.stderr_tail,
        )
        if i + 1 < len(attempts):
            next_encoder, next_cuda = attempts[i + 1]
            log(
                logger,
                "encode failed, retrying",
                reference=job.reference,
                encoder=encoder,
                decode="cuda" if cuda else "software",
                next_encoder=next_encoder,
                next_decode="cuda" if next_cuda else "software",
                stderr=outcome.stderr_tail[-500:],
            )
            if os.path.exists(dst):
                os.remove(dst)
    assert last is not None
    raise last


def _post_check(out: SourceInfo, plan: Plan, expected_w: int, expected_h: int, job: Job) -> None:
    rules = job.rules.video
    problems = []
    if plan.kind == ENCODE and plan.video_encode:
        if out.video_codec != rules.video_codec:
            problems.append(f"video_codec={out.video_codec}")
        if out.pixel_format != rules.pixel_format:
            problems.append(f"pixel_format={out.pixel_format}")
    if (out.width, out.height) != (expected_w, expected_h):
        problems.append(f"dimensions={out.width}x{out.height} expected {expected_w}x{expected_h}")
    if plan.kind == ENCODE and plan.video_encode and out.rotation != 0:
        problems.append(f"rotation={out.rotation}")
    if out.container != rules.container:
        problems.append(f"container={out.container}")
    if rules.faststart and out.faststart is not True:
        problems.append("faststart=missing")
    if plan.kind == ENCODE and plan.audio_encode and out.audio_codec != rules.audio_codec:
        problems.append(f"audio_codec={out.audio_codec}")
    if problems:
        raise JobError(errors.ENCODE_FAILED, "output failed post-check: " + ", ".join(problems))


def process_video(
    job: Job, input_path: str, workdir: str, settings: Settings, deadline: float, timing: dict
) -> VideoOutcome:
    t0 = time.monotonic()
    info = probe(settings.ffprobe_bin, input_path, timeout=min(60, max(_remaining(deadline), 1)))
    timing["probe"] = int((time.monotonic() - t0) * 1000)

    if info.duration_seconds is not None and info.duration_seconds > job.limits.max_duration_seconds:
        raise JobError(
            errors.DURATION_TOO_LONG,
            f"duration {info.duration_seconds:.1f}s exceeds {job.limits.max_duration_seconds}s",
        )
    if job.rules.video.container == "mp4" and job.output.content_type != "video/mp4":
        raise JobError(errors.INVALID_JOB, "output.content_type must be video/mp4 for an mp4 container")

    plan = plan_video(info, job.rules)
    log(
        logger,
        "plan",
        reference=job.reference,
        kind=plan.kind,
        actions=list(plan.actions),
        target=f"{plan.width}x{plan.height}",
    )

    output_path = None
    output = None
    decode = None
    encoder = None
    frame_source = input_path
    frame_w, frame_h = info.width, info.height

    if plan.kind != UNCHANGED:
        output_path = os.path.join(workdir, f"output.{job.rules.video.container}")
        t1 = time.monotonic()
        if plan.kind == REMUX:
            cmd = ffmpeg.build_remux(settings.ffmpeg_bin, input_path, output_path, info, job.rules.video)
            outcome = ffmpeg.run(cmd, timeout=_remaining(deadline))
            if outcome.returncode != 0:
                raise JobError(
                    errors.ENCODE_FAILED,
                    f"ffmpeg exited with status {outcome.returncode}",
                    stderr_tail=outcome.stderr_tail,
                )
        else:
            decode, encoder = _run_encode(job, plan, info, input_path, output_path, settings, deadline)
        timing["process"] = int((time.monotonic() - t1) * 1000)

        out_info = probe(settings.ffprobe_bin, output_path, timeout=min(60, max(_remaining(deadline), 1)))
        _post_check(out_info, plan, plan.width, plan.height, job)
        frame_source = output_path
        frame_w, frame_h = out_info.width, out_info.height
        output = {
            "width": out_info.width,
            "height": out_info.height,
            "duration_seconds": out_info.duration_seconds,
            "bytes": out_info.bytes or os.path.getsize(output_path),
            "content_type": job.output.content_type,
            "video_codec": out_info.video_codec,
            "audio_codec": out_info.audio_codec,
            "fps": out_info.fps,
        }

    thumbnail_path = None
    thumbnail = None
    if job.thumbnail:
        thumbnail_path = os.path.join(workdir, "thumbnail.jpg")
        duration = info.duration_seconds or 0
        ts = min(max(job.thumbnail.timestamp_seconds, 0), max(duration - 0.1, 0))
        cmd = ffmpeg.build_thumbnail(
            settings.ffmpeg_bin, frame_source, thumbnail_path, timestamp=ts, width=frame_w, height=frame_h
        )
        try:
            outcome = ffmpeg.run(cmd, timeout=max(min(_remaining(deadline), 120), 1))
        except JobError as err:
            outcome = None
            log(logger, "thumbnail failed", reference=job.reference, error=err.code)
        if outcome is not None and outcome.returncode == 0 and os.path.exists(thumbnail_path):
            thumbnail = {"width": frame_w, "height": frame_h, "bytes": os.path.getsize(thumbnail_path)}
        else:
            if outcome is not None:
                log(logger, "thumbnail failed", reference=job.reference, stderr=outcome.stderr_tail[-500:])
            thumbnail_path = None

    return VideoOutcome(
        status="unchanged" if plan.kind == UNCHANGED else "completed",
        actions=plan.actions,
        source=info,
        plan=plan,
        output_path=output_path,
        output=output,
        thumbnail_path=thumbnail_path,
        thumbnail=thumbnail,
        decode=decode,
        encoder=encoder,
    )
