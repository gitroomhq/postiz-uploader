"""process(job) -> result. Never raises; every failure becomes a `failed` result."""

from __future__ import annotations

import os
import shutil
import tempfile
import time

from postiz_uploader import __version__, errors, ffmpeg
from postiz_uploader.clip import process_clips
from postiz_uploader.config import Settings, get_settings
from postiz_uploader.download import download, host_allowed
from postiz_uploader.errors import JobError
from postiz_uploader.image import process_image
from postiz_uploader.ingest import process_ingest
from postiz_uploader.log import get_logger, log, redact_url, setup_logging
from postiz_uploader.schema import ClipJob, IngestJob, parse_job
from postiz_uploader.sniff import sniff
from postiz_uploader.upload import upload
from postiz_uploader.video import process_video

logger = get_logger(__name__)
SNIFF_BYTES = 512
DISK_WAIT_SECONDS = 5


def _worker_block(settings: Settings, decode: str | None, encoder: str | None = None) -> dict:
    return {
        # the encoder that actually ran, which differs from the configured one
        # after a fallback; the configured one when no encode ran
        "encoder": encoder or settings.encoder,
        "decode": decode,
        "ffmpeg": ffmpeg.version(settings.ffmpeg_bin),
        "gpu": ffmpeg.gpu_name() if settings.gpu else None,
        "version": __version__,
    }


def _result(reference: str, status: str, **fields) -> dict:
    base = {
        "version": 1,
        "reference": reference,
        "status": status,
        "actions": [],
        "source": None,
        "output": None,
        "thumbnail": None,
        "timing_ms": {},
        "worker": {},
        "failure": None,
    }
    base.update(fields)
    return base


def _ensure_disk(settings: Settings, needed: int) -> None:
    os.makedirs(settings.work_dir, exist_ok=True)
    for attempt in range(2):
        if shutil.disk_usage(settings.work_dir).free >= needed:
            return
        if attempt == 0:
            time.sleep(DISK_WAIT_SECONDS)
    raise JobError(errors.INTERNAL, "not enough free disk for this job", retryable=True)


def _capture(exc: BaseException) -> None:
    try:
        import sentry_sdk

        if sentry_sdk.is_initialized():
            sentry_sdk.capture_exception(exc)
    except Exception:  # noqa: BLE001 - reporting must never break a result
        pass


# what a result of each staged type looks like before the stage fills it in
_STAGED_BLANK = {
    "ingest": {"source": None, "video": None, "audio": None},
    "clip": {"source": None, "clips": []},
}


def _staged_result(job_type: str, reference: str, status: str, **fields) -> dict:
    base = {"version": 1, "type": job_type, "reference": reference, "status": status}
    base.update(_STAGED_BLANK[job_type])
    base.update({"timing_ms": {}, "worker": {}, "failure": None})
    base.update(fields)
    return base


def _process_staged(job: IngestJob | ClipJob, settings: Settings, started: float) -> dict:
    """ingest and clip: same guarantees as process() (never raises, always cleans up),
    but the stage owns its downloads and uploads because each has several outputs."""
    timing: dict[str, int] = {}
    # filled by the stage as facts become known, so a failure still reports them
    partial: dict = {}
    workdir = None
    try:
        deadline = started + job.limits.timeout_seconds
        max_bytes = min(job.limits.max_input_bytes, settings.max_input_bytes_cap)
        _ensure_disk(settings, 2 * max_bytes)
        workdir = tempfile.mkdtemp(prefix="job-", dir=settings.work_dir)

        if isinstance(job, IngestJob):
            fields = process_ingest(job, workdir, settings, deadline, timing, partial)
        else:
            if not host_allowed(job.source.url, settings.allowed_source_hosts):
                raise JobError(
                    errors.SOURCE_HOST_NOT_ALLOWED,
                    f"host of source.url is not allowed: {redact_url(job.source.url)}",
                )
            input_path = os.path.join(workdir, "input")
            t = time.monotonic()
            size = download(job.source.url, input_path, max_bytes=max_bytes, deadline=deadline)
            timing["download"] = int((time.monotonic() - t) * 1000)
            log(logger, "downloaded", reference=job.reference, bytes=size, url=redact_url(job.source.url))
            with open(input_path, "rb") as fh:
                sniffed = sniff(fh.read(SNIFF_BYTES))
            if sniffed is None or sniffed.kind != "video":
                raise JobError(errors.UNSUPPORTED_INPUT, "source is not a video file")
            fields = process_clips(job, input_path, workdir, settings, deadline, timing, partial)

        encoder = fields.pop("_encoder", None)
        decode = fields.pop("_decode", None)
        timing["total"] = int((time.monotonic() - started) * 1000)
        log(logger, "done", reference=job.reference, type=job.type, status=fields["status"], timing_ms=timing)
        return _staged_result(
            job.type,
            job.reference,
            fields.pop("status"),
            timing_ms=timing,
            worker=_worker_block(settings, decode, encoder),
            **fields,
        )

    except JobError as err:
        failure = err
    except Exception as err:  # noqa: BLE001 - a crash must still produce a result
        logger.exception("unhandled error", extra={"extra": {"reference": job.reference}})
        failure = JobError(errors.INTERNAL, f"{err.__class__.__name__}: {err}", retryable=True)
        _capture(err)
    finally:
        if workdir:
            shutil.rmtree(workdir, ignore_errors=True)

    timing["total"] = int((time.monotonic() - started) * 1000)
    log(
        logger,
        "failed",
        reference=job.reference,
        type=job.type,
        code=failure.code,
        message=failure.message,
        retryable=failure.retryable,
        stderr_tail=failure.stderr_tail,
    )
    return _staged_result(
        job.type,
        job.reference,
        "failed",
        timing_ms=timing,
        worker=_worker_block(settings, None),
        failure=failure.to_dict(),
        **partial,
    )


def process(raw_job: object) -> dict:
    setup_logging()
    settings = get_settings()
    started = time.monotonic()
    timing: dict[str, int] = {}
    reference = raw_job.get("reference", "") if isinstance(raw_job, dict) else ""
    if not isinstance(reference, str):
        reference = str(reference)

    try:
        job = parse_job(raw_job)
    except JobError as err:
        log(logger, "invalid job", reference=reference, code=err.code, message=err.message)
        worker = _worker_block(settings, None)
        staged = raw_job.get("type") if isinstance(raw_job, dict) else None
        if isinstance(staged, str) and staged in _STAGED_BLANK:
            return _staged_result(staged, reference, "failed", failure=err.to_dict(), worker=worker)
        return _result(reference, "failed", failure=err.to_dict(), worker=worker)

    if isinstance(job, (IngestJob, ClipJob)):
        return _process_staged(job, settings, started)

    workdir = None
    source: dict | None = None
    try:
        if not host_allowed(job.source.url, settings.allowed_source_hosts):
            raise JobError(
                errors.SOURCE_HOST_NOT_ALLOWED, f"host of source.url is not allowed: {redact_url(job.source.url)}"
            )

        max_bytes = min(job.limits.max_input_bytes, settings.max_input_bytes_cap)
        deadline = started + job.limits.timeout_seconds
        _ensure_disk(settings, 2 * max_bytes)
        workdir = tempfile.mkdtemp(prefix="job-", dir=settings.work_dir)
        input_path = os.path.join(workdir, "input")

        t = time.monotonic()
        size = download(job.source.url, input_path, max_bytes=max_bytes, deadline=deadline)
        timing["download"] = int((time.monotonic() - t) * 1000)
        log(logger, "downloaded", reference=job.reference, bytes=size, url=redact_url(job.source.url))

        with open(input_path, "rb") as fh:
            sniffed = sniff(fh.read(SNIFF_BYTES))
        if sniffed is None:
            raise JobError(errors.UNSUPPORTED_INPUT, "unrecognized file type")
        if sniffed.kind != job.type:
            raise JobError(
                errors.UNSUPPORTED_INPUT, f"job.type is {job.type} but the file is a {sniffed.kind} ({sniffed.format})"
            )

        decode = None
        encoder = None
        thumbnail = None
        thumbnail_path = None
        if job.type == "video":
            outcome = process_video(job, input_path, workdir, settings, deadline, timing)
            source = outcome.source.to_dict()
            decode = outcome.decode
            encoder = outcome.encoder
            thumbnail, thumbnail_path = outcome.thumbnail, outcome.thumbnail_path
        else:
            t = time.monotonic()
            outcome = process_image(job, input_path, workdir, settings)
            timing["process"] = int((time.monotonic() - t) * 1000)
            source = outcome.source

        if outcome.status == "completed" or thumbnail_path:
            if time.monotonic() > deadline:
                raise JobError(errors.TIMEOUT, "job timeout reached before upload", retryable=True)
            t = time.monotonic()
            if outcome.status == "completed":
                upload(outcome.output_path, job.output.url, job.output.content_type)
            # the thumbnail is uploaded even when the video itself is unchanged
            if thumbnail_path and job.thumbnail:
                upload(thumbnail_path, job.thumbnail.url, job.thumbnail.content_type)
            timing["upload"] = int((time.monotonic() - t) * 1000)

        timing["total"] = int((time.monotonic() - started) * 1000)
        log(
            logger,
            "done",
            reference=job.reference,
            status=outcome.status,
            actions=list(outcome.actions),
            timing_ms=timing,
        )
        return _result(
            job.reference,
            outcome.status,
            actions=list(outcome.actions),
            source=source,
            output=outcome.output,
            thumbnail=thumbnail,
            timing_ms=timing,
            worker=_worker_block(settings, decode, encoder),
        )

    except JobError as err:
        timing["total"] = int((time.monotonic() - started) * 1000)
        log(
            logger,
            "failed",
            reference=job.reference,
            code=err.code,
            message=err.message,
            retryable=err.retryable,
            stderr_tail=err.stderr_tail,
        )
        if err.code == errors.INTERNAL:
            _capture(err)
        return _result(
            job.reference,
            "failed",
            source=source,
            timing_ms=timing,
            worker=_worker_block(settings, None),
            failure=err.to_dict(),
        )

    except Exception as err:  # noqa: BLE001 - a crash must still produce a result
        timing["total"] = int((time.monotonic() - started) * 1000)
        logger.exception("unhandled error", extra={"extra": {"reference": job.reference}})
        _capture(err)
        internal = JobError(errors.INTERNAL, f"{err.__class__.__name__}: {err}", retryable=True)
        return _result(
            job.reference,
            "failed",
            source=source,
            timing_ms=timing,
            worker=_worker_block(settings, None),
            failure=internal.to_dict(),
        )

    finally:
        if workdir:
            shutil.rmtree(workdir, ignore_errors=True)
