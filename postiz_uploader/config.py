from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    encoder: str = "libx264"
    worker_concurrency: int = 1
    allowed_source_hosts: tuple[str, ...] = field(default_factory=tuple)
    work_dir: str = "/tmp/postiz-uploader"
    max_input_bytes_cap: int = 2 * 1024 * 1024 * 1024
    image_max_pixels: int = 100_000_000
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    sentry_dsn: str = ""
    log_level: str = "info"

    @property
    def gpu(self) -> bool:
        return self.encoder == "h264_nvenc"


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as err:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from err


def get_settings() -> Settings:
    """Read configuration from the environment on every call.

    Read lazily (not at import) so tests can set variables before the first job.
    """
    encoder = os.environ.get("ENCODER", "libx264").strip()
    if encoder not in ("libx264", "h264_nvenc"):
        raise RuntimeError(f"ENCODER must be libx264 or h264_nvenc, got {encoder!r}")

    hosts = tuple(h.strip().lower() for h in os.environ.get("ALLOWED_SOURCE_HOSTS", "").split(",") if h.strip())

    return Settings(
        encoder=encoder,
        worker_concurrency=_int("WORKER_CONCURRENCY", 8 if encoder == "h264_nvenc" else 1),
        allowed_source_hosts=hosts,
        work_dir=os.environ.get("WORK_DIR", "/tmp/postiz-uploader"),
        max_input_bytes_cap=_int("MAX_INPUT_BYTES_CAP", 2 * 1024 * 1024 * 1024),
        image_max_pixels=_int("IMAGE_MAX_PIXELS", 100_000_000),
        ffmpeg_bin=os.environ.get("FFMPEG_BIN", "ffmpeg"),
        ffprobe_bin=os.environ.get("FFPROBE_BIN", "ffprobe"),
        sentry_dsn=os.environ.get("SENTRY_DSN", ""),
        log_level=os.environ.get("LOG_LEVEL", "info"),
    )
