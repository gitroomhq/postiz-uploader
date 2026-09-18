from __future__ import annotations

import os
import re
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
    # ingest: hosts a `via: ytdlp` source may point at, and the proxy yt-dlp falls back to
    allowed_ingest_hosts: tuple[str, ...] = field(default_factory=tuple)
    ingest_proxies: tuple[str, ...] = field(default_factory=tuple)
    ingest_proxy_attempts: int = 2
    ingest_direct_first: bool = True
    # ingest: where the bgutil PO token server lives; empty means run without one
    pot_server_dir: str = ""
    # ingest via oxylabs: their login, the bucket they deliver into (with its key), and
    # the tallest rendition a job may ask for, since downloads are billed per GB
    oxylabs_username: str = ""
    oxylabs_password: str = ""
    oxylabs_storage_url: str = ""
    oxylabs_storage_region: str = "auto"
    oxylabs_max_height: int = 720
    # clip: where libass looks for caption fonts
    fonts_dir: str = ""

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


DEFAULT_INGEST_HOSTS = "youtube.com,*.youtube.com,youtu.be"


def _hosts(name: str, default: str) -> tuple[str, ...]:
    return tuple(h.strip().lower() for h in os.environ.get(name, default).split(",") if h.strip())


def parse_proxies(raw: str) -> tuple[str, ...]:
    """A proxy pool from one variable: URLs separated by commas or whitespace.

    Providers hand out lists as `host:port:user:pass` lines, so that shape is accepted
    too and read as an HTTP proxy.
    """
    proxies = []
    for item in re.split(r"[,\s]+", raw.strip()):
        if not item:
            continue
        if "://" not in item:
            parts = item.split(":")
            if len(parts) == 4:
                host, port, user, password = parts
                item = f"http://{user}:{password}@{host}:{port}"
            elif len(parts) == 2:
                item = f"http://{item}"
            else:
                raise RuntimeError("INGEST_PROXY entries must be URLs, host:port or host:port:user:pass")
        if item not in proxies:
            proxies.append(item)
    return tuple(proxies)


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def get_settings() -> Settings:
    """Read configuration from the environment on every call.

    Read lazily (not at import) so tests can set variables before the first job.
    """
    encoder = os.environ.get("ENCODER", "libx264").strip()
    if encoder not in ("libx264", "h264_nvenc"):
        raise RuntimeError(f"ENCODER must be libx264 or h264_nvenc, got {encoder!r}")

    hosts = _hosts("ALLOWED_SOURCE_HOSTS", "")

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
        allowed_ingest_hosts=_hosts("ALLOWED_INGEST_HOSTS", DEFAULT_INGEST_HOSTS),
        ingest_proxies=parse_proxies(os.environ.get("INGEST_PROXY", "")),
        ingest_proxy_attempts=max(_int("INGEST_PROXY_ATTEMPTS", 2), 1),
        ingest_direct_first=_bool("INGEST_DIRECT_FIRST", True),
        pot_server_dir=os.environ.get("POT_SERVER_DIR", "").strip(),
        oxylabs_username=os.environ.get("OXYLABS_USERNAME", "").strip(),
        oxylabs_password=os.environ.get("OXYLABS_PASSWORD", ""),
        oxylabs_storage_url=os.environ.get("OXYLABS_STORAGE_URL", "").strip(),
        oxylabs_storage_region=os.environ.get("OXYLABS_STORAGE_REGION", "auto").strip() or "auto",
        oxylabs_max_height=_int("OXYLABS_MAX_HEIGHT", 720),
        fonts_dir=os.environ.get("FONTS_DIR", "").strip(),
    )
