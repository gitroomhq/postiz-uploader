"""ffprobe wrapper. Produces a SourceInfo the planner can reason about."""

from __future__ import annotations

import json
import struct
import subprocess
from dataclasses import asdict, dataclass

from postiz_uploader import errors
from postiz_uploader.errors import JobError

HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}
WEBM_CODECS = {"vp8", "vp9", "av1", "opus", "vorbis"}


@dataclass(frozen=True)
class SourceInfo:
    container: str
    video_codec: str | None
    profile: str | None
    pixel_format: str | None
    color_transfer: str | None
    color_primaries: str | None
    width: int  # after rotation
    height: int  # after rotation
    rotation: int  # 0, 90, 180, 270
    fps: float | None
    duration_seconds: float | None
    bytes: int
    audio_codec: str | None
    audio_sample_rate: int | None
    faststart: bool | None  # None when not an ISO BMFF file

    @property
    def has_audio(self) -> bool:
        return self.audio_codec is not None

    @property
    def is_hdr(self) -> bool:
        return self.color_transfer in HDR_TRANSFERS or self.color_primaries == "bt2020"

    def to_dict(self) -> dict:
        return asdict(self)


def _fraction(value: str | None) -> float | None:
    if not value:
        return None
    if "/" in value:
        num, den = value.split("/", 1)
        try:
            num_f, den_f = float(num), float(den)
        except ValueError:
            return None
        if den_f == 0:
            return None
        return num_f / den_f
    try:
        return float(value)
    except ValueError:
        return None


def _rotation(stream: dict) -> int:
    """Rotation to apply for display, clockwise degrees in {0, 90, 180, 270}.

    ffprobe exposes it two ways with opposite signs: the display matrix side data
    reports the counter-clockwise angle (an iPhone portrait clip shows -90), while
    the legacy `rotate` tag is clockwise (the same clip shows 90). ffmpeg's own
    autorotate negates the display matrix value; mirror that so 90 always means
    `transpose=clock`.
    """
    value = None
    negate = False
    for side in stream.get("side_data_list") or []:
        if "rotation" in side:
            value = side["rotation"]
            negate = True
            break
    if value is None:
        value = (stream.get("tags") or {}).get("rotate")
    if value is None:
        return 0
    try:
        deg = int(round(float(value)))
    except (TypeError, ValueError):
        return 0
    if negate:
        deg = -deg
    return deg % 360


def _container(fmt: dict, streams: list[dict]) -> str:
    names = (fmt.get("format_name") or "").split(",")
    brand = ((fmt.get("tags") or {}).get("major_brand") or "").strip().lower()
    if "mp4" in names or "mov" in names:
        return "mov" if brand == "qt" else "mp4"
    if "matroska" in names or "webm" in names:
        codecs = {s.get("codec_name") for s in streams}
        return "webm" if codecs and codecs <= WEBM_CODECS else "mkv"
    return names[0] if names and names[0] else "unknown"


def isobmff_faststart(path: str) -> bool | None:
    """Whether `moov` precedes `mdat` at the top level. None if not an ISO BMFF file."""
    with open(path, "rb") as fh:
        head = fh.read(12)
        if len(head) < 12 or head[4:8] != b"ftyp":
            return None
        fh.seek(0)
        offset = 0
        seen_mdat = False
        while True:
            fh.seek(offset)
            header = fh.read(8)
            if len(header) < 8:
                return None
            size, kind = struct.unpack(">I4s", header)
            if size == 1:
                big = fh.read(8)
                if len(big) < 8:
                    return None
                size = struct.unpack(">Q", big)[0]
            elif size == 0:
                # atom extends to end of file
                return not seen_mdat if kind == b"moov" else None
            if kind == b"moov":
                return not seen_mdat
            if kind == b"mdat":
                seen_mdat = True
            if size < 8:
                return None
            offset += size


def probe(ffprobe_bin: str, path: str, *, timeout: float = 60, audio_only: bool = False) -> SourceInfo:
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        path,
    ]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as err:
        raise JobError(errors.PROBE_FAILED, "ffprobe timed out", retryable=True) from err
    except OSError as err:
        raise JobError(errors.INTERNAL, f"cannot run ffprobe: {err}", retryable=True) from err

    if completed.returncode != 0:
        raise JobError(
            errors.PROBE_FAILED,
            f"ffprobe exited with status {completed.returncode}",
            stderr_tail=completed.stderr[-4096:],
        )
    try:
        data = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as err:
        raise JobError(errors.PROBE_FAILED, "ffprobe produced invalid JSON") from err

    fmt = data.get("format") or {}
    streams = data.get("streams") or []
    # skip attached cover art (disposition.attached_pic), same as the 0:V stream specifier
    video = next(
        (s for s in streams if s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic")),
        None,
    )
    if video is None:
        video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video is None and audio_only and audio is not None:
        # an ingest that only wants the audio is handed an audio file (Oxylabs
        # delivers raw AAC for download_type=audio): there is no picture to describe
        return SourceInfo(
            container=_container(fmt, streams),
            video_codec=None,
            profile=None,
            pixel_format=None,
            color_transfer=None,
            color_primaries=None,
            width=0,
            height=0,
            rotation=0,
            fps=None,
            duration_seconds=_fraction(fmt.get("duration")) or _fraction(audio.get("duration")),
            bytes=int(fmt.get("size") or 0),
            audio_codec=audio.get("codec_name"),
            audio_sample_rate=int(audio["sample_rate"]) if audio.get("sample_rate") else None,
            faststart=isobmff_faststart(path),
        )
    if video is None:
        raise JobError(errors.UNSUPPORTED_INPUT, "no video stream")

    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    if width <= 0 or height <= 0:
        raise JobError(errors.PROBE_FAILED, "video stream has no dimensions")
    rotation = _rotation(video)
    if rotation in (90, 270):
        width, height = height, width

    fps = _fraction(video.get("avg_frame_rate")) or _fraction(video.get("r_frame_rate"))
    duration = _fraction(fmt.get("duration")) or _fraction(video.get("duration"))

    return SourceInfo(
        container=_container(fmt, streams),
        video_codec=video.get("codec_name"),
        profile=(video.get("profile") or None),
        pixel_format=video.get("pix_fmt"),
        color_transfer=video.get("color_transfer"),
        color_primaries=video.get("color_primaries"),
        width=width,
        height=height,
        rotation=rotation,
        fps=fps,
        duration_seconds=duration,
        bytes=int(fmt.get("size") or 0),
        audio_codec=audio.get("codec_name") if audio else None,
        audio_sample_rate=int(audio["sample_rate"]) if audio and audio.get("sample_rate") else None,
        faststart=isobmff_faststart(path),
    )
