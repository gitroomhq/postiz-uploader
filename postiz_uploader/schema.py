"""Job parsing and validation against the schemas under schema/v1.

`type` picks the schema: video and image share job.schema.json, ingest and clip have
their own files (and their own result schemas).

The JSON schema is the contract the caller vendors. This module validates against
it and then lifts the dict into dataclasses with every default applied, so the
rest of the code never touches raw dicts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import jsonschema

from postiz_uploader import errors
from postiz_uploader.errors import JobError

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schema"
SUPPORTED_VERSIONS = (1,)


@cache
def load_schema(version: int, name: str) -> dict:
    with open(SCHEMA_DIR / f"v{version}" / f"{name}.schema.json", encoding="utf-8") as fh:
        return json.load(fh)


@dataclass(frozen=True)
class Source:
    url: str
    content_type: str | None


@dataclass(frozen=True)
class Output:
    url: str
    content_type: str


@dataclass(frozen=True)
class Thumbnail:
    url: str
    timestamp_seconds: float
    content_type: str


@dataclass(frozen=True)
class VideoRules:
    container: str = "mp4"
    video_codec: str = "h264"
    profile: str = "high"
    pixel_format: str = "yuv420p"
    fps_max: float | None = 60
    quality: int = 23
    audio_codec: str = "aac"
    audio_bitrate_kbps: int = 128
    audio_sample_rate: int | None = 48000
    faststart: bool = True


@dataclass(frozen=True)
class ImageRules:
    jpeg_quality: int = 90
    keep_format: bool = True


@dataclass(frozen=True)
class Rules:
    short_side_min: int
    short_side_max: int
    long_side_max: int
    video: VideoRules
    image: ImageRules


@dataclass(frozen=True)
class Limits:
    max_input_bytes: int = 1024 * 1024 * 1024
    max_duration_seconds: float = 900
    timeout_seconds: float = 1200


@dataclass(frozen=True)
class Job:
    version: int
    type: str
    reference: str
    source: Source
    output: Output
    thumbnail: Thumbnail | None
    rules: Rules
    limits: Limits


@dataclass(frozen=True)
class IngestSource:
    url: str
    via: str = "direct"
    max_height: int = 1080
    proxy: str | None = None
    # oxylabs only: fetch this window instead of the whole video
    start_seconds: float | None = None
    end_seconds: float | None = None


@dataclass(frozen=True)
class AudioOutput:
    url: str
    content_type: str = "audio/ogg"
    bitrate_kbps: int = 24
    sample_rate: int = 16000
    # skip the audio (and its download) when the transcript output found captions
    unless_transcript: bool = False


@dataclass(frozen=True)
class TranscriptOutput:
    url: str
    content_type: str = "application/json"
    languages: tuple[str, ...] = ("en",)


@dataclass(frozen=True)
class IngestLimits:
    max_input_bytes: int = 2 * 1024 * 1024 * 1024
    max_duration_seconds: float = 7200
    timeout_seconds: float = 1200


@dataclass(frozen=True)
class IngestJob:
    version: int
    type: str
    reference: str
    source: IngestSource
    video: Output | None
    audio: AudioOutput | None
    transcript: TranscriptOutput | None
    limits: IngestLimits


@dataclass(frozen=True)
class Frame:
    width: int = 1080
    height: int = 1920
    fit: str = "crop"
    focus_x: float = 0.5
    focus_y: float = 0.5


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class CaptionStyle:
    font: str = "Montserrat"
    bold: bool = True
    font_size: int | None = None
    color: str = "#FFFFFF"
    highlight_color: str | None = "#FFE600"
    outline_color: str = "#000000"
    outline: float = 4
    shadow: float = 0
    position: str = "bottom"
    margin_v: int | None = None
    max_words: int = 4
    max_chars: int = 18
    uppercase: bool = False


@dataclass(frozen=True)
class Captions:
    words: tuple[Word, ...]
    style: CaptionStyle


@dataclass(frozen=True)
class Clip:
    reference: str
    start_seconds: float
    end_seconds: float
    output: Output
    thumbnail: Thumbnail | None
    focus_x: float | None = None
    focus_y: float | None = None


@dataclass(frozen=True)
class ClipLimits:
    max_input_bytes: int = 2 * 1024 * 1024 * 1024
    max_clip_seconds: float = 600
    timeout_seconds: float = 1200


@dataclass(frozen=True)
class ClipJob:
    version: int
    type: str
    reference: str
    source: Source
    clips: tuple[Clip, ...]
    frame: Frame
    captions: Captions | None
    video: VideoRules
    limits: ClipLimits


# job type -> (job schema, result schema)
SCHEMAS = {
    "video": ("job", "result"),
    "image": ("job", "result"),
    "ingest": ("ingest", "ingest-result"),
    "clip": ("clip", "clip-result"),
}


def _pick(data: dict, cls, **overrides):
    """Build a dataclass from a dict, keeping only known keys, applying defaults."""
    fields = cls.__dataclass_fields__
    kwargs = {k: v for k, v in data.items() if k in fields}
    kwargs.update(overrides)
    return cls(**kwargs)


def _thumbnail(thumb_raw: dict | None) -> Thumbnail | None:
    if not thumb_raw:
        return None
    return Thumbnail(
        url=thumb_raw["url"],
        timestamp_seconds=float(thumb_raw.get("timestamp_seconds", 0)),
        content_type=thumb_raw.get("content_type", "image/jpeg"),
    )


def _parse_ingest(raw: dict) -> IngestJob:
    video_raw = raw.get("video")
    audio_raw = raw.get("audio")
    transcript_raw = raw.get("transcript")
    if not video_raw and not audio_raw and not transcript_raw:
        raise JobError(errors.INVALID_JOB, "an ingest job needs a video, audio or transcript output")
    via = raw["source"].get("via") or "direct"
    if via != "oxylabs":
        if transcript_raw:
            raise JobError(errors.INVALID_JOB, "a transcript output needs source.via oxylabs")
        if raw["source"].get("start_seconds") is not None or raw["source"].get("end_seconds") is not None:
            raise JobError(errors.INVALID_JOB, "source.start_seconds and end_seconds need source.via oxylabs")
    return IngestJob(
        version=raw["version"],
        type="ingest",
        reference=raw["reference"],
        source=_pick({k: v for k, v in raw["source"].items() if v is not None}, IngestSource),
        video=Output(url=video_raw["url"], content_type=video_raw.get("content_type", "video/mp4"))
        if video_raw
        else None,
        audio=_pick(audio_raw, AudioOutput) if audio_raw else None,
        transcript=_pick(transcript_raw, TranscriptOutput, languages=tuple(transcript_raw.get("languages") or ("en",)))
        if transcript_raw
        else None,
        limits=_pick(raw.get("limits") or {}, IngestLimits),
    )


def _parse_clip(raw: dict) -> ClipJob:
    limits = _pick(raw.get("limits") or {}, ClipLimits)
    clips = []
    seen: set[str] = set()
    for i, item in enumerate(raw["clips"]):
        if item["end_seconds"] <= item["start_seconds"]:
            raise JobError(errors.INVALID_JOB, f"clips.{i}: end_seconds must be above start_seconds")
        if item["end_seconds"] - item["start_seconds"] > limits.max_clip_seconds:
            raise JobError(errors.INVALID_JOB, f"clips.{i}: longer than limits.max_clip_seconds")
        if item["reference"] in seen:
            raise JobError(errors.INVALID_JOB, f"clips.{i}: duplicate reference {item['reference']!r}")
        seen.add(item["reference"])
        clips.append(
            Clip(
                reference=item["reference"],
                start_seconds=float(item["start_seconds"]),
                end_seconds=float(item["end_seconds"]),
                output=Output(url=item["output"]["url"], content_type=item["output"].get("content_type", "video/mp4")),
                thumbnail=_thumbnail(item.get("thumbnail")),
                focus_x=item.get("focus_x"),
                focus_y=item.get("focus_y"),
            )
        )

    captions = None
    captions_raw = raw.get("captions")
    if captions_raw:
        words = tuple(
            Word(text=w["text"], start=float(w["start"]), end=float(w["end"])) for w in captions_raw["words"]
        )
        captions = Captions(words=words, style=_pick(captions_raw.get("style") or {}, CaptionStyle))

    return ClipJob(
        version=raw["version"],
        type="clip",
        reference=raw["reference"],
        source=Source(url=raw["source"]["url"], content_type=None),
        clips=tuple(clips),
        frame=_pick(raw.get("frame") or {}, Frame),
        captions=captions,
        video=_pick(raw.get("video") or {}, VideoRules),
        limits=limits,
    )


def parse_job(raw: object) -> Job | IngestJob | ClipJob:
    if not isinstance(raw, dict):
        raise JobError(errors.INVALID_JOB, "job must be an object")

    version = raw.get("version")
    if version not in SUPPORTED_VERSIONS:
        raise JobError(
            errors.UNSUPPORTED_VERSION,
            f"unsupported job version {version!r}; supported: {list(SUPPORTED_VERSIONS)}",
        )

    # an unknown type falls through to job.schema.json, whose enum names the valid ones
    job_type = raw.get("type")
    schema_name = SCHEMAS.get(job_type if isinstance(job_type, str) else "", ("job",))[0]
    validator = jsonschema.Draft202012Validator(load_schema(version, schema_name))
    error = jsonschema.exceptions.best_match(validator.iter_errors(raw))
    if error is not None:
        path = ".".join(str(p) for p in error.absolute_path) or "<root>"
        raise JobError(errors.INVALID_JOB, f"{path}: {error.message}")

    if job_type == "ingest":
        return _parse_ingest(raw)
    if job_type == "clip":
        return _parse_clip(raw)

    rules_raw = raw["rules"]
    rules = Rules(
        short_side_min=rules_raw["short_side_min"],
        short_side_max=rules_raw["short_side_max"],
        long_side_max=rules_raw["long_side_max"],
        video=_pick(rules_raw.get("video") or {}, VideoRules),
        image=_pick(rules_raw.get("image") or {}, ImageRules),
    )
    if rules.short_side_min > rules.short_side_max:
        raise JobError(errors.INVALID_JOB, "rules.short_side_min is above rules.short_side_max")

    return Job(
        version=version,
        type=raw["type"],
        reference=raw["reference"],
        source=Source(url=raw["source"]["url"], content_type=raw["source"].get("content_type")),
        output=Output(url=raw["output"]["url"], content_type=raw["output"]["content_type"]),
        thumbnail=_thumbnail(raw.get("thumbnail")),
        rules=rules,
        limits=_pick(raw.get("limits") or {}, Limits),
    )


def validate_result(result: dict) -> None:
    """Used by tests and the local runner; production trusts its own output."""
    name = SCHEMAS.get(result.get("type", "video"), ("job", "result"))[1]
    jsonschema.Draft202012Validator(load_schema(1, name)).validate(result)
