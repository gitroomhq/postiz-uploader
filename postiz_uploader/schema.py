"""Job parsing and validation against schema/v1/job.schema.json.

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


def _pick(data: dict, cls, **overrides):
    """Build a dataclass from a dict, keeping only known keys, applying defaults."""
    fields = cls.__dataclass_fields__
    kwargs = {k: v for k, v in data.items() if k in fields}
    kwargs.update(overrides)
    return cls(**kwargs)


def parse_job(raw: object) -> Job:
    if not isinstance(raw, dict):
        raise JobError(errors.INVALID_JOB, "job must be an object")

    version = raw.get("version")
    if version not in SUPPORTED_VERSIONS:
        raise JobError(
            errors.UNSUPPORTED_VERSION,
            f"unsupported job version {version!r}; supported: {list(SUPPORTED_VERSIONS)}",
        )

    validator = jsonschema.Draft202012Validator(load_schema(version, "job"))
    error = jsonschema.exceptions.best_match(validator.iter_errors(raw))
    if error is not None:
        path = ".".join(str(p) for p in error.absolute_path) or "<root>"
        raise JobError(errors.INVALID_JOB, f"{path}: {error.message}")

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

    thumb_raw = raw.get("thumbnail")
    thumbnail = None
    if thumb_raw:
        thumbnail = Thumbnail(
            url=thumb_raw["url"],
            timestamp_seconds=float(thumb_raw.get("timestamp_seconds", 0)),
            content_type=thumb_raw.get("content_type", "image/jpeg"),
        )

    return Job(
        version=version,
        type=raw["type"],
        reference=raw["reference"],
        source=Source(url=raw["source"]["url"], content_type=raw["source"].get("content_type")),
        output=Output(url=raw["output"]["url"], content_type=raw["output"]["content_type"]),
        thumbnail=thumbnail,
        rules=rules,
        limits=_pick(raw.get("limits") or {}, Limits),
    )


def validate_result(result: dict) -> None:
    """Used by tests and the local runner; production trusts its own output."""
    jsonschema.Draft202012Validator(load_schema(1, "result")).validate(result)
