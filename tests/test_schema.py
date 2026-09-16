from __future__ import annotations

import pytest

from postiz_uploader.errors import JobError
from postiz_uploader.schema import parse_job, validate_result

BASE = {
    "version": 1,
    "type": "video",
    "reference": "media_1",
    "source": {"url": "https://bucket.example.com/a.mov"},
    "output": {"url": "https://bucket.example.com/a.mp4?X-Amz-Signature=x", "content_type": "video/mp4"},
    "rules": {"short_side_min": 1080, "short_side_max": 1080, "long_side_max": 1920},
}


def test_defaults_applied():
    job = parse_job(BASE)
    assert job.rules.video.fps_max == 60
    assert job.rules.video.audio_sample_rate == 48000
    assert job.limits.timeout_seconds == 1200
    assert job.thumbnail is None


def test_unknown_fields_ignored():
    raw = dict(BASE, future_field=1)
    raw["rules"] = dict(BASE["rules"], video={"quality": 20, "not_yet": True})
    job = parse_job(raw)
    assert job.rules.video.quality == 20


def test_unsupported_version():
    with pytest.raises(JobError) as err:
        parse_job(dict(BASE, version=2))
    assert err.value.code == "UNSUPPORTED_VERSION"


@pytest.mark.parametrize(
    "mutate,fragment",
    [
        (lambda r: r.pop("reference"), "reference"),
        (lambda r: r.__setitem__("type", "audio"), "type"),
        (lambda r: r["source"].__setitem__("url", "ftp://x/y"), "source.url"),
        (lambda r: r["rules"].__setitem__("short_side_min", 0), "short_side_min"),
        (lambda r: r["rules"].update(short_side_min=1080, short_side_max=720), "short_side_min"),
    ],
)
def test_invalid_job(mutate, fragment):
    raw = {**BASE, "source": dict(BASE["source"]), "rules": dict(BASE["rules"])}
    mutate(raw)
    with pytest.raises(JobError) as err:
        parse_job(raw)
    assert err.value.code == "INVALID_JOB"
    assert fragment in err.value.message


def test_thumbnail_parsed():
    raw = dict(BASE, thumbnail={"url": "https://bucket.example.com/t.jpg", "timestamp_seconds": 2})
    job = parse_job(raw)
    assert job.thumbnail.timestamp_seconds == 2.0
    assert job.thumbnail.content_type == "image/jpeg"


def test_result_schema_accepts_failure_shape():
    validate_result(
        {
            "version": 1,
            "reference": "x",
            "status": "failed",
            "actions": [],
            "source": None,
            "output": None,
            "thumbnail": None,
            "timing_ms": {"total": 1},
            "worker": {"encoder": "libx264", "decode": None, "ffmpeg": None, "gpu": None, "version": "0.1.0"},
            "failure": {"code": "INTERNAL", "message": "x", "retryable": True, "stderr_tail": None},
        }
    )
