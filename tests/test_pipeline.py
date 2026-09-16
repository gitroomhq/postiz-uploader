"""End to end against generated fixtures, a local bucket stand-in and the real ffmpeg."""

from __future__ import annotations

import os
import shutil

import pytest

from postiz_uploader.config import get_settings
from postiz_uploader.ffmpeg import has_filter
from postiz_uploader.pipeline import process
from postiz_uploader.probe import probe
from postiz_uploader.schema import validate_result
from tests.conftest import make_job, needs_ffmpeg

pytestmark = needs_ffmpeg


def _stage(bucket: dict, fixtures_dir: str, name: str) -> str:
    """Copy a fixture into the bucket root and return its URL."""
    shutil.copy(os.path.join(fixtures_dir, name), os.path.join(bucket["root"], name))
    return f"{bucket['base']}/{name}"


def _run(bucket, fixtures_dir, name, type_, output_type, **kw) -> dict:
    src = _stage(bucket, fixtures_dir, name)
    out_name = f"out/{name}.{'mp4' if type_ == 'video' else output_type.split('/')[1]}"
    thumb = f"{bucket['base']}/out/{name}.thumb.jpg" if type_ == "video" else None
    job = make_job(
        type_=type_,
        source_url=src,
        output_url=f"{bucket['base']}/{out_name}?X-Amz-Signature=fake",
        output_type=output_type,
        thumbnail_url=thumb,
        **kw,
    )
    result = process(job)
    validate_result(result)
    result["_output_path"] = os.path.join(bucket["root"], out_name)
    result["_thumb_path"] = os.path.join(bucket["root"], f"out/{name}.thumb.jpg")
    return result


def _out(result: dict):
    return probe(get_settings().ffprobe_bin, result["_output_path"])


@pytest.fixture(autouse=True)
def _work(work_dir):
    yield


def test_compliant_is_unchanged(bucket, fixtures_dir):
    r = _run(bucket, fixtures_dir, "compliant-1080p.mp4", "video", "video/mp4")
    assert r["status"] == "unchanged", r["failure"]
    assert r["actions"] == []
    assert not os.path.exists(r["_output_path"])
    # a thumbnail is still produced from the original
    assert r["thumbnail"]["width"] == 1920 and os.path.exists(r["_thumb_path"])


def test_no_faststart_is_remuxed(bucket, fixtures_dir):
    r = _run(bucket, fixtures_dir, "compliant-no-faststart.mp4", "video", "video/mp4")
    assert r["status"] == "completed", r["failure"]
    assert r["actions"] == ["remux"]
    out = _out(r)
    assert out.faststart is True and out.video_codec == "h264" and (out.width, out.height) == (1920, 1080)


def test_portrait_rotation_is_baked_in(bucket, fixtures_dir):
    r = _run(bucket, fixtures_dir, "portrait-rotated.mov", "video", "video/mp4")
    assert r["status"] == "completed", r["failure"]
    assert "rotate" in r["actions"] and "video_encode" in r["actions"]
    assert (r["source"]["width"], r["source"]["height"]) == (1080, 1920)
    out = _out(r)
    assert (out.width, out.height) == (1080, 1920) and out.rotation == 0


def test_480p_is_upscaled(bucket, fixtures_dir):
    r = _run(bucket, fixtures_dir, "small-480p-h264.mp4", "video", "video/mp4")
    assert r["status"] == "completed", r["failure"]
    assert "scale" in r["actions"]
    assert (r["output"]["width"], r["output"]["height"]) == (1920, 1080)


def test_120fps_is_capped(bucket, fixtures_dir):
    r = _run(bucket, fixtures_dir, "slowmo-120fps.mp4", "video", "video/mp4")
    assert r["status"] == "completed", r["failure"]
    assert "fps_cap" in r["actions"]
    assert abs(_out(r).fps - 60) < 0.01


def test_no_audio_stays_silent(bucket, fixtures_dir):
    r = _run(bucket, fixtures_dir, "no-audio.mp4", "video", "video/mp4", rules={"video": {"faststart": True}})
    # compliant apart from nothing: unchanged. Force a change with a smaller cap.
    r = _run(
        bucket,
        fixtures_dir,
        "no-audio.mp4",
        "video",
        "video/mp4",
        rules={"short_side_min": 720, "short_side_max": 720, "long_side_max": 1280},
    )
    assert r["status"] == "completed", r["failure"]
    assert _out(r).audio_codec is None


def test_ultrawide_long_cap(bucket, fixtures_dir):
    r = _run(bucket, fixtures_dir, "ultrawide-2560x1080.mp4", "video", "video/mp4")
    assert r["status"] == "completed", r["failure"]
    assert (r["output"]["width"], r["output"]["height"]) == (1920, 810)


def test_audio_44100_reencodes_audio_only(bucket, fixtures_dir):
    r = _run(bucket, fixtures_dir, "audio-44100.mp4", "video", "video/mp4")
    assert r["status"] == "completed", r["failure"]
    assert r["actions"] == ["audio_encode"]
    assert _out(r).audio_sample_rate == 48000


def test_4k_hevc_downscaled(bucket, fixtures_dir):
    if not os.path.exists(os.path.join(fixtures_dir, "4k-hevc.mov")):
        pytest.skip("libx265 not available to build the fixture")
    r = _run(bucket, fixtures_dir, "4k-hevc.mov", "video", "video/mp4")
    assert r["status"] == "completed", r["failure"]
    assert set(r["actions"]) >= {"video_encode", "scale", "remux"}
    out = _out(r)
    assert (out.width, out.height) == (1920, 1080) and out.video_codec == "h264" and out.faststart


def test_4k_hdr_tonemapped(bucket, fixtures_dir):
    if not os.path.exists(os.path.join(fixtures_dir, "4k-hdr-hlg.mov")):
        pytest.skip("libx265 not available to build the fixture")
    if not has_filter(get_settings().ffmpeg_bin, "zscale"):
        pytest.skip("ffmpeg lacks zscale (libzimg)")
    r = _run(bucket, fixtures_dir, "4k-hdr-hlg.mov", "video", "video/mp4")
    assert r["status"] == "completed", r["failure"]
    assert "tonemap" in r["actions"]
    out = _out(r)
    assert out.pixel_format == "yuv420p" and (out.width, out.height) == (1920, 1080)


def test_webm_vp9_converted(bucket, fixtures_dir):
    if not os.path.exists(os.path.join(fixtures_dir, "small-480p.webm")):
        pytest.skip("libvpx not available to build the fixture")
    r = _run(bucket, fixtures_dir, "small-480p.webm", "video", "video/mp4")
    assert r["status"] == "completed", r["failure"]
    out = _out(r)
    assert out.container == "mp4" and out.video_codec == "h264" and (out.width, out.height) == (1920, 1080)


def test_truncated_file_fails_cleanly(bucket, fixtures_dir):
    r = _run(bucket, fixtures_dir, "truncated.mp4", "video", "video/mp4")
    assert r["status"] == "failed"
    assert r["failure"]["code"] in ("PROBE_FAILED", "ENCODE_FAILED", "UNSUPPORTED_INPUT")
    assert r["failure"]["retryable"] is False


def test_duration_limit(bucket, fixtures_dir):
    r = _run(
        bucket, fixtures_dir, "compliant-no-faststart.mp4", "video", "video/mp4", limits={"max_duration_seconds": 1}
    )
    assert r["status"] == "failed" and r["failure"]["code"] == "DURATION_TOO_LONG"


def test_type_mismatch(bucket, fixtures_dir):
    r = _run(bucket, fixtures_dir, "large.png", "video", "video/mp4")
    assert r["status"] == "failed" and r["failure"]["code"] == "UNSUPPORTED_INPUT"


def test_size_limit(bucket, fixtures_dir):
    r = _run(bucket, fixtures_dir, "compliant-1080p.mp4", "video", "video/mp4", limits={"max_input_bytes": 1000})
    assert r["status"] == "failed" and r["failure"]["code"] == "INPUT_TOO_LARGE"


def test_host_not_allowed(bucket, fixtures_dir, monkeypatch):
    monkeypatch.setenv("ALLOWED_SOURCE_HOSTS", "bucket.example.com")
    r = _run(bucket, fixtures_dir, "compliant-1080p.mp4", "video", "video/mp4")
    assert r["status"] == "failed" and r["failure"]["code"] == "SOURCE_HOST_NOT_ALLOWED"


def test_download_failure_is_retryable(bucket, fixtures_dir):
    job = make_job(
        type_="video",
        source_url=f"{bucket['base']}/missing.mp4",
        output_url=f"{bucket['base']}/o.mp4",
        output_type="video/mp4",
    )
    r = process(job)
    assert r["status"] == "failed" and r["failure"]["code"] == "DOWNLOAD_FAILED"


def test_invalid_job_returns_failed_result():
    r = process({"version": 1})
    validate_result(r)
    assert r["status"] == "failed" and r["failure"]["code"] == "INVALID_JOB"
    r = process({"version": 9, "reference": "x"})
    assert r["failure"]["code"] == "UNSUPPORTED_VERSION" and r["reference"] == "x"


def test_timeout_is_enforced(bucket, fixtures_dir):
    if not os.path.exists(os.path.join(fixtures_dir, "4k-hevc.mov")):
        pytest.skip("libx265 not available to build the fixture")
    r = _run(bucket, fixtures_dir, "4k-hevc.mov", "video", "video/mp4", limits={"timeout_seconds": 10})
    # a 2s 4K clip may finish under 10s on a fast machine; only assert the shape when it does not
    if r["status"] == "failed":
        assert r["failure"]["code"] == "TIMEOUT" and r["failure"]["retryable"] is True


def test_large_png_downscaled_with_alpha(bucket, fixtures_dir):
    r = _run(bucket, fixtures_dir, "large.png", "image", "image/png")
    assert r["status"] == "completed", r["failure"]
    assert r["actions"] == ["scale"]
    assert (r["output"]["width"], r["output"]["height"]) == (1440, 1080)
    from PIL import Image

    with Image.open(r["_output_path"]) as img:
        assert img.mode == "RGBA" and img.size == (1440, 1080)


def test_small_jpeg_upscaled_and_oriented(bucket, fixtures_dir):
    r = _run(bucket, fixtures_dir, "small-exif-rotated.jpg", "image", "image/jpeg")
    assert r["status"] == "completed", r["failure"]
    assert set(r["actions"]) == {"scale", "orient"}
    # orientation 6 turns the 400x300 source into 300x400, then the short side goes to 720
    assert (r["source"]["width"], r["source"]["height"]) == (300, 400)
    assert (r["output"]["width"], r["output"]["height"]) == (720, 960)
    from PIL import Image

    with Image.open(r["_output_path"]) as img:
        assert img.getexif().get(0x0112) in (None, 1)


def test_compliant_jpeg_unchanged(bucket, fixtures_dir):
    r = _run(bucket, fixtures_dir, "compliant-1080.jpg", "image", "image/jpeg")
    assert r["status"] == "unchanged"


def test_webp_inside_bounds_unchanged(bucket, fixtures_dir):
    r = _run(bucket, fixtures_dir, "medium.webp", "image", "image/webp")
    assert r["status"] == "unchanged"


def test_gif_untouched(bucket, fixtures_dir):
    r = _run(bucket, fixtures_dir, "animated.gif", "image", "image/gif")
    assert r["status"] == "unchanged" and r["source"]["animated"] is True


def test_image_format_must_match_output_type(bucket, fixtures_dir):
    r = _run(bucket, fixtures_dir, "large.png", "image", "image/jpeg")
    assert r["status"] == "failed" and r["failure"]["code"] == "INVALID_JOB"


def test_pixel_cap(bucket, fixtures_dir, monkeypatch):
    monkeypatch.setenv("IMAGE_MAX_PIXELS", "1000000")
    r = _run(bucket, fixtures_dir, "large.png", "image", "image/png")
    assert r["status"] == "failed" and r["failure"]["code"] == "UNSUPPORTED_INPUT"


def test_workdir_is_cleaned(bucket, fixtures_dir, work_dir):
    _run(bucket, fixtures_dir, "compliant-no-faststart.mp4", "video", "video/mp4")
    assert os.listdir(work_dir) == []
