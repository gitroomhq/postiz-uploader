"""clip jobs end to end: real ffmpeg, the local bucket stand-in, generated fixtures."""

from __future__ import annotations

import os
import shutil

import pytest

from postiz_uploader import errors
from postiz_uploader.config import get_settings
from postiz_uploader.ffmpeg import clip_filter, filter_path, has_filter
from postiz_uploader.pipeline import process
from postiz_uploader.probe import probe
from postiz_uploader.schema import Frame, parse_job, validate_result
from tests.conftest import needs_ffmpeg

SOURCE = "talk-8s.mp4"


@pytest.fixture(autouse=True)
def _work(work_dir):
    yield


def _job(bucket, clips, **extra) -> dict:
    items = []
    for ref, start, end in clips:
        items.append(
            {
                "reference": ref,
                "start_seconds": start,
                "end_seconds": end,
                "output": {"url": f"{bucket['base']}/clips/{ref}.mp4?X-Amz-Signature=fake"},
                "thumbnail": {"url": f"{bucket['base']}/clips/{ref}.jpg"},
            }
        )
    return {
        "version": 1,
        "type": "clip",
        "reference": "project_1",
        "source": {"url": f"{bucket['base']}/{SOURCE}"},
        "clips": items,
        **extra,
    }


def _stage(bucket, fixtures_dir):
    shutil.copy(os.path.join(fixtures_dir, SOURCE), os.path.join(bucket["root"], SOURCE))


def _probe(bucket, ref):
    return probe(get_settings().ffprobe_bin, os.path.join(bucket["root"], "clips", f"{ref}.mp4"))


@needs_ffmpeg
def test_two_clips_are_cut_reframed_and_uploaded(bucket, fixtures_dir):
    _stage(bucket, fixtures_dir)
    result = process(_job(bucket, [("a", 1.0, 3.5), ("b", 4.0, 7.0)]))
    validate_result(result)
    assert result["status"] == "completed", result
    assert result["source"]["width"] == 1280
    assert [c["status"] for c in result["clips"]] == ["completed", "completed"]

    a, b = _probe(bucket, "a"), _probe(bucket, "b")
    assert (a.width, a.height) == (1080, 1920) and a.faststart is True
    assert a.video_codec == "h264" and a.audio_codec == "aac"
    assert abs(a.duration_seconds - 2.5) < 0.2 and abs(b.duration_seconds - 3.0) < 0.2
    assert result["clips"][0]["thumbnail"]["height"] == 1920
    assert os.path.exists(os.path.join(bucket["root"], "clips", "a.jpg"))


@needs_ffmpeg
def test_blur_fit_and_a_custom_canvas(bucket, fixtures_dir):
    _stage(bucket, fixtures_dir)
    result = process(_job(bucket, [("square", 0, 2)], frame={"width": 720, "height": 720, "fit": "blur"}))
    assert result["status"] == "completed", result
    out = _probe(bucket, "square")
    assert (out.width, out.height) == (720, 720)


@needs_ffmpeg
def test_one_bad_clip_makes_the_job_partial_and_keeps_the_good_one(bucket, fixtures_dir):
    _stage(bucket, fixtures_dir)
    result = process(_job(bucket, [("good", 0, 2), ("past-the-end", 60, 70)]))
    validate_result(result)
    assert result["status"] == "partial" and result["failure"] is None
    good, bad = result["clips"]
    assert good["status"] == "completed"
    assert bad["status"] == "failed" and bad["failure"]["code"] == errors.INVALID_JOB
    assert os.path.exists(os.path.join(bucket["root"], "clips", "good.mp4"))


@needs_ffmpeg
def test_a_clip_running_past_the_source_is_clamped(bucket, fixtures_dir):
    _stage(bucket, fixtures_dir)
    result = process(_job(bucket, [("tail", 6, 20)]))
    assert result["status"] == "completed", result
    assert result["clips"][0]["end_seconds"] == pytest.approx(8, abs=0.2)
    assert abs(_probe(bucket, "tail").duration_seconds - 2) < 0.3


@needs_ffmpeg
def test_every_clip_failing_fails_the_job_with_the_first_failure(bucket, fixtures_dir):
    _stage(bucket, fixtures_dir)
    result = process(_job(bucket, [("x", 100, 110)]))
    validate_result(result)
    assert result["status"] == "failed" and result["failure"]["code"] == errors.INVALID_JOB


@needs_ffmpeg
def test_captions_are_burned_in_when_libass_is_available(bucket, fixtures_dir):
    if not has_filter(get_settings().ffmpeg_bin, "ass"):
        pytest.skip("this ffmpeg build has no libass")
    _stage(bucket, fixtures_dir)
    words = [
        {"text": "outside", "start": 0.1, "end": 0.5},
        {"text": "Hello", "start": 2.1, "end": 2.5},
        {"text": "from", "start": 2.5, "end": 2.8},
        {"text": "Postiz.", "start": 2.8, "end": 3.4},
        {"text": "Again", "start": 3.6, "end": 3.9},
    ]
    plain = process(_job(bucket, [("plain", 2, 4)]))
    captioned = process(_job(bucket, [("cap", 2, 4)], captions={"words": words, "style": {"font": "DejaVu Sans"}}))
    validate_result(captioned)
    assert captioned["status"] == "completed", captioned
    assert captioned["clips"][0]["caption_lines"] == 2
    # same cut, same encoder settings: burned-in text is the only thing that can change the bytes
    assert captioned["clips"][0]["output"]["bytes"] != plain["clips"][0]["output"]["bytes"]


def test_missing_source_fails_the_whole_job(bucket):
    job = _job(bucket, [("a", 0, 1)])
    job["source"]["url"] = f"{bucket['base']}/nope.mp4"
    result = process(job)
    validate_result(result)
    assert result["status"] == "failed" and result["failure"]["code"] == errors.DOWNLOAD_FAILED
    assert result["clips"] == []


@pytest.mark.parametrize(
    "mutate, fragment",
    [
        (lambda j: j["clips"][0].update(end_seconds=0.5, start_seconds=1), "end_seconds must be above"),
        (lambda j: j["clips"].append(dict(j["clips"][0])), "duplicate reference"),
        (lambda j: j["clips"][0].update(end_seconds=5000), "max_clip_seconds"),
        (lambda j: j.update(frame={"width": 1081}), "frame.width"),
        (lambda j: j.update(clips=[]), "clips"),
        (lambda j: j.update(captions={"words": [], "style": {"font": "A,B"}}), "captions.style.font"),
    ],
)
def test_invalid_clip_jobs_are_rejected(bucket, mutate, fragment):
    job = _job(bucket, [("a", 0, 1)])
    mutate(job)
    result = process(job)
    validate_result(result)
    assert result["type"] == "clip" and result["failure"]["code"] == errors.INVALID_JOB
    assert fragment in result["failure"]["message"]


def test_defaults_are_applied(bucket):
    job = parse_job(_job(bucket, [("a", 0, 1)]))
    assert (job.frame.width, job.frame.height, job.frame.fit) == (1080, 1920, "crop")
    assert job.captions is None and job.video.quality == 23 and job.limits.max_clip_seconds == 600


def test_filter_graph_shapes():
    crop = clip_filter(
        Frame(), focus_x=0.25, focus_y=0.5, fps_cap=30, tonemap=False, pixel_format="yuv420p", ass_path=None,
        fonts_dir=None,
    )
    assert crop.startswith("fps=fps=30,scale=1080:1920:force_original_aspect_ratio=increase")
    assert "crop=1080:1920:(iw-ow)*0.2500:(ih-oh)*0.5000" in crop and crop.endswith("setsar=1,format=yuv420p")

    blur = clip_filter(
        Frame(fit="blur"), focus_x=0.5, focus_y=0.5, fps_cap=None, tonemap=False, pixel_format="yuv420p",
        ass_path="/w/c.ass", fonts_dir="/fonts",
    )
    assert blur.startswith("split=2[bg][fg];") and "overlay=(W-w)/2:(H-h)/2" in blur
    assert "ass=filename='/w/c.ass':fontsdir='/fonts'" in blur


def test_filter_path_survives_both_parsing_passes():
    assert filter_path("/tmp/a:b") == "'/tmp/a\\:b'"
    assert filter_path("/tmp/it's") == "'/tmp/it\\'\\''s'"
