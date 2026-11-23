"""slideshow jobs end to end: real ffmpeg, the local bucket stand-in, generated fixtures."""

from __future__ import annotations

import os
import shutil

import pytest

from postiz_uploader import errors
from postiz_uploader.captions import build_cue_ass
from postiz_uploader.config import get_settings
from postiz_uploader.ffmpeg import build_slideshow, has_filter
from postiz_uploader.pipeline import process
from postiz_uploader.probe import probe
from postiz_uploader.schema import SubtitleCue, SubtitleStyle, parse_job, validate_result
from postiz_uploader.slideshow import timeline
from tests.conftest import needs_ffmpeg

IMAGE = "slide-64.png"


@pytest.fixture(autouse=True)
def _work(work_dir):
    yield


def _voice(fixtures_dir) -> str:
    return "voice-1s.mp3" if os.path.exists(os.path.join(fixtures_dir, "voice-1s.mp3")) else "voice-1s.wav"


def _stage(bucket, fixtures_dir) -> str:
    voice = _voice(fixtures_dir)
    for name in (IMAGE, voice):
        shutil.copy(os.path.join(fixtures_dir, name), os.path.join(bucket["root"], name))
    return voice


def _job(bucket, slides: int, voice: str = "voice-1s.mp3", name: str = "show", **extra) -> dict:
    return {
        "version": 1,
        "type": "slideshow",
        "reference": "video_1",
        "slides": [
            {"image": {"url": f"{bucket['base']}/{IMAGE}"}, "audio": {"url": f"{bucket['base']}/{voice}"}}
            for _ in range(slides)
        ],
        "output": {"url": f"{bucket['base']}/out/{name}.mp4?X-Amz-Signature=fake"},
        **extra,
    }


def _probe(bucket, name="show"):
    return probe(get_settings().ffprobe_bin, os.path.join(bucket["root"], "out", f"{name}.mp4"))


def test_timeline_subtracts_one_crossfade_per_boundary():
    starts, total = timeline([2.0, 3.0, 4.0], 0.5)
    assert starts == [0.0, 1.5, 4.0] and total == 8.0
    starts, total = timeline([2.0, 3.0], 0)
    assert starts == [0.0, 2.0] and total == 5.0
    assert timeline([2.0], 0.5) == ([0.0], 2.0)


def test_cues_are_clamped_to_the_video():
    cues = (SubtitleCue(0, 2, "one"), SubtitleCue(4.5, 7, "two {tag}"), SubtitleCue(6, 8, "late"))
    document, kept = build_cue_ass(cues, SubtitleStyle(), width=1080, height=1920, total=5.0)
    assert kept == 2
    assert "Dialogue: 0,0:00:00.00,0:00:02.00,Default,,0,0,0,,one" in document
    assert "Dialogue: 0,0:00:04.50,0:00:05.00,Default,,0,0,0,,two (tag)" in document
    assert "late" not in document and "Alignment, MarginL" in document and ",5," in document


def test_assemble_graph_shapes():
    rules = parse_job(_job({"base": "https://bucket.example.com"}, 1)).video
    joined = build_slideshow(
        "ffmpeg", ["a.mp4", "b.mp4", "c.mp4"], "o.mp4", offsets=[0, 1.5, 4.0], transition=0.5, rules=rules,
        encoder="libx264", ass_path="/w/s.ass", fonts_dir=None,
    )
    graph = joined[joined.index("-filter_complex") + 1]
    assert "[0:v][1:v]xfade=transition=fade:duration=0.5:offset=1.500[v1]" in graph
    assert "[v1][2:v]xfade=transition=fade:duration=0.5:offset=4.000[v2]" in graph
    assert "[a1][2:a]acrossfade=d=0.5[a2]" in graph and "[v2]ass=filename='/w/s.ass'[vs]" in graph
    assert "[vs]format=yuv420p[vout]" in graph
    assert joined[joined.index("-map") + 1] == "[vout]"

    cut = build_slideshow(
        "ffmpeg", ["a.mp4", "b.mp4"], "o.mp4", offsets=[0, 2], transition=0, rules=rules, encoder="libx264",
        ass_path=None, fonts_dir=None,
    )
    assert cut[cut.index("-filter_complex") + 1] == "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[vc][ac];[vc]format=yuv420p[vout]"


def test_defaults_are_applied(bucket):
    job = parse_job(_job(bucket, 2))
    assert (job.frame.width, job.frame.height, job.fit) == (1080, 1920, "contain")
    assert (job.audio_delay_seconds, job.tail_seconds, job.transition_seconds) == (0.5, 1.0, 0.5)
    assert job.subtitles is None and job.video.fps == 30 and job.limits.timeout_seconds == 1200
    assert job.output.content_type == "video/mp4" and job.slides[0].duration_seconds is None


@pytest.mark.parametrize(
    "mutate, fragment",
    [
        (lambda j: j.update(slides=[]), "slides"),
        (lambda j: j.update(slides=j["slides"] * 51), "slides"),
        (lambda j: j.update(fit="stretch"), "fit"),
        (lambda j: j.update(tail_seconds=0.2), "tail_seconds must not be below"),
        (lambda j: j.update(subtitles={"cues": [{"start": 2, "end": 1, "text": "x"}]}), "end must be above"),
        (lambda j: j.update(frame={"width": 1081}), "frame.width"),
        (lambda j: j["slides"][0].pop("audio"), "audio"),
    ],
)
def test_invalid_slideshow_jobs_are_rejected(bucket, mutate, fragment):
    job = _job(bucket, 1)
    mutate(job)
    result = process(job)
    validate_result(result)
    assert result["type"] == "slideshow" and result["status"] == "failed"
    assert result["failure"]["code"] == errors.INVALID_JOB and fragment in result["failure"]["message"]
    assert result["output"] is None


def test_missing_audio_fails_the_job(bucket, fixtures_dir):
    _stage(bucket, fixtures_dir)
    result = process(_job(bucket, 1, voice="nope.mp3"))
    validate_result(result)
    assert result["status"] == "failed" and result["failure"]["code"] == errors.DOWNLOAD_FAILED


@needs_ffmpeg
def test_three_slides_are_crossfaded_into_one_video(bucket, fixtures_dir):
    voice = _stage(bucket, fixtures_dir)
    result = process(_job(bucket, 3, voice=voice))
    validate_result(result)
    assert result["status"] == "completed", result
    assert result["output"]["slides"] == 3 and result["output"]["subtitle_cues"] == 0
    out = _probe(bucket)
    assert (out.width, out.height) == (1080, 1920) and out.faststart is True
    assert out.video_codec == "h264" and out.audio_codec == "aac" and abs(out.fps - 30) < 0.01
    # 3 x (1 s voice + 1 s tail) minus two 0.5 s crossfades
    assert out.duration_seconds == pytest.approx(5.0, abs=0.3)


@needs_ffmpeg
def test_hard_cuts_cover_fit_and_a_fixed_duration(bucket, fixtures_dir):
    voice = _stage(bucket, fixtures_dir)
    job = _job(
        bucket, 2, voice=voice, name="cut", transition_seconds=0, fit="cover", frame={"width": 1920, "height": 1080}
    )
    job["slides"][1]["duration_seconds"] = 3
    result = process(job)
    assert result["status"] == "completed", result
    out = _probe(bucket, "cut")
    assert (out.width, out.height) == (1920, 1080)
    assert out.duration_seconds == pytest.approx(5.0, abs=0.3)


@needs_ffmpeg
def test_subtitles_are_burned_in_when_libass_is_available(bucket, fixtures_dir):
    if not has_filter(get_settings().ffmpeg_bin, "ass"):
        pytest.skip("this ffmpeg build has no libass")
    voice = _stage(bucket, fixtures_dir)
    plain = process(_job(bucket, 2, voice=voice, name="plain"))
    subtitled = process(
        _job(
            bucket,
            2,
            voice=voice,
            name="sub",
            subtitles={
                "cues": [{"start": 0, "end": 2, "text": "Hello"}, {"start": 1.5, "end": 3.5, "text": "world"}],
                "style": {"font": "DejaVu Sans"},
            },
        )
    )
    validate_result(subtitled)
    assert subtitled["status"] == "completed", subtitled
    assert subtitled["output"]["subtitle_cues"] == 2
    # same slides, same encoder settings: burned-in text is the only thing that can change the bytes
    assert subtitled["output"]["bytes"] != plain["output"]["bytes"]


@needs_ffmpeg
def test_a_transition_longer_than_a_slide_is_rejected(bucket, fixtures_dir):
    voice = _stage(bucket, fixtures_dir)
    job = _job(bucket, 2, voice=voice, transition_seconds=3)
    job["slides"][0]["duration_seconds"] = 2.5
    result = process(job)
    assert result["status"] == "failed" and result["failure"]["code"] == errors.INVALID_JOB
    assert "transition_seconds" in result["failure"]["message"]


@needs_ffmpeg
def test_workdir_is_cleaned(bucket, fixtures_dir, work_dir):
    voice = _stage(bucket, fixtures_dir)
    process(_job(bucket, 1, voice=voice, name="one"))
    assert os.listdir(work_dir) == []
