"""The oxylabs ingest route. Their API is faked at `oxylabs._request`; the bucket they
deliver into is the local dev server, so signing, probing, download and cleanup are real."""

from __future__ import annotations

import datetime
import json
import os
import shutil
import subprocess

import pytest

from postiz_uploader import errors, oxylabs
from postiz_uploader.config import get_settings
from postiz_uploader.errors import JobError
from postiz_uploader.pipeline import process
from postiz_uploader.schema import validate_result
from tests.conftest import needs_ffmpeg

VIDEO_ID = "aqz-KE-bpKQ"
FIXTURE = "compliant-no-faststart.mp4"

ASR = {
    "events": [
        {"tStartMs": 0, "dDurationMs": 4000, "segs": [{"utf8": "hello"}, {"utf8": " big", "tOffsetMs": 600}]},
        {"tStartMs": 1500, "dDurationMs": 10, "aAppend": 1, "segs": [{"utf8": "\n"}]},
        {"tStartMs": 1500, "dDurationMs": 9000, "segs": [{"utf8": "world", "tOffsetMs": 0}]},
    ]
}
UPLOADER = "uploader_provided"
BIG_THUMB = "https://i.ytimg.com/hq.jpg"
UPLOADED = {"events": [{"tStartMs": 1000, "dDurationMs": 2500, "segs": [{"utf8": "Hello big\nworld"}]}]}


@pytest.fixture(autouse=True)
def _work(work_dir):
    yield


class FakeOxylabs:
    """Answers the three sources; a download "delivers" a fixture into the local bucket."""

    def __init__(
        self, bucket, fixtures_dir, *, captions=None, duration=600, extension="mp4", hints=True, bad_metadata=0
    ):
        self.bucket, self.fixtures_dir, self.bad_metadata = bucket, fixtures_dir, bad_metadata
        self.captions, self.duration, self.extension, self.hints = captions or {}, duration, extension, hints
        self.calls: list[dict] = []

    def __call__(self, settings, method, url, payload, timeout):
        if method == "GET":
            return {"status": "done"}
        self.calls.append(payload)
        context = {c["key"]: c["value"] for c in payload.get("context", [])}
        if payload["source"] == "youtube_metadata":
            assert payload["parse"] is True  # the source answers HTTP 400 without it
            if self.bad_metadata > 0:
                # seen live: 200 with content that is not the parsed object
                self.bad_metadata -= 1
                return {"results": [{"status_code": 200, "content": "<html>"}]}
            # the real thing: numbers and booleans arrive as strings
            data = {
                "title": "Big Buck Bunny",
                "uploader": "Blender",
                "duration": str(self.duration),
                "is_live": "False",
                "thumbnails": [{"url": "https://i.ytimg.com/s.jpg", "width": 120}, {"url": BIG_THUMB, "width": 480}],
            }
            if self.hints:
                data["is_transcript_available"] = str(bool(self.captions))
                data["user_subtitle_languages"] = [lang for lang, origin in self.captions if origin == UPLOADER]
            return {"results": [{"status_code": 200, "content": {"results": data}}]}
        if payload["source"] == "youtube_subtitles":
            language, origin = context["language_code"], context["subtitle_origin"]
            found = self.captions.get((language, origin))
            # the real thing: 200 either way, json3 nested under origin and language, {} for none
            content = {origin: {language: {"wireMagic": "pb3", **found}}} if found else {}
            return {"results": [{"status_code": 200, "content": content}]}
        assert payload["storage_type"] == "s3_compatible" and "SECRET" in payload["storage_url"]
        target = os.path.join(self.bucket["root"], "oxy", "in", f"{VIDEO_ID}_job1.{self.extension}")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if context.get("download_type") == "audio":
            # the real thing: an audio download is raw ADTS AAC, there is no picture in it
            subprocess.run(
                [get_settings().ffmpeg_bin, "-v", "error", "-y", "-i", os.path.join(self.fixtures_dir, FIXTURE),
                 "-vn", "-c:a", "aac", "-f", "adts", target],
                check=True,
            )
        else:
            shutil.copy(os.path.join(self.fixtures_dir, FIXTURE), target)
        return {"id": "job1", "status": "pending"}

    def downloads(self) -> list[dict]:
        return [
            {c["key"]: c["value"] for c in p["context"]} for p in self.calls if p["source"] == "youtube_download"
        ]


@pytest.fixture
def configured(bucket, monkeypatch):
    monkeypatch.setenv("OXYLABS_USERNAME", "user")
    monkeypatch.setenv("OXYLABS_PASSWORD", "hunter2")
    monkeypatch.setenv("OXYLABS_STORAGE_URL", bucket["base"].replace("http://", "http://KEY:SECRET@") + "/oxy/in")
    monkeypatch.setattr(oxylabs, "POLL_SECONDS", 0)
    monkeypatch.setattr(oxylabs, "METADATA_RETRY_SECONDS", 0)


@pytest.fixture
def fake(bucket, fixtures_dir, configured, monkeypatch):
    def build(**kw):
        api = FakeOxylabs(bucket, fixtures_dir, **kw)
        monkeypatch.setattr(oxylabs, "_request", api)
        return api

    return build


def _job(bucket, *, video=False, audio=False, transcript=False, **source) -> dict:
    job = {
        "version": 1,
        "type": "ingest",
        "reference": "source_1",
        "source": {"url": f"https://www.youtube.com/watch?v={VIDEO_ID}", "via": "oxylabs", **source},
    }
    if video:
        job["video"] = {"url": f"{bucket['base']}/out/source.mp4"}
    if audio:
        job["audio"] = {"url": f"{bucket['base']}/out/audio.ogg", "unless_transcript": True}
    if transcript:
        job["transcript"] = {"url": f"{bucket['base']}/out/transcript.json", "languages": ["de", "en"]}
    return job


# ---------------------------------------------------------------- pure pieces


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=aqz-KE-bpKQ&t=10s",
        "https://youtu.be/aqz-KE-bpKQ?si=x",
        "https://www.youtube.com/shorts/aqz-KE-bpKQ",
        "https://www.youtube.com/live/aqz-KE-bpKQ",
        "https://m.youtube.com/embed/aqz-KE-bpKQ",
    ],
)
def test_video_id_from_every_url_shape(url):
    assert oxylabs.video_id(url) == VIDEO_ID


@pytest.mark.parametrize("url", ["https://www.youtube.com/@channel", "https://www.youtube.com/watch?v=short"])
def test_a_url_without_a_video_id_is_unsupported(url):
    with pytest.raises(JobError) as err:
        oxylabs.video_id(url)
    assert err.value.code == errors.UNSUPPORTED_INPUT


def test_quality_is_capped_and_snaps_down_to_a_rendition_oxylabs_offers():
    assert oxylabs.quality(1080, 720) == "720"
    assert oxylabs.quality(600, 720) == "480"
    assert oxylabs.quality(100, 720) == "144"


def test_a_window_is_widened_to_whole_seconds():
    assert oxylabs.trim_window(187.4, 220.2) == (187, 221)
    assert oxylabs.clock(3725) == "01:02:05"
    assert oxylabs.trim_window(None, None) is None
    with pytest.raises(JobError):
        oxylabs.trim_window(10, 10)


def test_auto_generated_captions_give_word_timings():
    segments, words, word_level = oxylabs.parse_captions({"content": json.dumps(ASR)})
    assert word_level is True
    assert [(w["text"], w["start"], w["end"]) for w in words] == [
        ("hello", 0.0, 0.6),
        ("big", 0.6, 1.5),
        # the last word runs to the end of its event, but never longer than a held caption
        ("world", 1.5, 3.5),
    ]
    assert [s["text"] for s in segments] == ["hello big", "world"]
    assert segments[0]["end"] == 1.5  # rolling captions overlap; the next line is the real end


def test_uploaded_captions_give_lines_and_no_invented_word_timings():
    segments, words, word_level = oxylabs.parse_captions(UPLOADED)
    assert (word_level, words) == (False, [])
    assert segments == [{"start": 1.0, "end": 3.5, "text": "Hello big world"}]


@pytest.mark.parametrize("content", [None, "not json", {"events": []}, {"events": [{"tStartMs": 0}]}, 42])
def test_unreadable_captions_are_no_captions(content):
    assert oxylabs.parse_captions(content) is None


def test_presigned_urls_are_stable_sigv4():
    storage = oxylabs.parse_storage("https://KEY:SE%2FCRET@acc.r2.cloudflarestorage.com/bucket/in/box", "auto")
    assert (storage.bucket, storage.prefix, storage.secret_key) == ("bucket", "in/box", "SE/CRET")
    now = datetime.datetime(2026, 9, 18, 12, 0, 0, tzinfo=datetime.timezone.utc)
    url = oxylabs.presign(storage, "GET", "in/box/a b.mp4", now=now)
    assert url.startswith("https://acc.r2.cloudflarestorage.com/bucket/in/box/a%20b.mp4?X-Amz-Algorithm=")
    assert "X-Amz-Credential=KEY%2F20260918%2Fauto%2Fs3%2Faws4_request" in url
    assert url == oxylabs.presign(storage, "GET", "in/box/a b.mp4", now=now)
    assert url.rsplit("=", 1)[1] != oxylabs.presign(storage, "HEAD", "in/box/a b.mp4", now=now).rsplit("=", 1)[1]
    with pytest.raises(RuntimeError):
        oxylabs.parse_storage("https://KEY:SECRET@host/bucket-with-no-folder", "auto")


# ---------------------------------------------------------------- jobs


def test_captions_replace_the_audio_and_nothing_is_downloaded(bucket, fake):
    api = fake(captions={("en", "auto_generated"): ASR})
    result = process(_job(bucket, audio=True, transcript=True))
    validate_result(result)
    assert result["status"] == "completed", result
    assert api.downloads() == []
    assert result["audio"] is None and result["video"] is None
    assert result["transcript"]["origin"] == "auto_generated" and result["transcript"]["word_level"] is True
    assert result["source"]["title"] == "Big Buck Bunny" and result["source"]["language"] == "en"
    assert result["source"]["thumbnail_url"] == BIG_THUMB and result["source"]["duration_seconds"] == 600
    with open(os.path.join(bucket["root"], "out", "transcript.json"), encoding="utf-8") as fh:
        stored = json.load(fh)
    assert [w["text"] for w in stored["words"]] == ["hello", "big", "world"]
    # de was asked first; the metadata says nobody uploaded de captions, so that miss is skipped
    asked = [c["value"] for p in api.calls if p["source"] == "youtube_subtitles" for c in p["context"]]
    assert asked == ["de", "auto_generated", "en", "auto_generated"]


def test_uploaded_captions_are_found_and_reported_as_line_level(bucket, fake):
    api = fake(captions={("en", UPLOADER): UPLOADED})
    result = process(_job(bucket, audio=True, transcript=True))
    validate_result(result)
    assert result["transcript"]["origin"] == UPLOADER and result["transcript"]["word_level"] is False
    assert (result["transcript"]["segments"], result["transcript"]["words"]) == (1, 0)
    assert api.downloads() == []


def test_without_metadata_hints_every_language_and_origin_is_asked(bucket, fake):
    api = fake(captions={("en", "auto_generated"): ASR}, hints=False)
    assert process(_job(bucket, transcript=True))["status"] == "completed"
    asked = [c["value"] for p in api.calls if p["source"] == "youtube_subtitles" for c in p["context"]]
    assert asked == ["de", "auto_generated", "de", UPLOADER, "en", "auto_generated"]


def test_a_video_the_metadata_says_has_no_captions_costs_no_caption_requests(bucket, fake):
    api = fake()
    result = process(_job(bucket, transcript=True))
    assert result["status"] == "completed" and result["transcript"] is None
    assert [p["source"] for p in api.calls] == ["youtube_metadata"]


@needs_ffmpeg
def test_without_captions_only_the_audio_is_downloaded(bucket, fake):
    api = fake(extension="m4a")
    result = process(_job(bucket, audio=True, transcript=True))
    validate_result(result)
    assert result["status"] == "completed", result
    assert api.downloads() == [{"download_type": "audio"}]
    assert result["transcript"] is None and result["audio"]["codec"] == "opus"
    assert os.path.getsize(os.path.join(bucket["root"], "out", "audio.ogg")) > 0


@needs_ffmpeg
def test_a_clip_window_is_fetched_at_the_capped_quality(bucket, fake):
    api = fake()
    result = process(_job(bucket, video=True, max_height=1080, start_seconds=187.4, end_seconds=220.2))
    validate_result(result)
    assert result["status"] == "completed", result
    assert api.downloads() == [
        {"download_type": "audio_video", "video_quality": "720", "start_at": "00:03:07", "end_at": "00:03:41"}
    ]
    assert result["source"]["trim"] == {"start_seconds": 187, "end_seconds": 221}
    assert result["source"]["duration_seconds"] == 600  # the whole video, not the window
    assert result["video"]["width"] == 1920


def test_a_window_cuts_the_transcript_to_its_own_clock(bucket, fake):
    fake(captions={("en", "auto_generated"): ASR})
    job = _job(bucket, transcript=True, start_seconds=1, end_seconds=3)
    job["transcript"]["languages"] = ["en"]
    assert process(job)["status"] == "completed"
    with open(os.path.join(bucket["root"], "out", "transcript.json"), encoding="utf-8") as fh:
        words = json.load(fh)["words"]
    assert [(w["text"], w["start"]) for w in words] == [("big", -0.4), ("world", 0.5)]


def test_a_video_over_the_duration_limit_is_refused_before_anything_is_paid_for(bucket, fake):
    api = fake(duration=10_000)
    result = process(_job(bucket, video=True))
    assert result["failure"]["code"] == errors.DURATION_TOO_LONG
    assert [p["source"] for p in api.calls] == ["youtube_metadata"]


def test_rejected_credentials_fail_the_job_for_good(bucket, configured, monkeypatch):
    class Unauthorized:
        status_code = 401

    monkeypatch.setattr(oxylabs.requests, "request", lambda *a, **kw: Unauthorized())
    result = process(_job(bucket, video=True))
    assert result["failure"]["code"] == errors.DOWNLOAD_FAILED and result["failure"]["retryable"] is False
    assert "hunter2" not in str(result)


def test_an_unconfigured_worker_and_the_other_routes_refuse_oxylabs_fields(bucket, monkeypatch):
    monkeypatch.delenv("OXYLABS_USERNAME", raising=False)
    assert process(_job(bucket, video=True))["failure"]["code"] == errors.INVALID_JOB
    job = _job(bucket, video=True, transcript=True)
    job["source"]["via"] = "ytdlp"
    assert process(job)["failure"]["code"] == errors.INVALID_JOB


def test_a_storage_url_without_a_folder_fails_once_and_before_the_download_is_paid_for(bucket, fake, monkeypatch):
    api = fake()
    no_folder = bucket["base"].replace("http://", "http://KEY:s3cr3tvalue@") + "/bucket-only"
    monkeypatch.setenv("OXYLABS_STORAGE_URL", no_folder)
    result = process(_job(bucket, video=True))
    assert result["failure"]["code"] == errors.INVALID_JOB and result["failure"]["retryable"] is False
    assert "bucket/folder" in result["failure"]["message"] and "s3cr3tvalue" not in str(result)
    assert api.downloads() == []


def test_an_unparsed_metadata_answer_is_retried_inside_the_job(bucket, fake):
    api = fake(captions={("en", "auto_generated"): ASR}, bad_metadata=2)
    result = process(_job(bucket, transcript=True))
    assert result["status"] == "completed" and result["source"]["title"] == "Big Buck Bunny"
    assert [p["source"] for p in api.calls].count("youtube_metadata") == 3


def test_metadata_that_never_parses_fails_the_job_as_retryable(bucket, fake):
    api = fake(bad_metadata=99)
    result = process(_job(bucket, video=True))
    assert result["failure"]["code"] == errors.DOWNLOAD_FAILED and result["failure"]["retryable"] is True
    assert "3 times" in result["failure"]["message"] and api.downloads() == []
