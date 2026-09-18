"""ingest jobs end to end. yt-dlp's generic extractor accepts a direct media URL, so the
`ytdlp` route is exercised against the local bucket with no network."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess

import pytest

from postiz_uploader import errors
from postiz_uploader.config import get_settings
from postiz_uploader.pipeline import process
from postiz_uploader.probe import probe
from postiz_uploader.schema import validate_result
from postiz_uploader.ytdlp import classify
from tests.conftest import needs_ffmpeg

needs_ytdlp = pytest.mark.skipif(importlib.util.find_spec("yt_dlp") is None, reason="yt-dlp not installed")


@pytest.fixture(autouse=True)
def _work(work_dir):
    yield


def _job(bucket, name, *, via="direct", video=True, audio=True, **extra) -> dict:
    job = {
        "version": 1,
        "type": "ingest",
        "reference": "source_1",
        "source": {"url": f"{bucket['base']}/{name}", "via": via},
        **extra,
    }
    if video:
        job["video"] = {"url": f"{bucket['base']}/ingest/{via}-source.mp4?X-Amz-Signature=fake"}
    if audio:
        job["audio"] = {"url": f"{bucket['base']}/ingest/{via}-audio.ogg"}
    return job


def _streams(path: str) -> list[str]:
    cmd = [get_settings().ffprobe_bin, "-v", "error", "-show_entries", "stream=codec_name,sample_rate,channels",
           "-of", "csv=p=0", path]
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.split()


def _stage(bucket, fixtures_dir, name):
    shutil.copy(os.path.join(fixtures_dir, name), os.path.join(bucket["root"], name))


@needs_ffmpeg
def test_direct_source_gets_a_faststart_mp4_and_speech_audio(bucket, fixtures_dir):
    _stage(bucket, fixtures_dir, "compliant-no-faststart.mp4")
    result = process(_job(bucket, "compliant-no-faststart.mp4"))
    validate_result(result)
    assert result["status"] == "completed", result
    assert result["source"]["via"] == "direct" and result["source"]["width"] == 1920
    assert result["video"]["video_codec"] == "h264"

    video = probe(get_settings().ffprobe_bin, os.path.join(bucket["root"], "ingest/direct-source.mp4"))
    assert video.faststart is True and (video.width, video.height) == (1920, 1080)
    streams = _streams(os.path.join(bucket["root"], "ingest/direct-audio.ogg"))
    assert streams == ["opus,16000,1"] or streams == ["opus,48000,1"]  # ffprobe reports Opus at its 48k clock
    assert result["audio"]["sample_rate"] == 16000 and result["audio"]["bytes"] < 40_000


@needs_ffmpeg
def test_audio_only_ingest_uploads_no_video(bucket, fixtures_dir):
    _stage(bucket, fixtures_dir, "talk-8s.mp4")
    result = process(_job(bucket, "talk-8s.mp4", video=False))
    assert result["status"] == "completed", result
    assert result["video"] is None and result["audio"]["codec"] == "opus"


@needs_ffmpeg
def test_too_long_fails_but_still_reports_the_duration(bucket, fixtures_dir):
    _stage(bucket, fixtures_dir, "talk-8s.mp4")
    result = process(_job(bucket, "talk-8s.mp4", limits={"max_duration_seconds": 5}))
    validate_result(result)
    assert result["failure"]["code"] == errors.DURATION_TOO_LONG
    assert result["source"]["duration_seconds"] == pytest.approx(8, abs=0.2)


@needs_ffmpeg
def test_a_source_without_audio_cannot_feed_a_transcript(bucket, fixtures_dir):
    _stage(bucket, fixtures_dir, "no-audio.mp4")
    result = process(_job(bucket, "no-audio.mp4"))
    assert result["failure"]["code"] == errors.UNSUPPORTED_INPUT


@needs_ffmpeg
@needs_ytdlp
def test_ytdlp_route_downloads_through_the_generic_extractor(bucket, fixtures_dir, monkeypatch):
    monkeypatch.setenv("ALLOWED_INGEST_HOSTS", "127.0.0.1")
    _stage(bucket, fixtures_dir, "talk-8s.mp4")
    result = process(_job(bucket, "talk-8s.mp4", via="ytdlp"))
    validate_result(result)
    assert result["status"] == "completed", result
    assert result["source"]["via"] == "ytdlp" and result["source"]["proxied"] is False
    assert result["source"]["extractor"] and result["source"]["height"] == 720
    assert os.path.exists(os.path.join(bucket["root"], "ingest/ytdlp-source.mp4"))
    assert os.path.exists(os.path.join(bucket["root"], "ingest/ytdlp-audio.ogg"))


def test_ytdlp_hosts_have_their_own_allowlist(bucket):
    # 127.0.0.1 is an allowed *direct* host in tests, which must not leak into the ytdlp route
    result = process(_job(bucket, "x.mp4", via="ytdlp"))
    validate_result(result)
    assert result["failure"]["code"] == errors.SOURCE_HOST_NOT_ALLOWED


def test_an_ingest_job_needs_at_least_one_output(bucket):
    result = process(_job(bucket, "x.mp4", video=False, audio=False))
    validate_result(result)
    assert result["type"] == "ingest" and result["failure"]["code"] == errors.INVALID_JOB


@pytest.mark.parametrize(
    "stderr, code, retryable",
    [
        ("ERROR: [youtube] abc: Sign in to confirm you’re not a bot.", errors.SOURCE_BLOCKED, True),
        ("ERROR: unable to download video data: HTTP Error 403: Forbidden", errors.SOURCE_BLOCKED, True),
        ("ERROR: [youtube] abc: Private video. Sign in if you've been granted", errors.SOURCE_UNAVAILABLE, False),
        ("ERROR: [youtube] abc: Video unavailable", errors.SOURCE_UNAVAILABLE, False),
        ("ERROR: Unsupported URL: https://example.com", errors.UNSUPPORTED_INPUT, False),
        ("ERROR: Connection reset by peer", errors.DOWNLOAD_FAILED, True),
    ],
)
def test_ytdlp_failures_are_classified(stderr, code, retryable):
    assert classify(stderr) == (code, retryable)


def test_proxy_pool_accepts_urls_and_provider_lists():
    from postiz_uploader.config import parse_proxies

    raw = "http://u:p@1.1.1.1:80, socks5://2.2.2.2:1080\n3.3.3.3:12323:user:pass\n4.4.4.4:8080 http://u:p@1.1.1.1:80"
    assert parse_proxies(raw) == (
        "http://u:p@1.1.1.1:80",
        "socks5://2.2.2.2:1080",
        "http://user:pass@3.3.3.3:12323",
        "http://4.4.4.4:8080",
    )
    assert parse_proxies("  ") == ()
    with pytest.raises(RuntimeError):
        parse_proxies("a:b:c")


def _ingest_job(bucket, **source):
    from postiz_uploader.schema import parse_job

    job = _job(bucket, "x.mp4", via="ytdlp")
    job["source"].update(source)
    return parse_job(job)


def test_routes_go_direct_first_then_sample_the_pool(bucket, monkeypatch):
    from postiz_uploader.ingest import _routes

    pool = ",".join(f"http://10.0.0.{i}:80" for i in range(1, 9))
    monkeypatch.setenv("INGEST_PROXY", pool)
    seen = set()
    for _ in range(50):
        routes = _routes(_ingest_job(bucket), get_settings())
        assert routes[0] is None and len(routes) == 3 and len(set(routes)) == 3
        seen.update(routes[1:])
    assert len(seen) > 2  # sampled, not always the head of the list

    monkeypatch.setenv("INGEST_DIRECT_FIRST", "false")
    monkeypatch.setenv("INGEST_PROXY_ATTEMPTS", "5")
    assert None not in _routes(_ingest_job(bucket), get_settings())
    assert len(_routes(_ingest_job(bucket), get_settings())) == 5

    # a job's own proxy replaces the pool
    assert _routes(_ingest_job(bucket, proxy="http://9.9.9.9:1"), get_settings()) == ["http://9.9.9.9:1"]

    monkeypatch.delenv("INGEST_PROXY")
    assert _routes(_ingest_job(bucket), get_settings()) == [None]


def test_a_blocked_route_falls_through_to_the_next_and_credentials_stay_out_of_the_result(bucket, monkeypatch):
    from postiz_uploader import ytdlp
    from postiz_uploader.errors import JobError

    monkeypatch.setenv("ALLOWED_INGEST_HOSTS", "127.0.0.1")
    monkeypatch.setenv("INGEST_PROXY", "http://user:secret@10.0.0.1:80,http://user:secret@10.0.0.2:80")
    tried = []

    def blocked(url, workdir, *, proxy, **kw):
        tried.append(proxy)
        raise JobError(errors.SOURCE_BLOCKED, "Sign in to confirm you're not a bot", retryable=True)

    monkeypatch.setattr(ytdlp, "fetch_metadata", blocked)
    result = process(_job(bucket, "x.mp4", via="ytdlp"))
    validate_result(result)
    assert tried[0] is None and len(tried) == 3 and len(set(tried)) == 3
    assert result["failure"]["code"] == errors.SOURCE_BLOCKED and result["failure"]["retryable"] is True
    assert "secret" not in str(result)


def test_without_a_pot_server_nothing_starts_and_the_default_player_clients_are_kept(bucket, monkeypatch):
    from postiz_uploader import pot, ytdlp
    from postiz_uploader.errors import JobError

    monkeypatch.setenv("ALLOWED_INGEST_HOSTS", "127.0.0.1")
    monkeypatch.delenv("POT_SERVER_DIR", raising=False)
    monkeypatch.setattr(pot, "_spawn", lambda settings: pytest.fail("no POT_SERVER_DIR, nothing to start"))
    seen = []

    def blocked(url, workdir, *, player_clients, **kw):
        seen.append(player_clients)
        raise JobError(errors.SOURCE_BLOCKED, "Sign in to confirm you're not a bot", retryable=True)

    monkeypatch.setattr(ytdlp, "fetch_metadata", blocked)
    process(_job(bucket, "x.mp4", via="ytdlp"))
    assert seen == [None]


def test_a_running_pot_server_adds_the_player_client_that_uses_its_tokens(bucket, monkeypatch):
    from postiz_uploader import pot, ytdlp
    from postiz_uploader.errors import JobError

    monkeypatch.setenv("ALLOWED_INGEST_HOSTS", "127.0.0.1")
    monkeypatch.setattr(pot, "ensure", lambda settings, **kw: True)
    seen = []

    def blocked(url, workdir, *, player_clients, **kw):
        seen.append(player_clients)
        raise JobError(errors.SOURCE_BLOCKED, "Sign in to confirm you're not a bot", retryable=True)

    monkeypatch.setattr(ytdlp, "fetch_metadata", blocked)
    process(_job(bucket, "x.mp4", via="ytdlp"))
    assert seen == ["default,mweb"]


def test_a_pot_server_that_cannot_start_costs_the_token_not_the_job(tmp_path, monkeypatch):
    from postiz_uploader import pot

    # a directory with no server in it, then one whose process dies at once
    monkeypatch.setenv("POT_SERVER_DIR", str(tmp_path))
    monkeypatch.setattr(pot, "_listening", lambda: False)
    assert pot.ensure(get_settings()) is False

    monkeypatch.setattr(pot, "_spawn", lambda settings: subprocess.Popen(["false"]))
    assert pot.ensure(get_settings()) is False
    pot.stop()
