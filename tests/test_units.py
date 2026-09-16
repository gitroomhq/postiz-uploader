from __future__ import annotations

import os
import struct

import pytest

from postiz_uploader.download import host_allowed
from postiz_uploader.ffmpeg import build_encode, build_remux
from postiz_uploader.log import redact_url
from postiz_uploader.planner import ENCODE, Plan
from postiz_uploader.probe import _rotation, isobmff_faststart
from postiz_uploader.schema import VideoRules
from postiz_uploader.sniff import sniff
from tests.test_planner import info


@pytest.mark.parametrize(
    "url,allowed,ok",
    [
        ("https://bucket.example.com/x", ("bucket.example.com",), True),
        ("https://BUCKET.example.com/x", ("bucket.example.com",), True),
        ("https://other.example.com/x", ("bucket.example.com",), False),
        ("https://a.r2.example.com/x", ("*.r2.example.com",), True),
        ("https://r2.example.com/x", ("*.r2.example.com",), False),
        ("https://evil.com/x", (), False),
    ],
)
def test_host_allowed(url, allowed, ok):
    assert host_allowed(url, allowed) is ok


def test_redact_url_drops_query():
    assert redact_url("https://b.example.com/a.mp4?X-Amz-Signature=secret") == "https://b.example.com/a.mp4"


@pytest.mark.parametrize(
    "head,kind,fmt",
    [
        (b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00", "video", "mp4"),
        (b"\x00\x00\x00\x18ftypqt  \x00\x00\x02\x00", "video", "mp4"),
        (b"\x00\x00\x00\x18ftypheic\x00\x00\x02\x00", "image", "heic"),
        (b"\x1a\x45\xdf\xa3" + b"\x00" * 20, "video", "webm"),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "image", "webp"),
        (b"RIFF\x00\x00\x00\x00AVI LIST", "video", "avi"),
        (b"\x89PNG\r\n\x1a\n" + b"\x00" * 8, "image", "png"),
        (b"\xff\xd8\xff\xe1" + b"\x00" * 12, "image", "jpeg"),
        (b"GIF89a" + b"\x00" * 10, "image", "gif"),
    ],
)
def test_sniff(head, kind, fmt):
    got = sniff(head)
    assert got is not None
    assert (got.kind, got.format) == (kind, fmt)


def test_sniff_unknown():
    assert sniff(b"hello world, not a media file") is None


def _atom(kind: bytes, payload: bytes = b"") -> bytes:
    return struct.pack(">I4s", 8 + len(payload), kind) + payload


def test_faststart_detection(tmp_path):
    fast = tmp_path / "fast.mp4"
    fast.write_bytes(_atom(b"ftyp", b"isom" * 2) + _atom(b"moov", b"x" * 16) + _atom(b"mdat", b"y" * 64))
    slow = tmp_path / "slow.mp4"
    slow.write_bytes(_atom(b"ftyp", b"isom" * 2) + _atom(b"mdat", b"y" * 64) + _atom(b"moov", b"x" * 16))
    other = tmp_path / "other.bin"
    other.write_bytes(b"\x1a\x45\xdf\xa3" + b"\x00" * 64)
    assert isobmff_faststart(str(fast)) is True
    assert isobmff_faststart(str(slow)) is False
    assert isobmff_faststart(str(other)) is None


@pytest.mark.parametrize(
    "stream,expected",
    [
        ({}, 0),
        ({"tags": {"rotate": "90"}}, 90),
        ({"tags": {"rotate": "-90"}}, 270),
        ({"side_data_list": [{"rotation": -90}]}, 90),  # iPhone portrait
        ({"side_data_list": [{"rotation": 90}]}, 270),
        ({"side_data_list": [{"rotation": 180}]}, 180),
        ({"side_data_list": [{"rotation": -90}], "tags": {"rotate": "90"}}, 90),
    ],
)
def test_rotation_sign(stream, expected):
    assert _rotation(stream) == expected


def test_encode_command_cpu_shape():
    plan = Plan("encode", 1920, 1080, video_encode=True, audio_encode=True, scale=True, fps_cap=60, actions=())
    cmd = build_encode(
        "ffmpeg",
        "in",
        "out.mp4",
        plan,
        info(width=3840, height=2160, fps=120.0),
        VideoRules(),
        encoder="libx264",
        cuda_decode=False,
    )
    joined = " ".join(cmd)
    assert "-vf fps=fps=60,scale=1920:1080:flags=lanczos,setsar=1,format=yuv420p" in joined
    assert "-c:v libx264" in joined and "-crf 23" in joined
    assert "-c:a aac" in joined and "-ar 48000" in joined
    assert "-movflags +faststart" in joined
    assert "-hwaccel" not in joined


def test_encode_command_gpu_shape():
    plan = Plan("encode", 1080, 1920, video_encode=True, scale=True, rotate=True, actions=())
    cmd = build_encode(
        "ffmpeg",
        "in",
        "out.mp4",
        plan,
        info(width=1080, height=1920, rotation=90),
        VideoRules(),
        encoder="h264_nvenc",
        cuda_decode=True,
    )
    joined = " ".join(cmd)
    assert "-hwaccel cuda -hwaccel_output_format cuda -noautorotate -display_rotation 0" in joined
    assert "transpose_npp=dir=clock,scale_npp=1080:1920:interp_algo=lanczos:format=yuv420p,setsar=1" in joined
    assert "-c:v h264_nvenc" in joined and "-cq 23" in joined
    assert "-c:a copy" in joined


def test_encode_command_tonemap_uses_cpu_chain():
    plan = Plan("encode", 1920, 1080, video_encode=True, scale=True, tonemap=True, actions=())
    cmd = build_encode(
        "ffmpeg",
        "in",
        "out.mp4",
        plan,
        info(width=3840, height=2160),
        VideoRules(),
        encoder="h264_nvenc",
        cuda_decode=False,
    )
    joined = " ".join(cmd)
    assert "zscale=t=linear" in joined and "tonemap=hable" in joined
    assert "scale=1920:1080:flags=lanczos,setsar=1,zscale" in joined
    with pytest.raises(ValueError):
        build_encode("ffmpeg", "in", "out.mp4", plan, info(), VideoRules(), encoder="h264_nvenc", cuda_decode=True)


def test_audio_only_encode_copies_video():
    plan = Plan("encode", 1920, 1080, video_encode=False, audio_encode=True, actions=())
    cmd = build_encode(
        "ffmpeg",
        "in",
        "out.mp4",
        plan,
        info(audio_sample_rate=44100),
        VideoRules(),
        encoder="libx264",
        cuda_decode=False,
    )
    joined = " ".join(cmd)
    assert "-c:v copy" in joined and "-vf" not in joined
    assert "-c:a aac" in joined


def test_remux_without_audio():
    cmd = build_remux("ffmpeg", "in", "out.mp4", info(audio_codec=None), VideoRules())
    assert "0:a:0" not in cmd and "-c" in cmd and "copy" in cmd


def test_workdir_env_is_used(monkeypatch, tmp_path):
    from postiz_uploader.config import get_settings

    monkeypatch.setenv("WORK_DIR", str(tmp_path))
    monkeypatch.setenv("ENCODER", "h264_nvenc")
    settings = get_settings()
    assert settings.work_dir == str(tmp_path)
    assert settings.worker_concurrency == 8
    assert settings.gpu is True
    assert os.environ["ALLOWED_SOURCE_HOSTS"]


def _encode_attempts(monkeypatch, tmp_path, fail_until: int):
    """Drive _run_encode with a fake ffmpeg.run that fails the first `fail_until`
    attempts, returning the (encoder, cuda) ladder it walked."""
    from postiz_uploader import ffmpeg as ff
    from postiz_uploader import video
    from postiz_uploader.config import Settings
    from postiz_uploader.schema import parse_job

    calls: list[tuple[str, bool]] = []

    def fake_run(cmd, *, timeout):
        calls.append(("h264_nvenc" if "h264_nvenc" in cmd else "libx264", "-hwaccel" in cmd))
        ok = len(calls) > fail_until
        stderr = "" if ok else "Cannot load libnvidia-encode.so.1"
        return ff.RunOutcome(returncode=0 if ok else 255, stderr_tail=stderr, seconds=0)

    monkeypatch.setattr(video.ffmpeg, "run", fake_run)
    job = parse_job(
        {
            "version": 1,
            "type": "video",
            "reference": "t",
            "source": {"url": "https://bucket.example.com/in.mp4"},
            "output": {"url": "https://bucket.example.com/out.mp4", "content_type": "video/mp4"},
            "rules": {"short_side_min": 1080, "short_side_max": 1080, "long_side_max": 1920},
        }
    )
    plan = Plan(kind=ENCODE, width=1920, height=1030, actions=("scale", "video_encode"), scale=True, video_encode=True)
    settings = Settings(encoder="h264_nvenc")
    result = video._run_encode(job, plan, info(), str(tmp_path / "in"), str(tmp_path / "out"), settings, deadline=1e9)
    return calls, result


def test_encode_falls_back_to_libx264_when_nvenc_cannot_open(monkeypatch, tmp_path):
    calls, (decode, encoder) = _encode_attempts(monkeypatch, tmp_path, fail_until=2)
    assert calls == [("h264_nvenc", True), ("h264_nvenc", False), ("libx264", False)]
    assert (decode, encoder) == ("software", "libx264")


def test_encode_keeps_nvenc_when_only_cuda_decode_fails(monkeypatch, tmp_path):
    calls, (decode, encoder) = _encode_attempts(monkeypatch, tmp_path, fail_until=1)
    assert calls == [("h264_nvenc", True), ("h264_nvenc", False)]
    assert (decode, encoder) == ("software", "h264_nvenc")
