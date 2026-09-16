from __future__ import annotations

import pytest

from postiz_uploader.planner import ENCODE, REMUX, UNCHANGED, clamp_dimensions, plan_video
from postiz_uploader.probe import SourceInfo
from postiz_uploader.schema import ImageRules, Rules, VideoRules

VIDEO_RULES = Rules(1080, 1080, 1920, VideoRules(), ImageRules())
IMAGE_RULES = Rules(720, 1080, 1920, VideoRules(), ImageRules())


@pytest.mark.parametrize(
    "size,rules,even,expected",
    [
        ((1920, 1080), VIDEO_RULES, True, (1920, 1080)),  # already compliant
        ((3840, 2160), VIDEO_RULES, True, (1920, 1080)),  # 4K landscape down
        ((2160, 3840), VIDEO_RULES, True, (1080, 1920)),  # 4K portrait down
        ((854, 480), VIDEO_RULES, True, (1920, 1080)),  # 480p up
        ((640, 480), VIDEO_RULES, True, (1440, 1080)),  # 4:3 up
        ((1080, 1080), VIDEO_RULES, True, (1080, 1080)),  # square stays
        ((2560, 1080), VIDEO_RULES, True, (1920, 810)),  # ultrawide: long cap wins
        ((1079, 1919), VIDEO_RULES, True, (1080, 1920)),  # odd, just under
        ((1920, 1079), VIDEO_RULES, True, (1920, 1080)),  # short side 1079 -> 1080, long rounds to 1922 -> capped 1920
        ((400, 300), IMAGE_RULES, False, (960, 720)),  # small image up to 720
        ((1600, 900), IMAGE_RULES, False, (1600, 900)),  # image inside bounds
        ((4000, 3000), IMAGE_RULES, False, (1440, 1080)),  # large image down
        ((3000, 4000), IMAGE_RULES, False, (1080, 1440)),  # portrait image down
        ((8000, 1000), IMAGE_RULES, False, (1920, 240)),  # panorama: long cap pushes short below min
        (
            (1919, 1080),
            Rules(1080, 1080, 1919, VideoRules(), ImageRules()),
            True,
            (1918, 1080),
        ),  # odd cap never exceeded
        ((1920, 1081), Rules(1080, 1081, 1920, VideoRules(), ImageRules()), True, (1920, 1080)),  # odd short cap
    ],
)
def test_clamp_dimensions(size, rules, even, expected):
    assert clamp_dimensions(*size, rules, even=even) == expected


def info(**overrides) -> SourceInfo:
    base = dict(
        container="mp4",
        video_codec="h264",
        profile="High",
        pixel_format="yuv420p",
        color_transfer="bt709",
        color_primaries="bt709",
        width=1920,
        height=1080,
        rotation=0,
        fps=30.0,
        duration_seconds=10.0,
        bytes=1000,
        audio_codec="aac",
        audio_sample_rate=48000,
        faststart=True,
    )
    base.update(overrides)
    return SourceInfo(**base)


@pytest.mark.parametrize(
    "src,kind,actions",
    [
        (info(), UNCHANGED, ()),
        (info(faststart=False), REMUX, ("remux",)),
        (info(container="mov"), REMUX, ("remux",)),
        (info(audio_codec=None, faststart=False), REMUX, ("remux",)),
        (
            info(video_codec="hevc", width=3840, height=2160, container="mov"),
            ENCODE,
            ("video_encode", "scale", "remux"),
        ),
        (
            info(
                color_transfer="arib-std-b67",
                color_primaries="bt2020",
                pixel_format="yuv420p10le",
                video_codec="hevc",
                width=3840,
                height=2160,
            ),
            ENCODE,
            ("video_encode", "scale", "tonemap"),
        ),
        (info(rotation=90, width=1080, height=1920), ENCODE, ("video_encode", "rotate")),
        (
            info(width=854, height=480, video_codec="vp9", container="webm", audio_codec=None, faststart=None),
            ENCODE,
            ("video_encode", "scale", "remux"),
        ),
        (info(fps=120.0), ENCODE, ("video_encode", "fps_cap")),
        (info(fps=59.94), UNCHANGED, ()),
        (info(width=2560, height=1080), ENCODE, ("video_encode", "scale")),
        (info(audio_sample_rate=44100), ENCODE, ("audio_encode",)),
        (info(audio_codec="mp3"), ENCODE, ("audio_encode",)),
        (info(pixel_format="yuv422p"), ENCODE, ("video_encode",)),
    ],
)
def test_plan_video(src, kind, actions):
    plan = plan_video(src, VIDEO_RULES)
    assert plan.kind == kind
    assert plan.actions == actions


def test_audio_only_plan_copies_video():
    plan = plan_video(info(audio_sample_rate=44100), VIDEO_RULES)
    assert plan.video_encode is False
    assert plan.audio_encode is True
    assert (plan.width, plan.height) == (1920, 1080)


def test_null_sample_rate_rule_skips_audio_check():
    rules = Rules(1080, 1080, 1920, VideoRules(audio_sample_rate=None), ImageRules())
    assert plan_video(info(audio_sample_rate=44100), rules).kind == UNCHANGED


def test_null_fps_rule_skips_fps_cap():
    rules = Rules(1080, 1080, 1920, VideoRules(fps_max=None), ImageRules())
    assert plan_video(info(fps=240.0), rules).kind == UNCHANGED
