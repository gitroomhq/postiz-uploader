"""Pure decision logic: probe + rules -> plan. No ffmpeg here, fully unit-testable."""

from __future__ import annotations

from dataclasses import dataclass, field

from postiz_uploader.probe import SourceInfo
from postiz_uploader.schema import Rules

UNCHANGED = "unchanged"
REMUX = "remux"
ENCODE = "encode"


def _round_side(value: float, *, even: bool) -> int:
    """Nearest integer, or nearest even integer for video (H.264 needs even sides)."""
    if even:
        return max(int(round(value / 2.0)) * 2, 2)
    return max(int(round(value)), 1)


def clamp_dimensions(width: int, height: int, rules: Rules, *, even: bool) -> tuple[int, int]:
    """Apply the three bounds from README 4.1 with one scale factor.

    1. shorter side clamped into [short_side_min, short_side_max], aspect preserved
    2. then the longer side capped at long_side_max, scaling the whole frame down
       (this may leave the shorter side below its minimum; the long cap wins)
    3. video sides rounded to the nearest even number

    One factor for both sides keeps the aspect ratio exact before rounding, so the
    rounding error is never more than a pixel.
    """
    short, long = min(width, height), max(width, height)

    scale = 1.0
    if short < rules.short_side_min:
        scale = rules.short_side_min / short
    elif short > rules.short_side_max:
        scale = rules.short_side_max / short
    if long * scale > rules.long_side_max:
        scale = rules.long_side_max / long

    out_w = _round_side(width * scale, even=even)
    out_h = _round_side(height * scale, even=even)

    # nearest rounding can land one pixel above an odd cap; step the offending side back
    step = 2 if even else 1
    if max(out_w, out_h) > rules.long_side_max:
        if out_w >= out_h:
            out_w -= step
        else:
            out_h -= step
    if min(out_w, out_h) > rules.short_side_max:
        if out_w <= out_h:
            out_w -= step
        else:
            out_h -= step
    return out_w, out_h


@dataclass(frozen=True)
class Plan:
    kind: str  # UNCHANGED | REMUX | ENCODE
    width: int
    height: int
    video_encode: bool = False
    audio_encode: bool = False
    scale: bool = False
    tonemap: bool = False
    rotate: bool = False
    fps_cap: float | None = None
    actions: tuple[str, ...] = field(default_factory=tuple)


def plan_video(info: SourceInfo, rules: Rules) -> Plan:
    v = rules.video
    target_w, target_h = clamp_dimensions(info.width, info.height, rules, even=True)

    needs_scale = (target_w, target_h) != (info.width, info.height)
    needs_tonemap = info.is_hdr
    needs_rotate = info.rotation != 0
    fps_cap = v.fps_max if (v.fps_max and info.fps and info.fps > v.fps_max + 0.01) else None

    needs_video_encode = (
        needs_scale
        or needs_tonemap
        or needs_rotate
        or fps_cap is not None
        or info.video_codec != v.video_codec
        or info.pixel_format != v.pixel_format
    )
    needs_audio_encode = info.has_audio and (
        info.audio_codec != v.audio_codec
        or (v.audio_sample_rate is not None and info.audio_sample_rate != v.audio_sample_rate)
    )
    needs_remux = info.container != v.container or (v.faststart and info.faststart is False)

    if not (needs_video_encode or needs_audio_encode or needs_remux):
        return Plan(UNCHANGED, info.width, info.height)

    if not needs_video_encode and not needs_audio_encode:
        return Plan(REMUX, info.width, info.height, actions=("remux",))

    actions: list[str] = []
    if needs_video_encode:
        actions.append("video_encode")
    if needs_scale:
        actions.append("scale")
    if needs_tonemap:
        actions.append("tonemap")
    if needs_rotate:
        actions.append("rotate")
    if fps_cap is not None:
        actions.append("fps_cap")
    if needs_audio_encode:
        actions.append("audio_encode")
    if needs_remux:
        actions.append("remux")

    return Plan(
        ENCODE,
        target_w,
        target_h,
        video_encode=needs_video_encode,
        audio_encode=needs_audio_encode,
        scale=needs_scale,
        tonemap=needs_tonemap,
        rotate=needs_rotate,
        fps_cap=fps_cap,
        actions=tuple(actions),
    )
