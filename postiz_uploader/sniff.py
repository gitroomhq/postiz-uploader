"""Detect the real media type from the first bytes of a file."""

from __future__ import annotations

from dataclasses import dataclass

# ISO BMFF brands that mean a still image, not a video
_IMAGE_BRANDS = {b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1", b"avif", b"avis"}


@dataclass(frozen=True)
class Sniffed:
    kind: str  # "video" | "image"
    format: str  # "mp4", "webm", "avi", "mpegts", "png", "jpeg", "gif", "webp", "heic", "avif"


def sniff(head: bytes) -> Sniffed | None:
    if len(head) < 12:
        return None

    if head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand in _IMAGE_BRANDS:
            return Sniffed("image", "avif" if brand.startswith(b"avi") else "heic")
        return Sniffed("video", "mp4")

    if head[:4] == b"\x1a\x45\xdf\xa3":
        return Sniffed("video", "webm")

    if head[:4] == b"RIFF":
        if head[8:12] == b"WEBP":
            return Sniffed("image", "webp")
        if head[8:12] == b"AVI ":
            return Sniffed("video", "avi")
        return None

    if head[:4] == b"\x89PNG":
        return Sniffed("image", "png")
    if head[:3] == b"\xff\xd8\xff":
        return Sniffed("image", "jpeg")
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return Sniffed("image", "gif")

    if head[:4] == b"\x00\x00\x01\xba":
        return Sniffed("video", "mpegts")
    if len(head) > 188 and head[0] == 0x47 and head[188] == 0x47:
        return Sniffed("video", "mpegts")

    return None
