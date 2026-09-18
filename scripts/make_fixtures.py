"""Generate the synthetic fixture set with ffmpeg's lavfi sources and Pillow.

Real phone clips (HDR iPhone footage in particular) cannot be synthesized faithfully;
see README 10.1 and open decision 3. Everything here is small and deterministic.

    python scripts/make_fixtures.py fixtures/generated
"""

from __future__ import annotations

import os
import subprocess
import sys

from PIL import Image

FFMPEG = os.environ.get("FFMPEG_BIN", "ffmpeg")
DURATION = "2"


def _ffmpeg(*args: str) -> None:
    cmd = [FFMPEG, "-y", "-nostdin", "-hide_banner", "-loglevel", "error", *args]
    subprocess.run(cmd, check=True)


def _has_encoder(name: str) -> bool:
    out = subprocess.run([FFMPEG, "-hide_banner", "-encoders"], capture_output=True, text=True, check=False)
    return any(line.split()[1:2] == [name] for line in out.stdout.splitlines())


def _video(
    path: str,
    w: int,
    h: int,
    *,
    fps: int = 30,
    vcodec: list[str],
    audio: bool = True,
    extra: list[str] | None = None,
    faststart: bool = True,
    duration: str = DURATION,
) -> None:
    inputs = ["-f", "lavfi", "-i", f"testsrc2=size={w}x{h}:rate={fps}:duration={duration}"]
    if audio:
        inputs += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={duration}"]
    args = [*inputs, "-map", "0:v", *(["-map", "1:a"] if audio else []), *vcodec]
    if audio:
        args += ["-c:a", "aac", "-b:a", "96k", "-ar", "48000"]
    if faststart and path.endswith((".mp4", ".mov")):
        args += ["-movflags", "+faststart"]
    if extra:
        args += extra
    _ffmpeg(*args, path)


def _image(path: str, w: int, h: int, mode: str, fmt: str, **save) -> None:
    img = Image.new(mode, (w, h), (200, 90, 40, 255) if mode == "RGBA" else (200, 90, 40))
    for x in range(0, w, max(w // 8, 1)):
        for y in range(0, h, max(h // 8, 1)):
            img.paste(
                (30, 120, 220, 128) if mode == "RGBA" else (30, 120, 220),
                (x, y, x + max(w // 16, 1), y + max(h // 16, 1)),
            )
    img.save(path, format=fmt, **save)


def make(out_dir: str) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    x264 = ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-profile:v", "high"]
    made: dict[str, str] = {}

    def p(name: str) -> str:
        made[name] = os.path.join(out_dir, name)
        return made[name]

    _video(p("compliant-1080p.mp4"), 1920, 1080, vcodec=x264)
    _video(p("compliant-no-faststart.mp4"), 1920, 1080, vcodec=x264, faststart=False)
    _video(p("no-audio.mp4"), 1920, 1080, vcodec=x264, audio=False)
    _video(p("slowmo-120fps.mp4"), 1920, 1080, fps=120, vcodec=x264)
    _video(p("ultrawide-2560x1080.mp4"), 2560, 1080, vcodec=x264)
    _video(p("small-480p-h264.mp4"), 854, 480, vcodec=x264)
    # long enough to cut several clips out of; testsrc2 burns a running timecode in
    _video(p("talk-8s.mp4"), 1280, 720, vcodec=x264, duration="8")
    # display matrix like an iPhone portrait clip: 1920x1080 stored, shown as 1080x1920.
    # ffprobe reports it as rotation -90; the worker turns that into a clockwise 90.
    _ffmpeg(
        "-display_rotation",
        "-90",
        "-noautorotate",
        "-i",
        made["compliant-1080p.mp4"],
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        p("portrait-rotated.mov"),
    )
    # 44.1k audio triggers an audio-only encode with video copy
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size=1920x1080:rate=30:duration={DURATION}",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:sample_rate=44100:duration={DURATION}",
        "-map",
        "0:v",
        "-map",
        "1:a",
        *x264,
        "-c:a",
        "aac",
        "-ar",
        "44100",
        "-movflags",
        "+faststart",
        p("audio-44100.mp4"),
    )

    if _has_encoder("libx265"):
        x265 = [
            "-c:v",
            "libx265",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-tag:v",
            "hvc1",
            "-x265-params",
            "log-level=error",
        ]
        _video(p("4k-hevc.mov"), 3840, 2160, vcodec=x265)
        _video(
            p("4k-hdr-hlg.mov"),
            3840,
            2160,
            vcodec=[
                "-c:v",
                "libx265",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p10le",
                "-tag:v",
                "hvc1",
                "-color_primaries",
                "bt2020",
                "-color_trc",
                "arib-std-b67",
                "-colorspace",
                "bt2020nc",
                "-x265-params",
                "log-level=error:colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc",
            ],
        )
    if _has_encoder("libvpx-vp9"):
        _video(
            p("small-480p.webm"),
            854,
            480,
            vcodec=["-c:v", "libvpx-vp9", "-deadline", "realtime", "-cpu-used", "8", "-pix_fmt", "yuv420p"],
            audio=False,
        )

    with open(made["compliant-1080p.mp4"], "rb") as src, open(p("truncated.mp4"), "wb") as dst:
        dst.write(src.read(20_000))

    _image(p("large.png"), 4000, 3000, "RGBA", "PNG")
    _image(p("small-exif-rotated.jpg"), 400, 300, "RGB", "JPEG", quality=85)
    # write orientation 6 (rotate 90 CW on display) into the small jpeg
    with Image.open(made["small-exif-rotated.jpg"]) as img:
        exif = img.getexif()
        exif[0x0112] = 6
        img.save(made["small-exif-rotated.jpg"], format="JPEG", quality=85, exif=exif.tobytes())
    _image(p("compliant-1080.jpg"), 1920, 1080, "RGB", "JPEG", quality=85)
    _image(p("medium.webp"), 1600, 900, "RGBA", "WEBP", quality=85)
    frames = [Image.new("RGB", (64, 64), (i * 60, 100, 200 - i * 40)) for i in range(4)]
    frames[0].save(p("animated.gif"), format="GIF", save_all=True, append_images=frames[1:], duration=100, loop=0)
    return made


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "fixtures/generated"
    for name, path in make(target).items():
        print(f"{name:32} {os.path.getsize(path):>10} bytes")
