"""ffmpeg command builders and a runner with a hard timeout.

Commands are argument lists, never shell strings.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from functools import lru_cache

from postiz_uploader import errors
from postiz_uploader.config import Settings
from postiz_uploader.errors import JobError
from postiz_uploader.planner import Plan
from postiz_uploader.probe import SourceInfo
from postiz_uploader.schema import Frame, SlideshowVideo, VideoRules

STDERR_TAIL = 4096
BASE_ARGS = ["-y", "-nostdin", "-hide_banner", "-loglevel", "error"]

# HDR -> SDR on the CPU. Linearize, tone map, back to BT.709 8-bit.
TONEMAP_CPU = (
    "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
    "tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p"
)

_TRANSPOSE = {
    90: ["transpose_npp=dir=clock"],
    180: ["transpose_npp=dir=clock", "transpose_npp=dir=clock"],
    270: ["transpose_npp=dir=cclock"],
}


class Encoders:
    """The (encoder, cuda decode) combinations to try, best first.

    A combination that failed for one output will fail for the next one too (it is a
    property of the host and the source, not of the cut), so it is dropped for the
    rest of the job. `cuda_decode` adds the NVDEC rung, which only makes sense for a
    video source that is neither HDR nor a still image.
    """

    def __init__(self, settings: Settings, cuda_decode: bool):
        if settings.gpu:
            self.attempts = [(settings.encoder, False), ("libx264", False)]
            if cuda_decode:
                self.attempts.insert(0, (settings.encoder, True))
        else:
            self.attempts = [(settings.encoder, False)]
        self.used: tuple[str, bool] | None = None

    def drop(self, attempt: tuple[str, bool]) -> None:
        if len(self.attempts) > 1:
            self.attempts.remove(attempt)


@dataclass(frozen=True)
class RunOutcome:
    returncode: int
    stderr_tail: str
    seconds: float


def run(cmd: list[str], *, timeout: float) -> RunOutcome:
    """Run ffmpeg/ffprobe. Kill with SIGKILL when `timeout` passes."""
    if timeout <= 0:
        raise JobError(errors.TIMEOUT, "job timeout reached before ffmpeg started", retryable=True)
    started = time.monotonic()
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except OSError as err:
        raise JobError(errors.INTERNAL, f"cannot start {cmd[0]}: {err}", retryable=True) from err
    try:
        _, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise JobError(errors.TIMEOUT, f"{cmd[0]} exceeded {int(timeout)}s and was killed", retryable=True) from None
    tail = (stderr or b"")[-STDERR_TAIL:].decode("utf-8", errors="replace")
    return RunOutcome(proc.returncode, tail, time.monotonic() - started)


@lru_cache(maxsize=4)
def version(ffmpeg_bin: str) -> str | None:
    try:
        out = subprocess.run([ffmpeg_bin, "-version"], capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    first = (out.stdout or "").splitlines()[:1]
    if not first or not first[0].startswith("ffmpeg version"):
        return None
    return first[0].split()[2]


@lru_cache(maxsize=32)
def has_filter(ffmpeg_bin: str, name: str) -> bool:
    try:
        out = subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-filters"], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return any(line.split()[1:2] == [name] for line in out.stdout.splitlines())


@lru_cache(maxsize=4)
def gpu_name() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    name = (out.stdout or "").strip().splitlines()[:1]
    return name[0].strip() if name and out.returncode == 0 else None


def _audio_args(plan: Plan, info: SourceInfo, rules: VideoRules) -> list[str]:
    if not info.has_audio:
        return ["-an"]
    args = ["-map", "0:a:0"]
    if plan.audio_encode:
        args += ["-c:a", rules.audio_codec, "-b:a", f"{rules.audio_bitrate_kbps}k"]
        if rules.audio_sample_rate:
            args += ["-ar", str(rules.audio_sample_rate)]
        args += ["-ac", "2"]
    else:
        args += ["-c:a", "copy"]
    return args


def _encoder_args(encoder: str, rules: VideoRules | SlideshowVideo) -> list[str]:
    if encoder == "h264_nvenc":
        return [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p4",
            "-tune",
            "hq",
            "-rc",
            "vbr",
            "-cq",
            str(rules.quality),
            "-b:v",
            "0",
            "-profile:v",
            rules.profile,
        ]
    return [
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(rules.quality),
        "-profile:v",
        rules.profile,
        "-pix_fmt",
        rules.pixel_format,
    ]


def build_remux(ffmpeg_bin: str, src: str, dst: str, info: SourceInfo, rules: VideoRules) -> list[str]:
    cmd = [ffmpeg_bin, *BASE_ARGS, "-i", src, "-map", "0:V:0"]
    if info.has_audio:
        cmd += ["-map", "0:a:0"]
    cmd += ["-c", "copy"]
    if rules.faststart:
        cmd += ["-movflags", "+faststart"]
    cmd += ["-f", rules.container, dst]
    return cmd


def build_encode(
    ffmpeg_bin: str,
    src: str,
    dst: str,
    plan: Plan,
    info: SourceInfo,
    rules: VideoRules,
    *,
    encoder: str,
    cuda_decode: bool,
) -> list[str]:
    """One ffmpeg invocation for an ENCODE plan.

    cuda_decode keeps decode, rotate and scale on the GPU (frames never leave GPU
    memory). It is only valid with the nvenc encoder and never with tone mapping
    or a full-range source, which run on the CPU chain.
    """
    if cuda_decode and (encoder != "h264_nvenc" or plan.tonemap or info.full_range):
        raise ValueError("cuda_decode requires h264_nvenc, no tone mapping and a limited-range source")

    cmd = [ffmpeg_bin, *BASE_ARGS]
    if cuda_decode:
        cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        if plan.rotate:
            # transpose_npp does the rotation; the input display matrix must not
            # survive into the output or the file would be rotated twice on playback
            cmd += ["-noautorotate", "-display_rotation", "0"]
    # 0:V selects real video streams only, skipping attached cover art
    cmd += ["-i", src, "-map", "0:V:0"]

    if plan.video_encode:
        filters: list[str] = []
        if cuda_decode:
            if plan.rotate:
                filters += _TRANSPOSE.get(info.rotation, [])
            if plan.fps_cap:
                filters.append(f"fps=fps={plan.fps_cap:g}")
            filters.append(f"scale_npp={plan.width}:{plan.height}:interp_algo=lanczos:format={rules.pixel_format}")
            filters.append("setsar=1")
        else:
            # autorotate applies the rotate tag before user filters, so the target
            # dimensions (already post-rotation) are correct here
            if plan.fps_cap:
                filters.append(f"fps=fps={plan.fps_cap:g}")
            if plan.scale:
                filters.append(f"scale={plan.width}:{plan.height}:flags=lanczos")
                filters.append("setsar=1")
            if plan.tonemap:
                filters.append(TONEMAP_CPU)
            else:
                if info.full_range:
                    # format= keeps the full-range flag, and the output then probes as yuvj420p
                    filters.append("scale=out_range=tv")
                filters.append(f"format={rules.pixel_format}")
        cmd += ["-vf", ",".join(filters)]

        cmd += _encoder_args(encoder, rules)
    else:
        cmd += ["-c:v", "copy"]

    cmd += _audio_args(plan, info, rules)
    if rules.faststart:
        cmd += ["-movflags", "+faststart"]
    cmd += ["-f", rules.container, dst]
    return cmd


def build_thumbnail(ffmpeg_bin: str, src: str, dst: str, *, timestamp: float, width: int, height: int) -> list[str]:
    return [
        ffmpeg_bin,
        *BASE_ARGS,
        "-ss",
        f"{max(timestamp, 0):.3f}",
        "-i",
        src,
        "-frames:v",
        "1",
        "-vf",
        f"scale={width}:{height}:flags=lanczos",
        "-q:v",
        "3",
        "-f",
        "image2",
        dst,
    ]


def build_audio_extract(ffmpeg_bin: str, src: str, dst: str, *, bitrate_kbps: int, sample_rate: int) -> list[str]:
    """Mono Opus in Ogg, sized for speech-to-text rather than listening."""
    return [
        ffmpeg_bin,
        *BASE_ARGS,
        "-i",
        src,
        "-vn",
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "libopus",
        "-application",
        "voip",
        "-b:a",
        f"{bitrate_kbps}k",
        "-f",
        "ogg",
        dst,
    ]


def filter_path(path: str) -> str:
    """Quote a filesystem path for use as a filter option value.

    The value is parsed twice: the graph parser strips the quotes, then the option
    parser splits on `:` and unescapes backslashes. So inside the quotes every special
    character still carries the backslash the second pass needs.
    """
    return "'" + path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'\\''") + "'"


def clip_filter(
    frame: Frame,
    *,
    focus_x: float,
    focus_y: float,
    fps_cap: float | None,
    tonemap: bool,
    pixel_format: str,
    ass_path: str | None,
    fonts_dir: str | None,
) -> str:
    """The simple filtergraph that turns any source picture into the clip canvas."""
    w, h = frame.width, frame.height
    head: list[str] = []
    if fps_cap:
        head.append(f"fps=fps={fps_cap:g}")
    if tonemap:
        head.append(TONEMAP_CPU)

    if frame.fit == "blur":
        # the background is blurred at quarter size: cheaper, and softer once scaled back up
        bw, bh = max(w // 8 * 2, 2), max(h // 8 * 2, 2)
        pre = ",".join(head) + "," if head else ""
        # NVDEC hands frames back as nv12, whose two chroma channels share one plane;
        # boxblur blurs them as if they were one and the background turns green. Going
        # planar after the downscale costs a conversion of the small picture only
        graph = (
            f"{pre}split=2[bg][fg];"
            f"[bg]scale={bw}:{bh}:force_original_aspect_ratio=increase,format=yuv420p,"
            f"crop={bw}:{bh},boxblur=8:2,"
            f"scale={w}:{h}:flags=bilinear[bgb];"
            f"[fg]scale={w}:{h}:force_original_aspect_ratio=decrease:force_divisible_by=2:flags=lanczos[fgs];"
            f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2"
        )
    else:
        graph = ",".join(
            [
                *head,
                f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos",
                f"crop={w}:{h}:(iw-ow)*{focus_x:.4f}:(ih-oh)*{focus_y:.4f}",
            ]
        )

    tail = ["setsar=1"]
    if ass_path:
        ass = f"ass=filename={filter_path(ass_path)}"
        if fonts_dir:
            ass += f":fontsdir={filter_path(fonts_dir)}"
        tail.append(ass)
    tail.append(f"format={pixel_format}")
    return graph + "," + ",".join(tail)


def build_clip(
    ffmpeg_bin: str,
    src: str,
    dst: str,
    *,
    start: float,
    duration: float,
    vf: str,
    has_audio: bool,
    rules: VideoRules,
    encoder: str,
    cuda_decode: bool,
) -> list[str]:
    """Cut [start, start + duration) and re-encode it through `vf`.

    The seek is an input option, so it is frame accurate (ffmpeg decodes from the
    previous keyframe and drops frames up to `start`) and output timestamps begin at
    zero, which is what the caption file is timed against. Every filter in `vf` runs
    on the CPU, so cuda_decode only offloads the decode; frames come back to system
    memory.
    """
    cmd = [ffmpeg_bin, *BASE_ARGS]
    if cuda_decode:
        cmd += ["-hwaccel", "cuda"]
    cmd += ["-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", src, "-map", "0:V:0", "-vf", vf]
    cmd += _encoder_args(encoder, rules)
    if has_audio:
        cmd += ["-map", "0:a:0", "-c:a", rules.audio_codec, "-b:a", f"{rules.audio_bitrate_kbps}k"]
        if rules.audio_sample_rate:
            cmd += ["-ar", str(rules.audio_sample_rate)]
        cmd += ["-ac", "2"]
    else:
        cmd += ["-an"]
    cmd += ["-movflags", "+faststart", "-f", rules.container, dst]
    return cmd


def slide_filter(frame: Frame, *, fit: str, fps: float, pixel_format: str) -> str:
    """One still image -> the slideshow canvas, as a constant frame rate stream."""
    w, h = frame.width, frame.height
    if fit == "cover":
        picture = [
            f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos",
            f"crop={w}:{h}",
        ]
    else:
        picture = [
            f"scale={w}:{h}:force_original_aspect_ratio=decrease:force_divisible_by=2:flags=lanczos",
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black",
        ]
    return ",".join([*picture, "setsar=1", f"fps=fps={fps:g}", f"format={pixel_format}"])


def _slideshow_audio_args(rules: SlideshowVideo) -> list[str]:
    return ["-c:a", "aac", "-b:a", f"{rules.audio_bitrate_kbps}k", "-ar", str(rules.audio_sample_rate), "-ac", "2"]


def build_slide(
    ffmpeg_bin: str,
    image: str,
    audio: str,
    dst: str,
    *,
    duration: float,
    audio_delay: float,
    vf: str,
    rules: SlideshowVideo,
    encoder: str,
) -> list[str]:
    """One slide: the image looped for `duration`, the audio starting `audio_delay` in
    and padded with silence to the same length, so every part the assemble pass joins
    has one continuous audio track."""
    af = (
        f"aresample={rules.audio_sample_rate},aformat=channel_layouts=stereo,"
        f"adelay={int(round(audio_delay * 1000))}:all=1,apad"
    )
    return [
        ffmpeg_bin,
        *BASE_ARGS,
        "-loop",
        "1",
        "-framerate",
        f"{rules.fps:g}",
        "-i",
        image,
        "-i",
        audio,
        "-t",
        f"{duration:.3f}",
        "-filter_complex",
        f"[0:v]{vf}[v];[1:a]{af}[a]",
        "-map",
        "[v]",
        "-map",
        "[a]",
        "-r",
        f"{rules.fps:g}",
        *_encoder_args(encoder, rules),
        *_slideshow_audio_args(rules),
        "-movflags",
        "+faststart",
        "-f",
        "mp4",
        dst,
    ]


def build_slideshow(
    ffmpeg_bin: str,
    parts: list[str],
    dst: str,
    *,
    offsets: list[float],
    transition: float,
    rules: SlideshowVideo,
    encoder: str,
    ass_path: str | None,
    fonts_dir: str | None,
) -> list[str]:
    """Join the rendered slides into one video and burn the subtitles in.

    With a transition every boundary is an xfade/acrossfade pair whose offset is the
    start of the next slide on the output timeline (`offsets[k]`, see
    slideshow.timeline); without one the parts are concatenated as they are.
    """
    n = len(parts)
    chains: list[str] = []
    if transition > 0 and n > 1:
        video, audio = "[0:v]", "[0:a]"
        for k in range(1, n):
            chains.append(f"{video}[{k}:v]xfade=transition=fade:duration={transition:g}:offset={offsets[k]:.3f}[v{k}]")
            chains.append(f"{audio}[{k}:a]acrossfade=d={transition:g}[a{k}]")
            video, audio = f"[v{k}]", f"[a{k}]"
    else:
        streams = "".join(f"[{k}:v][{k}:a]" for k in range(n))
        chains.append(f"{streams}concat=n={n}:v=1:a=1[vc][ac]")
        video, audio = "[vc]", "[ac]"
    if ass_path:
        ass = f"ass=filename={filter_path(ass_path)}"
        if fonts_dir:
            ass += f":fontsdir={filter_path(fonts_dir)}"
        chains.append(f"{video}{ass}[vs]")
        video = "[vs]"
    chains.append(f"{video}format={rules.pixel_format}[vout]")
    video = "[vout]"

    cmd = [ffmpeg_bin, *BASE_ARGS]
    for part in parts:
        cmd += ["-i", part]
    cmd += ["-filter_complex", ";".join(chains), "-map", video, "-map", audio, "-r", f"{rules.fps:g}"]
    cmd += _encoder_args(encoder, rules)
    cmd += _slideshow_audio_args(rules)
    cmd += ["-movflags", "+faststart", "-f", "mp4", dst]
    return cmd
