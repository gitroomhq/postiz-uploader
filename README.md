# postiz-uploader

A stateless media normalization service. It takes a video or image by URL, makes it
fit a set of rules (resolution bounds, codec, container, frame rate), writes the result
to a URL it was given, and reports what it did.

It runs as a [RunPod Serverless](https://docs.runpod.io/serverless/overview) worker on
an NVIDIA L4, and as a plain CPU container for local development and as a fallback.

It also runs the two heavy steps of video clipping (sections 4.4 and 4.5): `ingest`
fetches a long source once and produces a faststart MP4 plus a small audio track for
speech-to-text, and `clip` cuts N segments out of a source, reframes them for vertical
feeds and burns in word-timed captions. Transcription, choosing what to cut and
scheduling the results are the caller's job.

The service knows nothing about Postiz. There are no media ids, no database, no storage
credentials and no callbacks into any API. URLs in, metadata out. This is a deliberate
boundary and every section below assumes it.

This document is the full specification. It is written so that the service can be
built from it without asking the author anything. Decisions that are still open are
marked **OPEN** and listed again at the end.

---

## 1. Why this exists

Postiz currently sends web uploads through Transloadit, which transcodes to 1080p before
the file reaches the R2 bucket. That is billed per gigabyte, so it is only enabled for
the web app, and API uploads are stored untouched. Not every platform accepts 4K or
odd codecs, so API customers hit rejections that web users never see.

The goal is to normalize every upload, web and API, at a cost that scales with GPU
seconds instead of gigabytes, and to do it with an autoscaling queue so a customer's
cron job dropping 500 videos at 3am does not delay a person uploading one file on the
web.

Sizing context (September 2026): the web path processes roughly 650 GB per month
through Transloadit. API volume is unbounded and bursty. An L4 decodes roughly 1000
minutes of 4K per hour, so even multiple terabytes per month is a handful of GPU hours.

---

## 2. Non-goals

- No knowledge of Postiz, organizations, plans, or social platforms. The **caller**
  decides the rules and passes them in every job.
- No persistent state. Nothing survives a job except logs and metrics.
- No storage credentials. Input is a URL the service can GET, output is a URL the
  service can PUT. The caller mints presigned URLs.
- No inbound upload from browsers. The original is already in the bucket when a job
  starts.
- No queueing, retry policy, or fairness logic of its own. RunPod Serverless provides
  the queue and autoscaling; the caller's workflow engine provides retries and
  per-tenant fairness.
- No callbacks or webhooks in v1. The caller polls RunPod's status endpoint.

---

## 3. Architecture

```
caller (Postiz orchestrator)
   |  POST https://api.runpod.ai/v2/<endpoint>/run   { input: <Job> }
   |  GET  https://api.runpod.ai/v2/<endpoint>/status/<id>
   v
RunPod Serverless endpoint  (queue + autoscaler, L4 workers, FlashBoot)
   |
   v
worker container (this repo)
   handler(job) -> download -> probe -> plan -> ffmpeg / Pillow -> upload -> result
   |                                                          |
   +--- GET source.url (presigned or public)                  +--- PUT output.url (presigned)
                                                              +--- PUT thumbnail.url (presigned)
```

Language: **Python 3.10+** (3.12 in the CPU image, the Ubuntu 22.04 system Python in
the GPU image). Reason: the official RunPod worker SDK (`runpod`) is Python, ffmpeg is
driven as a subprocess either way, and there is no code to share with the TypeScript
monorepo. If a TypeScript worker is strongly preferred, the same contract can be
implemented behind a thin HTTP handler, but that is not the default.

Layout:

```
handler.py                    RunPod entrypoint (worker loop / --rp_serve_api local API)
postiz_uploader/
  pipeline.py                 process(job) -> result; stages, timing, cleanup, never raises
  schema.py                   validate against schema/v1 and lift into dataclasses
  download.py  upload.py      bounded streaming GET, retried PUT
  sniff.py                    magic-byte type detection
  probe.py                    ffprobe -> SourceInfo (rotation, HDR, faststart)
  planner.py                  pure: clamp_dimensions, plan_video
  ffmpeg.py                   command builders (libx264 / nvenc / tonemap) and the runner
  video.py  image.py          the two normalization stages
  ingest.py  ytdlp.py         ingest stage; yt-dlp as a subprocess
  clip.py  captions.py        clip stage; word timings -> ASS subtitles (pure)
  devserver.py                GET/PUT file server used by tests, compose and scripts
schema/v1/                    job + result schemas per job type (vendored by the caller)
scripts/                      make_fixtures.py, run-local.py, bench.py
tests/                        planner/schema/unit tests + end-to-end on generated fixtures
```

Tools:

- `ffmpeg` and `ffprobe` for video. Two build flavours, see section 9.
- `Pillow` for still images. GIFs are passed through untouched.
- `runpod` Python SDK for the handler loop and the local test server.

---

## 4. Job contract

The contract is versioned. Both sides send `version` and the service rejects a
version it does not understand with `UNSUPPORTED_VERSION`.

RunPod wraps every request as `{ "input": <Job> }` and every result as
`{ "id": "...", "status": "COMPLETED" | "FAILED" | "IN_QUEUE" | "IN_PROGRESS", "output": <Result> }`.
The shapes below are the `input` and `output` payloads only.

### 4.1 Job (input)

```jsonc
{
  "version": 1,
  "type": "video",                       // "video" | "image"
  "reference": "media_abc123",           // opaque string, echoed back verbatim, never interpreted
  "source": {
    "url": "https://<bucket>/originals/abc.mov",   // GET; presigned or public
    "content_type": "video/quicktime"              // optional hint, the file is sniffed anyway
  },
  "output": {
    "url": "https://<bucket>/media/abc.mp4?X-Amz-...",   // presigned PUT
    "content_type": "video/mp4"                          // what the service must produce
  },
  "thumbnail": {                         // optional, video only
    "url": "https://<bucket>/media/abc-thumb.jpg?X-Amz-...",  // presigned PUT
    "timestamp_seconds": 0,              // frame to grab; clamped to duration
    "content_type": "image/jpeg"     // only jpeg is produced
  },
  "rules": {
    "short_side_min": 1080,              // upscale if the shorter edge is below this
    "short_side_max": 1080,              // downscale if the shorter edge is above this
    "long_side_max": 1920,               // additionally cap the longer edge
    "video": {
      "container": "mp4",
      "video_codec": "h264",
      "profile": "high",
      "pixel_format": "yuv420p",
      "fps_max": 60,
      "quality": 23,                     // CRF for libx264, CQ for nvenc
      "audio_codec": "aac",
      "audio_bitrate_kbps": 128,
      "audio_sample_rate": 48000,
      "faststart": true
    },
    "image": {
      "jpeg_quality": 90,
      "keep_format": true                // jpeg stays jpeg, png stays png, webp stays webp
    }
  },
  "limits": {
    "max_input_bytes": 1073741824,       // 1 GiB
    "max_duration_seconds": 900,         // 15 minutes
    "timeout_seconds": 1200              // hard wall-clock cap for the whole job
  }
}
```

Image jobs use `short_side_min`, `short_side_max`, `long_side_max` and `rules.image`.
Video jobs use the same three bounds and `rules.video`. Unknown fields are ignored so
the caller can add fields before the service understands them.

Rule semantics for the three bounds, both media types:

- Dimensions are measured **after rotation metadata is applied**. A portrait iPhone
  clip stored as 1920x1080 with a 90 degree rotate tag is 1080x1920.
- The shorter side is clamped into `[short_side_min, short_side_max]`, aspect ratio
  preserved.
- Then, if the longer side exceeds `long_side_max`, the whole frame is scaled down
  further so it fits. This can push the shorter side below `short_side_min`; that is
  accepted, the long cap wins.
- One scale factor is applied to both sides, then each side is rounded: to the
  nearest integer for images, to the nearest even integer for video (H.264 needs even
  sides). Rounding never lands above a cap; with an odd cap the side steps back to the
  even number below it. The aspect error is under a pixel.
- If no bound is violated the frame is not scaled.

### 4.2 Result (output)

```jsonc
{
  "version": 1,
  "reference": "media_abc123",
  "status": "completed",                 // "completed" | "unchanged" | "failed"
  "actions": ["scale", "video_encode", "audio_encode", "remux"],   // what was actually done
                                         // video: remux scale video_encode audio_encode tonemap fps_cap rotate
                                         // image: scale orient
  "source": {                            // probe of the original, always present unless probe failed
    "container": "mov",
    "video_codec": "hevc",
    "pixel_format": "yuv420p10le",
    "color_transfer": "arib-std-b67",    // HDR detection input
    "width": 3840, "height": 2160,       // after rotation
    "rotation": 90,
    "fps": 59.94,
    "duration_seconds": 61.2,
    "bytes": 412345678,
    "audio_codec": "aac",
    "audio_sample_rate": 44100,
    "faststart": false
  },
  "output": {                            // present when status == "completed"
    "width": 1080, "height": 1920,
    "duration_seconds": 61.2,
    "bytes": 98765432,
    "content_type": "video/mp4",
    "video_codec": "h264",
    "audio_codec": "aac",
    "fps": 59.94
  },
  "thumbnail": { "width": 1080, "height": 1920, "bytes": 123456 },   // when requested and produced
  "timing_ms": { "download": 8100, "probe": 120, "process": 9400, "upload": 3100, "total": 20800 },
  "worker": { "encoder": "h264_nvenc", "decode": "cuda", "ffmpeg": "7.1", "gpu": "NVIDIA L4", "version": "0.1.0" },
  "failure": null
}
```

The failure block is named `failure` and not `error` on purpose: the RunPod SDK strips
a top-level `error` key from handler output and reports the job as platform-level
`FAILED`, which would hide the code and stderr from the caller.

`worker.decode` is `cuda` when NVDEC decoded the source, `software` when ffmpeg fell
back to CPU decode (see 5.5), and null when no encode ran. The transport may drop keys
whose value is null, so the caller treats a missing key as null; only `version`,
`reference`, `status` and `actions` are guaranteed.

`status: "unchanged"` means the original already satisfies every rule. **Nothing is
uploaded to `output.url`** and `output` is absent (a requested thumbnail is still
produced from the original and uploaded, see 5.6). The caller keeps the original as the final file.
This is the cheapest and most common outcome for phone videos that are already 1080p
H.264 MP4 with faststart, and for images already inside the bounds.

On failure:

```jsonc
{
  "version": 1,
  "reference": "media_abc123",
  "status": "failed",
  "actions": [],
  "source": { ...probe if it succeeded... },
  "failure": {
    "code": "ENCODE_FAILED",
    "message": "ffmpeg exited with status 1",
    "retryable": false,
    "stderr_tail": "...last 4 KB of ffmpeg stderr..."
  }
}
```

The handler **returns** the failure payload as a normal result. It does not raise, so
RunPod reports the job as `COMPLETED` with a `failed` status inside. This keeps the
probe and stderr available to the caller. The only RunPod-level `FAILED` is an
unhandled crash, which is a bug.

### 4.3 Error codes

| code | retryable | meaning |
|---|---|---|
| `UNSUPPORTED_VERSION` | no | `version` not understood |
| `INVALID_JOB` | no | schema validation failed; `message` names the field |
| `SOURCE_HOST_NOT_ALLOWED` | no | `source.url` host not in the allowlist |
| `DOWNLOAD_FAILED` | yes | network error or non-2xx while fetching the source |
| `SOURCE_BLOCKED` | yes | ingest: the platform refused this worker (bot check, 403, 429) on every route |
| `SOURCE_UNAVAILABLE` | no | ingest: the video is private, removed, age or region restricted |
| `INPUT_TOO_LARGE` | no | declared or actual size over `limits.max_input_bytes` |
| `UNSUPPORTED_INPUT` | no | not a video/image we can read, or type does not match `job.type` |
| `PROBE_FAILED` | no | ffprobe could not parse the file |
| `DURATION_TOO_LONG` | no | over `limits.max_duration_seconds` |
| `ENCODE_FAILED` | no | ffmpeg or Pillow failed; see `stderr_tail` |
| `UPLOAD_FAILED` | yes | PUT to `output.url` or `thumbnail.url` failed |
| `TIMEOUT` | yes | exceeded `limits.timeout_seconds`; ffmpeg was killed |
| `INTERNAL` | yes | anything else; a bug |

### 4.4 Ingest jobs (`type: "ingest"`)

Schemas: `schema/v1/ingest.schema.json` and `ingest-result.schema.json`. Same
`version`, same envelope, same error block as above.

```jsonc
{
  "version": 1,
  "type": "ingest",
  "reference": "clip_project_42",
  "source": {
    "url": "https://www.youtube.com/watch?v=...",
    "via": "ytdlp",              // "direct" (default): plain GET of a file, ALLOWED_SOURCE_HOSTS
                                 // "ytdlp": a video page, ALLOWED_INGEST_HOSTS
    "max_height": 1080,          // ytdlp only; never pull a taller rendition
    "proxy": null                // ytdlp only; overrides INGEST_PROXY, never logged
  },
  "video": { "url": "<presigned PUT>" },                       // faststart MP4; optional
  "audio": { "url": "<presigned PUT>", "bitrate_kbps": 24, "sample_rate": 16000 },  // mono Opus in Ogg; optional
  "limits": { "max_input_bytes": 2147483648, "max_duration_seconds": 7200, "timeout_seconds": 1200 }
}
```

At least one of `video` and `audio` is required. A caller whose source is already in
its bucket sends `via: "direct"` with only `audio`.

- **ytdlp route.** One metadata pass, then a download that reuses it
  (`--load-info-json`), so the page is resolved once. Live and upcoming streams are
  rejected. Duration and estimated size are checked **before** any media is fetched,
  and again after for pages that do not declare them. Format choice is
  `-S res:<max_height>,vcodec:h264,acodec:aac`: the stored source plays in a browser
  and the merge is a remux.
- **Proxy.** Datacenter addresses get bot-checked. With a proxy configured the worker
  still tries its own address first and only retries through the proxy on
  `SOURCE_BLOCKED`, because proxy bandwidth is the most expensive part of an ingest
  (`INGEST_DIRECT_FIRST=false` sends everything through it). `INGEST_PROXY` is a pool:
  each job samples `INGEST_PROXY_ATTEMPTS` addresses at random and moves to the next
  one on `SOURCE_BLOCKED`. The worker keeps no state between jobs, so random choice is
  what spreads load and keeps one flagged address from failing every job.
  `source.proxied` in the result says which kind of route won; proxy credentials never
  reach logs or results.
- **Audio.** 24 kbps mono Opus is about 11 MB per hour, small enough for any
  speech-to-text API to fetch by URL.
- The result's `source` block carries `title`, `description` (first 5000 chars),
  `uploader`, `thumbnail_url`, `duration_seconds` and the probed stream facts. It is
  filled as soon as it is known, so a `DURATION_TOO_LONG` failure still reports the
  duration. A caller metering by the minute can set `max_duration_seconds` to the
  customer's remaining balance and read the real length off the failure.

### 4.5 Clip jobs (`type: "clip"`)

Schemas: `schema/v1/clip.schema.json` and `clip-result.schema.json`.

```jsonc
{
  "version": 1,
  "type": "clip",
  "reference": "clip_project_42",
  "source": { "url": "<presigned GET of the ingested MP4>" },
  "clips": [                                   // 1 to 50
    {
      "reference": "clip_1",                   // unique within the job, echoed back
      "start_seconds": 312.4,
      "end_seconds": 356.0,                    // clamped to the end of the source
      "output": { "url": "<presigned PUT>" },  // video/mp4
      "thumbnail": { "url": "<presigned PUT>", "timestamp_seconds": 0 },   // optional, clip-relative
      "focus_x": 0.35                          // optional per-clip override of frame.focus_x
    }
  ],
  "frame": { "width": 1080, "height": 1920, "fit": "crop", "focus_x": 0.5, "focus_y": 0.5 },
  "captions": {                                // optional
    "words": [ { "text": "Hello", "start": 312.6, "end": 312.9 } ],   // SOURCE timeline
    "style": { "font": "Montserrat", "highlight_color": "#FFE600", "position": "bottom",
               "max_words": 4, "max_chars": 18, "uppercase": false }
  },
  "video": { "quality": 23, "fps_max": 60, "audio_bitrate_kbps": 128 },
  "limits": { "max_input_bytes": 2147483648, "max_clip_seconds": 600, "timeout_seconds": 1200 }
}
```

- **One download, N cuts.** The source is fetched once. Each cut seeks on the input
  (`-ss` before `-i`), which is frame accurate because the clip is always re-encoded.
- **`frame.fit`.** `crop` fills the canvas and cuts the overflow around the focus
  point (0 is left/top, 1 is right/bottom). `blur` fits the whole picture over a
  blurred, quarter-resolution copy of itself. The focus point is an input so that a
  caller with face tracking can steer the crop without this service knowing about it.
- **Captions.** `words` is the whole transcript on the source timeline; the worker
  slices it per clip, so the same array serves every clip. Words are grouped into
  lines (at most `max_words` / `max_chars`, and always broken on a pause over 0.8 s or
  sentence punctuation) and written as ASS with one event per word so the spoken word
  carries `highlight_color`. `highlight_color: null` gives plain lines. Rendering is
  libass, so right-to-left and non-Latin scripts work through fontconfig fallback
  (Noto is installed; CJK is not). Braces and backslashes in the transcript are
  neutralised so text can never inject ASS override tags.
- **Partial success.** Each clip is uploaded as soon as it is rendered. The job status
  is `completed` when every clip succeeded, `partial` when some did, `failed` when none
  did (the top-level `failure` then repeats the first clip's). Per-clip `status` and
  `failure` say which to resubmit. A job that runs out of time keeps the clips it
  finished; the rest fail with a retryable `TIMEOUT`.
- **Encoder.** Same fallback ladder as normalization (NVDEC + NVENC, software decode
  + NVENC, libx264). Every filter here runs on the CPU, so CUDA only offloads the
  decode and encode. A rung that fails is dropped for the remaining clips of the job.

---

## 5. Processing pipeline

Every job runs these stages in order inside a per-job temporary directory that is
deleted in a `finally` block no matter what.

### 5.1 Validate

- Parse and validate the job against the schema for its `version`.
- Check `source.url` host against `ALLOWED_SOURCE_HOSTS` (section 8). Presigned
  query strings are never logged.

### 5.2 Download

- Stream `source.url` to `<workdir>/input` with a bounded read. Abort as soon as the
  byte count passes `max_input_bytes`, regardless of `Content-Length`. Redirects are
  not followed, since the allowlist was checked against the exact host.
- Connect timeout 10 s, read timeout 60 s per chunk, total bounded by the job timeout.
- Sniff the real type from the first bytes. Reject if it does not match `job.type`.

### 5.3 Probe (video)

Run `ffprobe -v error -print_format json -show_format -show_streams` and extract:

- container (from `format_name`), duration, size
- first video stream: codec, profile, pixel format, width, height, rotation (from
  `side_data_list` `rotation` or the `rotate` tag), average frame rate, color transfer,
  color primaries
- first audio stream if any: codec, sample rate, channels
- faststart: whether the `moov` atom precedes `mdat`. Check by scanning the first
  bytes of the file for the atom order. ffprobe does not report this directly.

HDR detection: `color_transfer` in `smpte2084` (PQ) or `arib-std-b67` (HLG), or
`color_primaries == bt2020`, or a 10-bit pixel format with bt2020 primaries.

### 5.4 Plan (video)

The planner is a **pure function** from probe plus rules to a plan. It must have unit
tests with no ffmpeg involved.

```
target_w, target_h = clamp(probe.width, probe.height, rules)   # section 4.1 semantics

needs_scale        = (target_w, target_h) != (probe.width, probe.height)
needs_tonemap      = probe.is_hdr
needs_video_encode = needs_scale
                  or needs_tonemap
                  or probe.video_codec != rules.video_codec
                  or probe.pixel_format != rules.pixel_format
                  or probe.fps > rules.fps_max
                  or probe.rotation != 0          # bake rotation in, some platforms ignore the tag
needs_audio_encode = probe.has_audio and (
                     probe.audio_codec != rules.audio_codec
                  or probe.audio_sample_rate != rules.audio_sample_rate)
needs_remux        = probe.container != rules.container
                  or (rules.faststart and not probe.faststart)

if not (needs_video_encode or needs_audio_encode or needs_remux):
    plan = UNCHANGED
elif not needs_video_encode and not needs_audio_encode:
    plan = REMUX          # -c copy -movflags +faststart
else:
    plan = ENCODE(video=needs_video_encode, audio=needs_audio_encode, scale, tonemap, fps_cap)
```

`actions` in the result is derived from the plan: `remux`, `scale`, `video_encode`,
`audio_encode`, `tonemap`, `fps_cap`, `rotate`.

### 5.5 Execute (video)

All commands are argument lists, never shell strings. ffmpeg runs with `-y -nostdin
-hide_banner -loglevel error`, stdout discarded, stderr captured for the result, and is
killed with SIGKILL when the job timeout passes. Video streams are selected with the
`0:V:0` specifier so attached cover art is never picked.

Remux only:

```
ffmpeg -i input -map 0:v:0 -map 0:a:0? -c copy -movflags +faststart output.mp4
```

Encode, CPU flavour (`ENCODER=libx264`):

```
ffmpeg -i input
  -map 0:v:0 -map 0:a:0?
  -vf "<fps>,<scale>,setsar=1,format=yuv420p"          # scale=W:H:flags=lanczos, fps=fps=60 only when needed
  -c:v libx264 -preset veryfast -crf <quality> -profile:v high -pix_fmt yuv420p
  -c:a aac -b:a <audio_bitrate>k -ar <sample_rate>     # or -c:a copy when audio is already compliant
  -movflags +faststart
  output.mp4
```

Encode, GPU flavour (`ENCODER=h264_nvenc`), decode and scale stay on the GPU:

```
ffmpeg -hwaccel cuda -hwaccel_output_format cuda -i input
  -map 0:v:0 -map 0:a:0?
  -vf "<fps>,scale_npp=W:H:interp_algo=lanczos,setsar=1"  # frames never leave GPU memory (scale_cuda needs an nvcc build the base image lacks)
  -c:v h264_nvenc -preset p4 -tune hq -rc vbr -cq <quality> -b:v 0 -profile:v high -pix_fmt yuv420p
  -c:a aac -b:a <audio_bitrate>k -ar <sample_rate>
  -movflags +faststart
  output.mp4
```

HDR tone mapping (`needs_tonemap`), CPU path, replaces the scale filter:

```
-vf "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,
     zscale=t=bt709:m=bt709:r=tv,format=yuv420p,scale=W:H:flags=lanczos"
```

Square pixels: `setsar=1` follows every scale. Rounding W and H to even numbers
makes the target a hair off the source ratio, and ffmpeg's scale filters compensate
by writing a sample aspect ratio (1310x702 -> 1920x1028 got 33667:33696, shown as
1918x1028 by players). Platforms disagree on whether to honour that field, so the
worker forces 1:1 and accepts the sub-pixel ratio drift instead.

GPU path for tone mapping is **OPEN** (section 12). Until it is decided, HDR jobs on
the GPU flavour use software decode plus the CPU filter chain plus NVENC. They are
correct, just slower, and `actions` records `tonemap` so the caller can see them.

Encoder fallback: some RunPod hosts start the container without
`libnvidia-encode.so.1` even though the GPU is present, so `h264_nvenc` cannot
open at all. When both nvenc attempts fail the same plan runs once more with
`libx264` on the CPU, and `worker.encoder` reports `libx264` so the caller can
see it happened. It is slow but correct; a host that does this often should be
excluded from the endpoint.

CUDA decode fallback: NVDEC cannot decode every source (4:2:2 10-bit HEVC from
iPhones is the usual case, VP8 another). When the CUDA command fails, the worker
retries the same plan with software decode and NVENC encode, and reports
`worker.decode: "software"`. A failure on both paths is `ENCODE_FAILED` with the
second attempt's stderr.

Rotation: on the CPU path ffmpeg applies the rotate tag automatically before the
user filters (the target dimensions are already post-rotation) and the output has no
tag. On the CUDA path autorotate is disabled and `transpose_npp` does the same on GPU
frames. When remuxing, the tag is preserved; the planner forces an encode when
rotation is non-zero, so remux only ever sees unrotated files.

Frame rate cap: `fps=fps=<fps_max>` only when `probe.fps > fps_max`. Never raise a
frame rate.

Audio: when the source has no audio stream, produce no audio stream. Do not synthesize
silence.

Post-check: probe the output and confirm codec, pixel format, dimensions and faststart
match the rules. A mismatch is `ENCODE_FAILED` with a message naming the field; never
upload a file that fails the post-check.

### 5.6 Thumbnail (video, optional)

```
ffmpeg -ss <timestamp> -i <output or input> -frames:v 1 -vf "scale=W:H" -q:v 3 thumb.jpg
```

Take the frame from the **output** when one was produced, otherwise from the input.
`timestamp_seconds` is clamped to `[0, duration - 0.1]`. Upload to `thumbnail.url`.
The thumbnail is produced and uploaded even when the video itself is `unchanged`, so
the caller always gets one when it asks. A thumbnail failure is logged and reported in
`thumbnail: null`, it does not fail the job.

### 5.7 Images

- Open with Pillow, apply EXIF orientation with `ImageOps.exif_transpose`, then drop
  EXIF from the output.
- GIFs and any animated image (animated WebP included): return `unchanged`. Never
  re-encode a GIF or flatten an animation.
- Compute target dimensions with the same clamp as video, no even-number rounding.
- Resize with `Image.LANCZOS` in both directions.
- Preserve alpha for PNG and WebP. JPEG output is `quality=rules.image.jpeg_quality`,
  `optimize=True`, `progressive=True`. PNG output is `optimize=True`.
- Preserve the ICC profile if present.
- `keep_format: true` keeps the source format. `output.content_type` must agree; if it
  does not, the job is `INVALID_JOB`.
- If no resize is needed and the orientation was already upright, return `unchanged`.
  If only the orientation needed baking in, that counts as a change and is uploaded.
  `actions` is `scale` and/or `orient`.
- `source.width` and `source.height` are reported after orientation, like video.

Pillow limits: the pixel count is checked against `IMAGE_MAX_PIXELS` (default 100 MP)
right after the header is read, so a decompression bomb is `UNSUPPORTED_INPUT` rather
than an OOM.

### 5.8 Upload

- PUT the output file to `output.url` with `Content-Type: output.content_type` and
  `Content-Length`. Stream from disk, never read the file into memory.
- Retry a failed PUT up to 3 times with backoff inside the job before reporting
  `UPLOAD_FAILED`. Presigned URLs are idempotent to PUT.
- Same for the thumbnail.

### 5.9 Cleanup

Delete the job directory. Verify free disk before starting a job; if less than
`2 * max_input_bytes` is free, wait briefly and re-check, then fail with `INTERNAL` so
the caller retries elsewhere.

---

## 6. Handler and concurrency

```python
import asyncio
import runpod

async def handler(event):
    # async + a thread per job: the SDK runs a sync handler on its event loop,
    # which would serialize every job behind one ffmpeg
    return await asyncio.to_thread(process, event["input"])   # never raises; returns a Result dict

def concurrency_modifier(current):
    return int(os.environ.get("WORKER_CONCURRENCY", "8"))

runpod.serverless.start({
    "handler": handler,
    "concurrency_modifier": concurrency_modifier,
})
```

- `WORKER_CONCURRENCY` defaults to 8 on the GPU flavour and 1 on the CPU flavour.
- Each job is CPU-bound in ffmpeg's muxing and I/O even when the encode is on the
  GPU. Around one vCPU per job is the practical guide, so 8 to 12 on an L4 pod with
  12 vCPUs. Tune with the benchmark in section 10, watching `nvidia-smi` encoder and
  decoder utilization alongside CPU.
- The handler must be safe to run concurrently: everything is per-job and lives in
  the job's own directory. No module-level mutable state.

---

## 7. Logging and observability

- One structured JSON log line per stage with `reference`, stage, duration and
  outcome. URLs are logged **without** query strings.
- On failure log the last 4 KB of ffmpeg stderr, and include it in the result.
- Optional Sentry via `SENTRY_DSN`. Capture only `INTERNAL` errors and unhandled
  exceptions; expected failures are results, not exceptions.
- Emit the `timing_ms` block on every result so the caller can track where time goes.

---

## 8. Configuration

All configuration is environment variables. There are no config files.

| variable | default | purpose |
|---|---|---|
| `ENCODER` | `libx264` | `libx264` or `h264_nvenc`. Selects the ffmpeg command template. |
| `WORKER_CONCURRENCY` | `1` | Jobs per worker; see section 6. |
| `ALLOWED_SOURCE_HOSTS` | required | Comma-separated hostnames allowed for `source.url`. Exact match or `*.suffix`. |
| `WORK_DIR` | `/tmp/postiz-uploader` | Scratch root. On RunPod this is the container disk. |
| `MAX_INPUT_BYTES_CAP` | `2147483648` | Upper bound the service enforces even if a job asks for more. |
| `IMAGE_MAX_PIXELS` | `100000000` | Images above this many pixels are rejected as `UNSUPPORTED_INPUT`. |
| `FFMPEG_BIN` / `FFPROBE_BIN` | `ffmpeg` / `ffprobe` | Override binaries for local testing. |
| `ALLOWED_INGEST_HOSTS` | `youtube.com,*.youtube.com,youtu.be` | Hosts a `via: ytdlp` source may point at. Separate from the list above on purpose: yt-dlp's generic extractor will fetch any URL it is given. |
| `INGEST_PROXY` | empty | Proxy pool yt-dlp falls back to when blocked: URLs (`http://`, `socks5://`) or `host:port:user:pass` entries, separated by commas or whitespace. A job's `source.proxy` replaces the pool. |
| `INGEST_PROXY_ATTEMPTS` | `2` | How many different proxies from the pool one job may try. |
| `INGEST_DIRECT_FIRST` | `true` | Try the worker's own address before the proxy. |
| `FONTS_DIR` | empty | Extra directory of caption fonts for libass, on top of fontconfig. |
| `SENTRY_DSN` | empty | Optional error reporting. |
| `LOG_LEVEL` | `info` | |

There are intentionally **no storage credentials** and no caller API keys. The only
secrets are the RunPod endpoint key, which lives on the caller, and the optional
ingest proxy URL.

---

## 9. Docker images

Two build targets from one Dockerfile, same Python code.

- `cpu`: `python:3.12-slim` plus a pinned static ffmpeg build for the target
  architecture (amd64 and arm64, so it builds on Apple Silicon too) with libx264,
  libzimg for `zscale`, and `tonemap`. Used for local development, CI, the
  benchmark on a laptop, and as a CPU fallback fleet if ever needed.
- `gpu`: the same `python:3.12-slim` base plus `ffmpeg`/`ffprobe` and the shared
  libraries they resolve, copied out of `jrottenberg/ffmpeg:7.1-nvidia2204` (built with
  `--enable-libnpp --enable-nvenc --enable-nvdec`). The prebuilt image is 2.6 GB because
  it carries the whole CUDA runtime (cuBLAS, cuSPARSE, cuFFT, NCCL, ...); ffmpeg needs
  about 260 MB of it, mostly the NPP image libraries behind `scale_npp`. Copying only
  what `ldd` reports gives a 0.7 GB image (0.33 GB download), which cuts a fresh
  RunPod cold pull from ~80 s to ~15 s. The CUDA compat package is kept so hosts on the
  525/535 driver branches still work, and `NVIDIA_REQUIRE_CUDA` mirrors the base.
  The build fails if `ldd` reports a missing library or if `h264_nvenc`, `scale_npp`,
  `zscale` or `tonemap` are absent, so a broken build fails in CI rather than on the
  first job.

Both images also carry what clipping needs: `yt-dlp` with a pinned Deno (it needs a
JavaScript runtime for YouTube's player challenges), and fontconfig with Montserrat
and Noto for captions. The build asserts the `ass` filter and `libopus` exist and runs
`scripts/check-caption-font.sh`, which renders one subtitle and fails unless libass
resolved Montserrat to the real font file (a broken fontconfig otherwise falls back
silently).

Both images:

- run as a non-root user,
- set `ENCODER` appropriately,
- have `HEALTHCHECK` disabled (RunPod manages worker health),
- pin every dependency version.

Tag scheme: `ghcr.io/gitroomhq/postiz-uploader:<semver>-cpu` and `<semver>-gpu`. The
RunPod endpoint references an explicit semver tag, never `latest`.

---

## 10. Testing and benchmark

### 10.1 Fixtures

`scripts/make_fixtures.py` generates the whole set synthetically with ffmpeg's `lavfi`
sources and Pillow, so nothing binary is committed and the tests build it into a temp
directory on every run. Real phone clips (in particular HDR iPhone footage and 4:2:2
HEVC) cannot be synthesized faithfully and are still an open decision (section 12).
Fixtures that need `libx265` or `libvpx` are skipped when the local ffmpeg lacks them;
the HDR test additionally needs `zscale`, which the Docker images have.

| fixture | exercises |
|---|---|
| `compliant-1080p.mp4` | 1080p H.264 AAC faststart, expect `unchanged` |
| `compliant-no-faststart.mp4` | expect `remux` only |
| `iphone-4k-hevc.mov` | 4K HEVC, downscale plus codec change |
| `iphone-4k-hdr-hlg.mov` | tone mapping |
| `portrait-rotated.mov` | 1920x1080 with rotate 90, expect 1080x1920 output |
| `small-480p.webm` | VP9 upscale to 1080 |
| `slowmo-120fps.mp4` | frame-rate cap |
| `no-audio.mp4` | no audio stream in output |
| `ultrawide-2560x1080.mp4` | long-side cap |
| `truncated.mp4` | corrupt file, expect `PROBE_FAILED` or `ENCODE_FAILED` |
| `large.png` | downscale with alpha preserved |
| `small-exif-rotated.jpg` | upscale to 720 with orientation baked in |
| `animated.gif` | `unchanged` |
| `bomb.png` | decompression bomb, expect `UNSUPPORTED_INPUT` |

### 10.2 Tests

- **Unit**: the planner (section 5.4) and the clamp function, table-driven, no ffmpeg.
  Every row above has a planner test asserting the expected `actions`.
- **Integration**: run the full pipeline on each fixture against a local HTTP server
  that serves the source and accepts PUTs, and assert the post-check plus the result
  document. Runs on the `cpu` image in CI.
- **Contract**: JSON schema files for Job and Result under `schema/v1/`, validated in
  tests. The caller repo will vendor these files.

### 10.3 Benchmark

`scripts/bench.py` runs every video fixture N times at a given concurrency and prints a
table: fixture, plan, wall time, realtime factor, peak RSS, and on the GPU flavour the
encoder and decoder utilization sampled from `nvidia-smi`. Its output is what sets
`WORKER_CONCURRENCY` and is attached to every release note.

### 10.4 Local development

```
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/ruff check . && .venv/bin/pytest                 # needs ffmpeg/ffprobe on PATH

.venv/bin/python scripts/make_fixtures.py fixtures/generated
ALLOWED_SOURCE_HOSTS=127.0.0.1 .venv/bin/python scripts/run-local.py fixtures/generated/4k-hevc.mov
ALLOWED_SOURCE_HOSTS=127.0.0.1 .venv/bin/python scripts/bench.py fixtures/generated --concurrency 2

docker compose up --build                                  # worker API on :8000, bucket stand-in on :8080
.venv/bin/python scripts/run-local.py clip.mov --endpoint http://localhost:8000 --bucket-host host.docker.internal
```

- `scripts/run-local.py` runs one job in-process by default (no RunPod involved),
  serving the file from a throwaway local server and moving outputs to
  `bench-results/`. With `--endpoint` it posts the same job to a running worker API.
- `docker compose up` builds the `cpu` image and starts the RunPod local API
  (`python handler.py --rp_serve_api`) on port 8000, plus `postiz_uploader.devserver`
  on port 8080 as the bucket stand-in.
- `postiz_uploader.devserver` is the GET/PUT file server used by the tests, compose and
  the scripts. It is dev tooling and is never reachable in production.

---

## 11. Deployment on RunPod

Endpoint settings for production:

| setting | value | note |
|---|---|---|
| GPU | NVIDIA L4 | 2 NVENC, 4 NVDEC, 12 vCPU, 50 GB RAM. No session limit. |
| Cloud | Secure | Small premium over Community on this card, no evictions. |
| Min workers | 1 | One warm worker avoids cold starts on the first upload after a lull. **OPEN** whether to start at 0. |
| Max workers | 5 | Cost cap. Raise with volume. |
| Idle timeout | 60 s | Flex workers spin down after this. |
| FlashBoot | on | |
| Container disk | 50 GB | `WORKER_CONCURRENCY * 2 * max_input_bytes` with headroom. |
| Execution timeout | 1500 s | Above `limits.timeout_seconds` so the service's own timeout fires first and returns a proper `TIMEOUT` result. |
| Job TTL | 3600 s | Queue time plus execution. Expired jobs return 404 to the caller, which must treat that as retryable. |
| Env | `ENCODER=h264_nvenc`, `WORKER_CONCURRENCY=8`, `ALLOWED_SOURCE_HOSTS=<bucket hosts>` | |
| Region | closest to the R2 bucket's location hint | Download and upload are the only latency that matters. |

**Two endpoints, one image.** Clipping runs on its own endpoint (`postiz-clipper`) with
the same image tag. A clip job downloads up to 2 GB and renders for minutes; on a
shared queue five of them would hold every worker while a person waits on a web
upload. The split also keeps the settings honest:

| setting | normalization | clipping |
|---|---|---|
| `WORKER_CONCURRENCY` | 8 | 2 (each job runs a full-frame CPU filter chain) |
| `ALLOWED_SOURCE_HOSTS` | bucket hosts | bucket hosts |
| `ALLOWED_INGEST_HOSTS` | unused | default (YouTube) |
| `INGEST_PROXY` | unset | the proxy, only here |
| Container disk | 50 GB | 50 GB |

Release flow: GitHub Actions builds both images on a tag, pushes to GHCR, runs the
integration suite on the CPU image, and the endpoint's image tag is updated by hand (or
by the RunPod API in a later iteration). Never point production at a tag that has not
passed the suite.

---

## 12. Open decisions

1. **GPU tone mapping.** Options are `tonemap_opencl` (needs OpenCL in the image),
   `libplacebo` with Vulkan (best quality, heaviest image), or accept the CPU path for
   HDR only. Decide after the benchmark shows what share of real uploads is HDR.
2. **Min workers 0 or 1.** Whether a 30 to 60 second cold start on the first job after
   a quiet period is acceptable for web users. Cost of a warm L4 is roughly $350 per
   month.
3. **Fixture hosting.** Git LFS in this repo versus a fixtures bucket with a download
   script.
4. **Image library.** Pillow is the default. Switch to `pyvips` only if a benchmark
   shows Pillow is too slow on 30 MB PNGs at the target concurrency.

---

## 13. What the caller will do (out of scope here)

For orientation only. None of this lives in this repo and none of it is built yet.

- Add a processing status to its media records, additive migration, default ready.
- On every upload path, web and API, save the original, mark the record processing,
  and start a workflow.
- The workflow mints presigned GET and PUT URLs, submits a Job to the RunPod endpoint,
  polls status, and on `completed` swaps the record's path to the output and deletes
  the original; on `unchanged` keeps the original; on `failed` keeps the original and
  records the error. The presigned PUT must be signed with the same `Content-Type`
  the service sends (`output.content_type`, `image/jpeg` for the thumbnail), and the
  output key should carry the final extension (`<id>.mp4`) so the caller's own
  extension-based type detection keeps working.
- Web uploads wait in the uploader until the record is ready, as they do today with
  Transloadit. API clients poll a get-media endpoint, and scheduled posts wait for
  readiness before publishing.
- Fairness across tenants, per-plan limits, and dedupe by content hash are the
  caller's job. The service processes whatever it is handed.

---

## 14. Milestones

1. **Done.** Repo skeleton, Job and Result schemas under `schema/v1/`, planner with
   table-driven unit tests.
2. **Done.** CPU pipeline end to end on the generated fixture set, integration tests,
   `docker compose` local loop, `scripts/run-local.py`, `scripts/bench.py`, the RunPod
   local API smoke-tested with `curl`.
3. **Partly done.** `0.1.1-gpu` ran on a real L4 (2026-09-16): a 720p H.264 clip
   was upscaled to 1080p with `h264_nvenc` and passed the post-check. The CUDA
   decode path (`-hwaccel cuda` + `scale_npp`) failed with ffmpeg error -38 and the
   worker fell back to software decode as designed (`worker.decode: "software"`).
   Getting NVDEC decode working and the benchmark-driven concurrency setting are
   still open.
4. **Done.** Endpoint `postiz-uploader` (id `otkdrlw2yudos9`) in Secure Cloud, pinned
   to the L4 inside pool `AMPERE_24`, image `ghcr.io/gitroomhq/postiz-uploader:0.1.1-gpu`,
   FlashBoot on, 50 GB disk, execution timeout 1500 s, min 0 / max 5 workers, idle
   60 s. `ALLOWED_SOURCE_HOSTS` is a placeholder (`*.r2.cloudflarestorage.com`) until
   the real bucket hosts are set. Min workers is 0 until the GPU path is validated
   further; raise to 1 for a warm worker (about $0.69/h).
5. Hand the schema files and endpoint details to the Postiz integration work.
6. **Done (0.2.0).** `ingest` and `clip` job types, their schemas, yt-dlp + Deno and
   caption fonts in both images, tests on generated fixtures including the yt-dlp
   route through its generic extractor.
7. **Done.** Endpoint `postiz-clipper` (id `or9bfqyb37w9ze`), same GPU pinning, image
   `0.2.0-gpu`, `WORKER_CONCURRENCY=2`, min 0 / max 10 workers, idle 60 s
   (`postiz-uploader` is max 20; the account quota is 30). First L4 run, 2026-09-18:
   a two-clip blur-fit job with captions rendered in 11 s with `decode: cuda` +
   `h264_nvenc`, and a YouTube ingest succeeded on the direct route with no proxy
   (`proxied: false`). One video from one address proves little about bot checks at
   volume; `INGEST_PROXY` is still unset.
