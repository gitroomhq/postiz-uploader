# syntax=docker/dockerfile:1.7
#
# Two targets, one codebase:
#   docker build --target cpu -t postiz-uploader:cpu .
#   docker build --target gpu -t postiz-uploader:gpu .

ARG FFMPEG_STATIC_VERSION=7.0.2
ARG FFMPEG_GPU_IMAGE=jrottenberg/ffmpeg:7.1-nvidia2204
# yt-dlp needs a JavaScript runtime to solve YouTube's player challenges
ARG DENO_IMAGE=denoland/deno:bin-2.5.0

FROM ${DENO_IMAGE} AS deno

# ---------------------------------------------------------------- shared python deps
FROM python:3.12-slim AS deps
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------------------------------------------------------------- cpu
FROM python:3.12-slim AS cpu
ARG FFMPEG_STATIC_VERSION
ARG TARGETARCH
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    ENCODER=libx264 WORKER_CONCURRENCY=1 WORK_DIR=/work
# fonts: Montserrat is the default caption face, Noto covers non-Latin scripts through
# fontconfig fallback (CJK is not included, it would add ~100 MB)
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl xz-utils \
      fontconfig fonts-montserrat fonts-noto-core \
    && rm -rf /var/lib/apt/lists/* \
    # keep the four basic Montserrat faces only: the other weights also register under the
    # family name "Montserrat", and libass versions disagree on which one "bold" means
    && find /usr/share/fonts -ipath '*montserrat*' -type f \
         ! -name 'Montserrat-Regular.otf' ! -name 'Montserrat-Bold.otf' \
         ! -name 'Montserrat-Italic.otf' ! -name 'Montserrat-BoldItalic.otf' -delete \
    && fc-cache -f
RUN set -eux; \
    case "$TARGETARCH" in amd64|arm64) arch="$TARGETARCH" ;; *) echo "unsupported arch $TARGETARCH" >&2; exit 1 ;; esac; \
    curl -fsSL "https://johnvansickle.com/ffmpeg/releases/ffmpeg-${FFMPEG_STATIC_VERSION}-${arch}-static.tar.xz" -o /tmp/ffmpeg.tar.xz; \
    mkdir -p /tmp/ffmpeg && tar -xJf /tmp/ffmpeg.tar.xz -C /tmp/ffmpeg --strip-components=1; \
    install -m 0755 /tmp/ffmpeg/ffmpeg /tmp/ffmpeg/ffprobe /usr/local/bin/; \
    rm -rf /tmp/ffmpeg /tmp/ffmpeg.tar.xz; \
    ffmpeg -hide_banner -encoders | grep -q ' libx264 '; \
    ffmpeg -hide_banner -filters | grep -q ' zscale '; \
    ffmpeg -hide_banner -filters | grep -q ' tonemap '; \
    ffmpeg -hide_banner -filters | grep -q ' ass '; \
    ffmpeg -hide_banner -encoders | grep -q ' libopus '
COPY scripts/check-caption-font.sh /usr/local/bin/
RUN check-caption-font.sh
COPY --from=deno /deno /usr/local/bin/deno
COPY --from=deps /install /usr/local
RUN deno --version && python -m yt_dlp --version
WORKDIR /app
COPY postiz_uploader ./postiz_uploader
COPY schema ./schema
COPY handler.py .
RUN useradd --create-home --uid 1000 worker && mkdir -p /work && chown worker:worker /work
USER worker
CMD ["python", "handler.py"]

# ---------------------------------------------------------------- gpu
# The prebuilt ffmpeg image drags in the whole CUDA runtime (cuBLAS, cuSPARSE, cuFFT,
# NCCL, ...), 2.6 GB of which ffmpeg uses about 260 MB. Stage 1 collects ffmpeg,
# ffprobe and every shared library they resolve; stage 2 drops them into the same
# python:3.12-slim base the cpu image uses. Result: ~0.6 GB instead of 2.6 GB, and a
# cold pull on RunPod of ~15 s instead of ~80 s.
FROM ${FFMPEG_GPU_IMAGE} AS ffmpeg-gpu
RUN set -eux; \
    mkdir -p /slim/bin /slim/lib; \
    cp /usr/local/bin/ffmpeg /usr/local/bin/ffprobe /slim/bin/; \
    # every resolved dependency except what the target base already provides
    # (glibc, libgcc, libstdc++, libgomp, openssl, zlib, expat)
    ldd /usr/local/bin/ffmpeg /usr/local/bin/ffprobe \
      | awk '/=> \//{print $3}' | sort -u \
      | grep -Ev '/(ld-linux[^/]*|libc|libm|libdl|libpthread|librt|libmvec|libresolv|libgcc_s|libstdc\+\+|libgomp|libcrypto|libssl|libz|libexpat)\.so' \
      | while read -r lib; do cp -L "$lib" /slim/lib/; done; \
    ls /slim/lib | wc -l

FROM python:3.12-slim AS gpu
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    ENCODER=h264_nvenc WORKER_CONCURRENCY=8 WORK_DIR=/work \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,video,utility \
    # same constraint the CUDA 12.3 base declares: the driver must speak CUDA 12.3, or be
    # one of the LTS branches (470/525/535) that the compat package below covers
    NVIDIA_REQUIRE_CUDA="cuda>=12.3 brand=tesla,driver>=470,driver<471 brand=unknown,driver>=470,driver<471 brand=nvidia,driver>=470,driver<471 brand=nvidiartx,driver>=470,driver<471 brand=geforce,driver>=470,driver<471 brand=geforcertx,driver>=470,driver<471 brand=quadro,driver>=470,driver<471 brand=quadrortx,driver>=470,driver<471 brand=titan,driver>=470,driver<471 brand=titanrtx,driver>=470,driver<471 brand=tesla,driver>=525,driver<526 brand=unknown,driver>=525,driver<526 brand=nvidia,driver>=525,driver<526 brand=nvidiartx,driver>=525,driver<526 brand=geforce,driver>=525,driver<526 brand=geforcertx,driver>=525,driver<526 brand=quadro,driver>=525,driver<526 brand=quadrortx,driver>=525,driver<526 brand=titan,driver>=525,driver<526 brand=titanrtx,driver>=525,driver<526 brand=tesla,driver>=535,driver<536 brand=unknown,driver>=535,driver<536 brand=nvidia,driver>=535,driver<536 brand=nvidiartx,driver>=535,driver<536 brand=geforce,driver>=535,driver<536 brand=geforcertx,driver>=535,driver<536 brand=quadro,driver>=535,driver<536 brand=quadrortx,driver>=535,driver<536 brand=titan,driver>=535,driver<536 brand=titanrtx,driver>=535,driver<536"
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libstdc++6 libgomp1 libexpat1 \
      fontconfig fonts-montserrat fonts-noto-core \
    && rm -rf /var/lib/apt/lists/* \
    # keep the four basic Montserrat faces only: the other weights also register under the
    # family name "Montserrat", and libass versions disagree on which one "bold" means
    && find /usr/share/fonts -ipath '*montserrat*' -type f \
         ! -name 'Montserrat-Regular.otf' ! -name 'Montserrat-Bold.otf' \
         ! -name 'Montserrat-Italic.otf' ! -name 'Montserrat-BoldItalic.otf' -delete \
    && fc-cache -f
COPY --from=ffmpeg-gpu /slim/bin/ /opt/ffmpeg/bin/
COPY --from=ffmpeg-gpu /slim/lib/ /opt/ffmpeg/lib/
# forward-compat driver libs: the nvidia container runtime mounts these over the host
# driver when the host is on an older LTS branch (see NVIDIA_REQUIRE_CUDA above)
COPY --from=ffmpeg-gpu /usr/local/cuda/compat/ /usr/local/cuda/compat/
# zz- so the base's own libs win and /opt/ffmpeg/lib only fills the gaps
RUN set -eux; \
    echo /opt/ffmpeg/lib > /etc/ld.so.conf.d/zz-ffmpeg.conf; ldconfig; \
    ln -s /opt/ffmpeg/bin/ffmpeg /opt/ffmpeg/bin/ffprobe /usr/local/bin/; \
    if ldd /usr/local/bin/ffmpeg /usr/local/bin/ffprobe | grep 'not found'; then exit 1; fi; \
    ffmpeg -hide_banner -encoders | grep -q ' h264_nvenc '; \
    ffmpeg -hide_banner -filters | grep -q ' scale_npp '; \
    ffmpeg -hide_banner -filters | grep -q ' zscale '; \
    ffmpeg -hide_banner -filters | grep -q ' tonemap '; \
    ffmpeg -hide_banner -filters | grep -q ' ass '; \
    ffmpeg -hide_banner -encoders | grep -q ' libopus '; \
    (ffmpeg -hide_banner -filters | grep -q ' transpose_npp ' || echo "WARNING: transpose_npp missing, rotated clips will use software decode")
COPY scripts/check-caption-font.sh /usr/local/bin/
RUN check-caption-font.sh
COPY --from=deno /deno /usr/local/bin/deno
COPY --from=deps /install /usr/local
RUN deno --version && python -m yt_dlp --version
WORKDIR /app
COPY postiz_uploader ./postiz_uploader
COPY schema ./schema
COPY handler.py .
RUN useradd --create-home --uid 1000 worker && mkdir -p /work && chown worker:worker /work
USER worker
CMD ["python", "handler.py"]
