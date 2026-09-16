# syntax=docker/dockerfile:1.7
#
# Two targets, one codebase:
#   docker build --target cpu -t postiz-uploader:cpu .
#   docker build --target gpu -t postiz-uploader:gpu .

ARG FFMPEG_STATIC_VERSION=7.0.2
ARG FFMPEG_GPU_IMAGE=jrottenberg/ffmpeg:7.1-nvidia2204

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
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl xz-utils \
    && rm -rf /var/lib/apt/lists/*
RUN set -eux; \
    case "$TARGETARCH" in amd64|arm64) arch="$TARGETARCH" ;; *) echo "unsupported arch $TARGETARCH" >&2; exit 1 ;; esac; \
    curl -fsSL "https://johnvansickle.com/ffmpeg/releases/ffmpeg-${FFMPEG_STATIC_VERSION}-${arch}-static.tar.xz" -o /tmp/ffmpeg.tar.xz; \
    mkdir -p /tmp/ffmpeg && tar -xJf /tmp/ffmpeg.tar.xz -C /tmp/ffmpeg --strip-components=1; \
    install -m 0755 /tmp/ffmpeg/ffmpeg /tmp/ffmpeg/ffprobe /usr/local/bin/; \
    rm -rf /tmp/ffmpeg /tmp/ffmpeg.tar.xz; \
    ffmpeg -hide_banner -encoders | grep -q ' libx264 '; \
    ffmpeg -hide_banner -filters | grep -q ' zscale '; \
    ffmpeg -hide_banner -filters | grep -q ' tonemap '
COPY --from=deps /install /usr/local
WORKDIR /app
COPY postiz_uploader ./postiz_uploader
COPY schema ./schema
COPY handler.py .
RUN useradd --create-home --uid 1000 worker && mkdir -p /work && chown worker:worker /work
USER worker
CMD ["python", "handler.py"]

# ---------------------------------------------------------------- gpu
FROM ${FFMPEG_GPU_IMAGE} AS gpu
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    ENCODER=h264_nvenc WORKER_CONCURRENCY=8 WORK_DIR=/work \
    NVIDIA_DRIVER_CAPABILITIES=compute,video,utility
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ffmpeg -hide_banner -encoders | grep -q ' h264_nvenc ' \
    && ffmpeg -hide_banner -filters | grep -q ' scale_cuda ' \
    && ffmpeg -hide_banner -filters | grep -q ' zscale ' \
    && ffmpeg -hide_banner -filters | grep -q ' tonemap ' \
    && (ffmpeg -hide_banner -filters | grep -q ' transpose_npp ' || echo "WARNING: transpose_npp missing, rotated clips will use software decode")
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY postiz_uploader ./postiz_uploader
COPY schema ./schema
COPY handler.py .
RUN useradd --create-home --uid 1000 worker && mkdir -p /work && chown worker:worker /work
USER worker
# the ffmpeg base image sets ffmpeg as the entrypoint
ENTRYPOINT []
CMD ["python3", "handler.py"]
