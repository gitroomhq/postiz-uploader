"""Run one job against a local file and print the result.

In-process by default (no RunPod involved), which is the day-to-day loop for
touching ffmpeg flags:

    python scripts/run-local.py path/to/clip.mov
    python scripts/run-local.py photo.jpg --type image --rules rules.json

With --endpoint the same job is submitted to a running worker HTTP API instead
(`docker compose up`, or `python handler.py --rp_serve_api`):

    python scripts/run-local.py clip.mov --endpoint http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from postiz_uploader.devserver import serve  # noqa: E402

VIDEO_RULES = {"short_side_min": 1080, "short_side_max": 1080, "long_side_max": 1920}
IMAGE_RULES = {"short_side_min": 720, "short_side_max": 1080, "long_side_max": 1920}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file")
    parser.add_argument("--type", choices=["video", "image"], help="defaults from the file's mime type")
    parser.add_argument("--rules", help="JSON file with a rules object to merge over the defaults")
    parser.add_argument("--out", default="bench-results", help="where outputs are written (bucket stand-in root)")
    parser.add_argument("--endpoint", help="worker HTTP API base URL; omit to run in-process")
    parser.add_argument("--no-thumbnail", action="store_true")
    parser.add_argument("--bucket-host", default="127.0.0.1", help="host the worker should use to reach this machine")
    args = parser.parse_args()

    mime = mimetypes.guess_type(args.file)[0] or ""
    kind = args.type or ("image" if mime.startswith("image/") else "video")
    rules = dict(IMAGE_RULES if kind == "image" else VIDEO_RULES)
    if args.rules:
        with open(args.rules, encoding="utf-8") as fh:
            rules.update(json.load(fh))

    os.makedirs(args.out, exist_ok=True)
    root = tempfile.mkdtemp(prefix="run-local-")
    name = os.path.basename(args.file)
    shutil.copy(args.file, os.path.join(root, name))
    server, _ = serve(root, "0.0.0.0", 0)
    port = server.server_address[1]
    base = f"http://{args.bucket_host}:{port}"

    stem, ext = os.path.splitext(name)
    out_type = "video/mp4" if kind == "video" else mime
    out_name = f"{stem}.out{'.mp4' if kind == 'video' else ext}"
    job = {
        "version": 1,
        "type": kind,
        "reference": f"local_{stem}",
        "source": {"url": f"{base}/{name}"},
        "output": {"url": f"{base}/out/{out_name}", "content_type": out_type},
        "rules": rules,
    }
    if kind == "video" and not args.no_thumbnail:
        job["thumbnail"] = {"url": f"{base}/out/{stem}.thumb.jpg", "timestamp_seconds": 0}

    started = time.monotonic()
    try:
        if args.endpoint:
            import requests

            r = requests.post(f"{args.endpoint.rstrip('/')}/runsync", json={"input": job}, timeout=3600)
            r.raise_for_status()
            body = r.json()
            result = body.get("output", body)
        else:
            os.environ.setdefault("ALLOWED_SOURCE_HOSTS", args.bucket_host)
            from postiz_uploader.pipeline import process

            result = process(job)
    finally:
        server.shutdown()

    print(json.dumps(result, indent=2))
    out_dir = os.path.join(root, "out")
    if os.path.isdir(out_dir):
        for f in os.listdir(out_dir):
            shutil.move(os.path.join(out_dir, f), os.path.join(args.out, f))
            print(f"wrote {os.path.join(args.out, f)}", file=sys.stderr)
    shutil.rmtree(root, ignore_errors=True)
    print(f"wall {time.monotonic() - started:.1f}s", file=sys.stderr)
    return 0 if result.get("status") != "failed" else 1


if __name__ == "__main__":
    sys.exit(main())
