"""Throughput benchmark: every video fixture, N times, at a given concurrency.

    python scripts/make_fixtures.py fixtures/generated
    python scripts/bench.py fixtures/generated --runs 3 --concurrency 4
    python scripts/bench.py /path/to/real/phone/clips --concurrency 8

Prints one row per fixture: plan, wall time, realtime factor (video seconds
processed per wall second), and on a GPU worker the encoder/decoder utilization
sampled from nvidia-smi while the batch ran. Its output sets WORKER_CONCURRENCY.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from postiz_uploader.devserver import serve  # noqa: E402

VIDEO_EXT = (".mp4", ".mov", ".webm", ".mkv", ".m4v")


class GpuSampler(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.samples: list[tuple[int, int]] = []
        self.stop = threading.Event()

    def run(self) -> None:
        while not self.stop.is_set():
            try:
                out = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.encoder,utilization.decoder",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                enc, dec = (int(x) for x in out.stdout.strip().split(","))
                self.samples.append((enc, dec))
            except Exception:  # noqa: BLE001
                return
            self.stop.wait(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dir", help="directory of video files")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--bucket-host", default="127.0.0.1")
    parser.add_argument("--json", help="also write raw results to this file")
    args = parser.parse_args()

    os.environ.setdefault("ALLOWED_SOURCE_HOSTS", args.bucket_host)
    from postiz_uploader.pipeline import process

    files = sorted(f for f in os.listdir(args.dir) if f.lower().endswith(VIDEO_EXT))
    if not files:
        print("no video files found", file=sys.stderr)
        return 1

    root = tempfile.mkdtemp(prefix="bench-")
    for f in files:
        shutil.copy(os.path.join(args.dir, f), os.path.join(root, f))
    server, _ = serve(root, "0.0.0.0", 0)
    base = f"http://{args.bucket_host}:{server.server_address[1]}"

    jobs = []
    for i in range(args.runs):
        for f in files:
            jobs.append(
                (
                    f,
                    {
                        "version": 1,
                        "type": "video",
                        "reference": f"bench_{i}_{f}",
                        "source": {"url": f"{base}/{f}"},
                        "output": {"url": f"{base}/out/{i}/{f}.mp4", "content_type": "video/mp4"},
                        "rules": {"short_side_min": 1080, "short_side_max": 1080, "long_side_max": 1920},
                    },
                )
            )

    sampler = GpuSampler()
    sampler.start()
    started = time.monotonic()

    def one(item):
        name, job = item
        t = time.monotonic()
        result = process(job)
        return name, time.monotonic() - t, result

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(one, jobs))
    wall_total = time.monotonic() - started
    sampler.stop.set()
    server.shutdown()
    shutil.rmtree(root, ignore_errors=True)

    by_name: dict[str, list] = {}
    for name, wall, result in results:
        by_name.setdefault(name, []).append((wall, result))

    print(f"{'fixture':34} {'plan':38} {'wall s':>8} {'x realtime':>11} {'status'}")
    for name, rows in by_name.items():
        walls = [w for w, _ in rows]
        r = rows[0][1]
        duration = (r.get("source") or {}).get("duration_seconds") or 0
        median = statistics.median(walls)
        factor = f"{duration / median:.1f}" if duration and median else "-"
        plan = ",".join(r.get("actions") or []) or r.get("status")
        status = r["status"] if r["status"] != "failed" else f"failed:{r['error']['code']}"
        print(f"{name:34} {plan:38} {median:8.1f} {factor:>11} {status}")

    print(f"\ntotal wall {wall_total:.1f}s for {len(jobs)} jobs at concurrency {args.concurrency}")
    peak_children_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    unit = 1024 * 1024 if sys.platform == "darwin" else 1024
    print(f"peak child RSS {peak_children_rss / unit:.0f} MB")
    if sampler.samples:
        enc = max(s[0] for s in sampler.samples)
        dec = max(s[1] for s in sampler.samples)
        print(f"nvidia-smi peak utilization: encoder {enc}% decoder {dec}%")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump([{"name": n, "wall": w, "result": r} for n, w, r in results], fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
