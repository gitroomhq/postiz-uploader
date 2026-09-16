from __future__ import annotations

import os
import shutil
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

# Configuration must be in place before the first get_settings() call
os.environ.setdefault("ALLOWED_SOURCE_HOSTS", "127.0.0.1,localhost")
os.environ.setdefault("ENCODER", "libx264")

from postiz_uploader.devserver import serve  # noqa: E402

HAS_FFMPEG = shutil.which(os.environ.get("FFMPEG_BIN", "ffmpeg")) is not None
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")


@pytest.fixture(scope="session")
def fixtures_dir(tmp_path_factory) -> str:
    """Generate the synthetic fixture set once per session."""
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg not installed")
    from make_fixtures import make

    out = str(tmp_path_factory.mktemp("fixtures"))
    make(out)
    return out


@pytest.fixture(scope="session")
def bucket(tmp_path_factory):
    """A local stand-in for the bucket: GET serves from it, PUT writes into it."""
    root = str(tmp_path_factory.mktemp("bucket"))
    server, _ = serve(root, "127.0.0.1", 0)
    host, port = server.server_address[:2]
    yield {"root": root, "base": f"http://{host}:{port}"}
    server.shutdown()


@pytest.fixture
def work_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("WORK_DIR", str(tmp_path / "work"))
    return str(tmp_path / "work")


def make_job(
    *,
    type_: str,
    source_url: str,
    output_url: str,
    output_type: str,
    thumbnail_url: str | None = None,
    reference: str = "ref_1",
    rules: dict | None = None,
    limits: dict | None = None,
) -> dict:
    job = {
        "version": 1,
        "type": type_,
        "reference": reference,
        "source": {"url": source_url},
        "output": {"url": output_url, "content_type": output_type},
        "rules": {"short_side_min": 1080, "short_side_max": 1080, "long_side_max": 1920},
    }
    if type_ == "image":
        job["rules"] = {"short_side_min": 720, "short_side_max": 1080, "long_side_max": 1920}
    if rules:
        job["rules"].update(rules)
    if thumbnail_url:
        job["thumbnail"] = {"url": thumbnail_url, "timestamp_seconds": 0.5}
    if limits:
        job["limits"] = limits
    return job
