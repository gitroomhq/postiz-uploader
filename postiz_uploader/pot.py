"""The PO token server behind yt-dlp's bgutil plugin.

YouTube scores requests that carry a proof-of-origin token as more legitimate. The
plugin (a pip dependency) asks http://127.0.0.1:4416 for one on every extraction and
carries on without when nobody answers, so everything here is best effort: a server
that will not start costs the token, never the job.

A token does not unblock an address YouTube has already flagged. It only slows down
how fast a clean one gets there.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import threading
import time

from postiz_uploader.config import Settings
from postiz_uploader.log import get_logger, log

logger = get_logger(__name__)

PORT = 4416  # the plugin's default base_url; changing it means passing extractor args too
READY_TIMEOUT = 15.0
LOG_TAIL = 2000

_lock = threading.Lock()
_proc: subprocess.Popen | None = None


def _log_path(settings: Settings) -> str:
    return os.path.join(settings.work_dir, "pot-server.log")


def _listening() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", PORT), timeout=0.5):
            return True
    except OSError:
        return False


def _tail(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read()[-LOG_TAIL:]
    except OSError:
        return ""


def _spawn(settings: Settings) -> subprocess.Popen | None:
    home = settings.pot_server_dir
    deno = shutil.which("deno")
    if not deno or not os.path.isfile(os.path.join(home, "src", "main.ts")):
        log(logger, "pot server not installed, continuing without PO tokens", dir=home, deno=bool(deno))
        return None
    modules = os.path.join(home, "node_modules")
    os.makedirs(settings.work_dir, exist_ok=True)
    # a file, not a pipe: nobody drains a pipe, and the server logs every token
    with open(_log_path(settings), "w", encoding="utf-8") as out:
        return subprocess.Popen(
            [
                deno,
                "run",
                "--allow-env",
                "--allow-net",
                f"--allow-ffi={modules}",
                f"--allow-read={modules}",
                os.path.join(home, "src", "main.ts"),
                "--port",
                str(PORT),
            ],
            cwd=home,
            env={
                **os.environ,
                "DENO_DIR": os.path.join(home, ".cache", "deno"),
                "DENO_NO_PROMPT": "1",
                "DENO_NO_UPDATE_CHECK": "1",
            },
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT,
        )


def ensure(settings: Settings, *, wait: bool = True) -> bool:
    """Start the server if it is not running. True once it accepts connections.

    Called without waiting when the worker boots and again before each yt-dlp ingest,
    which also restarts a server that died. Never raises.
    """
    global _proc
    if not settings.pot_server_dir:
        return False
    try:
        with _lock:
            if _proc is not None and _proc.poll() is not None:
                log(logger, "pot server exited", code=_proc.returncode, tail=_tail(_log_path(settings)))
                _proc = None
            if _proc is None and not _listening():
                _proc = _spawn(settings)
                if _proc is None:
                    return False
                log(logger, "pot server started", pid=_proc.pid)
            if not wait:
                return _listening()
            give_up = time.monotonic() + READY_TIMEOUT
            while time.monotonic() < give_up:
                if _listening():
                    return True
                if _proc is not None and _proc.poll() is not None:
                    break
                time.sleep(0.2)
            log(logger, "pot server not ready, continuing without PO tokens", tail=_tail(_log_path(settings)))
            return False
    except Exception as err:  # noqa: BLE001 - best effort by design, see module docstring
        log(logger, "pot server failed to start", error=str(err))
        return False


def stop() -> None:
    global _proc
    with _lock:
        if _proc is not None and _proc.poll() is None:
            _proc.terminate()
            try:
                _proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _proc.kill()
        _proc = None
