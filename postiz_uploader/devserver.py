"""A tiny file server for local development and tests.

GET  /<path>  serves files under the root directory
PUT  /<path>  writes the body to root/<path> (directories created as needed)

It stands in for the bucket: sources are fetched from it and outputs are PUT to it.
"""

from __future__ import annotations

import argparse
import os
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class Handler(SimpleHTTPRequestHandler):
    def do_PUT(self) -> None:  # noqa: N802 - http.server naming
        rel = self.path.split("?", 1)[0].lstrip("/")
        target = os.path.normpath(os.path.join(self.directory, rel))
        root = os.path.abspath(self.directory)
        if os.path.commonpath([root, target]) != root:
            self.send_error(403)
            return
        length = int(self.headers.get("Content-Length", "0"))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        remaining = length
        with open(target, "wb") as fh:
            while remaining > 0:
                chunk = self.rfile.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                fh.write(chunk)
                remaining -= len(chunk)
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        if os.environ.get("DEVSERVER_LOG"):
            super().log_message(format, *args)


def serve(root: str, host: str = "127.0.0.1", port: int = 0) -> tuple[ThreadingHTTPServer, threading.Thread]:
    root = os.path.abspath(root)
    server = ThreadingHTTPServer((host, port), partial(Handler, directory=root))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    server, thread = serve(args.root, args.host, args.port)
    print(f"serving {os.path.abspath(args.root)} on http://{args.host}:{args.port}", flush=True)
    try:
        thread.join()
    except KeyboardInterrupt:
        server.shutdown()
