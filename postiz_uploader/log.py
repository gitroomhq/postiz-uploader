from __future__ import annotations

import json
import logging
import os
import sys
import time
from urllib.parse import urlsplit, urlunsplit


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": round(time.time(), 3),
            "level": record.levelname.lower(),
            "msg": record.getMessage(),
        }
        extra = getattr(record, "extra", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging() -> None:
    level = os.environ.get("LOG_LEVEL", "info").upper()
    root = logging.getLogger()
    if any(isinstance(h.formatter, JsonFormatter) for h in root.handlers):
        root.setLevel(level)
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.handlers = [handler]
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log(logger: logging.Logger, msg: str, **fields) -> None:
    """One structured line. Fields land at the top level of the JSON."""
    logger.info(msg, extra={"extra": fields})


def redact_url(url: str) -> str:
    """Drop the query string: presigned URLs carry their signature there."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return "<invalid-url>"
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
