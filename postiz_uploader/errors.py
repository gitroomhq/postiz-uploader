from __future__ import annotations


class JobError(Exception):
    """An expected failure. It becomes a `failed` result, never a crash.

    `retryable` tells the caller whether resubmitting the same job may succeed.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        stderr_tail: str | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.stderr_tail = stderr_tail

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "stderr_tail": self.stderr_tail,
        }


# Codes, mirrored in README section 4.3 and schema/v1/result.schema.json
UNSUPPORTED_VERSION = "UNSUPPORTED_VERSION"
INVALID_JOB = "INVALID_JOB"
SOURCE_HOST_NOT_ALLOWED = "SOURCE_HOST_NOT_ALLOWED"
DOWNLOAD_FAILED = "DOWNLOAD_FAILED"
INPUT_TOO_LARGE = "INPUT_TOO_LARGE"
UNSUPPORTED_INPUT = "UNSUPPORTED_INPUT"
PROBE_FAILED = "PROBE_FAILED"
DURATION_TOO_LONG = "DURATION_TOO_LONG"
ENCODE_FAILED = "ENCODE_FAILED"
UPLOAD_FAILED = "UPLOAD_FAILED"
TIMEOUT = "TIMEOUT"
INTERNAL = "INTERNAL"
