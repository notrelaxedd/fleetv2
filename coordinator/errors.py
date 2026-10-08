"""Exceptions raised by queue operations; the API maps them to HTTP codes."""


class QueueError(Exception):
    """Base class; `status` is the HTTP status the API should answer with."""

    status = 400

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class BadRequest(QueueError):
    """Invalid input (unknown kind, bad role, unknown setting key)."""

    status = 400


class Unauthorized(QueueError):
    """Bad or missing credentials."""

    status = 401


class Forbidden(QueueError):
    """Authenticated but not allowed (owner mismatch, bad Origin)."""

    status = 403


class NotFound(QueueError):
    """Worker or job does not exist."""

    status = 404


class Conflict(QueueError):
    """Lease token mismatch or wrong job status."""

    status = 409


class Upstream(QueueError):
    """An external source (Alpaca) could not be reached."""

    status = 502
