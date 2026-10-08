"""Tiny JSON HTTP client over urllib.

Two exception types: HttpError for an answer with a non-2xx status (carries status
and detail) and HttpConnectionError when no answer arrived at all (refused, timeout,
DNS, reset). Proxies from the environment are ignored on purpose: workers talk to the
host over the tailnet.

get_json_etag() is the conditional GET used for the games cache: it sends
If-None-Match and returns a Response whose status is 304 (body None) when the server
says the document is unchanged.
"""

from __future__ import annotations

import http.client
import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

DEFAULT_TIMEOUT = 4.0


class HttpError(Exception):
    """The server answered with a non-2xx status."""

    def __init__(self, status: int, detail: str, url: str = "") -> None:
        super().__init__(f"HTTP {status} from {url}: {detail}")
        self.status = status
        self.detail = detail
        self.url = url


class HttpConnectionError(Exception):
    """No usable answer: connection refused, timeout, DNS failure, reset."""


@dataclass
class Response:
    """Status, decoded JSON body (None for 304 or an empty body) and the ETag header."""

    status: int
    body: Any
    etag: str | None


_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _headers(token: str | None, has_body: bool, extra: dict[str, str] | None = None) -> dict[str, str]:
    headers = {"Accept": "application/json", "User-Agent": "fleet-worker"}
    if has_body:
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = "Bearer " + token
    if extra:
        headers.update(extra)
    return headers


def _detail_from_body(raw: bytes) -> str:
    text = raw.decode("utf-8", "replace")
    try:
        data = json.loads(text)
    except ValueError:
        return text.strip()[:500]
    if isinstance(data, dict) and "detail" in data:
        return str(data["detail"])
    return text.strip()[:500]


def _open(
    method: str,
    url: str,
    body: Any,
    token: str | None,
    timeout: float,
    extra_headers: dict[str, str] | None = None,
    allow: tuple[int, ...] = (),
) -> tuple[int, bytes, str | None]:
    """One request; (status, raw body, etag). Statuses in `allow` (e.g. 304) are
    returned instead of raised."""
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method, headers=_headers(token, data is not None, extra_headers))
    try:
        with _opener.open(req, timeout=timeout) as resp:
            return resp.status, resp.read(), resp.headers.get("ETag")
    except urllib.error.HTTPError as exc:
        raw = b""
        try:
            raw = exc.read()
        except Exception:
            pass
        if exc.code in allow:
            return exc.code, raw, exc.headers.get("ETag")
        raise HttpError(exc.code, _detail_from_body(raw), url) from None
    except (urllib.error.URLError, http.client.HTTPException, socket.timeout, OSError) as exc:
        raise HttpConnectionError(f"{method} {url}: {exc}") from None


def _decode(method: str, url: str, raw: bytes) -> Any:
    if not raw.strip():
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except ValueError as exc:
        raise HttpConnectionError(f"{method} {url}: invalid JSON in response: {exc}") from None


def request_bytes(
    method: str,
    url: str,
    body: Any = None,
    token: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> bytes:
    """Perform one request and return the raw response body."""
    return _open(method, url, body, token, timeout)[1]


def request_json(
    method: str,
    url: str,
    body: Any = None,
    token: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Any:
    """Perform one request and decode the JSON body (an empty body decodes to None)."""
    return _decode(method, url, request_bytes(method, url, body=body, token=token, timeout=timeout))


def get_json(url: str, token: str | None = None, timeout: float = DEFAULT_TIMEOUT) -> Any:
    """GET a JSON document."""
    return request_json("GET", url, token=token, timeout=timeout)


def get_json_etag(
    url: str,
    token: str | None = None,
    etag: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Response:
    """Conditional GET: send If-None-Match when an etag is given. Returns Response(200,
    body, etag) or Response(304, None, etag) when unchanged; other statuses raise."""
    extra = {"If-None-Match": etag} if etag else None
    status, raw, new_etag = _open("GET", url, None, token, timeout, extra_headers=extra, allow=(304,))
    if status == 304:
        return Response(304, None, new_etag or etag)
    return Response(status, _decode("GET", url, raw), new_etag)


def post_json(url: str, body: Any, token: str | None = None, timeout: float = DEFAULT_TIMEOUT) -> Any:
    """POST a JSON body and decode the JSON answer."""
    return request_json("POST", url, body=body, token=token, timeout=timeout)


def get_bytes(url: str, token: str | None = None, timeout: float = 60.0) -> bytes:
    """GET a binary document (tarballs)."""
    return request_bytes("GET", url, token=token, timeout=timeout)
