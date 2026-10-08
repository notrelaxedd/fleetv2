"""Server-rendered HTML helpers (as in v1): the Jinja2 environment, error pages, paths.

Templates live in coordinator/templates and static files in coordinator/static.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from jinja2 import Environment, FileSystemLoader, select_autoescape

PACKAGE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = PACKAGE_DIR / "templates"
STATIC_DIR = PACKAGE_DIR / "static"
NO_STORE = {"Cache-Control": "no-store", "Pragma": "no-cache"}
STATUS_TEXT = {400: "Bad request", 401: "Not signed in", 403: "Forbidden", 404: "Not found",
               409: "Conflict", 413: "Too large", 500: "Server error", 502: "Alpaca unavailable"}

env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=select_autoescape(["html"]))


def render(name: str, context: dict[str, Any], status_code: int = 200) -> HTMLResponse:
    """Render a template with no-store headers (pages change every few seconds)."""
    return HTMLResponse(env.get_template(name).render(**context), status_code=status_code, headers=NO_STORE)


def error_response(request: Request, status: int, message: str) -> Response:
    """JSON {"detail"} under /api and /dl; a small plain page on the dashboard."""
    path = request.url.path
    if path.startswith("/api/") or path.startswith("/dl/") or "application/json" in request.headers.get("accept", ""):
        return JSONResponse({"detail": message}, status_code=status)
    title = STATUS_TEXT.get(status, "Error")
    body = (f"<!doctype html><meta charset=utf-8><title>{title}</title>"
            f"<body style='background:#0D1013;color:#E8EDF2;font-family:sans-serif;padding:24px'>"
            f"<h1>{title}</h1><p>{_escape(message)}</p></body>")
    return HTMLResponse(body, status_code=status, headers=NO_STORE)


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
