"""Worker state directory, worker.conf and the status file (copied from v1, trimmed).

status.json is rewritten after every heartbeat, so it lives in the runtime directory
(FLEET_RUN_DIR, /run/fleet2 under systemd: tmpfs, no flash wear). Without that variable
it falls back to /run/fleet2 when that exists next to the default state dir, else to
the state dir itself (tests). Nothing else here is written more than once per job:
price data is never cached on disk (workers hold it in memory)."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

DEFAULT_STATE_DIR = "/var/lib/fleet2"
DEFAULT_RUN_DIR = "/run/fleet2"
CONF_NAME = "worker.conf"
STATUS_NAME = "status.json"
PENDING_POSTS_NAME = "pending_posts.json"


class ConfMissing(Exception):
    """worker.conf does not exist or is unusable."""


def state_dir() -> str:
    """State directory from FLEET_STATE_DIR, default /var/lib/fleet2."""
    return os.environ.get("FLEET_STATE_DIR") or DEFAULT_STATE_DIR


def conf_path(directory: str) -> str:
    return os.path.join(directory, CONF_NAME)


def run_dir(directory: str) -> str:
    """Where per-heartbeat files go: FLEET_RUN_DIR, /run/fleet2 for the default state
    dir when it exists, else the state dir."""
    env = os.environ.get("FLEET_RUN_DIR")
    if env:
        return env
    if os.path.abspath(directory) == DEFAULT_STATE_DIR and os.path.isdir(DEFAULT_RUN_DIR):
        return DEFAULT_RUN_DIR
    return directory


def status_path(directory: str) -> str:
    return os.path.join(run_dir(directory), STATUS_NAME)


def app_dir(directory: str) -> str:
    return os.path.join(directory, "app")


def pending_posts_path(directory: str) -> str:
    return os.path.join(directory, PENDING_POSTS_NAME)


def _write_private_json(path: str, data: Any, mode: int) -> None:
    """Write JSON atomically with the given file mode."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_conf(directory: str) -> dict[str, Any]:
    """Load worker.conf; raise ConfMissing when absent or incomplete."""
    path = conf_path(directory)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        raise ConfMissing(f"{path}: {exc}") from None
    if not isinstance(data, dict):
        raise ConfMissing(f"{path}: not a JSON object")
    for key in ("host_url", "worker_id", "worker_token"):
        if not data.get(key):
            raise ConfMissing(f"{path}: missing {key}")
    data["host_url"] = str(data["host_url"]).rstrip("/")
    return data


def save_conf(directory: str, conf: dict[str, Any]) -> None:
    """Write worker.conf with mode 0600."""
    _write_private_json(conf_path(directory), conf, 0o600)


def save_status(directory: str, status: dict[str, Any]) -> None:
    """Write status.json (best effort; errors are swallowed)."""
    try:
        _write_private_json(status_path(directory), status, 0o644)
    except OSError:
        pass


def load_status(directory: str) -> dict[str, Any] | None:
    try:
        with open(status_path(directory), "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def redacted(conf: dict[str, Any]) -> dict[str, Any]:
    """Copy of the conf with the token hidden."""
    out = dict(conf)
    token = str(out.get("worker_token", ""))
    out["worker_token"] = (token[:4] + "..." + token[-2:]) if len(token) > 8 else "***"
    return out


def save_pending_posts(directory: str, posts: list[dict[str, Any]]) -> None:
    """Persist unsent /complete and /fail calls (best effort) so a crash does not lose them."""
    path = pending_posts_path(directory)
    try:
        if not posts:
            if os.path.exists(path):
                os.unlink(path)
            return
        _write_private_json(path, posts, 0o600)
    except OSError:
        pass


def load_pending_posts(directory: str) -> list[dict[str, Any]]:
    """Unsent posts saved by an earlier run, or an empty list."""
    try:
        with open(pending_posts_path(directory), "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    return [p for p in data if isinstance(p, dict) and p.get("path") and isinstance(p.get("body"), dict)]
